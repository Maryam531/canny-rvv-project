#include "gaussian.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

// ════════════════════════════════════════════════════════════════════════
//  LMUL EXPERIMENT VERSION: LMUL4
//  Root load:    vuint8m2_t
//  Widen step 1: vuint16m4_t
//  Accumulator:  vint32m8_t
//
//  This is the "effective LMUL" reporting convention: since this is a
//  widening kernel (u8 -> u16 -> u32), we label the version by the LMUL of
//  the dominant accumulator/compute step, not the literal root load type.
//  Every vector type in the 25-tap MAC loop is exactly one LMUL step ABOVE
//  the LMUL2 baseline (m1->m2, m2->m4, m4->m8).
//
//  IMPORTANT ARCHITECTURAL NOTE — why the divide stage looks different here:
//  The divide-by-273 step needs to widen the i32 accumulator to i64 for
//  correctness (max product 69615 * 122911 ≈ 8.5e9, overflows i32). RVV's
//  widening rule is SEW -> 2*SEW requires LMUL -> 2*LMUL. At the LMUL2
//  baseline that's i32m4 -> i64m8, which is valid (m8 is the max group
//  size). At THIS tier the accumulator is already i32m8, and there is no
//  m16 group — RVV does not define LMUL beyond 8. So vint32m8_t cannot be
//  widened to i64 in a single instruction.
//
//  Fix: split the m8 accumulator into its two constituent m4 halves with
//  __riscv_vget_v_i32m8_i32m4(), widen EACH half to i64m8 independently
//  (m4 -> m8 is a valid widening step), do the multiply-shift-clamp-narrow
//  on each half, and store both halves contiguously. The 25-tap
//  accumulation loop itself still runs at full m8 width — only the final
//  divide/narrow epilogue is split — so the wide-load/wide-MAC benefit of
//  this LMUL tier is preserved; only the unavoidable last step pays a
//  two-way split.
// ════════════════════════════════════════════════════════════════════════

// ── Kernel constants ──────────────────────────────────────────────────────
// The assignment specifies this exact non-separable 5×5 kernel, sum = 273.
// Do NOT change these values — the kernel is NOT the outer product of K1D
// and cannot be reproduced by two separable 1D passes.
static constexpr int32_t KERNEL_SUM = 273;
static constexpr int     RADIUS     = 2;
static constexpr int     KSIZE      = 2 * RADIUS + 1;   // 5

// Flattened row-major kernel weights (int16 for use with vmacc).
// Row 0 is the top row, row 4 is the bottom row.
// max accumulator = 255 * 273 = 69 615  (fits int32, comfortably)
static constexpr int16_t KERNEL2D[KSIZE][KSIZE] = {
    { 1,  4,  7,  4, 1},
    { 4, 16, 26, 16, 4},
    { 7, 26, 41, 26, 7},
    { 4, 16, 26, 16, 4},
    { 1,  4,  7,  4, 1}
};

// ── Reciprocal-multiply for ÷273 ─────────────────────────────────────────
// floor(x / 273) == floor(x * 122911 >> 25)
// Exhaustively verified for x in [0, 69615] (max scalar acc = 255*273):
//   0 errors.
// 69615 * 122911 = 8,557,513,665 < 2^63  → safe as int64 intermediate.
static constexpr int64_t RECIP_M = 122911;
static constexpr int     RECIP_S = 25;

static inline int32_t div273_scalar(int32_t x) {
    return (int32_t)(((int64_t)x * RECIP_M) >> RECIP_S);
}

static inline size_t round_up64(size_t n) {
    return (n + 63u) & ~size_t(63);
}

// ── Scalar pixel (border fallback, zero-padding) ──────────────────────────
// Uses the identical KERNEL2D and div273 as the scalar reference so that
// border pixels are bit-exact with gaussian.cpp.
static inline uint8_t scalar_pixel(const uint8_t* in, int W, int H, int x, int y) {
    int32_t acc = 0;
    for (int ky = -RADIUS; ky <= RADIUS; ky++) {
        const int sy = y + ky;
        if (sy < 0 || sy >= H) continue;
        const uint8_t* row = in + sy * W;
        for (int kx = -RADIUS; kx <= RADIUS; kx++) {
            const int sx = x + kx;
            if (sx < 0 || sx >= W) continue;
            acc += (int32_t)row[sx] * (int32_t)KERNEL2D[ky + RADIUS][kx + RADIUS];
        }
    }
    const int32_t r = div273_scalar(acc);
    return (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
}

// ── Static persistent scratch (row cache, allocated once via init) ────────
// We cache the 5 source rows that contribute to the current output row as
// uint8 pointers — no format conversion needed, since we load u8 directly
// in the vector loop. No ring buffer arithmetic required: for interior rows
// we just point into the input image.
static int s_ring_W = 0;
static int s_ring_H = 0;

void gaussian_blur_rvv_init(int W, int H) {
    // No persistent allocation needed for the 2D path — we read directly
    // from the input image. Keep W/H for the assert in _into().
    s_ring_W = W;
    s_ring_H = H;
}

void gaussian_blur_rvv_free() {
    s_ring_W = 0;
    s_ring_H = 0;
}

// ── Core: true 25-tap 2D RVV convolution ─────────────────────────────────
//
// Algorithm:
//   For each interior output pixel (x, y) [i.e. RADIUS ≤ x,y < W,H-RADIUS]:
//     acc = Σ_{ky=-2}^{2} Σ_{kx=-2}^{2}  in[y+ky][x+kx] * KERNEL2D[ky+2][kx+2]
//     out = clamp(acc * RECIP_M >> RECIP_S, 0, 255)
//
//   The inner kx loop is vectorised: for each kernel row ky, we load a
//   vector of pixels (shifted by kx in [-2..2]) and multiply-accumulate
//   into a vint32 accumulator.  After all 25 taps, we widen to i64, apply
//   the reciprocal multiply-shift, clamp, and narrow back to u8.
//
//   Border rows/columns (within RADIUS of any edge) fall back to the scalar
//   path, which uses the same KERNEL2D and div273, giving bit-exact results.
//
// Overflow analysis:
//   max acc = 255 * 273 = 69 615 → fits int32 (no widening needed for acc).
//   For the divide: 69615 * 122911 = 8,557,513,665 < 2^63 → i64 is safe.
//
// RVV notes (LMUL4 version):
//   - Root load:    vuint8m2   (2 native registers' worth of u8 lanes)
//   - Widen step 1: vuint16m4  (u8m2 -> u16m4, one widening conversion)
//   - Widen step 2: vuint32m8  (u16m4 -> u32m8, second widening conversion)
//   - Accumulator:  vint32m8
//   - Divide:       i32m8 split into two i32m4 halves via vget, each
//                    independently widened to vint64m8 (m4 -> m8 is the
//                    valid widening step; m8 -> m16 does not exist)
//   - Each kx offset is a separate vle8 load (stride=1, offset pointer);
//     no gather needed because kx offsets are small constants.

static void gaussian_blur_rvv_core(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   int W, int H)
{
    for (int y = 0; y < H; y++) {
        uint8_t* out_row = out + (size_t)y * W;

        // ── Border rows: pure scalar ──────────────────────────────────────
        if (y < RADIUS || y >= H - RADIUS) {
            for (int x = 0; x < W; x++)
                out_row[x] = scalar_pixel(in, W, H, x, y);
            continue;
        }

        // ── Interior row ──────────────────────────────────────────────────
        // Left and right border columns: scalar
        for (int x = 0; x < RADIUS; x++)
            out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W; x++)
            out_row[x] = scalar_pixel(in, W, H, x, y);

        // Hoist the 5 source-row base pointers for this output row.
        // y >= RADIUS so all sy = y + ky are in [0, H-1] for the interior.
        const uint8_t* src_rows[KSIZE];
        for (int ky = -RADIUS; ky <= RADIUS; ky++)
            src_rows[ky + RADIUS] = in + (size_t)(y + ky) * W;

        // ── Vectorised interior columns ────────────────────────────────────
        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            // Use e32m8 for the accumulator — the widest 32-bit group RVV
            // defines, giving this tier its largest per-iteration vl.
            size_t vl = __riscv_vsetvl_e32m8((size_t)(vec_end - x));

            vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vl);

            // 25 taps: iterate over all (ky, kx) pairs
            for (int ky = 0; ky < KSIZE; ky++) {
                const uint8_t* row = src_rows[ky];   // points at column 0
                for (int kx = 0; kx < KSIZE; kx++) {
                    int16_t w = KERNEL2D[ky][kx];
                    if (w == 0) continue;             // skip zero weights (none here, but defensive)

                    // Load vl u8 pixels starting at row[x + (kx - RADIUS)].
                    // x >= RADIUS and x+vl-1 <= W-RADIUS-1, so x+(kx-RADIUS)
                    // is always in [0, W-1] for all kx in [0, KSIZE).
                    vuint8m2_t vp8 = __riscv_vle8_v_u8m2(row + x + (kx - RADIUS), vl);

                    // Zero-extend u8 → u16 → u32, then view as i32 for vmacc.
                    // Pixel values 0-255 never set the sign bit of i16/i32.
                    vuint16m4_t vp16 = __riscv_vwcvtu_x_x_v_u16m4(vp8, vl);
                    vuint32m8_t vp32u = __riscv_vwcvtu_x_x_v_u32m8(vp16, vl);
                    vint32m8_t  vp32  = __riscv_vreinterpret_v_u32m8_i32m8(vp32u);

                    vacc = __riscv_vmacc_vx_i32m8(vacc, (int32_t)w, vp32, vl);
                }
            }

            // ── Divide by 273 via widening i64 multiply-shift ─────────────
            // max(vacc) = 69615; 69615 * 122911 = 8,557,513,665 < 2^63 ✓
            //
            // vacc is i32m8 — there is no m16 group, so we cannot widen the
            // whole thing to i64 in one step (m8 -> m16 is not valid). We
            // split into two i32m4 halves (vget index 0 / 1) and widen each
            // half to i64m8 separately (m4 -> m8 IS valid). The split point
            // is vl/2 elements in; if vl is odd (only possible on the very
            // last partial chunk of a row) the second half simply carries
            // one fewer logical element and we narrow each half with its
            // own correctly-sized vl.
            size_t vl_lo = vl / 2;
            size_t vl_hi = vl - vl_lo;

            vint32m4_t vacc_lo = __riscv_vget_v_i32m8_i32m4(vacc, 0);
            vint32m4_t vacc_hi = __riscv_vget_v_i32m8_i32m4(vacc, 1);

            // ── Low half ───────────────────────────────────────────────
            vint64m8_t v64_lo  = __riscv_vsext_vf2_i64m8(vacc_lo, vl_lo);
            vint64m8_t vmul_lo = __riscv_vmul_vx_i64m8(v64_lo, RECIP_M, vl_lo);
            vint64m8_t vres_lo = __riscv_vsra_vx_i64m8(vmul_lo, (unsigned)RECIP_S, vl_lo);
            vres_lo = __riscv_vmax_vx_i64m8(vres_lo, 0,   vl_lo);
            vres_lo = __riscv_vmin_vx_i64m8(vres_lo, 255, vl_lo);

            vint32m4_t vn32_lo = __riscv_vncvt_x_x_w_i32m4(vres_lo, vl_lo);
            vint16m2_t vn16_lo = __riscv_vncvt_x_x_w_i16m2(vn32_lo, vl_lo);
            vuint8m1_t vn8_lo  = __riscv_vncvt_x_x_w_u8m1(
                                     __riscv_vreinterpret_v_i16m2_u16m2(vn16_lo), vl_lo);
            __riscv_vse8_v_u8m1(out_row + x, vn8_lo, vl_lo);

            // ── High half ──────────────────────────────────────────────
            if (vl_hi > 0) {
                vint64m8_t v64_hi  = __riscv_vsext_vf2_i64m8(vacc_hi, vl_hi);
                vint64m8_t vmul_hi = __riscv_vmul_vx_i64m8(v64_hi, RECIP_M, vl_hi);
                vint64m8_t vres_hi = __riscv_vsra_vx_i64m8(vmul_hi, (unsigned)RECIP_S, vl_hi);
                vres_hi = __riscv_vmax_vx_i64m8(vres_hi, 0,   vl_hi);
                vres_hi = __riscv_vmin_vx_i64m8(vres_hi, 255, vl_hi);

                vint32m4_t vn32_hi = __riscv_vncvt_x_x_w_i32m4(vres_hi, vl_hi);
                vint16m2_t vn16_hi = __riscv_vncvt_x_x_w_i16m2(vn32_hi, vl_hi);
                vuint8m1_t vn8_hi  = __riscv_vncvt_x_x_w_u8m1(
                                         __riscv_vreinterpret_v_i16m2_u16m2(vn16_hi), vl_hi);
                __riscv_vse8_v_u8m1(out_row + x + vl_lo, vn8_hi, vl_hi);
            }

            x += (int)vl;
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────

void gaussian_blur_rvv_into(const Image& img, uint8_t* out_data) {
    assert(img.data  != nullptr && out_data != nullptr);
    assert(s_ring_W  == img.width && s_ring_H == img.height);
    gaussian_blur_rvv_core(img.data, out_data, img.width, img.height);
}

Image gaussian_blur_rvv(const Image& img) {
    assert(img.data   != nullptr);
    assert(img.width  >  0);
    assert(img.height >  0);

    const int W = img.width;
    const int H = img.height;

    Image out;
    out.width  = W;
    out.height = H;
    out.data   = (uint8_t*)aligned_alloc(64, round_up64((size_t)W * H));
    assert(out.data != nullptr);

    gaussian_blur_rvv_core(img.data, out.data, W, H);
    return out;
}
