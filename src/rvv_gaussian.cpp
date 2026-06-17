#include "gaussian.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

// ── Kernel constants ──────────────────────────────────────────────────────
// This project uses a fixed 5×5 kernel with a total weight of 273.
// Please keep these values exactly as they are, because the kernel is not
// separable and cannot be recreated by applying two 1D passes.

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
// RVV notes:
//   - Accumulator: vint32m4  (LMUL=4, handles up to 4*VLEN/32 elements/iter)
//   - Pixel load: vuint8m1 widened to vint32m4 via two-step extension
//   - Divide: widened to vint64m8, then narrowed back i64→i32→i16→u8
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
            // Set the active vector length (VL) for 32-bit elements using LMUL=4.
            // LMUL=4 is required so the same VL stays valid through the widened
            // i64m8 divide stage further down (vl is reused, not recomputed).
            // If VLEN changes (128, 256, or 512 bits), the returned VL changes
            // automatically, so this same line works without modification.
            size_t vl = __riscv_vsetvl_e32m4((size_t)(vec_end - x));

            // Initialize the accumulator vector to zero before the 25-tap sum.
            // LMUL=4 matches the accumulator width that vmacc writes into below.
            // A larger VLEN zeros more accumulator lanes per call, but the
            // per-pixel result stays identical regardless of VLEN.
            vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

            // 25 taps: iterate over all (ky, kx) pairs
            for (int ky = 0; ky < KSIZE; ky++) {
                const uint8_t* row = src_rows[ky];   // points at column 0
                for (int kx = 0; kx < KSIZE; kx++) {
                    int16_t w = KERNEL2D[ky][kx];
                    if (w == 0) continue;             // skip zero weights (none here, but defensive)

                    // Load VL unsigned 8-bit pixels for this tap, offset by (kx - RADIUS).
                    // LMUL=1 keeps register pressure low since this is the raw,
                    // un-widened pixel data coming straight from the input image.
                    // A larger VLEN simply loads more pixels per iteration
                    // automatically; the loaded values and final output don't change.
                    vuint8m1_t vp8 = __riscv_vle8_v_u8m1(row + x + (kx - RADIUS), vl);

                    // Widen 8-bit pixels to 16-bit before multiplication to avoid overflow.
                    // LMUL increases from m1 to m2 because widening doubles the
                    // register group needed to keep the same VL.
                    // A larger VLEN widens more elements each iteration but
                    // produces identical output.
                    vuint16m2_t vp16 = __riscv_vwcvtu_x_x_v_u16m2(vp8, vl);
                    // Widen again, 16-bit to 32-bit, to match the accumulator's width.
                    // LMUL increases from m2 to m4, the same doubling rule
                    // applied a second time to keep VL constant.
                    // VLEN size only changes how many pixels widen together each
                    // pass, never the resulting values.
                    vuint32m4_t vp32u = __riscv_vwcvtu_x_x_v_u32m4(vp16, vl);
                    // Reinterpret the unsigned widened pixels as signed so that
                    // vmacc_vx, which expects a signed accumulator, will accept them.
                    // LMUL stays m4 — a reinterpret never changes the register
                    // grouping, only the type used to read the same bits.
                    // VLEN has no effect here since no data is moved or computed.
                    vint32m4_t  vp32  = __riscv_vreinterpret_v_u32m4_i32m4(vp32u);

                    // Multiply vector elements by this tap's kernel coefficient and accumulate.
                    // LMUL=4 is required because the accumulation uses 32-bit
                    // values, matching the widened pixel data above.
                    // With a larger VLEN, more pixels are accumulated per
                    // iteration while producing the same Gaussian result.
                    vacc = __riscv_vmacc_vx_i32m4(vacc, (int32_t)w, vp32, vl);
                }
            }

            // ── Divide by 273 via widening i64 multiply-shift ─────────────
            // max(vacc) = 69615; 69615 * 122911 = 8,557,513,665 < 2^63 ✓
            // Sign-extend the 32-bit accumulator to 64-bit before the
            // reciprocal-multiply divide, since the product needs more range.
            // LMUL doubles from m4 to m8 to keep VL constant as the element
            // width doubles again (m8 is the largest LMUL the spec allows).
            // A different VLEN changes how many lanes are extended per call,
            // never the extended values themselves.
            vint64m8_t v64  = __riscv_vsext_vf2_i64m8(vacc, vl);
            // Multiply the accumulator by the reciprocal constant (122911)
            // to perform the ÷273 approximation.
            // LMUL=8 is required because the product (up to ~8.5 billion)
            // only fits in the full 64-bit range.
            // A different VLEN changes how many of these multiplies run per
            // instruction, not the math or the result.
            vint64m8_t vmul = __riscv_vmul_vx_i64m8(v64, RECIP_M, vl);
            // Arithmetic right-shift by 25, the ">> 25" half of the
            // "x * 122911 >> 25 ≈ x / 273" trick.
            // LMUL stays at m8 to match the 64-bit product being shifted.
            // VLEN only changes how many lanes are shifted per call; the
            // shift amount and result stay fixed.
            vint64m8_t vres = __riscv_vsra_vx_i64m8(vmul, (unsigned)RECIP_S, vl);

            // Clamp [0, 255] and narrow i64 → u8 in three steps
            // Clamp the divide result to a minimum of 0.
            // LMUL stays m8 since clamping happens before any narrowing.
            // VLEN changes only how many lanes are clamped per call, never
            // the clamped values themselves.
            vres = __riscv_vmax_vx_i64m8(vres, 0,   vl);
            // Clamp the divide result to a maximum of 255.
            // LMUL stays m8, same reasoning as the min-clamp above.
            // VLEN changes only how many lanes are clamped per call, never
            // the clamped values themselves.
            vres = __riscv_vmin_vx_i64m8(vres, 255, vl);

            // Narrow the clamped 64-bit result down to 32-bit.
            // LMUL halves from m8 to m4 because narrowing halves the
            // register group to keep VL constant as SEW halves.
            // VLEN changes how many lanes are narrowed per call, not the
            // output values, since clamping already happened.
            vint32m4_t  vn32 = __riscv_vncvt_x_x_w_i32m4(vres, vl);
            // Narrow again, 32-bit to 16-bit, continuing the descent back
            // toward a single output byte.
            // LMUL halves from m4 to m2, mirroring the widening done
            // earlier in reverse.
            // VLEN changes only how many lanes move per call, not the result.
            vint16m2_t  vn16 = __riscv_vncvt_x_x_w_i16m2(vn32, vl);
            // Narrow from 16-bit down to the final 8-bit pixel value; the
            // reinterpret just before it relabels the signed 16-bit result
            // as unsigned so this narrow's unsigned-source requirement is met.
            // LMUL halves from m2 to m1, landing back at the same width the
            // pixels were originally loaded at (the vle8 call above).
            // VLEN changes how many output pixels are produced per call,
            // not their values.
            vuint8m1_t  vn8  = __riscv_vncvt_x_x_w_u8m1(
                                   __riscv_vreinterpret_v_i16m2_u16m2(vn16), vl);

            // Store VL finished 8-bit pixels back to the output row.
            // LMUL=1 matches the load at the top of this loop, keeping the
            // pipeline symmetric (m1 in, m1 out).
            // A larger VLEN stores more finished pixels per call
            // automatically; the source code itself never changes.
            __riscv_vse8_v_u8m1(out_row + x, vn8, vl);
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
