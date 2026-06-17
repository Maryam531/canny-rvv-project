// rvv_gaussian_lmul1.cpp
//
// LMUL SWEEP — LMUL=1 variant.
//
// Vector type chain (all intrinsics used here are base RVV ops present
// since the earliest stable intrinsic specs — no segment-extraction
// (vget), no vsetvlmax, no triple-widen chains):
//
//   pixel load          : vuint8m1_t   (LMUL=1)
//   widen u8  -> u16     : vuint16m2_t  (LMUL=2)
//   widen u16 -> u32     : vuint32m4_t  (LMUL=4)
//   accumulator           : vuint32m4_t  (LMUL=4)
//   divide (see below)    : vuint32m4_t  (LMUL=4, NO further widen needed)
//
// DIVIDE-BY-273 WITHOUT WIDENING TO i64:
//   The original approach widened the i32 accumulator to i64 to do a
//   reciprocal-multiply-then-shift (M=122911, shift=25). That extra widen
//   step is what makes LMUL=2 and LMUL=4 awkward (i32m8 -> i64m16 is
//   illegal, and so is i32m16 for LMUL=4's two-widen load chain).
//
//   Instead we pick a DIFFERENT reciprocal constant where the shift
//   amount is exactly 32. For an unsigned 32-bit value x, the high 32
//   bits of the 32x32->64 product x*M are exactly floor(x*M / 2^32),
//   which is precisely what the RVV base intrinsic vmulhu computes —
//   entirely within 32-bit vector registers, no widening at all:
//
//       floor(x / 273) == (x * 15732481) >> 32   for all x in [0, 69615]
//
//   This was exhaustively verified for every x in [0, 69615] (the max
//   possible accumulator value, 255*273) with zero mismatches.
//
// This removes the i64 stage entirely, which means the divide step now
// imposes NO additional LMUL constraint beyond what the load/widen chain
// already requires — at any LMUL level, if the load+widen-to-u32 chain is
// legal, the divide is automatically legal too (same width, same LMUL,
// single vmulhu call).

#include "gaussian.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

// ── Kernel constants (unchanged from reference) ───────────────────────────
static constexpr int32_t KERNEL_SUM = 273;
static constexpr int     RADIUS     = 2;
static constexpr int     KSIZE      = 2 * RADIUS + 1;   // 5

static constexpr int16_t KERNEL2D[KSIZE][KSIZE] = {
    { 1,  4,  7,  4, 1},
    { 4, 16, 26, 16, 4},
    { 7, 26, 41, 26, 7},
    { 4, 16, 26, 16, 4},
    { 1,  4,  7,  4, 1}
};

// Original reciprocal pair (shift=25), kept ONLY for the scalar border
// fallback so border pixels stay bit-exact with gaussian.cpp's own scalar
// path (gaussian.cpp uses its own divide; this is a local scalar helper
// used only for the few border pixels outside the vectorised region).
static constexpr int64_t RECIP_M_SCALAR = 122911;
static constexpr int     RECIP_S_SCALAR = 25;

// Vector-path reciprocal pair (shift=32 exactly, so a single vmulhu call
// gives the answer with no extra shift instruction and no widening).
// Exhaustively verified for all x in [0, 69615]: 0 mismatches against
// floor(x/273).
static constexpr uint32_t RECIP_M_VEC = 15732481u;

static inline int32_t div273_scalar(int32_t x) {
    return (int32_t)(((int64_t)x * RECIP_M_SCALAR) >> RECIP_S_SCALAR);
}

static inline size_t round_up64(size_t n) {
    return (n + 63u) & ~size_t(63);
}

// ── Scalar pixel (border fallback, zero-padding) ──────────────────────────
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

static int s_ring_W = 0;
static int s_ring_H = 0;

void gaussian_blur_rvv_init(int W, int H) {
    s_ring_W = W;
    s_ring_H = H;
}

void gaussian_blur_rvv_free() {
    s_ring_W = 0;
    s_ring_H = 0;
}

// ── Core: 25-tap 2D RVV convolution — LMUL=1 load width ──────────────────
static void gaussian_blur_rvv_core(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   int W, int H)
{
    for (int y = 0; y < H; y++) {
        uint8_t* out_row = out + (size_t)y * W;

        if (y < RADIUS || y >= H - RADIUS) {
            for (int x = 0; x < W; x++)
                out_row[x] = scalar_pixel(in, W, H, x, y);
            continue;
        }

        for (int x = 0; x < RADIUS; x++)
            out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W; x++)
            out_row[x] = scalar_pixel(in, W, H, x, y);

        const uint8_t* src_rows[KSIZE];
        for (int ky = -RADIUS; ky <= RADIUS; ky++)
            src_rows[ky + RADIUS] = in + (size_t)(y + ky) * W;

        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            // LMUL=1 root: vsetvl at e8m1 sized off the pixel load width.
            size_t vl = __riscv_vsetvl_e8m1((size_t)(vec_end - x));

            vuint32m4_t vacc = __riscv_vmv_v_x_u32m4(0, vl);

            for (int ky = 0; ky < KSIZE; ky++) {
                const uint8_t* row = src_rows[ky];
                for (int kx = 0; kx < KSIZE; kx++) {
                    int16_t w = KERNEL2D[ky][kx];
                    if (w == 0) continue;

                    // LMUL=1 pixel load
                    vuint8m1_t vp8 = __riscv_vle8_v_u8m1(row + x + (kx - RADIUS), vl);

                    // widen x2: u8m1 -> u16m2
                    vuint16m2_t vp16 = __riscv_vwcvtu_x_x_v_u16m2(vp8, vl);
                    // widen x2: u16m2 -> u32m4
                    vuint32m4_t vp32 = __riscv_vwcvtu_x_x_v_u32m4(vp16, vl);

                    // weight is always positive (KERNEL2D has no negative
                    // taps), so an unsigned multiply-accumulate is exact.
                    vacc = __riscv_vmacc_vx_u32m4(vacc, (uint32_t)w, vp32, vl);
                }
            }

            // Divide by 273 via a single high-multiply — no widening.
            // Base intrinsic, stable across RVV intrinsic spec versions.
            vuint32m4_t vres = __riscv_vmulhu_vx_u32m4(vacc, RECIP_M_VEC, vl);

            // Clamp [0, 255]. vres is unsigned and already >= 0, so only
            // the upper bound needs clamping (acc max 69615 -> result max
            // 254, which is already <= 255 for this kernel, but clamp is
            // kept for safety/clarity in case constants change).
            vres = __riscv_vminu_vx_u32m4(vres, 255u, vl);

            // Narrow u32 -> u16 -> u8 (two single-step narrows, both base
            // intrinsics, no widening involved).
            vuint16m2_t vn16 = __riscv_vncvt_x_x_w_u16m2(vres, vl);
            vuint8m1_t  vn8  = __riscv_vncvt_x_x_w_u8m1(vn16, vl);

            __riscv_vse8_v_u8m1(out_row + x, vn8, vl);
            x += (int)vl;
        }
    }
}

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
