// rvv_gaussian_lmul2.cpp
//
// LMUL SWEEP — LMUL=2 variant.
//
// Identical structure to rvv_gaussian_lmul1.cpp — every vector type is
// simply bumped up one LMUL notch (the root pixel load moves from m1 to
// m2, and everything downstream follows the standard widening rule):
//
//   pixel load          : vuint8m2_t   (LMUL=2)
//   widen u8  -> u16     : vuint16m4_t  (LMUL=4)
//   widen u16 -> u32     : vuint32m8_t  (LMUL=8)
//   accumulator           : vuint32m8_t  (LMUL=8)
//   divide                 : vuint32m8_t  (LMUL=8, single vmulhu, no widen)
//
// LMUL=8 is the maximum legal group multiplier in RVV, so this chain is
// exactly at the ceiling but still fully legal — no splitting required.
// (This is precisely why removing the i64-widen divide step matters: the
// old i64-based approach would have needed i64m16 here, which is illegal.
// The vmulhu approach never leaves u32, so it never needs to widen past
// m8 at all.)
//
// Divide-by-273 constant (RECIP_M_VEC = 15732481, implicit shift = 32 via
// vmulhu) is identical to and exhaustively verified the same way as in
// rvv_gaussian_lmul1.cpp — see that file's header comment for the proof.
//
// Public API names are identical to rvv_gaussian.cpp — build this as its
// own executable, do not link alongside rvv_gaussian.cpp or
// rvv_gaussian_lmul1.cpp (duplicate symbol link errors).

#include "gaussian.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

static constexpr int32_t KERNEL_SUM = 273;
static constexpr int     RADIUS     = 2;
static constexpr int     KSIZE      = 2 * RADIUS + 1;

static constexpr int16_t KERNEL2D[KSIZE][KSIZE] = {
    { 1,  4,  7,  4, 1},
    { 4, 16, 26, 16, 4},
    { 7, 26, 41, 26, 7},
    { 4, 16, 26, 16, 4},
    { 1,  4,  7,  4, 1}
};

static constexpr int64_t RECIP_M_SCALAR = 122911;
static constexpr int     RECIP_S_SCALAR = 25;
static constexpr uint32_t RECIP_M_VEC   = 15732481u;

static inline int32_t div273_scalar(int32_t x) {
    return (int32_t)(((int64_t)x * RECIP_M_SCALAR) >> RECIP_S_SCALAR);
}

static inline size_t round_up64(size_t n) {
    return (n + 63u) & ~size_t(63);
}

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

// ── Core: 25-tap 2D RVV convolution — LMUL=2 load width ───────────────────
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
            // LMUL=2 root: vsetvl at e8m2 sized off the pixel load width.
            size_t vl = __riscv_vsetvl_e8m2((size_t)(vec_end - x));

            vuint32m8_t vacc = __riscv_vmv_v_x_u32m8(0, vl);

            for (int ky = 0; ky < KSIZE; ky++) {
                const uint8_t* row = src_rows[ky];
                for (int kx = 0; kx < KSIZE; kx++) {
                    int16_t w = KERNEL2D[ky][kx];
                    if (w == 0) continue;

                    // LMUL=2 pixel load
                    vuint8m2_t vp8 = __riscv_vle8_v_u8m2(row + x + (kx - RADIUS), vl);

                    // widen x2: u8m2 -> u16m4
                    vuint16m4_t vp16 = __riscv_vwcvtu_x_x_v_u16m4(vp8, vl);
                    // widen x2: u16m4 -> u32m8
                    vuint32m8_t vp32 = __riscv_vwcvtu_x_x_v_u32m8(vp16, vl);

                    vacc = __riscv_vmacc_vx_u32m8(vacc, (uint32_t)w, vp32, vl);
                }
            }

            // Divide by 273 via a single high-multiply — no widening
            // needed, so this stays legal at m8 (the LMUL ceiling).
            vuint32m8_t vres = __riscv_vmulhu_vx_u32m8(vacc, RECIP_M_VEC, vl);
            vres = __riscv_vminu_vx_u32m8(vres, 255u, vl);

            // Narrow u32 -> u16 -> u8
            vuint16m4_t vn16 = __riscv_vncvt_x_x_w_u16m4(vres, vl);
            vuint8m2_t  vn8  = __riscv_vncvt_x_x_w_u8m2(vn16, vl);

            __riscv_vse8_v_u8m2(out_row + x, vn8, vl);
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
