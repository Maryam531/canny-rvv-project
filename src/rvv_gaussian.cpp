#include "gaussian.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

// ── Kernel constants ──────────────────────────────────────────────────────
// Assignment-specified non-separable 5×5 kernel, sum = 273.
// These values are fixed by the project spec.
static constexpr int32_t KERNEL_SUM = 273;
static constexpr int     RADIUS     = 2;
static constexpr int     KSIZE      = 5;

static constexpr uint16_t KERNEL2D[KSIZE][KSIZE] = {
    { 1,  4,  7,  4, 1},
    { 4, 16, 26, 16, 4},
    { 7, 26, 41, 26, 7},
    { 4, 16, 26, 16, 4},
    { 1,  4,  7,  4, 1}
};

// ── Reciprocal-multiply for ÷273 ─────────────────────────────────────────
// floor(x / 273) == floor(x * 122911 >> 25)
// Exhaustively verified for x in [0, 69615] (max acc = 255*273): 0 errors.
// 69615 * 122911 = 8,557,513,665 — does NOT fit int32, requires int64.
static constexpr int64_t RECIP_M = 122911;
static constexpr int     RECIP_S = 25;

static inline int32_t div273_scalar(int32_t x) {
    return (int32_t)(((int64_t)x * RECIP_M) >> RECIP_S);
}

static inline size_t round_up64(size_t n) {
    return (n + 63u) & ~size_t(63);
}

// ── Scalar border fallback (zero-padding) ────────────────────────────────
// Must use the same KERNEL2D and div273 as gaussian.cpp for bit-exact output.
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

// ── Init / Free ───────────────────────────────────────────────────────────
// No persistent allocation — we read directly from the input image.
static int s_W = 0, s_H = 0;

void gaussian_blur_rvv_init(int W, int H) { s_W = W; s_H = H; }
void gaussian_blur_rvv_free()             { s_W = 0; s_H = 0; }

// ── Core: optimised 25-tap 2D RVV convolution ────────────────────────────
//
// Key optimisations vs the previous version:
//
//  1. ONE widen instead of TWO per tap.
//     Old: vle8(u8m1) → vwcvtu(u16m2) → vwcvtu(u32m4) → vreinterpret → vmacc
//          = 5 instructions × 25 taps = 125 instructions
//     New: vle8(u8m1) → vwcvtu(u16m2) → vwmaccu_vx(u16m2 * scalar → u32m4 acc)
//          = 3 instructions × 25 taps = 75 instructions  (-40%)
//     vwmaccu_vx widens u16 pixels and multiplies by a u16 scalar weight,
//     accumulating into u32 in a single instruction. Safe because:
//       • all weights are positive (1–41), so u16 scalar is fine
//       • max acc = 255*273 = 69615 << 2^32 (u32 never overflows)
//       • 69615 < 2^31 so reinterpreting u32→i32 before the divide is safe
//         (the sign bit is never set)
//
//  2. Full loop unroll of the 5×5 kernel.
//     Eliminates all loop-overhead branches in the inner hot path.
//     Lets the compiler/CPU pipeline the 25 independent loads freely.
//
//  3. Precomputed tap pointers.
//     Each of the 25 tap base addresses (src_rows[ky] + x + kx_offset) is
//     computed once before the strip-mining loop and reused every iteration.
//     Avoids 25 pointer additions inside the while loop.
//
//  4. u32 accumulator, reinterpret to i32 once before the divide.
//     Avoids a signed/unsigned conversion chain inside the 25-tap accumulate.
//
// Overflow analysis (unchanged from original):
//   max acc (u32)  = 255 * 273 = 69,615  < 2^32   (no u32 overflow)
//   max acc (i32)  = 69,615              < 2^31   (safe reinterpret)
//   divide product = 69,615 * 122,911    = 8,557,513,665 < 2^63  (i64 safe)
//
// VLEN-agnostic: vsetvl_e32m4 adapts vl automatically to any VLEN (128/256/512).

static void gaussian_blur_rvv_core(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   int W, int H)
{
    for (int y = 0; y < H; y++) {
        uint8_t* out_row = out + (size_t)y * W;

        // Border rows → scalar fallback
        if (y < RADIUS || y >= H - RADIUS) {
            for (int x = 0; x < W; x++)
                out_row[x] = scalar_pixel(in, W, H, x, y);
            continue;
        }

        // Border columns → scalar fallback
        for (int x = 0;          x < RADIUS;     x++) out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W;          x++) out_row[x] = scalar_pixel(in, W, H, x, y);

        // ── Precompute the 25 tap base pointers (optimisation 3) ─────────
        // tap[ky][kx] = pointer to column 0 of source row (y+ky), offset by kx.
        // Inside the strip-mining loop we just add x to get the right address.
        // This moves 25 pointer additions out of the while loop entirely.
        const uint8_t* tap[KSIZE][KSIZE];
        for (int ky = 0; ky < KSIZE; ky++) {
            const uint8_t* row = in + (size_t)(y + ky - RADIUS) * W;
            for (int kx = 0; kx < KSIZE; kx++)
                tap[ky][kx] = row + (kx - RADIUS);   // offset baked in
        }

        // ── Strip-mining loop over interior columns ───────────────────────
        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            // vl = number of pixels this iteration handles (e32m4 elements).
            // At VLEN=128: vl≤16; VLEN=256: vl≤32; VLEN=512: vl≤64.
            // Using e32m4 because the accumulator is u32 (LMUL=4) and the
            // subsequent i64m8 divide stage doubles to m8 (legal maximum).
            size_t vl = __riscv_vsetvl_e32m4((size_t)(vec_end - x));

            // Accumulator: u32m4, zeroed. All 25 taps accumulate here.
            // u32 is safe: max value 69615 << 2^32.
            vuint32m4_t vacc = __riscv_vmv_v_x_u32m4(0, vl);

            // ── 25-tap fully unrolled accumulation ──
            // Each tap: load u8 → widen to u16 → vwmaccu into u32 accumulator.
            // vwmaccu_vx: vacc[i] += (u32)(u16_pixel[i]) * (u32)(scalar_weight)
            // This is ONE widening instruction instead of two, saving 50
            // instructions across the full 25-tap sum.
            //
            // Macro to keep the 25 lines readable:
            #define TAP(KY, KX) do { \
                vuint8m1_t  _p8  = __riscv_vle8_v_u8m1(tap[KY][KX] + x, vl); \
                vuint16m2_t _p16 = __riscv_vwcvtu_x_x_v_u16m2(_p8, vl); \
                vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[KY][KX], _p16, vl); \
            } while(0)

            TAP(0,0); TAP(0,1); TAP(0,2); TAP(0,3); TAP(0,4);
            TAP(1,0); TAP(1,1); TAP(1,2); TAP(1,3); TAP(1,4);
            TAP(2,0); TAP(2,1); TAP(2,2); TAP(2,3); TAP(2,4);
            TAP(3,0); TAP(3,1); TAP(3,2); TAP(3,3); TAP(3,4);
            TAP(4,0); TAP(4,1); TAP(4,2); TAP(4,3); TAP(4,4);

            #undef TAP

            // ── Divide by 273 via widening i64 multiply-shift ─────────────
            // Reinterpret u32→i32 (safe: max=69615, sign bit never set), then
            // widen to i64 for the multiply-shift. Must use i64 because
            // 69615 * 122911 = 8.55B > INT32_MAX.

            // u32m4 → i32m4 (reinterpret, no data movement)
            vint32m4_t  vi32 = __riscv_vreinterpret_v_u32m4_i32m4(vacc);
            // i32m4 → i64m8 (sign-extend, LMUL doubles m4→m8)
            vint64m8_t  v64  = __riscv_vsext_vf2_i64m8(vi32, vl);
            // Multiply by reciprocal constant
            vint64m8_t  vmul = __riscv_vmul_vx_i64m8(v64, RECIP_M, vl);
            // Arithmetic right shift = the ">> 25" of the approximation
            vint64m8_t  vres = __riscv_vsra_vx_i64m8(vmul, (unsigned)RECIP_S, vl);

            // Clamp to [0, 255] then narrow i64→i32→i16→u8 (three steps,
            // RVV can only halve element width per narrowing instruction).
            vres            = __riscv_vmax_vx_i64m8(vres, 0,   vl);
            vres            = __riscv_vmin_vx_i64m8(vres, 255, vl);
            vint32m4_t vn32 = __riscv_vncvt_x_x_w_i32m4(vres, vl);
            vint16m2_t vn16 = __riscv_vncvt_x_x_w_i16m2(vn32, vl);
            vuint8m1_t vn8  = __riscv_vncvt_x_x_w_u8m1(
                                  __riscv_vreinterpret_v_i16m2_u16m2(vn16), vl);

            // Store vl output pixels
            __riscv_vse8_v_u8m1(out_row + x, vn8, vl);
            x += (int)vl;
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────

void gaussian_blur_rvv_into(const Image& img, uint8_t* out_data) {
    assert(img.data != nullptr && out_data != nullptr);
    assert(s_W == img.width && s_H == img.height);
    gaussian_blur_rvv_core(img.data, out_data, img.width, img.height);
}

Image gaussian_blur_rvv(const Image& img) {
    assert(img.data != nullptr && img.width > 0 && img.height > 0);
    const int W = img.width, H = img.height;
    Image out;
    out.width  = W;
    out.height = H;
    out.data   = (uint8_t*)aligned_alloc(64, round_up64((size_t)W * H));
    assert(out.data != nullptr);
    gaussian_blur_rvv_core(img.data, out.data, W, H);
    return out;
}
