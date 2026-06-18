#include "gaussian.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

// ── Kernel constants ──────────────────────────────────────────────────────
// This is the assignment-specified non-separable 5x5 kernel (sum = 273).
// Don't change these values, they're fixed by the project spec.
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
// Division is slow, even more so in a vector loop, so we replace it with
// a multiply + shift: floor(x / 273) == floor(x * 122911 >> 25).
// We checked this exhaustively for every x in [0, 69615] (the largest the
// accumulator can ever be) and it matches plain integer division with zero
// off-by-one errors. Note 69615 * 122911 = 8,557,513,665, which is bigger
// than what int32 can hold, so the multiply has to happen in int64.
static constexpr int64_t RECIP_M = 122911;
static constexpr int     RECIP_S = 25;

static inline int32_t div273_scalar(int32_t x) {
    return (int32_t)(((int64_t)x * RECIP_M) >> RECIP_S);
}

static inline size_t round_up64(size_t n) {
    return (n + 63u) & ~size_t(63);
}

// ── Scalar border fallback (zero-padding) ────────────────────────────────
// Pixels within 2 rows/cols of the edge fall back to this. It has to use
// the exact same KERNEL2D and div273 as gaussian.cpp, otherwise the borders
// of the scalar and RVV outputs will never line up.
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
// No persistent allocation needed — we read straight from the input image.
static int s_W = 0, s_H = 0;

void gaussian_blur_rvv_init(int W, int H) { s_W = W; s_H = H; }
void gaussian_blur_rvv_free()             { s_W = 0; s_H = 0; }


// ═══════════════════════════════════════════════════════════════════════
//  LMUL SWEEP
//  ──────────
//  Phase 6 asks us to implement the same Gaussian kernel at LMUL=1, 2,
//  and 4 and compare. The three functions below (_lmul1, _lmul2, _lmul4)
//  are otherwise identical — same 25-tap kernel, same divide trick, same
//  border handling — the only thing that changes is how many vector
//  registers each logical variable is allowed to span.
//
//  Widening always doubles LMUL, so picking the *final* accumulator LMUL
//  fixes everything upstream:
//
//      target acc LMUL | u16 pixel LMUL | u8 pixel LMUL | i64 divide LMUL
//      ----------------+-----------------+----------------+-----------------
//      m4 (this file)  |  m2             |  m1            |  m8
//      m2              |  m1             |  mf2           |  m4
//      m1              |  mf2            |  mf4           |  m2
//
//  "mf2"/"mf4" mean fractional LMUL (half/quarter of one register) — RVV
//  lets you do this when SEW is small enough that a full register would
//  be wasted.
// ═══════════════════════════════════════════════════════════════════════

// Which variant gaussian_blur_rvv_into() actually calls during the
// benchmark. Switch this to compare LMUL=1 / 2 / 4 end to end without
// touching main.cpp.
#ifndef GAUSSIAN_RVV_LMUL
#define GAUSSIAN_RVV_LMUL 4
#endif


// ── LMUL=4 variant (the "go wide" version) ────────────────────────────────
static void gaussian_blur_rvv_core_lmul4(const uint8_t* __restrict__ in,
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

        for (int x = 0;          x < RADIUS; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W;      x++) out_row[x] = scalar_pixel(in, W, H, x, y);

        // Five source-row pointers, one per kernel row, computed once per
        // output row instead of once per pixel.
        const uint8_t* tap[KSIZE][KSIZE];
        for (int ky = 0; ky < KSIZE; ky++) {
            const uint8_t* row = in + (size_t)(y + ky - RADIUS) * W;
            for (int kx = 0; kx < KSIZE; kx++)
                tap[ky][kx] = row + (kx - RADIUS);
        }

        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            // Ask the hardware how many 32-bit-wide accumulator elements
            // it can give us this round, at LMUL=4 (i.e. spread across 4
            // physical vector registers worth of space). On a VLEN=128
            // machine that's 16 pixels; VLEN=256 doubles it to 32; VLEN=512
            // doubles again to 64. We never write a number here ourselves —
            // the hardware tells us, and the loop just adapts.
            size_t vl = __riscv_vsetvl_e32m4((size_t)(vec_end - x));

            // Zero the accumulator before the 25-tap sum starts. LMUL=4
            // because that's the width we picked for this variant — every
            // vector below has to agree on this width or the intrinsics
            // won't even compile. At a bigger VLEN this just means more
            // lanes get zeroed per call, the math doesn't change.
            vuint32m4_t vacc = __riscv_vmv_v_x_u32m4(0, vl);

            // The 25-tap accumulation, fully unrolled by hand. Each tap
            // does three things: load a row of pixels as bytes, widen them
            // to 16-bit so they can be safely multiplied, then multiply by
            // this tap's kernel weight and add into the running total —
            // all three pixels-widening-and-accumulate in one instruction
            // (vwmaccu). We can get away with the "u" (unsigned) widening
            // macc because every weight in this kernel is positive and
            // every pixel is non-negative, so there's never a sign to lose.
            //
            //   load u8m1     -> 8-bit pixels, one register's worth (LMUL=1)
            //   widen to u16m2 -> doubles LMUL because we doubled the width
            //   vwmaccu to u32m4 -> doubles LMUL again, lands at our target
            //
            // None of this changes with VLEN — a bigger VLEN just means
            // each of these steps touches more pixels per call.
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

            // We've got the raw 25-tap sum sitting in vacc (max value
            // 69615, well inside u32). Now we need to divide by 273 and
            // squash the result into [0, 255]. The multiply-by-122911 step
            // needs 64-bit math, so everything below widens up to m8 —
            // the largest LMUL RVV allows — which is exactly why we chose
            // m4 as the accumulator width in this variant: m4 doubled once
            // is m8, the ceiling, so there's no room left to widen further
            // after this. (This is the actual reason LMUL=4 can't be pushed
            // any higher for this kernel — see the LMUL=1/2 variants below
            // for what happens with more headroom.)

            // Relabel the bits as signed (no values actually change, the
            // sign bit was never set since max=69615 is well under 2^31).
            vint32m4_t  vi32 = __riscv_vreinterpret_v_u32m4_i32m4(vacc);
            // Sign-extend 32-bit to 64-bit. LMUL doubles m4->m8 because
            // doubling the element width always doubles LMUL to keep the
            // same number of elements in flight.
            vint64m8_t  v64  = __riscv_vsext_vf2_i64m8(vi32, vl);
            // Multiply by the reciprocal constant (the "x * 122911" half
            // of the ÷273 trick). Stays at m8 since the values are already
            // 64-bit here.
            vint64m8_t  vmul = __riscv_vmul_vx_i64m8(v64, RECIP_M, vl);
            // Shift right 25 — the other half of the divide trick.
            vint64m8_t  vres = __riscv_vsra_vx_i64m8(vmul, (unsigned)RECIP_S, vl);

            // Clamp into [0, 255] before we start throwing bits away during
            // narrowing, otherwise an out-of-range value could wrap into
            // something that looks valid.
            vres            = __riscv_vmax_vx_i64m8(vres, 0,   vl);
            vres            = __riscv_vmin_vx_i64m8(vres, 255, vl);
            // Narrow back down to a byte. RVV only lets you halve the
            // width per instruction, so this takes three steps: 64->32,
            // 32->16, 16->8. LMUL halves each time right along with it
            // (m8->m4->m2->m1), ending back where the pixel loads started.
            vint32m4_t vn32 = __riscv_vncvt_x_x_w_i32m4(vres, vl);
            vint16m2_t vn16 = __riscv_vncvt_x_x_w_i16m2(vn32, vl);
            vuint8m1_t vn8  = __riscv_vncvt_x_x_w_u8m1(
                                  __riscv_vreinterpret_v_i16m2_u16m2(vn16), vl);

            // Write the finished pixels out. m1 matches the pixel loads
            // above — symmetric in, symmetric out. A bigger VLEN just
            // means more finished pixels land in memory per call.
            __riscv_vse8_v_u8m1(out_row + x, vn8, vl);
            x += (int)vl;
        }
    }
}


// ── LMUL=2 variant ─────────────────────────────────────────────────────────
// Same algorithm, half the register footprint per vector. The accumulator
// is u32m2 instead of u32m4, so vsetvl hands us roughly half as many
// pixels per iteration — but the compiler has more spare registers to
// juggle (fewer of the 32 physical registers are tied up per logical
// vector), so there's less risk of spilling. Whether that trade is a net
// win is exactly what the LMUL sweep is supposed to measure.
static void gaussian_blur_rvv_core_lmul2(const uint8_t* __restrict__ in,
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

        for (int x = 0;          x < RADIUS; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W;      x++) out_row[x] = scalar_pixel(in, W, H, x, y);

        const uint8_t* tap[KSIZE][KSIZE];
        for (int ky = 0; ky < KSIZE; ky++) {
            const uint8_t* row = in + (size_t)(y + ky - RADIUS) * W;
            for (int kx = 0; kx < KSIZE; kx++)
                tap[ky][kx] = row + (kx - RADIUS);
        }

        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            // Target accumulator is u32m2 this time, so we ask for that
            // directly. At VLEN=128 this gives 8 pixels/iter (half of the
            // LMUL=4 variant's 16); VLEN=256 gives 16; VLEN=512 gives 32.
            // Same scaling rule, just shifted down because LMUL is smaller.
            size_t vl = __riscv_vsetvl_e32m2((size_t)(vec_end - x));

            vuint32m2_t vacc = __riscv_vmv_v_x_u32m2(0, vl);

            // Pixel loads now sit at "mf2" — half a physical register —
            // since u32m2 needs u16m1 pixels underneath it (one widening
            // step down from m2), which in turn needs u8mf2 pixels (one
            // more widening step down). RVV allows fractional LMUL exactly
            // for cases like this, where the element type is small enough
            // that a whole register would go half-unused.
            #define TAP(KY, KX) do { \
                vuint8mf2_t _p8  = __riscv_vle8_v_u8mf2(tap[KY][KX] + x, vl); \
                vuint16m1_t _p16 = __riscv_vwcvtu_x_x_v_u16m1(_p8, vl); \
                vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[KY][KX], _p16, vl); \
            } while(0)

            TAP(0,0); TAP(0,1); TAP(0,2); TAP(0,3); TAP(0,4);
            TAP(1,0); TAP(1,1); TAP(1,2); TAP(1,3); TAP(1,4);
            TAP(2,0); TAP(2,1); TAP(2,2); TAP(2,3); TAP(2,4);
            TAP(3,0); TAP(3,1); TAP(3,2); TAP(3,3); TAP(3,4);
            TAP(4,0); TAP(4,1); TAP(4,2); TAP(4,3); TAP(4,4);

            #undef TAP

            // Same divide-by-273 trick, just starting from m2 instead of
            // m4. Doubling twice (m2->m4->m8 wait — no, here we only need
            // one doubling: m2->m4 for the i64 stage. There's a full extra
            // doubling of headroom left over compared to the LMUL=4
            // variant, which is the up-side this version is trading for
            // fewer pixels per iteration.
            vint32m2_t  vi32 = __riscv_vreinterpret_v_u32m2_i32m2(vacc);
            vint64m4_t  v64  = __riscv_vsext_vf2_i64m4(vi32, vl);
            vint64m4_t  vmul = __riscv_vmul_vx_i64m4(v64, RECIP_M, vl);
            vint64m4_t  vres = __riscv_vsra_vx_i64m4(vmul, (unsigned)RECIP_S, vl);

            vres            = __riscv_vmax_vx_i64m4(vres, 0,   vl);
            vres            = __riscv_vmin_vx_i64m4(vres, 255, vl);
            vint32m2_t vn32 = __riscv_vncvt_x_x_w_i32m2(vres, vl);
            vint16m1_t vn16 = __riscv_vncvt_x_x_w_i16m1(vn32, vl);
            vuint8mf2_t vn8 = __riscv_vncvt_x_x_w_u8mf2(
                                  __riscv_vreinterpret_v_i16m1_u16m1(vn16), vl);

            __riscv_vse8_v_u8mf2(out_row + x, vn8, vl);
            x += (int)vl;
        }
    }
}


// ── LMUL=1 variant (the "go narrow" version) ──────────────────────────────
// Smallest register footprint of the three. Fewest pixels per iteration,
// but also the least register pressure — the compiler has the most room
// to keep other things (like our 25 tap pointers) in registers without
// spilling to the stack.
static void gaussian_blur_rvv_core_lmul1(const uint8_t* __restrict__ in,
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

        for (int x = 0;          x < RADIUS; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W;      x++) out_row[x] = scalar_pixel(in, W, H, x, y);

        const uint8_t* tap[KSIZE][KSIZE];
        for (int ky = 0; ky < KSIZE; ky++) {
            const uint8_t* row = in + (size_t)(y + ky - RADIUS) * W;
            for (int kx = 0; kx < KSIZE; kx++)
                tap[ky][kx] = row + (kx - RADIUS);
        }

        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            // u32m1 accumulator: one physical register, the smallest
            // grouping RVV offers. Fewest pixels per call of the three
            // variants — at VLEN=128 this is just 4 pixels/iter.
            size_t vl = __riscv_vsetvl_e32m1((size_t)(vec_end - x));

            vuint32m1_t vacc = __riscv_vmv_v_x_u32m1(0, vl);

            // One more notch down on fractional LMUL than the LMUL=2
            // version: pixels load at u8mf4 (a quarter register), widen to
            // u16mf2, then vwmaccu lands at our u32m1 target. Same widening
            // logic as the other two variants, just shifted down a step.
            #define TAP(KY, KX) do { \
                vuint8mf4_t _p8  = __riscv_vle8_v_u8mf4(tap[KY][KX] + x, vl); \
                vuint16mf2_t _p16 = __riscv_vwcvtu_x_x_v_u16mf2(_p8, vl); \
                vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[KY][KX], _p16, vl); \
            } while(0)

            TAP(0,0); TAP(0,1); TAP(0,2); TAP(0,3); TAP(0,4);
            TAP(1,0); TAP(1,1); TAP(1,2); TAP(1,3); TAP(1,4);
            TAP(2,0); TAP(2,1); TAP(2,2); TAP(2,3); TAP(2,4);
            TAP(3,0); TAP(3,1); TAP(3,2); TAP(3,3); TAP(3,4);
            TAP(4,0); TAP(4,1); TAP(4,2); TAP(4,3); TAP(4,4);

            #undef TAP

            // Two full doublings of headroom before we hit the m8 ceiling
            // here (m1->m2->m4), the most spare room of the three variants —
            // but it doesn't buy us anything extra since the final divide
            // only ever needs one doubling (m1->m2) to reach i64.
            vint32m1_t  vi32 = __riscv_vreinterpret_v_u32m1_i32m1(vacc);
            vint64m2_t  v64  = __riscv_vsext_vf2_i64m2(vi32, vl);
            vint64m2_t  vmul = __riscv_vmul_vx_i64m2(v64, RECIP_M, vl);
            vint64m2_t  vres = __riscv_vsra_vx_i64m2(vmul, (unsigned)RECIP_S, vl);

            vres             = __riscv_vmax_vx_i64m2(vres, 0,   vl);
            vres             = __riscv_vmin_vx_i64m2(vres, 255, vl);
            vint32m1_t vn32  = __riscv_vncvt_x_x_w_i32m1(vres, vl);
            vint16mf2_t vn16 = __riscv_vncvt_x_x_w_i16mf2(vn32, vl);
            vuint8mf4_t vn8  = __riscv_vncvt_x_x_w_u8mf4(
                                   __riscv_vreinterpret_v_i16mf2_u16mf2(vn16), vl);

            __riscv_vse8_v_u8mf4(out_row + x, vn8, vl);
            x += (int)vl;
        }
    }
}


// Picks which LMUL variant actually runs. Compile with
// -DGAUSSIAN_RVV_LMUL=1 or =2 to switch — default is 4, matching the
// version we benchmark against scalar in main.cpp.
static void gaussian_blur_rvv_core(const uint8_t* __restrict__ in,
                                    uint8_t* __restrict__ out,
                                    int W, int H)
{
#if   GAUSSIAN_RVV_LMUL == 1
    gaussian_blur_rvv_core_lmul1(in, out, W, H);
#elif GAUSSIAN_RVV_LMUL == 2
    gaussian_blur_rvv_core_lmul2(in, out, W, H);
#else
    gaussian_blur_rvv_core_lmul4(in, out, W, H);
#endif
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
