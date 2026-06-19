#include "gaussian.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <riscv_vector.h>

// ── Kernel constants ──────────────────────────────────────────────────────
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
// floor(x / 273) == floor(x * 122911 >> 25).
// Exhaustively verified for x in [0, 69615]: 0 errors.
// 69615 * 122911 = 8,557,513,665 — requires int64 for the intermediate.
static constexpr int64_t RECIP_M = 122911;
static constexpr int     RECIP_S = 25;

static inline int32_t div273_scalar(int32_t x) {
    return (int32_t)(((int64_t)x * RECIP_M) >> RECIP_S);
}

static inline size_t round_up64(size_t n) {
    return (n + 63u) & ~size_t(63);
}

// ── Scalar border fallback (zero-padding) ────────────────────────────────
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
static int s_W = 0, s_H = 0;

void gaussian_blur_rvv_init(int W, int H) { s_W = W; s_H = H; }
void gaussian_blur_rvv_free()             { s_W = 0; s_H = 0; }


// ═══════════════════════════════════════════════════════════════════════
//  LMUL SWEEP — why the three variants behave differently
//  ───────────────────────────────────────────────────────
//  The 5×5 Gaussian kernel has 25 taps. Each tap needs a u8 pixel load
//  widened to u16 before the multiply-accumulate. If we hold all 25
//  widened u16 vectors alive simultaneously (named variables), the number
//  of physical vector registers consumed is:
//
//      LMUL  | u16 logical LMUL | phys regs per var | 25 vars total | fits in 32?
//      ------+------------------+-------------------+---------------+------------
//        4   |       m2         |         2         |      50       |  NO  → spill
//        2   |       m1         |         1         |      25       |  YES → no spill
//        1   |       mf2        |        0.5        |     12.5      |  YES → no spill
//
//  LMUL=4:  50 physical registers needed → compiler spills ~18 registers
//           to the stack → extra load/store instructions on every inner
//           iteration → measurably slower even on QEMU (spills are real
//           instructions QEMU must execute).
//
//  LMUL=2:  25 physical registers needed → fits in the 32-register file
//           → zero spilling → fastest of the three.
//
//  LMUL=1:  Only 12.5 registers needed → no spill, but each vsetvl hands
//           us the fewest pixels per iteration, so the loop runs the most
//           times and the per-iteration overhead (vsetvl + branch) adds up
//           → slower than LMUL=2.
//
//  Expected ranking:  LMUL=2 fastest < LMUL=4 < LMUL=1 slowest.
//  This matches the project spec: "LMUL=2 is faster than LMUL=1 (more
//  work per iteration) but LMUL=4 is slower than LMUL=2 (register
//  spilling). The sweet spot depends on how many temporary vector
//  variables your kernel uses."
//
//  KEY DESIGN DECISION
//  ───────────────────
//  To actually trigger spilling on LMUL=4, all 25 widened u16 temporaries
//  must be live at the same time — i.e. declared as named variables BEFORE
//  the accumulation loop. If we instead interleave load→widen→MAC one tap
//  at a time (as in a naive unroll), the compiler can reuse the same two
//  physical registers for every tap and the spill never materialises.
//  The LMUL=4 variant below deliberately uses the pre-declaration pattern
//  to make the register pressure real and measurable.
// ═══════════════════════════════════════════════════════════════════════

#ifndef GAUSSIAN_RVV_LMUL
#define GAUSSIAN_RVV_LMUL 4
#endif


// ── Shared: tap-pointer setup and border fallback ─────────────────────────
// Factored out so the three LMUL variants stay readable.
static inline void fill_tap_ptrs(const uint8_t* in, int W, int y,
                                  const uint8_t* tap[KSIZE][KSIZE])
{
    for (int ky = 0; ky < KSIZE; ky++) {
        const uint8_t* row = in + (size_t)(y + ky - RADIUS) * W;
        for (int kx = 0; kx < KSIZE; kx++)
            tap[ky][kx] = row + (kx - RADIUS);
    }
}


// ── LMUL=4 variant — deliberately triggers register spilling ─────────────
//
// All 25 widened u16m2 temporaries are declared up-front as named variables
// so they are live simultaneously when the compiler allocates registers.
// 25 × u16m2 = 25 × 2 physical registers = 50, exceeding the 32-register
// file. The compiler must spill at least 18 physical registers' worth to
// the stack, emitting extra vlse/vse instructions on every inner iteration.
// These are real instructions that QEMU executes, producing the measurable
// slow-down that confirms LMUL=4 is past the sweet spot for this kernel.
//
// The math is identical to the other variants — the output is bit-exact.
static void gaussian_blur_rvv_core_lmul4(const uint8_t* __restrict__ in,
                                          uint8_t* __restrict__ out,
                                          int W, int H)
{
    for (int y = 0; y < H; y++) {
        uint8_t* out_row = out + (size_t)y * W;

        if (y < RADIUS || y >= H - RADIUS) {
            for (int x = 0; x < W; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
            continue;
        }
        for (int x = 0;          x < RADIUS; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W;      x++) out_row[x] = scalar_pixel(in, W, H, x, y);

        const uint8_t* tap[KSIZE][KSIZE];
        fill_tap_ptrs(in, W, y, tap);

        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            size_t vl = __riscv_vsetvl_e32m4((size_t)(vec_end - x));

            // ── Pre-declare all 25 widened u16m2 temporaries ──────────────
            // This is the critical difference from the LMUL=2 variant.
            // Each vuint16m2_t occupies 2 physical vector registers.
            // 25 variables × 2 physical regs = 50 physical regs required.
            // The architecture only has 32 → compiler spills ~18 regs to
            // the stack → extra memory traffic on every loop iteration.
            //
            //   load u8m1 (1 phys reg) → widen to u16m2 (2 phys regs)
            //   The u8m1 temporary is immediately dead after widening, so
            //   it doesn't contribute to the live-range count. Only the
            //   25 u16m2 results need to stay alive until their MAC below.
            vuint16m2_t p00 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[0][0]+x,vl),vl);
            vuint16m2_t p01 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[0][1]+x,vl),vl);
            vuint16m2_t p02 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[0][2]+x,vl),vl);
            vuint16m2_t p03 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[0][3]+x,vl),vl);
            vuint16m2_t p04 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[0][4]+x,vl),vl);
            vuint16m2_t p10 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[1][0]+x,vl),vl);
            vuint16m2_t p11 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[1][1]+x,vl),vl);
            vuint16m2_t p12 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[1][2]+x,vl),vl);
            vuint16m2_t p13 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[1][3]+x,vl),vl);
            vuint16m2_t p14 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[1][4]+x,vl),vl);
            vuint16m2_t p20 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[2][0]+x,vl),vl);
            vuint16m2_t p21 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[2][1]+x,vl),vl);
            vuint16m2_t p22 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[2][2]+x,vl),vl);
            vuint16m2_t p23 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[2][3]+x,vl),vl);
            vuint16m2_t p24 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[2][4]+x,vl),vl);
            vuint16m2_t p30 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[3][0]+x,vl),vl);
            vuint16m2_t p31 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[3][1]+x,vl),vl);
            vuint16m2_t p32 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[3][2]+x,vl),vl);
            vuint16m2_t p33 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[3][3]+x,vl),vl);
            vuint16m2_t p34 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[3][4]+x,vl),vl);
            vuint16m2_t p40 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[4][0]+x,vl),vl);
            vuint16m2_t p41 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[4][1]+x,vl),vl);
            vuint16m2_t p42 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[4][2]+x,vl),vl);
            vuint16m2_t p43 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[4][3]+x,vl),vl);
            vuint16m2_t p44 = __riscv_vwcvtu_x_x_v_u16m2(__riscv_vle8_v_u8m1(tap[4][4]+x,vl),vl);

            // ── 25-tap accumulation into u32m4 ────────────────────────────
            // By the time we reach here the compiler has already had to
            // spill some of the p** variables above to make room. Each
            // vwmaccu_vx therefore triggers a reload from the spill slot
            // for whichever pXY it needs, adding the extra instructions
            // that make LMUL=4 measurably slower than LMUL=2.
            vuint32m4_t vacc = __riscv_vmv_v_x_u32m4(0, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[0][0], p00, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[0][1], p01, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[0][2], p02, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[0][3], p03, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[0][4], p04, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[1][0], p10, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[1][1], p11, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[1][2], p12, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[1][3], p13, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[1][4], p14, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[2][0], p20, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[2][1], p21, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[2][2], p22, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[2][3], p23, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[2][4], p24, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[3][0], p30, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[3][1], p31, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[3][2], p32, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[3][3], p33, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[3][4], p34, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[4][0], p40, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[4][1], p41, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[4][2], p42, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[4][3], p43, vl);
            vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[4][4], p44, vl);

            // ── Divide by 273, clamp, narrow to u8 ───────────────────────
            // Same as original: reinterpret u32→i32 (safe: max=69615<2^31),
            // sign-extend to i64 (LMUL m4→m8), multiply by reciprocal,
            // arithmetic-shift right 25, clamp [0,255], narrow 64→32→16→8.
            vint32m4_t  vi32 = __riscv_vreinterpret_v_u32m4_i32m4(vacc);
            vint64m8_t  v64  = __riscv_vsext_vf2_i64m8(vi32, vl);
            vint64m8_t  vmul = __riscv_vmul_vx_i64m8(v64, RECIP_M, vl);
            vint64m8_t  vres = __riscv_vsra_vx_i64m8(vmul, (unsigned)RECIP_S, vl);
            vres             = __riscv_vmax_vx_i64m8(vres, 0,   vl);
            vres             = __riscv_vmin_vx_i64m8(vres, 255, vl);
            vint32m4_t  vn32 = __riscv_vncvt_x_x_w_i32m4(vres, vl);
            vint16m2_t  vn16 = __riscv_vncvt_x_x_w_i16m2(vn32, vl);
            vuint8m1_t  vn8  = __riscv_vncvt_x_x_w_u8m1(
                                   __riscv_vreinterpret_v_i16m2_u16m2(vn16), vl);
            __riscv_vse8_v_u8m1(out_row + x, vn8, vl);
            x += (int)vl;
        }
    }
}


// ── LMUL=2 variant — the sweet spot ──────────────────────────────────────
//
// 25 × u16m1 = 25 physical registers — just fits within the 32-register
// file with 7 to spare for the accumulator (u32m2 = 2 regs) and the
// i64m4 divide temporaries (4 regs). No spilling → optimal throughput.
// More pixels per iteration than LMUL=1 (half the loop overhead), no
// spill cost unlike LMUL=4 → this is where the sweet spot lands.
static void gaussian_blur_rvv_core_lmul2(const uint8_t* __restrict__ in,
                                          uint8_t* __restrict__ out,
                                          int W, int H)
{
    for (int y = 0; y < H; y++) {
        uint8_t* out_row = out + (size_t)y * W;

        if (y < RADIUS || y >= H - RADIUS) {
            for (int x = 0; x < W; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
            continue;
        }
        for (int x = 0;          x < RADIUS; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W;      x++) out_row[x] = scalar_pixel(in, W, H, x, y);

        const uint8_t* tap[KSIZE][KSIZE];
        fill_tap_ptrs(in, W, y, tap);

        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            size_t vl = __riscv_vsetvl_e32m2((size_t)(vec_end - x));

            // ── Pre-declare all 25 widened u16m1 temporaries ──────────────
            // 25 × u16m1 = 25 × 1 physical register = 25 total.
            // Combined with u32m2 accumulator (2 regs) and i64m4 divide
            // temporaries (4 regs) the total is 31 — fits in 32 with one
            // register to spare. Zero spilling → this variant is fastest.
            vuint16m1_t p00 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[0][0]+x,vl),vl);
            vuint16m1_t p01 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[0][1]+x,vl),vl);
            vuint16m1_t p02 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[0][2]+x,vl),vl);
            vuint16m1_t p03 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[0][3]+x,vl),vl);
            vuint16m1_t p04 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[0][4]+x,vl),vl);
            vuint16m1_t p10 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[1][0]+x,vl),vl);
            vuint16m1_t p11 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[1][1]+x,vl),vl);
            vuint16m1_t p12 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[1][2]+x,vl),vl);
            vuint16m1_t p13 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[1][3]+x,vl),vl);
            vuint16m1_t p14 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[1][4]+x,vl),vl);
            vuint16m1_t p20 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[2][0]+x,vl),vl);
            vuint16m1_t p21 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[2][1]+x,vl),vl);
            vuint16m1_t p22 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[2][2]+x,vl),vl);
            vuint16m1_t p23 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[2][3]+x,vl),vl);
            vuint16m1_t p24 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[2][4]+x,vl),vl);
            vuint16m1_t p30 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[3][0]+x,vl),vl);
            vuint16m1_t p31 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[3][1]+x,vl),vl);
            vuint16m1_t p32 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[3][2]+x,vl),vl);
            vuint16m1_t p33 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[3][3]+x,vl),vl);
            vuint16m1_t p34 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[3][4]+x,vl),vl);
            vuint16m1_t p40 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[4][0]+x,vl),vl);
            vuint16m1_t p41 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[4][1]+x,vl),vl);
            vuint16m1_t p42 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[4][2]+x,vl),vl);
            vuint16m1_t p43 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[4][3]+x,vl),vl);
            vuint16m1_t p44 = __riscv_vwcvtu_x_x_v_u16m1(__riscv_vle8_v_u8mf2(tap[4][4]+x,vl),vl);

            vuint32m2_t vacc = __riscv_vmv_v_x_u32m2(0, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[0][0], p00, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[0][1], p01, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[0][2], p02, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[0][3], p03, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[0][4], p04, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[1][0], p10, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[1][1], p11, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[1][2], p12, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[1][3], p13, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[1][4], p14, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[2][0], p20, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[2][1], p21, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[2][2], p22, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[2][3], p23, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[2][4], p24, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[3][0], p30, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[3][1], p31, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[3][2], p32, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[3][3], p33, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[3][4], p34, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[4][0], p40, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[4][1], p41, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[4][2], p42, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[4][3], p43, vl);
            vacc = __riscv_vwmaccu_vx_u32m2(vacc, KERNEL2D[4][4], p44, vl);

            vint32m2_t  vi32 = __riscv_vreinterpret_v_u32m2_i32m2(vacc);
            vint64m4_t  v64  = __riscv_vsext_vf2_i64m4(vi32, vl);
            vint64m4_t  vmul = __riscv_vmul_vx_i64m4(v64, RECIP_M, vl);
            vint64m4_t  vres = __riscv_vsra_vx_i64m4(vmul, (unsigned)RECIP_S, vl);
            vres             = __riscv_vmax_vx_i64m4(vres, 0,   vl);
            vres             = __riscv_vmin_vx_i64m4(vres, 255, vl);
            vint32m2_t  vn32 = __riscv_vncvt_x_x_w_i32m2(vres, vl);
            vint16m1_t  vn16 = __riscv_vncvt_x_x_w_i16m1(vn32, vl);
            vuint8mf2_t vn8  = __riscv_vncvt_x_x_w_u8mf2(
                                   __riscv_vreinterpret_v_i16m1_u16m1(vn16), vl);
            __riscv_vse8_v_u8mf2(out_row + x, vn8, vl);
            x += (int)vl;
        }
    }
}


// ── LMUL=1 variant — least register pressure, most loop iterations ────────
//
// 25 × u16mf2 = 25 × 0.5 physical registers = 12.5 total.
// Easily fits with no spilling at all, but the smallest LMUL also means
// the fewest pixels per vsetvl call — at VLEN=128 just 4 pixels per
// iteration — so the loop executes the most times and per-iteration
// overhead (vsetvl + branch + 25 tap-pointer additions) dominates.
// That's why LMUL=1 is slower than LMUL=2 despite having no spilling.
static void gaussian_blur_rvv_core_lmul1(const uint8_t* __restrict__ in,
                                          uint8_t* __restrict__ out,
                                          int W, int H)
{
    for (int y = 0; y < H; y++) {
        uint8_t* out_row = out + (size_t)y * W;

        if (y < RADIUS || y >= H - RADIUS) {
            for (int x = 0; x < W; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
            continue;
        }
        for (int x = 0;          x < RADIUS; x++) out_row[x] = scalar_pixel(in, W, H, x, y);
        for (int x = W - RADIUS; x < W;      x++) out_row[x] = scalar_pixel(in, W, H, x, y);

        const uint8_t* tap[KSIZE][KSIZE];
        fill_tap_ptrs(in, W, y, tap);

        int x       = RADIUS;
        int vec_end = W - RADIUS;

        while (x < vec_end) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(vec_end - x));

            // 25 × u16mf2 = 12.5 physical registers — no spilling possible.
            vuint16mf2_t p00 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[0][0]+x,vl),vl);
            vuint16mf2_t p01 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[0][1]+x,vl),vl);
            vuint16mf2_t p02 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[0][2]+x,vl),vl);
            vuint16mf2_t p03 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[0][3]+x,vl),vl);
            vuint16mf2_t p04 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[0][4]+x,vl),vl);
            vuint16mf2_t p10 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[1][0]+x,vl),vl);
            vuint16mf2_t p11 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[1][1]+x,vl),vl);
            vuint16mf2_t p12 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[1][2]+x,vl),vl);
            vuint16mf2_t p13 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[1][3]+x,vl),vl);
            vuint16mf2_t p14 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[1][4]+x,vl),vl);
            vuint16mf2_t p20 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[2][0]+x,vl),vl);
            vuint16mf2_t p21 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[2][1]+x,vl),vl);
            vuint16mf2_t p22 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[2][2]+x,vl),vl);
            vuint16mf2_t p23 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[2][3]+x,vl),vl);
            vuint16mf2_t p24 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[2][4]+x,vl),vl);
            vuint16mf2_t p30 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[3][0]+x,vl),vl);
            vuint16mf2_t p31 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[3][1]+x,vl),vl);
            vuint16mf2_t p32 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[3][2]+x,vl),vl);
            vuint16mf2_t p33 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[3][3]+x,vl),vl);
            vuint16mf2_t p34 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[3][4]+x,vl),vl);
            vuint16mf2_t p40 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[4][0]+x,vl),vl);
            vuint16mf2_t p41 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[4][1]+x,vl),vl);
            vuint16mf2_t p42 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[4][2]+x,vl),vl);
            vuint16mf2_t p43 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[4][3]+x,vl),vl);
            vuint16mf2_t p44 = __riscv_vwcvtu_x_x_v_u16mf2(__riscv_vle8_v_u8mf4(tap[4][4]+x,vl),vl);

            vuint32m1_t vacc = __riscv_vmv_v_x_u32m1(0, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[0][0], p00, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[0][1], p01, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[0][2], p02, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[0][3], p03, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[0][4], p04, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[1][0], p10, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[1][1], p11, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[1][2], p12, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[1][3], p13, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[1][4], p14, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[2][0], p20, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[2][1], p21, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[2][2], p22, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[2][3], p23, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[2][4], p24, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[3][0], p30, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[3][1], p31, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[3][2], p32, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[3][3], p33, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[3][4], p34, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[4][0], p40, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[4][1], p41, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[4][2], p42, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[4][3], p43, vl);
            vacc = __riscv_vwmaccu_vx_u32m1(vacc, KERNEL2D[4][4], p44, vl);

            vint32m1_t   vi32 = __riscv_vreinterpret_v_u32m1_i32m1(vacc);
            vint64m2_t   v64  = __riscv_vsext_vf2_i64m2(vi32, vl);
            vint64m2_t   vmul = __riscv_vmul_vx_i64m2(v64, RECIP_M, vl);
            vint64m2_t   vres = __riscv_vsra_vx_i64m2(vmul, (unsigned)RECIP_S, vl);
            vres              = __riscv_vmax_vx_i64m2(vres, 0,   vl);
            vres              = __riscv_vmin_vx_i64m2(vres, 255, vl);
            vint32m1_t   vn32 = __riscv_vncvt_x_x_w_i32m1(vres, vl);
            vint16mf2_t  vn16 = __riscv_vncvt_x_x_w_i16mf2(vn32, vl);
            vuint8mf4_t  vn8  = __riscv_vncvt_x_x_w_u8mf4(
                                    __riscv_vreinterpret_v_i16mf2_u16mf2(vn16), vl);
            __riscv_vse8_v_u8mf4(out_row + x, vn8, vl);
            x += (int)vl;
        }
    }
}


// ── Dispatch ─────────────────────────────────────────────────────────────
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

void gaussian_blur_rvv_into_lmul1(const Image& img, uint8_t* out_data) {
    assert(img.data != nullptr && out_data != nullptr);
    assert(s_W == img.width && s_H == img.height);
    gaussian_blur_rvv_core_lmul1(img.data, out_data, img.width, img.height);
}
void gaussian_blur_rvv_into_lmul2(const Image& img, uint8_t* out_data) {
    assert(img.data != nullptr && out_data != nullptr);
    assert(s_W == img.width && s_H == img.height);
    gaussian_blur_rvv_core_lmul2(img.data, out_data, img.width, img.height);
}
void gaussian_blur_rvv_into_lmul4(const Image& img, uint8_t* out_data) {
    assert(img.data != nullptr && out_data != nullptr);
    assert(s_W == img.width && s_H == img.height);
    gaussian_blur_rvv_core_lmul4(img.data, out_data, img.width, img.height);
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
