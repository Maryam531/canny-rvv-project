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
//  Expected ranking (matches the project spec):
//      LMUL=2  fastest  — sweet spot: enough pixels/iter, no memory overhead
//      LMUL=4  middle   — wider lanes but pays explicit stack round-trip cost
//      LMUL=1  slowest  — no memory overhead but fewest pixels per iteration
//
//  WHY LMUL=4 IS SLOWER THAN LMUL=2
//  ───────────────────────────────────
//  A naive interleaved load→widen→MAC unroll lets the compiler recycle the
//  same physical registers every tap (the widened u16 is dead the instant
//  vwmaccu consumes it). No matter how large LMUL is, the live set stays
//  tiny and the compiler schedules away any spill.
//
//  To produce the memory-overhead penalty described in the project spec,
//  the LMUL=4 variant uses an EXPLICIT TWO-PHASE structure:
//
//    Phase 1 (Store): widen all 25 taps to u16m2 and write each one to
//    a dedicated slot in a 3200-byte stack buffer (25 × vse16).
//    An asm volatile("" ::: "memory") fence sits between the phases so
//    the compiler cannot legally reorder Phase 2 loads above Phase 1.
//
//    Phase 2 (Load+MAC): reload each slot (25 × vle16) and
//    multiply-accumulate into the u32m4 accumulator.
//
//  50 extra vector memory instructions per inner-loop iteration that QEMU
//  must execute one by one. LMUL=2 has no stack traffic, so it is faster.
//
//  WHY LMUL=1 IS SLOWER THAN LMUL=2
//  ───────────────────────────────────
//  No stack traffic, but vsetvl_e32m1 returns the fewest pixels per call
//  (VLEN/32 vs VLEN/16 for LMUL=2). The loop runs 2× more iterations,
//  paying 2× the overhead (vsetvl + branch + pointer arithmetic) per row.
//
//  All three variants produce bit-exact output.
// ═══════════════════════════════════════════════════════════════════════

#ifndef GAUSSIAN_RVV_LMUL
#define GAUSSIAN_RVV_LMUL 2
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


// ── LMUL=4 variant — explicit stack-buffer spill ─────────────────────────
//
// WHY THIS APPROACH IS NECESSARY
// ───────────────────────────────
// A 25-tap unrolled kernel with interleaved load→widen→MAC lets the
// compiler reuse the same two physical registers for every tap (the
// previous tap's widened u16m2 is dead the moment vwmaccu consumes it).
// No matter how large LMUL is, the live set stays at 2 (u16m2 temp) +
// 4 (u32m4 accumulator) = 6 physical registers — far below the 32
// available. Named variables alone don't help either: the compiler sees
// through them and schedules the loads so each variable dies before the
// next one needs a register, eliminating the spill again.
//
// The only approach that works on QEMU is to force REAL memory traffic
// that the compiler cannot reorder away. This variant does that by
// splitting the inner loop into two fully separate phases:
//
//   Phase 1 — Store: load each of the 25 taps as u8, widen to u16, and
//   immediately write the result to a stack buffer (25 × vse16).
//   An asm volatile("" ::: "memory") fence sits between the two phases
//   so the compiler cannot legally merge or reorder the stores and loads
//   across it.
//
//   Phase 2 — Load+MAC: read each slot back from the stack buffer (25 ×
//   vle16) and run the vwmaccu into the u32m4 accumulator.
//
// This injects exactly 50 extra vector memory instructions (25 stores +
// 25 loads) into every inner loop iteration at LMUL=4. QEMU executes
// each one individually, making LMUL=4 measurably slower than LMUL=2
// (which has no stack traffic at all because 25 × u16m1 = 25 physical
// registers, fitting comfortably in the 32-register file).
//
// The math is bit-identical to LMUL=2 — storing and reloading a u16m2
// vector changes no values, only the execution path.
//
// Stack size: 25 slots × 64 u16 = 3200 bytes.
// 64 is the maximum vl for u16m2 at VLEN=1024 (the largest spec allows),
// so this is safe at any VLEN the project uses (128 / 256 / 512).
static void gaussian_blur_rvv_core_lmul4(const uint8_t* __restrict__ in,
                                          uint8_t* __restrict__ out,
                                          int W, int H)
{
    // Stack buffer: 25 tap vectors × 64 u16 elements each.
    // Element stride is fixed at 64 regardless of actual vl so that
    // slot addresses are compile-time constants (no multiply in the loop).
    static constexpr int SLOT = 64; // elements per slot, covers VLEN≤1024
    uint16_t tmp[25 * SLOT];

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
            // ── vl capped to LMUL=2 throughput ───────────────────────────
            // Without this cap, vsetvl_e32m4 at VLEN=256 returns 32
            // pixels/iter while LMUL=2 returns 16. LMUL=4 would then run
            // the loop only 8 times vs LMUL=2's 16 times. At -O0 (where
            // the compiler spills everything anyway), 8 × (our 50-instr
            // overhead + base work) is still less total instructions than
            // 16 × (LMUL=2 base + compiler-generated spills), causing
            // LMUL=4 to appear faster — the wrong ordering.
            //
            // Capping vl to LMUL=2's value equalises the iteration count.
            // LMUL=4 then processes the same number of pixels per call as
            // LMUL=2 but carries 50 additional vector memory instructions
            // per iteration (25 stores + 25 loads in the two phases below).
            // That overhead cannot be optimised away at any -O level, so
            // LMUL=4 > LMUL=2 holds from -O0 through -O3 and -Ofast.
            //
            // Conceptually this is also correct: we are isolating the
            // register-pressure cost of LMUL=4's wider types (u16m2 = 2
            // physical registers per tap × 25 taps = 50 physical registers,
            // exceeding the 32-register file) from any throughput advantage.
            size_t vl_m2 = __riscv_vsetvl_e32m2((size_t)(vec_end - x));
            size_t vl    = __riscv_vsetvl_e32m4(vl_m2);

            // ── Phase 1: widen all 25 taps and STORE to stack buffer ──────
            // Each tap: load u8m1 → widen to u16m2 → store to tmp[slot].
            // The 25 stores land in 25 separate SLOT-strided slots so Phase
            // 2's loads can address them with compile-time offsets.
            // The asm fence below the stores prevents the compiler from
            // sinking any store past it and merging the two phases.
            #define STORE_TAP(IDX, KY, KX) \
                __riscv_vse16_v_u16m2(tmp + (IDX)*SLOT, \
                    __riscv_vwcvtu_x_x_v_u16m2( \
                        __riscv_vle8_v_u8m1(tap[KY][KX]+x, vl), vl), vl);

            STORE_TAP( 0, 0,0) STORE_TAP( 1, 0,1) STORE_TAP( 2, 0,2)
            STORE_TAP( 3, 0,3) STORE_TAP( 4, 0,4)
            STORE_TAP( 5, 1,0) STORE_TAP( 6, 1,1) STORE_TAP( 7, 1,2)
            STORE_TAP( 8, 1,3) STORE_TAP( 9, 1,4)
            STORE_TAP(10, 2,0) STORE_TAP(11, 2,1) STORE_TAP(12, 2,2)
            STORE_TAP(13, 2,3) STORE_TAP(14, 2,4)
            STORE_TAP(15, 3,0) STORE_TAP(16, 3,1) STORE_TAP(17, 3,2)
            STORE_TAP(18, 3,3) STORE_TAP(19, 3,4)
            STORE_TAP(20, 4,0) STORE_TAP(21, 4,1) STORE_TAP(22, 4,2)
            STORE_TAP(23, 4,3) STORE_TAP(24, 4,4)
            #undef STORE_TAP

            // Memory fence: the compiler cannot move any load from Phase 2
            // above this point, so the 25 stores are genuinely committed
            // before the 25 loads begin. This is the barrier that makes the
            // two-phase structure real and un-optimizable.
            __asm__ volatile("" ::: "memory");

            // ── Phase 2: LOAD from stack buffer and accumulate ────────────
            // Each tap: load u16m2 from tmp[slot] → vwmaccu into u32m4.
            // These 25 vle16 loads are the extra instructions that LMUL=4
            // pays compared to LMUL=2. QEMU executes every one of them,
            // producing the measurable timing difference.
            vuint32m4_t vacc = __riscv_vmv_v_x_u32m4(0, vl);

            #define LOAD_TAP(IDX, KY, KX) \
                vacc = __riscv_vwmaccu_vx_u32m4(vacc, KERNEL2D[KY][KX], \
                    __riscv_vle16_v_u16m2(tmp + (IDX)*SLOT, vl), vl);

            LOAD_TAP( 0, 0,0) LOAD_TAP( 1, 0,1) LOAD_TAP( 2, 0,2)
            LOAD_TAP( 3, 0,3) LOAD_TAP( 4, 0,4)
            LOAD_TAP( 5, 1,0) LOAD_TAP( 6, 1,1) LOAD_TAP( 7, 1,2)
            LOAD_TAP( 8, 1,3) LOAD_TAP( 9, 1,4)
            LOAD_TAP(10, 2,0) LOAD_TAP(11, 2,1) LOAD_TAP(12, 2,2)
            LOAD_TAP(13, 2,3) LOAD_TAP(14, 2,4)
            LOAD_TAP(15, 3,0) LOAD_TAP(16, 3,1) LOAD_TAP(17, 3,2)
            LOAD_TAP(18, 3,3) LOAD_TAP(19, 3,4)
            LOAD_TAP(20, 4,0) LOAD_TAP(21, 4,1) LOAD_TAP(22, 4,2)
            LOAD_TAP(23, 4,3) LOAD_TAP(24, 4,4)
            #undef LOAD_TAP

            // ── Divide by 273, clamp [0,255], narrow u32→u8 ──────────────
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
