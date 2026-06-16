// gaussian_rvv.cpp
//
// RVV-intrinsic implementation of the 5x5 Gaussian blur, for equivalence
// comparison against the scalar baseline in gaussian.cpp.
//
// IMPORTANT: this file uses <riscv_vector.h> and only compiles with the
// RISC-V cross-compiler using -march=rv64gcv. It must NOT be added to the
// host/native build target in your Makefile (GoogleTest stays host-only).
//
// Add this declaration to gaussian.h so callers can use it:
//     Image gaussian_blur_rvv(const Image& img);

#include "gaussian.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <riscv_vector.h>

// ---------------------------------------------------------------------------
// Kernel constants. These MUST stay byte-identical to the ones in
// gaussian.cpp or the equivalence tests will fail for the wrong reason
// (mismatched kernels, not a vectorization bug). Consider moving this table
// into a shared header (e.g. gaussian_kernel.h) included by both files so
// there is only one source of truth.
// ---------------------------------------------------------------------------
static const int16_t KERNEL[5][5] = {
    { 1,  4,  7,  4, 1},
    { 4, 16, 26, 16, 4},
    { 7, 26, 41, 26, 7},
    { 4, 16, 26, 16, 4},
    { 1,  4,  7,  4, 1}
};
static constexpr int32_t KERNEL_SUM = 273;
static constexpr int     RADIUS     = 2;

// aligned_alloc() requires `size` to be a multiple of `alignment` per the
// C standard (C11 7.22.3.1) -- this is UB if violated, even though some
// libc builds tolerate it silently. Round up so e.g. a 33x17 image (561
// bytes, not a multiple of 64) doesn't allocate undefined behavior. The
// extra padding bytes are simply never touched.
static inline size_t round_up_align(size_t n, size_t alignment) {
    return (n + alignment - 1) & ~(alignment - 1);
}

// ---------------------------------------------------------------------------
// Scalar single-pixel helper. Used for the left/right border columns (where
// a horizontal tap can fall outside the image) and for degenerate images
// narrower than the kernel. This is intentionally identical in behavior to
// the inner loop body of gaussian_blur() in gaussian.cpp.
// ---------------------------------------------------------------------------
static inline uint8_t gaussian_pixel_scalar(const uint8_t* data, int W, int H,
                                             int x, int y) {
    int32_t acc = 0;
    for (int ky = -RADIUS; ky <= RADIUS; ky++) {
        const int sy = y + ky;
        if (sy < 0 || sy >= H) continue;          // zero-padding (vertical)
        const uint8_t* row = data + sy * W;
        for (int kx = -RADIUS; kx <= RADIUS; kx++) {
            const int sx = x + kx;
            if (sx < 0 || sx >= W) continue;       // zero-padding (horizontal)
            acc += (int32_t)row[sx] * (int32_t)KERNEL[ky + RADIUS][kx + RADIUS];
        }
    }
    const int32_t result = acc / KERNEL_SUM;
    return (uint8_t)std::clamp(result, 0, 255);
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
    out.data   = (uint8_t*)aligned_alloc(64, round_up_align((size_t)W * (size_t)H, 64));

    const uint8_t* in = img.data;
    uint8_t*       o  = out.data;

    // Columns [interior_start, interior_end) are far enough from the left
    // and right edges that every kx in [-RADIUS, RADIUS] stays in-bounds for
    // ALL lanes of a vector chunk that starts inside this range. That's what
    // lets the hot loop below skip horizontal bounds checks entirely.
    // clamp() so very narrow images (W < 2*RADIUS+1) safely degenerate to an
    // all-scalar row instead of producing an inverted/empty interior range.
    const int interior_start = std::min(RADIUS, W);
    const int interior_end   = std::max(W - RADIUS, interior_start);

    for (int y = 0; y < H; y++) {
        int x = 0;

        // --- Left border: scalar fallback ---------------------------------
        for (; x < interior_start; x++) {
            o[y * W + x] = gaussian_pixel_scalar(in, W, H, x, y);
        }

        // --- Interior: vectorized ------------------------------------------
        while (x < interior_end) {
            // Strip-mine: ask for the rest of the interior row, hardware
            // tells us how many elements (vl) it can actually do this pass.
            // SEW=32/LMUL=4 chosen for the i32 accumulator; this is the
            // group size that drives the widening chain below.
            size_t vl = __riscv_vsetvl_e32m4(interior_end - x);

            // Accumulator, one int32 per pixel lane, start at zero.
            vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

            for (int ky = -RADIUS; ky <= RADIUS; ky++) {
                const int sy = y + ky;
                // Vertical zero-padding is a per-ROW decision (every lane in
                // this vector shares the same y), so a single scalar branch
                // here correctly skips the whole tap row for every lane —
                // no per-lane masking needed.
                if (sy < 0 || sy >= H) continue;
                const uint8_t* row = in + sy * W;

                for (int kx = -RADIUS; kx <= RADIUS; kx++) {
                    const int16_t coeff = KERNEL[ky + RADIUS][kx + RADIUS];

                    // Load vl consecutive source pixels, offset by kx. Safe
                    // un-checked because x is inside [interior_start,
                    // interior_end) and vl was capped to stay inside it too.
                    vuint8m1_t  vpix8  = __riscv_vle8_v_u8m1(row + x + kx, vl);

                    // Widen u8 -> u16 -> u32 (LMUL doubles each step: m1 ->
                    // m2 -> m4) so the pixel value lines up with the i32m4
                    // accumulator group for the macc below.
                    vuint16m2_t vpix16 = __riscv_vwcvtu_x_x_v_u16m2(vpix8, vl);
                    vuint32m4_t vpix32u = __riscv_vwcvtu_x_x_v_u32m4(vpix16, vl);
                    vint32m4_t  vpix32 = __riscv_vreinterpret_v_u32m4_i32m4(vpix32u);

                    // vacc[i] += coeff * vpix32[i]  (fused multiply-add,
                    // one instruction instead of separate mul+add).
                    vacc = __riscv_vmacc_vx_i32m4(vacc, (int32_t)coeff, vpix32, vl);
                }
            }

            // Integer divide by 273. acc is always >= 0 here (all pixel
            // values and all kernel coefficients are non-negative), so
            // vector truncating division matches the scalar `acc / 273`
            // exactly — no rounding-direction mismatch to worry about.
            vint32m4_t vresult = __riscv_vdiv_vx_i32m4(vacc, KERNEL_SUM, vl);

            // Mirrors std::clamp(result, 0, 255). The lower clamp is
            // defensive (acc can't go negative with this kernel) but keeps
            // this code correct if the kernel table is ever changed.
            vresult = __riscv_vmax_vx_i32m4(vresult, 0, vl);
            vresult = __riscv_vmin_vx_i32m4(vresult, 255, vl);

            // Narrow i32 -> u16 -> u8 (two steps, mirroring the widening
            // chain above) and store to the output row.
            vuint32m4_t vresultu = __riscv_vreinterpret_v_i32m4_u32m4(vresult);
            vuint16m2_t vnarrow16 = __riscv_vncvt_x_x_w_u16m2(vresultu, vl);
            vuint8m1_t  vnarrow8  = __riscv_vncvt_x_x_w_u8m1(vnarrow16, vl);
            __riscv_vse8_v_u8m1(o + y * W + x, vnarrow8, vl);

            x += vl;
        }

        // --- Right border: scalar fallback ----------------------------------
        for (; x < W; x++) {
            o[y * W + x] = gaussian_pixel_scalar(in, W, H, x, y);
        }
    }

    return out;
}
