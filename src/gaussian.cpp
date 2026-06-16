#include "gaussian.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>

// ---------------------------------------------------------------------------
// 5x5 Gaussian kernel (sigma ~= 1.0, integer coefficients, sum = 273)
//
//   [ 1  4  7  4  1 ]
//   [ 4 16 26 16  4 ]
//   [ 7 26 41 26  7 ]   / 273
//   [ 4 16 26 16  4 ]
//   [ 1  4  7  4  1 ]
//
// Boundary handling: ZERO-PADDING (out-of-bounds pixels treated as 0).
// aligned_alloc(64) ensures 64-byte alignment needed for RVV vector loads.
// ---------------------------------------------------------------------------

static const int16_t KERNEL[5][5] = {
    { 1,  4,  7,  4, 1},
    { 4, 16, 26, 16, 4},
    { 7, 26, 41, 26, 7},
    { 4, 16, 26, 16, 4},
    { 1,  4,  7,  4, 1}
};
static constexpr int KERNEL_SUM = 273;
static constexpr int RADIUS     = 2;

Image gaussian_blur(const Image& img) {
    assert(img.data   != nullptr);
    assert(img.width  >  0);
    assert(img.height >  0);

    const int W = img.width;
    const int H = img.height;

    Image out;
    out.width  = W;
    out.height = H;
    out.data   = (uint8_t*)aligned_alloc(64, W * H);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int32_t acc = 0;

            for (int ky = -RADIUS; ky <= RADIUS; ky++) {
                for (int kx = -RADIUS; kx <= RADIUS; kx++) {
                    const int sy = y + ky;
                    const int sx = x + kx;

                    // Zero-padding: out-of-bounds pixels contribute 0
                    if (sx < 0 || sx >= W || sy < 0 || sy >= H)
                        continue;

                    acc += (int32_t)img.data[sy * W + sx]
                         * (int32_t)KERNEL[ky + RADIUS][kx + RADIUS];
                }
            }

            const int32_t result = acc / KERNEL_SUM;
            out.data[y * W + x]  = (uint8_t)std::clamp(result, 0, 255);
        }
    }
    return out;
}
