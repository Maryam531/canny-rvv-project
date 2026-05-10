// gaussian.cpp
// Implements gaussian_blur() declared in gaussian.h
//
// Assumed Image layout (adjust field names here if yours differ):
//
//   struct Image {
//       uint8_t* data;   // row-major pixel buffer, one byte per pixel
//       int      width;
//       int      height;
//   };
//
// If your Image uses different names (e.g. .pixels, .w, .h, .cols, .rows),
// do a find-and-replace in this file:
//   img.data   -> img.pixels  (or whatever the buffer field is called)
//   img.width  -> img.w       (or .cols)
//   img.height -> img.h       (or .rows)

#include "gaussian.h"

#include <algorithm>   // std::clamp
#include <cassert>
#include <cstdint>     // uint8_t, int16_t, int32_t

// ---------------------------------------------------------------------------
// 5x5 Gaussian kernel  (sigma ~= 1.0, integer coefficients, sum = 273)
//
//   [ 1  4  7  4  1 ]
//   [ 4 16 26 16  4 ]
//   [ 7 26 41 26  7 ]   / 273
//   [ 4 16 26 16  4 ]
//   [ 1  4  7  4  1 ]
//
// Derived from the outer product of the 1-D Pascal kernel [1 4 6 4 1]
// with itself.  All arithmetic stays in integers; no floating-point needed.
// ---------------------------------------------------------------------------

static const int16_t kCoeff[5][5] = {
    { 1,  4,  7,  4,  1 },
    { 4, 16, 26, 16,  4 },
    { 7, 26, 41, 26,  7 },
    { 4, 16, 26, 16,  4 },
    { 1,  4,  7,  4,  1 },
};
static constexpr int32_t kDivisor = 273;
static constexpr int     kRadius  = 2;      // (5-1)/2

// ---------------------------------------------------------------------------
// gaussian_blur
//
// Boundary handling: ZERO-PADDING.
//   Pixels outside the image boundary are treated as 0 (black).
//   Out-of-bounds taps are simply skipped (equivalent to adding 0),
//   keeping the inner loop branch-free for fully interior rows.
//
// Template parameters used internally:
//   PixelT  = uint8_t   (one byte per grayscale pixel)
//   AccumT  = int32_t   (max accumulator value = 255 * 273 = 69615, safe)
//   KernelT = int16_t   (coefficients fit comfortably in 16 bits)
// ---------------------------------------------------------------------------
Image gaussian_blur(const Image& img)
{
    assert(img.data   != nullptr);
    assert(img.width  >  0);
    assert(img.height >  0);

    const int W = img.width;
    const int H = img.height;

    // Allocate output image with the same dimensions.
    // Adjust this block if your Image has a constructor or factory function,
    // e.g.:  Image out(W, H);  or  Image out = Image::create(W, H);
    Image out;
    out.width  = W;
    out.height = H;
    out.data   = new uint8_t[static_cast<std::size_t>(W) * H];

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {

            int32_t acc = 0;

            for (int ky = -kRadius; ky <= kRadius; ++ky) {
                const int sy = y + ky;

                for (int kx = -kRadius; kx <= kRadius; ++kx) {
                    const int sx = x + kx;

                    // Zero-padding: out-of-bounds pixels contribute 0.
                    if (sx < 0 || sx >= W || sy < 0 || sy >= H)
                        continue;

                    acc += static_cast<int32_t>(img.data[sy * W + sx])
                         * static_cast<int32_t>(kCoeff[ky + kRadius][kx + kRadius]);
                }
            }

            // Divide by kernel sum, clamp to [0, 255].
            const int32_t result  = acc / kDivisor;
            out.data[y * W + x]   = static_cast<uint8_t>(
                std::clamp(result, int32_t{0}, int32_t{255}));
        }
    }

    return out;
}