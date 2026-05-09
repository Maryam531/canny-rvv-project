#include "gaussian.h"
#include <cstdint>
#include <cstdlib>

// 5x5 Gaussian kernel coefficients (sigma ~ 1.0)
// Sum of all coefficients = 273
static const int16_t KERNEL[5][5] = {
    { 1,  4,  7,  4, 1},
    { 4, 16, 26, 16, 4},
    { 7, 26, 41, 26, 7},
    { 4, 16, 26, 16, 4},
    { 1,  4,  7,  4, 1}
};
static const int KERNEL_SUM = 273;

Image gaussian_blur(const Image& img) {
    int width  = img.width;
    int height = img.height;

    // Allocate output image
    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int32_t sum = 0;

            // Apply 5x5 kernel
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int ny = y + ky;
                    int nx = x + kx;

                    // Zero padding - treat out of bounds as 0
                    uint8_t pixel = 0;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width)
                        pixel = img.data[ny * width + nx];

                    sum += pixel * KERNEL[ky + 2][kx + 2];
                }
            }

            // Divide by kernel sum and clamp to [0, 255]
            int32_t result = sum / KERNEL_SUM;
            if (result < 0)   result = 0;
            if (result > 255) result = 255;
            out.data[y * width + x] = (uint8_t)result;
        }
    }
    return out;
}
