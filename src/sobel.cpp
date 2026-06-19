#include "sobel.h"
#include <cstring>
#include <cmath>
#include <cstdlib> // For std::abs

void sobel(
    const Image& img,
    int16_t* gx,
    int16_t* gy
)
{
    int width = img.width;
    int height = img.height;
    const uint8_t* data = img.data;

    // 1. Zero out the entire output buffers to guarantee clean borders
    memset(gx, 0, width * height * sizeof(int16_t));
    memset(gy, 0, width * height * sizeof(int16_t));

    // 2. Compute gradients only on valid interior pixels (ignoring the 1-pixel border)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;

            // Sobel X Kernel (Detects horizontal changes / vertical edges)
            int16_t gx_val =
                -data[(y - 1) * width + (x - 1)]
                + data[(y - 1) * width + (x + 1)]
                -2 * data[y * width + (x - 1)]
                +2 * data[y * width + (x + 1)]
                -data[(y + 1) * width + (x - 1)]
                + data[(y + 1) * width + (x + 1)];

            // Sobel Y Kernel (Detects vertical changes / horizontal edges)
            int16_t gy_val =
                -data[(y - 1) * width + (x - 1)]
                -2 * data[(y - 1) * width + x]
                -data[(y - 1) * width + (x + 1)]
                + data[(y + 1) * width + (x - 1)]
                +2 * data[(y + 1) * width + x]
                + data[(y + 1) * width + (x + 1)];

            // Save computed raw gradient values to the buffers
            gx[idx] = gx_val;
            gy[idx] = gy_val;
        }
    }
}
