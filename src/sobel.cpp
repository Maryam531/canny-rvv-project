#include "sobel.h"


void sobel(
    const Image& img,
    int16_t* gx,
    int16_t* gy
)
{
    int width = img.width;
    int height = img.height;

    uint8_t* data = img.data;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            int idx = y * width + x;

            int16_t gx_val =
                -data[(y - 1) * width + (x - 1)]
                + data[(y - 1) * width + (x + 1)]

                -2 * data[y * width + (x - 1)]
                +2 * data[y * width + (x + 1)]

                -data[(y + 1) * width + (x - 1)]
                + data[(y + 1) * width + (x + 1)];

            int16_t gy_val =
                -data[(y - 1) * width + (x - 1)]
                -2 * data[(y - 1) * width + x]
                -data[(y - 1) * width + (x + 1)]

                +data[(y + 1) * width + (x - 1)]
                +2 * data[(y + 1) * width + x]
                +data[(y + 1) * width + (x + 1)];

            gx[idx] = gx_val;
            gy[idx] = gy_val;
        }
    }
}