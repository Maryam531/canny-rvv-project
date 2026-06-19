#include "image.h"
#include "image_io.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>
// Creates a black image with a white rectangle in the middle
Image create_rectangle_image(int width, int height) {
    Image img;
    img.width  = width;
    img.height = height;
    img.data   = (uint8_t*)aligned_alloc(64, width * height);
    memset(img.data, 0, width * height);

    // Draw white rectangle in center
    int x1 = width  / 4;
    int x2 = width  * 3 / 4;
    int y1 = height / 4;
    int y2 = height * 3 / 4;

    for (int y = y1; y < y2; y++)
        for (int x = x1; x < x2; x++)
            img.data[y * width + x] = 255;

    return img;
}

// Creates image with vertical edge (left=black, right=white)
Image create_vertical_edge_image(int width, int height) {
    Image img;
    img.width  = width;
    img.height = height;
    img.data   = (uint8_t*)aligned_alloc(64, width * height);

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            img.data[y * width + x] = (x < width / 2) ? 0 : 255;

    return img;
}

// Creates image with horizontal edge (top=black, bottom=white)
Image create_horizontal_edge_image(int width, int height) {
    Image img;
    img.width  = width;
    img.height = height;
    img.data   = (uint8_t*)aligned_alloc(64, width * height);

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            img.data[y * width + x] = (y < height / 2) ? 0 : 255;

    return img;
}

// Creates uniform gray image (for testing blur invariant)
Image create_uniform_image(int width, int height, uint8_t value) {
    Image img;
    img.width  = width;
    img.height = height;
    img.data   = (uint8_t*)aligned_alloc(64, width * height);
    memset(img.data, value, width * height);
    return img;
}
//Test circle image
Image create_circle_image(int width, int height, int radius = -1) {
    Image img;
    img.width  = width;
    img.height = height;
    img.data   = (uint8_t*)aligned_alloc(64, width * height);
    memset(img.data, 0, width * height);
    int cx = width / 2;
    int cy = height / 2;
    int r  = (radius > 0) ? radius : (std::min(width, height) / 4);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dx = x - cx;
            int dy = y - cy;
            if (dx * dx + dy * dy <= r * r) {
                img.data[y * width + x] = 255;
            }
        }
    }
    return img;
}
//Test triangle
Image create_triangle_image(int width, int height) {
    Image img;
    img.width  = width;
    img.height = height;
    img.data   = (uint8_t*)aligned_alloc(64, width * height);
    memset(img.data, 0, width * height);

    int x0 = width / 2,     y0 = height / 8;
    int x1 = width / 8,     y1 = height * 7 / 8;
    int x2 = width * 7 / 8, y2 = height * 7 / 8;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            auto sign = [](int px, int py, int ax, int ay, int bx, int by) {
                return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
            };
            int d1 = sign(x, y, x0, y0, x1, y1);
            int d2 = sign(x, y, x1, y1, x2, y2);
            int d3 = sign(x, y, x2, y2, x0, y0);

            bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
            bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

            if (!(has_neg && has_pos)) {
                img.data[y * width + x] = 255;
            }
        }
    }
    return img;
}
int main() {
    // Generate all test images into images/ folder
    Image rect = create_rectangle_image(256, 256);
    save_image("images/rectangle.raw", rect);
    free_image(rect);

    Image vedge = create_vertical_edge_image(256, 256);
    save_image("images/vertical_edge.raw", vedge);
    free_image(vedge);

    Image hedge = create_horizontal_edge_image(256, 256);
    save_image("images/horizontal_edge.raw", hedge);
    free_image(hedge);

    Image uniform = create_uniform_image(256, 256, 128);
    save_image("images/uniform.raw", uniform);

    printf("Test images generated in images/ folder\n");
    return 0;
}
