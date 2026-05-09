#include "image.h"
#include "image_io.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
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
    free_image(uniform);

    printf("Test images generated in images/ folder\n");
    return 0;
}
