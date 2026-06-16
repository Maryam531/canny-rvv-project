#include <gtest/gtest.h>
#include "gaussian.h"
#include <cstdlib>
#include <cstring>

static Image make_image(int w, int h, uint8_t fill) {
    Image img;
    img.width  = w;
    img.height = h;
    img.data   = (uint8_t*)aligned_alloc(64, w * h);
    memset(img.data, fill, w * h);
    return img;
}
static void free_image(Image& img) { free(img.data); img.data = nullptr; }

TEST(GaussianBlur, UniformImageStaysUniform) {
    Image input = make_image(64, 64, 128);
    Image output = gaussian_blur(input);
    for (int y = 2; y < 62; y++)
        for (int x = 2; x < 62; x++)
            EXPECT_NEAR(output.data[y * 64 + x], 128, 1);
    free_image(input); free_image(output);
}

TEST(GaussianBlur, AllBlackStaysBlack) {
    Image input = make_image(64, 64, 0);
    Image output = gaussian_blur(input);
    for (int i = 0; i < 64 * 64; i++)
        EXPECT_EQ(output.data[i], 0);
    free_image(input); free_image(output);
}

TEST(GaussianBlur, ImpulseSpreadToNeighbors) {
    Image input = make_image(64, 64, 0);
    input.data[32 * 64 + 32] = 255;
    Image output = gaussian_blur(input);
    EXPECT_GT(output.data[32 * 64 + 32], 0);
    EXPECT_LT(output.data[32 * 64 + 32], 255);
    EXPECT_GT(output.data[31 * 64 + 32], 0);
    EXPECT_GT(output.data[33 * 64 + 32], 0);
    EXPECT_EQ(output.data[27 * 64 + 32], 0);
    free_image(input); free_image(output);
}

TEST(GaussianBlur, ImpulseResponseIsSymmetric) {
    Image input = make_image(32, 32, 0);
    input.data[15 * 32 + 15] = 255;
    Image output = gaussian_blur(input);
    for (int d = 1; d <= 2; d++) {
        EXPECT_EQ(output.data[(15-d)*32+15], output.data[(15+d)*32+15]);
        EXPECT_EQ(output.data[15*32+(15-d)], output.data[15*32+(15+d)]);
    }
    free_image(input); free_image(output);
}

TEST(GaussianBlur, OutputClampedTo0_255) {
    Image input = make_image(64, 64, 255);
    Image output = gaussian_blur(input);
    for (int i = 0; i < 64 * 64; i++) {
        EXPECT_GE(output.data[i], 0);
        EXPECT_LE(output.data[i], 255);
    }
    free_image(input); free_image(output);
}

TEST(GaussianBlur, OutputDimensionsMatchInput) {
    Image input = make_image(100, 75, 128);
    Image output = gaussian_blur(input);
    EXPECT_EQ(output.width,  100);
    EXPECT_EQ(output.height, 75);
    free_image(input); free_image(output);
}
