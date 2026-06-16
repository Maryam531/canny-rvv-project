#include <gtest/gtest.h>
#include "sobel.h"
#include <cstdlib>
#include <cstring>
#include <cmath>

static Image make_image(int w, int h, uint8_t fill) {
    Image img;
    img.width  = w;
    img.height = h;
    img.data   = (uint8_t*)aligned_alloc(64, w * h);
    memset(img.data, fill, w * h);
    return img;
}
static void free_image(Image& img) { free(img.data); img.data = nullptr; }
static int16_t* make_grad(int w, int h) { return (int16_t*)calloc(w*h, sizeof(int16_t)); }

TEST(SobelGradient, UniformImageZeroGradient) {
    Image input = make_image(64, 64, 128);
    int16_t* gx = make_grad(64, 64);
    int16_t* gy = make_grad(64, 64);
    sobel(input, gx, gy);
    for (int y = 1; y < 63; y++)
        for (int x = 1; x < 63; x++) {
            EXPECT_EQ(gx[y*64+x], 0);
            EXPECT_EQ(gy[y*64+x], 0);
        }
    free_image(input); free(gx); free(gy);
}

TEST(SobelGradient, VerticalEdgeLargeGx) {
    Image input = make_image(64, 64, 0);
    for (int y = 0; y < 64; y++)
        for (int x = 32; x < 64; x++)
            input.data[y*64+x] = 255;
    int16_t* gx = make_grad(64, 64);
    int16_t* gy = make_grad(64, 64);
    sobel(input, gx, gy);
    for (int y = 5; y < 59; y++) {
        EXPECT_GT(gx[y*64+32], 100);
        EXPECT_NEAR(gy[y*64+32], 0, 10);
    }
    free_image(input); free(gx); free(gy);
}

TEST(SobelGradient, HorizontalEdgeLargeGy) {
    Image input = make_image(64, 64, 0);
    for (int y = 32; y < 64; y++)
        for (int x = 0; x < 64; x++)
            input.data[y*64+x] = 255;
    int16_t* gx = make_grad(64, 64);
    int16_t* gy = make_grad(64, 64);
    sobel(input, gx, gy);
    for (int x = 5; x < 59; x++) {
        EXPECT_GT(gy[32*64+x], 100);
        EXPECT_NEAR(gx[32*64+x], 0, 10);
    }
    free_image(input); free(gx); free(gy);
}

TEST(SobelGradient, AllBlackZeroGradient) {
    Image input = make_image(64, 64, 0);
    int16_t* gx = make_grad(64, 64);
    int16_t* gy = make_grad(64, 64);
    sobel(input, gx, gy);
    for (int i = 0; i < 64*64; i++) {
        EXPECT_EQ(gx[i], 0);
        EXPECT_EQ(gy[i], 0);
    }
    free_image(input); free(gx); free(gy);
}

TEST(SobelGradient, FlippedEdgeFlipsSign) {
    Image input1 = make_image(64, 64, 0);
    Image input2 = make_image(64, 64, 255);
    for (int y = 0; y < 64; y++) {
        for (int x = 32; x < 64; x++) input1.data[y*64+x] = 255;
        for (int x = 32; x < 64; x++) input2.data[y*64+x] = 0;
    }
    int16_t* gx1 = make_grad(64,64); int16_t* gy1 = make_grad(64,64);
    int16_t* gx2 = make_grad(64,64); int16_t* gy2 = make_grad(64,64);
    sobel(input1, gx1, gy1);
    sobel(input2, gx2, gy2);
    for (int y = 5; y < 59; y++) {
        EXPECT_GT(gx1[y*64+32],  0);
        EXPECT_LT(gx2[y*64+32],  0);
        EXPECT_EQ(gx1[y*64+32], -gx2[y*64+32]);
    }
    free_image(input1); free_image(input2);
    free(gx1); free(gy1); free(gx2); free(gy2);
}
