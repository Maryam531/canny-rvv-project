#include <gtest/gtest.h>
#include "magnitude.h"
#include <cstdlib>
#include <cstring>

static int16_t* make_grad(int w, int h) {
    return (int16_t*)calloc(w * h, sizeof(int16_t));
}

static void free_image(Image& img) {
    free(img.data);
    img.data = nullptr;
}

TEST(Magnitude, ZeroGradientL1GivesBlackImage) {
    int16_t* gx = make_grad(64,64);
    int16_t* gy = make_grad(64,64);

    Image out = magnitude_l1(gx, gy, 64, 64);

    for (int i = 0; i < 64 * 64; i++)
        EXPECT_EQ(out.data[i], 0);

    free_image(out);
    free(gx);
    free(gy);
}

TEST(Magnitude, ZeroGradientL2GivesBlackImage) {
    int16_t* gx = make_grad(64,64);
    int16_t* gy = make_grad(64,64);

    Image out = magnitude_l2(gx, gy, 64, 64);

    for (int i = 0; i < 64 * 64; i++)
        EXPECT_EQ(out.data[i], 0);

    free_image(out);
    free(gx);
    free(gy);
}

TEST(Magnitude, L1DetectsVerticalEdge) {
    int16_t* gx = make_grad(64,64);
    int16_t* gy = make_grad(64,64);

    gx[32 * 64 + 32] = 500;

    Image out = magnitude_l1(gx, gy, 64, 64);

    EXPECT_GT(out.data[32 * 64 + 32], 0);

    free_image(out);
    free(gx);
    free(gy);
}

TEST(Magnitude, L2DetectsVerticalEdge) {
    int16_t* gx = make_grad(64,64);
    int16_t* gy = make_grad(64,64);

    gx[32 * 64 + 32] = 500;

    Image out = magnitude_l2(gx, gy, 64, 64);

    EXPECT_GT(out.data[32 * 64 + 32], 0);

    free_image(out);
    free(gx);
    free(gy);
}

TEST(Magnitude, OutputDimensionsMatchInput) {
    int16_t* gx = make_grad(100,75);
    int16_t* gy = make_grad(100,75);

    Image out = magnitude_l1(gx, gy, 100, 75);

    EXPECT_EQ(out.width, 100);
    EXPECT_EQ(out.height, 75);

    free_image(out);
    free(gx);
    free(gy);
}

TEST(Magnitude, OutputClampedTo255) {
    int16_t* gx = make_grad(64,64);
    int16_t* gy = make_grad(64,64);

    gx[0] = 32767;
    gy[0] = 32767;

    Image out = magnitude_l1(gx, gy, 64, 64);

    for (int i = 0; i < 64 * 64; i++) {
        EXPECT_GE(out.data[i], 0);
        EXPECT_LE(out.data[i], 255);
    }

    free_image(out);
    free(gx);
    free(gy);
}
