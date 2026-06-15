#include <gtest/gtest.h>
#include "direction.h"

// Pure horizontal gradients -> DIR_0
TEST(Direction, HorizontalGradientGivesDir0)
{
    EXPECT_EQ(quantiseDirection(100, 0), Direction::DIR_0);
    EXPECT_EQ(quantiseDirection(-100, 0), Direction::DIR_0);
}

// Pure vertical gradients -> DIR_90
TEST(Direction, VerticalGradientGivesDir90)
{
    EXPECT_EQ(quantiseDirection(0, 100), Direction::DIR_90);
    EXPECT_EQ(quantiseDirection(0, -100), Direction::DIR_90);
}

// Same-sign diagonals -> DIR_45
TEST(Direction, SameSignDiagonalGivesDir45)
{
    EXPECT_EQ(quantiseDirection(100, 100), Direction::DIR_45);
    EXPECT_EQ(quantiseDirection(-100, -100), Direction::DIR_45);
}

// Opposite-sign diagonals -> DIR_135
TEST(Direction, OppositeSignDiagonalGivesDir135)
{
    EXPECT_EQ(quantiseDirection(100, -100), Direction::DIR_135);
    EXPECT_EQ(quantiseDirection(-100, 100), Direction::DIR_135);
}

// Boundary near 22.5 degrees
TEST(Direction, BoundaryNear22Degrees)
{
    EXPECT_EQ(quantiseDirection(100, 39), Direction::DIR_0);
    EXPECT_NE(quantiseDirection(100, 41), Direction::DIR_0);
}

// Boundary near 67.5 degrees
TEST(Direction, BoundaryNear67Degrees)
{
    EXPECT_EQ(quantiseDirection(100, 239), Direction::DIR_45);
    EXPECT_EQ(quantiseDirection(100, 241), Direction::DIR_90);
}

// Full direction-map test
TEST(Direction, ComputeDirectionMap)
{
    constexpr int N = 4;

    int16_t gx[N] = {100, 0, 100, -100};
    int16_t gy[N] = {0, 100, 100, 100};

    Direction dirs[N];

    computeGradientDirections(gx, gy, dirs, N, 1);

    EXPECT_EQ(dirs[0], Direction::DIR_0);
    EXPECT_EQ(dirs[1], Direction::DIR_90);
    EXPECT_EQ(dirs[2], Direction::DIR_45);
    EXPECT_EQ(dirs[3], Direction::DIR_135);
}
