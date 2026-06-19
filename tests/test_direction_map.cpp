#include <gtest/gtest.h>
#include "direction.h"

TEST(DirectionMapTest, AllHorizontal)
{
    const int W = 4;
    const int H = 4;

    int16_t gx[16];
    int16_t gy[16];
    Direction dirs[16];

    for(int i=0;i<16;i++)
    {
        gx[i] = 100;
        gy[i] = 0;
    }

    computeGradientDirections(
        gx,
        gy,
        dirs,
        W,
        H);

    for(int i=0;i<16;i++)
    {
        EXPECT_EQ(
            dirs[i],
            Direction::DIR_0);
    }
}
