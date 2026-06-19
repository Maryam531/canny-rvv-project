#include <gtest/gtest.h>
#include <cstdint>

#include "nms.h"
#include "direction.h"


TEST(NMS, MaximumPreserved)
{
    int W = 3;
    int H = 3;


    uint8_t magnitude[9] =
    {
        0,10,0,
        0,20,0,
        0,15,0
    };


    Direction direction[9] =
    {
        Direction::DIR_90, Direction::DIR_90, Direction::DIR_90,
        Direction::DIR_90, Direction::DIR_90, Direction::DIR_90,
        Direction::DIR_90, Direction::DIR_90, Direction::DIR_90
    };


    uint8_t output[9]={0};


    nonMaximumSuppression(
        magnitude,
        direction,
        output,
        W,
        H
    );


    EXPECT_EQ(output[4],20);
}


TEST(NMS, NonMaximumRemoved)
{

    int W=3;
    int H=3;


    uint8_t magnitude[9]=
    {
        0,10,0,
        0,20,0,
        0,15,0
    };


    Direction direction[9]=
    {
        Direction::DIR_90, Direction::DIR_90, Direction::DIR_90,
        Direction::DIR_90, Direction::DIR_90, Direction::DIR_90,
        Direction::DIR_90, Direction::DIR_90, Direction::DIR_90
    };


    uint8_t output[9]={0};


    nonMaximumSuppression(
        magnitude,
        direction,
        output,
        W,
        H
    );


    EXPECT_EQ(output[7],0);

}
