#include <gtest/gtest.h>
#include <cstdint>

#include "threshold.h"


TEST(Threshold, StrongWeakSuppressedEdges)
{

    uint8_t input[6]=
    {
        120,
        70,
        20,

        150,
        40,
        80
    };


    uint8_t output[6]={0};


    doubleThreshold(
        input,
        output,
        3,
        2,
        50,
        100
    );


    EXPECT_EQ(output[0],255);
    EXPECT_EQ(output[1],75);
    EXPECT_EQ(output[2],0);

    EXPECT_EQ(output[3],255);
    EXPECT_EQ(output[4],0);
    EXPECT_EQ(output[5],75);

}
