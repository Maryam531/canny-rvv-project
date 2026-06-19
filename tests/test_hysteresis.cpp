#include <gtest/gtest.h>
#include <cstdint>

#include "hysteresis.h"
#include "threshold.h"


TEST(Hysteresis, WeakConnectedToStrongIsKept)
{
    // 3x3 grid:
    //   0   0   0
    //   75 255   0     <- weak pixel (idx 3) is 8-connected to strong (idx 4)
    //   0   0   0
    int W = 3;
    int H = 3;

    uint8_t classified[9] =
    {
        EDGE_NONE,   EDGE_NONE,   EDGE_NONE,
        EDGE_WEAK,   EDGE_STRONG, EDGE_NONE,
        EDGE_NONE,   EDGE_NONE,   EDGE_NONE
    };

    uint8_t output[9] = {0};

    hysteresisEdgeTracing(classified, output, W, H);

    EXPECT_EQ(output[3], EDGE_STRONG); // promoted weak pixel
    EXPECT_EQ(output[4], EDGE_STRONG); // original strong pixel stays
}


TEST(Hysteresis, IsolatedWeakIsSuppressed)
{
    // Weak pixel with no strong pixel anywhere nearby must be dropped.
    int W = 3;
    int H = 3;

    uint8_t classified[9] =
    {
        EDGE_NONE, EDGE_NONE, EDGE_NONE,
        EDGE_NONE, EDGE_WEAK, EDGE_NONE,
        EDGE_NONE, EDGE_NONE, EDGE_NONE
    };

    uint8_t output[9] = {0};

    hysteresisEdgeTracing(classified, output, W, H);

    EXPECT_EQ(output[4], EDGE_NONE);
}


TEST(Hysteresis, ChainOfWeakPixelsConnectsToStrong)
{
    // A straight chain of weak pixels leading away from a single strong
    // pixel should be fully promoted (transitive 8-connectivity).
    //
    //   255  75  75  75  0
    int W = 5;
    int H = 1;

    uint8_t classified[5] =
    {
        EDGE_STRONG, EDGE_WEAK, EDGE_WEAK, EDGE_WEAK, EDGE_NONE
    };

    uint8_t output[5] = {0};

    hysteresisEdgeTracing(classified, output, W, H);

    EXPECT_EQ(output[0], EDGE_STRONG);
    EXPECT_EQ(output[1], EDGE_STRONG);
    EXPECT_EQ(output[2], EDGE_STRONG);
    EXPECT_EQ(output[3], EDGE_STRONG);
    EXPECT_EQ(output[4], EDGE_NONE); // never classified as an edge at all
}


TEST(Hysteresis, WeakChainBlockedByGapIsSuppressed)
{
    // Same chain, but a NONE pixel breaks the connection to the strong
    // pixel, so the far weak pixels must be suppressed.
    //
    //   255  0  75  75
    int W = 4;
    int H = 1;

    uint8_t classified[4] =
    {
        EDGE_STRONG, EDGE_NONE, EDGE_WEAK, EDGE_WEAK
    };

    uint8_t output[4] = {0};

    hysteresisEdgeTracing(classified, output, W, H);

    EXPECT_EQ(output[0], EDGE_STRONG);
    EXPECT_EQ(output[1], EDGE_NONE);
    EXPECT_EQ(output[2], EDGE_NONE);
    EXPECT_EQ(output[3], EDGE_NONE);
}


TEST(Hysteresis, CannyFinalEdgesWrapperRunsFullPipeline)
{
    // Exercises the doubleThreshold + hysteresisEdgeTracing convenience
    // wrapper directly on raw NMS-style magnitude input.
    int W = 3;
    int H = 3;

    uint8_t nms_output[9] =
    {
          0,  10,   0,
        120, 200,   0,
          0,   0,   0
    };

    uint8_t* result = cannyFinalEdges(nms_output, W, H, /*low=*/50, /*high=*/150);

    ASSERT_NE(result, nullptr);

    EXPECT_EQ(result[4], EDGE_STRONG); // 200 >= high
    EXPECT_EQ(result[3], EDGE_STRONG); // 120 is weak, but 8-connected to idx 4
    EXPECT_EQ(result[1], EDGE_NONE);   // 10 is below low, never an edge

    free(result);
}
