// threshold.cpp — Double Thresholding
//
// Classifies each pixel from the NMS output into one of three classes:
//   STRONG (255) : definite edge, magnitude >= high threshold
//   WEAK   (75)  : candidate edge, magnitude in [low, high)
//   NONE   (0)   : not an edge, magnitude < low threshold
//
// The weak edges are then resolved by hysteresis: a weak pixel becomes a
// final edge only if it is 8-connected to at least one strong pixel.

#include "threshold.h"

void doubleThreshold(
    const uint8_t* input,
    uint8_t*       output,
    int            width,
    int            height,
    uint8_t        low,
    uint8_t        high)
{
    const int N = width * height;

    for (int i = 0; i < N; i++)
    {
        if (input[i] >= high)
            output[i] = EDGE_STRONG;
        else if (input[i] >= low)
            output[i] = EDGE_WEAK;
        else
            output[i] = EDGE_NONE;
    }
}
