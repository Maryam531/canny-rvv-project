#include "threshold.h"

void doubleThreshold(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    uint8_t low,
    uint8_t high)
{
    int size = width * height;

    for(int i = 0; i < size; i++)
    {
        if(input[i] >= high)
            output[i] = 255;
        else if(input[i] >= low)
            output[i] = 75;
        else
            output[i] = 0;
    }
}
