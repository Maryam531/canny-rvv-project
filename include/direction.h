#ifndef DIRECTION_H
#define DIRECTION_H
#include <cstdint>

// Four canonical gradient directions used by Canny NMS.
enum class Direction : uint8_t {
    DIR_0   = 0,
    DIR_45  = 1,
    DIR_90  = 2,
    DIR_135 = 3
};

// Quantise a single gradient vector into one of the four directions.
Direction quantiseDirection(int gx, int gy);

// Compute a direction map from Sobel gradient images.
void computeGradientDirections(
    const int16_t* gx,
    const int16_t* gy,
    Direction* dirs,
    int width,
    int height);

