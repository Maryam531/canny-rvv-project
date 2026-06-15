#ifndef DIRECTION_H
#define DIRECTION_H
#include <cstdint>
enum class Direction : uint8_t {
    DIR_0   = 0,   // horizontal  (0°)
    DIR_45  = 1,   // diagonal    (45°)
    DIR_90  = 2,   // vertical    (90°)
    DIR_135 = 3,   // anti-diagonal (135°)
};

Direction quantiseDirection(int gx, int gy);
void computeGradientDirections(
    const int16_t* gx,
    const int16_t* gy,
    Direction*     dirs,
    int            width,
    int            height);

#endif
