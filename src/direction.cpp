// gradient_direction.cpp
// Section 2.5 — Gradient Direction Quantisation
//
// Quantises the 2-D gradient vector (Gx, Gy) at every pixel into one of
#include "direction.h"

#include <cassert>
#include <cstdlib>

// -----------------------------------------------------------------------------
// Quantise one gradient vector into {0°,45°,90°,135°}
//
// Uses integer arithmetic only.
// Boundary angles:
//   22.5° ≈ 2/5
//   67.5° ≈ 12/5
// -----------------------------------------------------------------------------
Direction quantiseDirection(int gx, int gy)
{
    const int ax = std::abs(gx);
    const int ay = std::abs(gy);

    // Near horizontal (0°)
    if (ay * 5 < ax * 2)
        return Direction::DIR_0;

    // Near diagonal (45° or 135°)
    if (ay * 5 < ax * 12)
    {
        const bool sameSign = (gx >= 0) == (gy >= 0);

        return sameSign
            ? Direction::DIR_45
            : Direction::DIR_135;
    }

    // Near vertical (90°)
    return Direction::DIR_90;
}

// -----------------------------------------------------------------------------
// Compute full direction map
// -----------------------------------------------------------------------------
void computeGradientDirections(
    const int16_t* gx,
    const int16_t* gy,
    Direction* dirs,
    int width,
    int height)
{
    assert(gx   != nullptr);
    assert(gy   != nullptr);
    assert(dirs != nullptr);
    assert(width  > 0);
    assert(height > 0);

    const int N = width * height;

    for (int i = 0; i < N; ++i)
    {
        dirs[i] = quantiseDirection(gx[i], gy[i]);
    }
}
