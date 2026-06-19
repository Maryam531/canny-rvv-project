#ifndef NMS_H
#define NMS_H

#include <cstdint>
#include "direction.h"

// Non-maximum suppression using the Direction enum from direction.h.
// Pixels that are NOT local maxima along their gradient direction are zeroed.
// Border pixels (1-pixel wide) are always set to 0.
void nonMaximumSuppression(
    const uint8_t*   magnitude,  // normalised magnitude map [0,255]
    const Direction* direction,  // quantised gradient direction per pixel
    uint8_t*         output,     // suppressed output (caller-allocated, W*H bytes)
    int              width,
    int              height);

#endif // NMS_H
