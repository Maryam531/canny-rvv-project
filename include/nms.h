#ifndef NMS_H
#define NMS_H

#include <stdint.h>

void nonMaximumSuppression(
    const uint8_t* magnitude,
    const uint8_t* direction,
    uint8_t* output,
    int width,
    int height);

#endif
