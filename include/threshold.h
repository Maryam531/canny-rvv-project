#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <stdint.h>

void doubleThreshold(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    uint8_t low,
    uint8_t high);

#endif
