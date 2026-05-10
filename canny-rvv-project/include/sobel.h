#ifndef SOBEL_H
#define SOBEL_H

#include "image.h"
#include <cstdint>

void sobel(
    const Image& img,
    int16_t* gx,
    int16_t* gy
);

#endif
