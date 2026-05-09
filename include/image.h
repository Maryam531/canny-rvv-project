#ifndef IMAGE_H
#define IMAGE_H

#include <cstdint>

struct Image {
    int width;
    int height;
    uint8_t* data;
};

#endif
