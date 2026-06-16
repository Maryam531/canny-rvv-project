#include "magnitude.h"
#include <cstdlib>
#include <cmath>

// L1 norm: |Gx| + |Gy| — fast, no floating point
Image magnitude_l1(const int16_t* gx, const int16_t* gy,
                   int width, int height) {
    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, width * height);

    // First pass: find maximum magnitude
    int32_t max_mag = 1;
    for (int i = 0; i < width * height; i++) {
        int32_t mag = abs(gx[i]) + abs(gy[i]);
        if (mag > max_mag) max_mag = mag;
    }

    // Second pass: normalize to [0, 255]
    for (int i = 0; i < width * height; i++) {
        int32_t mag = abs(gx[i]) + abs(gy[i]);
        out.data[i] = (uint8_t)(mag * 255 / max_mag);
    }

    return out;
}

// L2 norm: sqrt(Gx^2 + Gy^2) — more accurate
Image magnitude_l2(const int16_t* gx, const int16_t* gy,
                   int width, int height) {
    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, width * height);

    // First pass: find maximum magnitude
    float max_mag = 1.0f;
    for (int i = 0; i < width * height; i++) {
        float mag = sqrtf((float)gx[i]*gx[i] + (float)gy[i]*gy[i]);
        if (mag > max_mag) max_mag = mag;
    }

    // Second pass: normalize to [0, 255]
    for (int i = 0; i < width * height; i++) {
        float mag = sqrtf((float)gx[i]*gx[i] + (float)gy[i]*gy[i]);
        out.data[i] = (uint8_t)(mag * 255.0f / max_mag);
    }

    return out;
}
