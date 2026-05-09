#include "magnitude.h"
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>

// ─────────────────────────────────────────────
// L1 Norm: |Gx| + |Gy|
// Input:  Gx, Gy  — signed 16-bit Sobel outputs
// Output: out     — normalized 8-bit magnitude
// ─────────────────────────────────────────────
void gradient_magnitude_L1(
    const int16_t* Gx,      // horizontal gradient buffer
    const int16_t* Gy,      // vertical gradient buffer
    uint8_t*       out,     // output magnitude (0–255)
    int            width,
    int            height)
{
    int n = width * height;

    // Temporary buffer to hold raw (un-normalized) magnitudes
    // Using int32_t to avoid overflow: max L1 = |32767| + |32767| = 65534
    int32_t* mag = new int32_t[n];

    // ── Pass 1: compute raw magnitudes and find the maximum ──
    int32_t max_mag = 0;

    for (int i = 0; i < n; i++) {
        // std::abs on int16_t gives the absolute value
        int32_t m = std::abs((int32_t)Gx[i]) + std::abs((int32_t)Gy[i]);
        mag[i]  = m;
        if (m > max_mag) max_mag = m;  // track global max
    }

    // ── Pass 2: normalize to [0, 255] ──
    // Each pixel's normalized value = (raw_mag / max_mag) * 255
    // We rearrange to avoid float: (raw_mag * 255) / max_mag
    if (max_mag == 0) {
        // Flat image — no edges at all
        std::memset(out, 0, n);
    } else {
        for (int i = 0; i < n; i++) {
            out[i] = (uint8_t)((mag[i] * 255) / max_mag);
        }
    }

    delete[] mag;
}


// ─────────────────────────────────────────────
// L2 Norm: sqrt(Gx² + Gy²)
// ─────────────────────────────────────────────
void gradient_magnitude_L2(
    const int16_t* Gx,
    const int16_t* Gy,
    uint8_t*       out,
    int            width,
    int            height)
{
    int n = width * height;

    float* mag = new float[n];
    float  max_mag = 0.0f;

    // ── Pass 1 ──
    for (int i = 0; i < n; i++) {
        float gx = (float)Gx[i];
        float gy = (float)Gy[i];
        float m  = std::sqrt(gx * gx + gy * gy);
        mag[i]   = m;
        if (m > max_mag) max_mag = m;
    }

    // ── Pass 2: normalize ──
    if (max_mag == 0.0f) {
        std::memset(out, 0, n);
    } else {
        for (int i = 0; i < n; i++) {
            out[i] = (uint8_t)((mag[i] / max_mag) * 255.0f);
        }
    }

    delete[] mag;
}