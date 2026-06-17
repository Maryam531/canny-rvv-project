#include "magnitude.h"
#include <cstdlib>
#include <cmath>
#include <cassert>

static inline size_t round_up64(size_t n) { return (n + 63u) & ~size_t(63); }

// ── L1: |Gx| + |Gy|, normalised to [0, 255] ─────────────────────────────
Image magnitude_l1(const int16_t* gx, const int16_t* gy,
                   int width, int height)
{
    assert(gx && gy && width > 0 && height > 0);

    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, round_up64((size_t)width * height));
    assert(out.data);

    const int N = width * height;

    // First pass: find maximum
    int32_t max_mag = 1;
    for (int i = 0; i < N; i++) {
        int32_t m = (int32_t)abs(gx[i]) + (int32_t)abs(gy[i]);
        if (m > max_mag) max_mag = m;
    }

    // Second pass: normalise
    for (int i = 0; i < N; i++) {
        int32_t m = (int32_t)abs(gx[i]) + (int32_t)abs(gy[i]);
        int32_t v = m * 255 / max_mag;
        out.data[i] = (uint8_t)(v > 255 ? 255 : v);
    }

    return out;
}

// ── L2: sqrt(Gx² + Gy²), normalised to [0, 255] ─────────────────────────
Image magnitude_l2(const int16_t* gx, const int16_t* gy,
                   int width, int height)
{
    assert(gx && gy && width > 0 && height > 0);

    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, round_up64((size_t)width * height));
    assert(out.data);

    const int N = width * height;

    // First pass: find maximum
    float max_mag = 1.0f;
    for (int i = 0; i < N; i++) {
        float fx = (float)gx[i];
        float fy = (float)gy[i];
        float m  = sqrtf(fx * fx + fy * fy);
        if (m > max_mag) max_mag = m;
    }

    // Second pass: normalise
    const float scale = 255.0f / max_mag;
    for (int i = 0; i < N; i++) {
        float fx = (float)gx[i];
        float fy = (float)gy[i];
        float m  = sqrtf(fx * fx + fy * fy) * scale;
        int32_t v = (int32_t)m;
        out.data[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }

    return out;
}
