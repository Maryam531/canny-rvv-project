#ifndef HYSTERESIS_H
#define HYSTERESIS_H

#include <cstdint>
#include "threshold.h"   // for EDGE_STRONG / EDGE_WEAK / EDGE_NONE

// Hysteresis edge tracing (Stage 5 of the Canny algorithm).
//
// Input  : the classified map produced by doubleThreshold()
//          pixels are EDGE_STRONG (255), EDGE_WEAK (75), or EDGE_NONE (0)
// Output : final binary edge map
//          pixels are either 255 (final edge) or 0 (not an edge)
//
// Algorithm:
//   1. Every STRONG pixel is a confirmed edge.
//   2. Every WEAK pixel that is 8-connected to at least one STRONG pixel
//      becomes a confirmed edge.
//   3. All remaining WEAK pixels are suppressed to 0.
//
// The function uses an iterative flood-fill seeded from strong pixels.
// The output buffer must be caller-allocated with width * height bytes.
// It is safe to call with output == input (in-place operation).
void hysteresisEdgeTracing(
    const uint8_t* classified,   // input from doubleThreshold()
    uint8_t*       output,       // final binary edge map
    int            width,
    int            height);

// Convenience wrapper: runs doubleThreshold + hysteresisEdgeTracing in one call.
// Allocates and returns a new image buffer (caller must free()).
// low and high are passed directly to doubleThreshold.
uint8_t* cannyFinalEdges(
    const uint8_t* nms_output,   // pixel data after NMS
    int            width,
    int            height,
    uint8_t        low,
    uint8_t        high);

#endif // HYSTERESIS_H
