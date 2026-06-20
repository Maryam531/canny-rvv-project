#ifndef THRESHOLD_H

#define THRESHOLD_H



#include <cstdint>



// Pixel classification values written by doubleThreshold

static constexpr uint8_t EDGE_STRONG = 255;

static constexpr uint8_t EDGE_WEAK   =  75;

static constexpr uint8_t EDGE_NONE   =   0;



// Double thresholding:

//   pixel >= high  → EDGE_STRONG (255)

//   pixel >= low   → EDGE_WEAK   (75)

//   otherwise      → EDGE_NONE   (0)

//

// Typical defaults: low = 51 (20% of 255), high = 102 (40% of 255).

// Output is used directly as input to hysteresis edge tracing.

void doubleThreshold(
<<<<<<< HEAD

    const uint8_t* input,

    uint8_t*       output,

    int            width,

    int            height,

    uint8_t        low,

    uint8_t        high);



#endif 
