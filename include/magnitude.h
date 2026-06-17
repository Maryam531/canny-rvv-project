#ifndef MAGNITUDE_H
#define MAGNITUDE_H
#include "image.h"
#include <cstdint>

Image magnitude_l1(const int16_t* gx, const int16_t* gy,
                   int width, int height);

Image magnitude_l2(const int16_t* gx, const int16_t* gy,
                   int width, int height);

               /*RVV_implementation*/

Image magnitude_l1_rvv(const int16_t* gx, const int16_t* gy,
                        int width, int height);

Image magnitude_l2_rvv(const int16_t* gx, const int16_t* gy,
                        int width, int height);

#endif
