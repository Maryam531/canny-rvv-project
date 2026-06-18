#include "nms.h"

void nonMaximumSuppression(
    const uint8_t* mag,
    const uint8_t* dir,
    uint8_t* out,
    int width,
    int height)
{
    for(int y = 1; y < height - 1; y++)
    {
        for(int x = 1; x < width - 1; x++)
        {
            int idx = y * width + x;

            uint8_t current = mag[idx];

            uint8_t n1 = 0;
            uint8_t n2 = 0;

            switch(dir[idx])
            {
                case 0:
                    n1 = mag[idx - 1];
                    n2 = mag[idx + 1];
                    break;

                case 1:
                    n1 = mag[(y - 1) * width + (x + 1)];
                    n2 = mag[(y + 1) * width + (x - 1)];
                    break;

                case 2:
                    n1 = mag[(y - 1) * width + x];
                    n2 = mag[(y + 1) * width + x];
                    break;

                case 3:
                    n1 = mag[(y - 1) * width + (x - 1)];
                    n2 = mag[(y + 1) * width + (x + 1)];
                    break;
            }

            if(current >= n1 && current >= n2)
                out[idx] = current;
            else
                out[idx] = 0;
        }
    }
}
