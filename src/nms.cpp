// nms.cpp — Non-Maximum Suppression
//
// For every interior pixel, compares the gradient magnitude against its two
// neighbours along the gradient direction.  If the pixel is the local maximum
// it is kept; otherwise it is zeroed.  This thins edges to single-pixel width.
//
// Direction mapping (matches Direction enum in direction.h):
//   DIR_0   (0°)  → compare left/right           neighbours: (y, x-1) and (y, x+1)
//   DIR_45  (45°) → compare top-right/bot-left   neighbours: (y-1,x+1) and (y+1,x-1)
//   DIR_90  (90°) → compare top/bottom            neighbours: (y-1,x) and (y+1,x)
//   DIR_135 (135°)→ compare top-left/bot-right   neighbours: (y-1,x-1) and (y+1,x+1)

#include "nms.h"
#include <cstring>

void nonMaximumSuppression(
    const uint8_t*   mag,
    const Direction* dir,
    uint8_t*         out,
    int              width,
    int              height)
{
    // Zero the entire output first — border pixels stay 0
    std::memset(out, 0, (size_t)width * height);

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            const int idx     = y * width + x;
            const uint8_t cur = mag[idx];

            uint8_t n1 = 0, n2 = 0;

            switch (dir[idx])
            {
                case Direction::DIR_0:
                    // Horizontal gradient → edge runs vertically
                    // compare left and right neighbours
                    n1 = mag[idx - 1];
                    n2 = mag[idx + 1];
                    break;

                case Direction::DIR_45:
                    // Diagonal (bottom-left to top-right)
                    n1 = mag[(y - 1) * width + (x + 1)];
                    n2 = mag[(y + 1) * width + (x - 1)];
                    break;

                case Direction::DIR_90:
                    // Vertical gradient → edge runs horizontally
                    // compare top and bottom neighbours
                    n1 = mag[(y - 1) * width + x];
                    n2 = mag[(y + 1) * width + x];
                    break;

                case Direction::DIR_135:
                    // Diagonal (top-left to bottom-right)
                    n1 = mag[(y - 1) * width + (x - 1)];
                    n2 = mag[(y + 1) * width + (x + 1)];
                    break;
            }

            // Keep pixel only if it is the local maximum
            if (cur >= n1 && cur >= n2)
                out[idx] = cur;
            // else out[idx] already 0 from memset
        }
    }
}
