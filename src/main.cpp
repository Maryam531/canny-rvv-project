#include <cstdio>
#include <cstdlib>
#include "image.h"
#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "timer.h"

#define ITERATIONS 100

int main() {
    // Load test image
    Image img = load_image("images/rectangle.raw", 256, 256);
    if (!img.data) {
        printf("Error: could not load image\n");
        return 1;
    }
    printf("Loaded image: %dx%d\n", img.width, img.height);

    int size = img.width * img.height;
    int16_t* gx = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));
    int16_t* gy = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));

    // ── Stage 1: Gaussian blur ──────────────────────────────────────────
    double t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        Image blurred = gaussian_blur(img);
        if (i < ITERATIONS - 1) free_image(blurred);
        else {
            // Stage 2: Sobel
            double t1 = get_time_ms();
            printf("Gaussian blur:  %.3f ms (avg over %d runs)\n",
                   (t1 - t0) / ITERATIONS, ITERATIONS);

            double t2 = get_time_ms();
            for (int j = 0; j < ITERATIONS; j++)
                sobel(blurred, gx, gy);
            double t3 = get_time_ms();
            printf("Sobel:          %.3f ms (avg over %d runs)\n",
                   (t3 - t2) / ITERATIONS, ITERATIONS);

            // Stage 3: Magnitude L1
            double t4 = get_time_ms();
            for (int j = 0; j < ITERATIONS; j++) {
                Image m = magnitude_l1(gx, gy, img.width, img.height);
                free_image(m);
            }
            double t5 = get_time_ms();
            printf("Magnitude L1:   %.3f ms (avg over %d runs)\n",
                   (t5 - t4) / ITERATIONS, ITERATIONS);

            // Stage 4: Magnitude L2
            double t6 = get_time_ms();
            for (int j = 0; j < ITERATIONS; j++) {
                Image m = magnitude_l2(gx, gy, img.width, img.height);
                free_image(m);
            }
            double t7 = get_time_ms();
            printf("Magnitude L2:   %.3f ms (avg over %d runs)\n",
                   (t7 - t6) / ITERATIONS, ITERATIONS);

            // Save results
            Image mag_l1 = magnitude_l1(gx, gy, img.width, img.height);
            Image mag_l2 = magnitude_l2(gx, gy, img.width, img.height);
            save_image("images/blurred.raw", blurred);
            save_image("images/mag_l1.raw",  mag_l1);
            save_image("images/mag_l2.raw",  mag_l2);
            printf("Results saved to images/\n");

            free_image(blurred);
            free_image(mag_l1);
            free_image(mag_l2);
        }
    }

    free_image(img);
    free(gx);
    free(gy);
    return 0;
}
