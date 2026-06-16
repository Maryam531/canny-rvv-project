#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "image.h"
#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "timer.h"

#define ITERATIONS 100

int main() {
    Image img = load_image("images/rectangle.raw", 256, 256);

    if (!img.data) {
        printf("Error: could not load image\n");
        return 1;
    }

    printf("Loaded image: %dx%d\n", img.width, img.height);

    int size = img.width * img.height;

    int16_t* gx = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));
    int16_t* gy = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));

    // Variables to accumulate total elapsed times for each stage
    double total_gaussian_time = 0.0;
    double total_sobel_time = 0.0;
    double total_mag_l1_time = 0.0;
    double total_mag_l2_time = 0.0;

    // ── Stage 1: Gaussian ─────────────────────────────
    double t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        Image blurred = gaussian_blur(img);

        if (i < ITERATIONS - 1) {
            free_image(blurred);
        } else {
            double t1 = get_time_ms();
            total_gaussian_time = t1 - t0;
            printf("Gaussian blur:  %.3f ms (avg)\n", total_gaussian_time / ITERATIONS);

            // ── Stage 2: Sobel ─────────────────────────────
            double t2 = get_time_ms();
            for (int j = 0; j < ITERATIONS; j++) {
                sobel(blurred, gx, gy);
            }
            double t3 = get_time_ms();
            total_sobel_time = t3 - t2;
            printf("Sobel:          %.3f ms (avg)\n", total_sobel_time / ITERATIONS);

            // ── Convert Sobel outputs to images ────────────
            Image sobel_x;
            Image sobel_y;

            sobel_x.width = img.width;
            sobel_x.height = img.height;
            sobel_x.data = (uint8_t*)malloc(size);

            sobel_y.width = img.width;
            sobel_y.height = img.height;
            sobel_y.data = (uint8_t*)malloc(size);

            for (int k = 0; k < size; k++) {
                int vx = abs(gx[k]);
                int vy = abs(gy[k]);

                if (vx > 255) vx = 255;
                if (vy > 255) vy = 255;

                sobel_x.data[k] = (uint8_t)vx;
                sobel_y.data[k] = (uint8_t)vy;
            }

            // ── Stage 3: Magnitude L1 ─────────────────────
            double t4 = get_time_ms();
            for (int j = 0; j < ITERATIONS; j++) {
                Image m = magnitude_l1(gx, gy, img.width, img.height);
                free_image(m);
            }
            double t5 = get_time_ms();
            total_mag_l1_time = t5 - t4;
            printf("Magnitude L1:   %.3f ms (avg)\n", total_mag_l1_time / ITERATIONS);

            // ── Stage 4: Magnitude L2 ─────────────────────
            double t6 = get_time_ms();
            for (int j = 0; j < ITERATIONS; j++) {
                Image m = magnitude_l2(gx, gy, img.width, img.height);
                free_image(m);
            }
            double t7 = get_time_ms();
            total_mag_l2_time = t7 - t6;
            printf("Magnitude L2:   %.3f ms (avg)\n", total_mag_l2_time / ITERATIONS);

            // ── Save outputs ───────────────────────────────
            Image mag_l1 = magnitude_l1(gx, gy, img.width, img.height);
            Image mag_l2 = magnitude_l2(gx, gy, img.width, img.height);

            save_image("images/blurred.raw", blurred);
            save_image("images/sobel_x.raw", sobel_x);
            save_image("images/sobel_y.raw", sobel_y);
            save_image("images/mag_l1.raw", mag_l1);
            save_image("images/mag_l2.raw", mag_l2);

            printf("Results saved to images/\n");

            // ── Phase 5: Profiling Breakdown ────────────────
            double total_pipeline_time = total_gaussian_time + total_sobel_time + total_mag_l1_time + total_mag_l2_time;

            printf("\n--- Phase 5: Profiling Breakdown ---\n");
            printf("Gaussian:     %.1f%%\n", (total_gaussian_time / total_pipeline_time) * 100.0);
            printf("Sobel:        %.1f%%\n", (total_sobel_time / total_pipeline_time) * 100.0);
            printf("Magnitude L1: %.1f%%\n", (total_mag_l1_time / total_pipeline_time) * 100.0);
            printf("Magnitude L2: %.1f%%\n", (total_mag_l2_time / total_pipeline_time) * 100.0);
            printf("Total Pipeline Running Cost: %.3f ms (avg per full pipeline run)\n", total_pipeline_time / ITERATIONS);

            // Clean up stage-allocated images
            free_image(blurred);
            free_image(sobel_x);
            free_image(sobel_y);
            free_image(mag_l1);
            free_image(mag_l2);
        }
    }

    free_image(img);
    free(gx);
    free(gy);

    return 0;
}
