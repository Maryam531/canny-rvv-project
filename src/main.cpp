#include <cstdio>
#include <cstdlib>
#include "image.h"
#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "timer.h"

int main() {
    // Load test image
    Image img = load_image("images/rectangle.raw", 256, 256);
    if (!img.data) {
        printf("Error: could not load image\n");
        return 1;
    }
    printf("Loaded image: %dx%d\n", img.width, img.height);

    // Stage 1: Gaussian blur
    double t0 = get_time_ms();
    Image blurred = gaussian_blur(img);
    double t1 = get_time_ms();
    printf("Gaussian blur:  %.3f ms\n", t1 - t0);

    // Stage 2: Sobel gradients
    int size = img.width * img.height;
    int16_t* gx = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));
    int16_t* gy = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));

    double t2 = get_time_ms();
    sobel(blurred, gx, gy);
    double t3 = get_time_ms();
    printf("Sobel:          %.3f ms\n", t3 - t2);

    // Stage 3: Magnitude (L1 and L2)
    double t4 = get_time_ms();
    Image mag_l1 = magnitude_l1(gx, gy, img.width, img.height);
    double t5 = get_time_ms();
    printf("Magnitude L1:   %.3f ms\n", t5 - t4);

    double t6 = get_time_ms();
    Image mag_l2 = magnitude_l2(gx, gy, img.width, img.height);
    double t7 = get_time_ms();
    printf("Magnitude L2:   %.3f ms\n", t7 - t6);

    // Save results
    save_image("images/blurred.raw",  blurred);
    save_image("images/mag_l1.raw",   mag_l1);
    save_image("images/mag_l2.raw",   mag_l2);
    printf("Results saved to images/\n");

    // Cleanup
    free_image(img);
    free_image(blurred);
    free_image(mag_l1);
    free_image(mag_l2);
    free(gx);
    free(gy);

    return 0;
}
