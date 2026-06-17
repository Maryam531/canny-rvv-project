// main.cpp – Phase 6 v4: RVV vs Scalar Performance Harness (FAIR BENCHMARK)
//
// Correctness fix (v3 → v4):
//   Root cause: gaussian.cpp used M=30724, S=23 for its div-by-273
//   reciprocal, which is INCORRECT. e.g. x=273 → (273*30724)>>23 = 0,
//   but 273/273 = 1 (wrong). This produced 1137 wrong accumulator values
//   in the scalar path's range [0, 69615].
//
//   rvv_gaussian.cpp's scalar_pixel() already used the correct pair
//   M=122911, S=25 (verified exhaustively for x in [0, 73695], 0 errors).
//
//   Fix: gaussian.cpp now uses the same M=122911, S=25 reciprocal,
//   making both scalar_pixel() implementations identical and ensuring
//   border pixels match between scalar and RVV paths.
//
//   The i64 widening in the RVV vertical pass (also M=122911, S=25) was
//   already correct in v3 and is unchanged here.
//
// NOTE on QEMU results:
//   On QEMU, RVV is typically slower than scalar because QEMU emulates each
//   vector instruction sequentially — it does not simulate parallel execution.
//   A speedup > 1x from RVV requires real RISC-V vector hardware (e.g. SiFive
//   X280, StarFive JH7110, or an FPGA softcore with V extension). The numbers
//   here reflect QEMU overhead, NOT the performance on real hardware.

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
    // ── Load image ────────────────────────────────────────────────────────
    Image img = load_image("images/rectangle.raw", 256, 256);
    if (!img.data) {
        printf("Error: could not load image\n");
        return 1;
    }

    const int W    = img.width;
    const int H    = img.height;
    const int size = W * H;

    printf("Loaded image size: %dx%d (%d pixels)\n\n", W, H, size);

    // ── Pre-allocate reusable output buffers ──────────────────────────────
    uint8_t* scalar_out_data = (uint8_t*)aligned_alloc(64, (size_t)size);
    uint8_t* rvv_out_data    = (uint8_t*)aligned_alloc(64, (size_t)size);

    // ── Init both paths OUTSIDE timing ───────────────────────────────────
    gaussian_blur_init(W, H);       // scalar: no-op, keeps API symmetric
    gaussian_blur_rvv_init(W, H);   // RVV: allocates ring buffer once

    // Shared Sobel buffers
    int16_t* gx = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));
    int16_t* gy = (int16_t*)aligned_alloc(64, size * sizeof(int16_t));

    // ── Scalar Gaussian Benchmark ─────────────────────────────────────────
    printf("Running Scalar Gaussian Benchmark (%d iterations)... ", ITERATIONS);
    fflush(stdout);

    gaussian_blur_into(img, scalar_out_data);   // warm-up

    double t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        gaussian_blur_into(img, scalar_out_data);
    }
    double t1 = get_time_ms();
    double scalar_gaussian_ms = (t1 - t0) / ITERATIONS;
    printf("Done.\n");

    // ── RVV Gaussian Benchmark ────────────────────────────────────────────
    printf("Running RVV Vector Gaussian Benchmark (%d iterations)... ", ITERATIONS);
    fflush(stdout);

    gaussian_blur_rvv_into(img, rvv_out_data);  // warm-up

    double t2 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        gaussian_blur_rvv_into(img, rvv_out_data);
    }
    double t3 = get_time_ms();
    double rvv_gaussian_ms = (t3 - t2) / ITERATIONS;
    printf("Done.\n");

    // ── Correctness check ─────────────────────────────────────────────────
    int mismatches = 0;
    for (int i = 0; i < size; i++) {
        if (scalar_out_data[i] != rvv_out_data[i]) mismatches++;
    }
    if (mismatches == 0)
        printf("Correctness: PASS (scalar and RVV outputs are identical)\n\n");
    else
        printf("Correctness: FAIL (%d pixels differ between scalar and RVV)\n\n", mismatches);

    // ── Print comparison ──────────────────────────────────────────────────
    printf("==================================================\n");
    printf("              PERFORMANCE COMPARISON\n");
    printf("==================================================\n");
    printf("Gaussian Blur (Scalar):   %8.3f ms (avg per run)\n", scalar_gaussian_ms);
    printf("Gaussian Blur (Vector):   %8.3f ms (avg per run)\n", rvv_gaussian_ms);
    printf("--------------------------------------------------\n");
    if (rvv_gaussian_ms < scalar_gaussian_ms)
        printf("VECTOR SPEEDUP:           %8.2fx faster\n",
               scalar_gaussian_ms / rvv_gaussian_ms);
    else
        printf("VECTOR OVERHEAD:          %8.2fx slower (expected on QEMU)\n",
               rvv_gaussian_ms / scalar_gaussian_ms);
    printf("==================================================\n");
    printf("NOTE: QEMU emulates vector instructions sequentially.\n");
    printf("      Speedup requires real RISC-V V-extension hardware.\n\n");

    // ── Full pipeline breakdown (using RVV Gaussian output) ───────────────
    Image rvv_blurred;
    rvv_blurred.width  = W;
    rvv_blurred.height = H;
    rvv_blurred.data   = rvv_out_data;   // NOT owned here

    // Sobel
    double t4 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) sobel(rvv_blurred, gx, gy);
    double t5       = get_time_ms();
    double sobel_ms = (t5 - t4) / ITERATIONS;

    // Magnitude L1
    double t6 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) { Image m = magnitude_l1(gx, gy, W, H); free_image(m); }
    double t7        = get_time_ms();
    double mag_l1_ms = (t7 - t6) / ITERATIONS;

    // Magnitude L2
    double t8 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) { Image m = magnitude_l2(gx, gy, W, H); free_image(m); }
    double t9        = get_time_ms();
    double mag_l2_ms = (t9 - t8) / ITERATIONS;

    double total_ms = rvv_gaussian_ms + sobel_ms + mag_l1_ms + mag_l2_ms;

    printf("--- Full Pipeline Breakdown (Using RVV Gaussian) ---\n");
    printf("Gaussian (RVV): %.1f%%\n", 100.0 * rvv_gaussian_ms / total_ms);
    printf("Sobel:          %.1f%%\n", 100.0 * sobel_ms          / total_ms);
    printf("Magnitude L1:   %.1f%%\n", 100.0 * mag_l1_ms         / total_ms);
    printf("Magnitude L2:   %.1f%%\n", 100.0 * mag_l2_ms         / total_ms);
    printf("Total Pipeline Latency:   %.3f ms (avg per run)\n", total_ms);

    // ── Save outputs ──────────────────────────────────────────────────────
    Image mag_l1 = magnitude_l1(gx, gy, W, H);
    Image mag_l2 = magnitude_l2(gx, gy, W, H);
    save_image("images/blurred.raw", rvv_blurred);
    save_image("images/mag_l1.raw",  mag_l1);
    save_image("images/mag_l2.raw",  mag_l2);
    free_image(mag_l1);
    free_image(mag_l2);

    // ── Cleanup ───────────────────────────────────────────────────────────
    gaussian_blur_free();
    gaussian_blur_rvv_free();
    free_image(img);
    free(scalar_out_data);
    free(rvv_out_data);
    free(gx);
    free(gy);

    return 0;
}
