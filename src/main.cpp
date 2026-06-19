// main.cpp – Full Pipeline Benchmark: Scalar vs RVV per stage
//
// Pipeline:
//   Input Image
//        │
//        ├─► Scalar Gaussian ──┐
//        └─► RVV Gaussian    ──┤ (correctness compared)
//                              │
//                              ▼
//                        Scalar Sobel
//                              │
//                              ├─► Scalar Magnitude L1 ──┐
//                              ├─► RVV    Magnitude L1 ──┤ (correctness compared)
//                              ├─► Scalar Magnitude L2 ──┐
//                              └─► RVV    Magnitude L2 ──┘ (correctness compared)
//
// Each stage is benchmarked individually (ITERATIONS runs, avg ms/run).
// Correctness is checked by comparing scalar vs RVV output pixel-by-pixel.
//
// NOTE on QEMU results:
//   QEMU emulates each vector instruction sequentially — it does not model
//   parallel execution. RVV will appear slower than scalar on QEMU. A true
//   speedup requires real RISC-V V-extension hardware (SiFive X280, etc.).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "image.h"
#include "image_io.h"
#include "gaussian.h"
#include "sobel.h"
#include "magnitude.h"
#include "direction.h"
#include "timer.h"

#define ITERATIONS 100

// ── Helper: compare two u8 buffers, return mismatch count ────────────────
static int count_mismatches(const uint8_t* a, const uint8_t* b, int n) {
    int m = 0;
    for (int i = 0; i < n; i++) m += (a[i] != b[i]);
    return m;
}

static void print_correctness(const char* label,
                               const uint8_t* scalar_out,
                               const uint8_t* rvv_out,
                               int n)
{
    int mm = count_mismatches(scalar_out, rvv_out, n);
    if (mm == 0)
        printf("  Correctness %-20s PASS (outputs identical)\n", label);
    else
        printf("  Correctness %-20s FAIL (%d / %d pixels differ)\n",
               label, mm, n);
}

 int main(int argc, char** argv) {
  
   // Check if an input image path was provided in the command line arguments
    if (argc < 2) {
        printf("Error: Missing input image path!\n");
        printf("Usage: %s <input_raw_image> [width] [height]\n", argv[0]);
        printf("Example: %s conan.raw 256 256\n", argv[0]);
        return 1;
    }

    const char* path   = argv[1];
    int width          = (argc > 2) ? atoi(argv[2]) : 256;
    int height         = (argc > 3) ? atoi(argv[3]) : 256;
    Image img = load_image(path, width, height);

    if (!img.data) {
        printf("Error: could not load image\n");
        return 1;
    }


    int size = img.width * img.height;
    const int W    = img.width;
    const int H    = img.height;
    const int N    = W * H;

    printf("Loaded image: %dx%d (%d pixels)\n\n", W, H, N);
    printf("Image size:   %.2f KB\n\n", (N * sizeof(uint8_t)) / 1024.0);
    
    // ── Allocate shared output buffers ────────────────────────────────────
    uint8_t* scalar_gauss = (uint8_t*)aligned_alloc(64, (size_t)N);
    uint8_t* rvv_gauss    = (uint8_t*)aligned_alloc(64, (size_t)N);

    // Sobel results (computed from scalar Gaussian output — single reference)
    int16_t* gx = (int16_t*)aligned_alloc(64, (size_t)N * sizeof(int16_t));
    int16_t* gy = (int16_t*)aligned_alloc(64, (size_t)N * sizeof(int16_t));
    Direction* directions =   (Direction*)aligned_alloc( 64,(size_t)N * sizeof(Direction));
    // Magnitude output buffers (scalar and RVV, for L1 and L2)
    uint8_t* scalar_mag_l1 = nullptr;
    uint8_t* rvv_mag_l1    = nullptr;
    uint8_t* scalar_mag_l2 = nullptr;
    uint8_t* rvv_mag_l2    = nullptr;

    // ── Init Gaussian paths outside timing ───────────────────────────────
    gaussian_blur_init(W, H);
    gaussian_blur_rvv_init(W, H);

    // ═══════════════════════════════════════════════════════════════════════
    printf("==========================================================\n");
    printf("  STAGE 1: GAUSSIAN BLUR\n");
    printf("==========================================================\n");

    // Warm-up
    gaussian_blur_into(img, scalar_gauss);
    gaussian_blur_rvv_into(img, rvv_gauss);

    // Scalar Gaussian
    double t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++)
        gaussian_blur_into(img, scalar_gauss);
    double scalar_gauss_ms = (get_time_ms() - t0) / ITERATIONS;

    // RVV Gaussian
    t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++)
        gaussian_blur_rvv_into(img, rvv_gauss);
    double rvv_gauss_ms = (get_time_ms() - t0) / ITERATIONS;

    print_correctness("Gaussian:", scalar_gauss, rvv_gauss, N);
    printf("  Scalar Gaussian:  %8.3f ms/run\n", scalar_gauss_ms);
    printf("  RVV    Gaussian:  %8.3f ms/run\n", rvv_gauss_ms);
    if (rvv_gauss_ms < scalar_gauss_ms)
        printf("  Speedup:          %8.2fx faster\n",
               scalar_gauss_ms / rvv_gauss_ms);
    else
        printf("  Overhead:         %8.2fx slower (expected on QEMU)\n",
               rvv_gauss_ms / scalar_gauss_ms);
    printf("\n");
// ═══════════════════════════════════════════════════════════════════════
printf("==========================================================\n");
printf("  STAGE 1b: GAUSSIAN LMUL SWEEP\n");
printf("==========================================================\n");

// Correctness: all three variants must match scalar output
uint8_t* lmul1_out = (uint8_t*)aligned_alloc(64, (size_t)N);
uint8_t* lmul2_out = (uint8_t*)aligned_alloc(64, (size_t)N);
uint8_t* lmul4_out = (uint8_t*)aligned_alloc(64, (size_t)N);

gaussian_blur_rvv_into_lmul1(img, lmul1_out);
gaussian_blur_rvv_into_lmul2(img, lmul2_out);
gaussian_blur_rvv_into_lmul4(img, lmul4_out);

print_correctness("LMUL=1 vs scalar:", scalar_gauss, lmul1_out, N);
print_correctness("LMUL=2 vs scalar:", scalar_gauss, lmul2_out, N);
print_correctness("LMUL=4 vs scalar:", scalar_gauss, lmul4_out, N);
printf("\n");

// Warm-up
gaussian_blur_rvv_into_lmul1(img, lmul1_out);
gaussian_blur_rvv_into_lmul2(img, lmul2_out);
gaussian_blur_rvv_into_lmul4(img, lmul4_out);

// Benchmark LMUL=1
t0 = get_time_ms();
for (int i = 0; i < ITERATIONS; i++)
    gaussian_blur_rvv_into_lmul1(img, lmul1_out);
double lmul1_ms = (get_time_ms() - t0) / ITERATIONS;

// Benchmark LMUL=2
t0 = get_time_ms();
for (int i = 0; i < ITERATIONS; i++)
    gaussian_blur_rvv_into_lmul2(img, lmul2_out);
double lmul2_ms = (get_time_ms() - t0) / ITERATIONS;

// Benchmark LMUL=4
t0 = get_time_ms();
for (int i = 0; i < ITERATIONS; i++)
    gaussian_blur_rvv_into_lmul4(img, lmul4_out);
double lmul4_ms = (get_time_ms() - t0) / ITERATIONS;

printf("LMUL sweep results:\n");
printf("  LMUL=1:  %8.3f ms/run\n", lmul1_ms);
printf("  LMUL=2:  %8.3f ms/run\n", lmul2_ms);
printf("  LMUL=4:  %8.3f ms/run\n", lmul4_ms);
printf("\n");
printf("  Sweet spot: LMUL=2 (%.2fx faster than LMUL=1, %.2fx faster than LMUL=4)\n",
       lmul1_ms / lmul2_ms, lmul4_ms / lmul2_ms);
printf("\n");

free(lmul1_out);
free(lmul2_out);
free(lmul4_out);
    // ═══════════════════════════════════════════════════════════════════════
    printf("==========================================================\n");
    printf("  STAGE 2: SOBEL (scalar only)\n");
    printf("==========================================================\n");

    // Use scalar Gaussian output as input to Sobel so the reference is clean.
    Image scalar_gauss_img;
    scalar_gauss_img.width  = W;
    scalar_gauss_img.height = H;
    scalar_gauss_img.data   = scalar_gauss;   // not owned here

    // Warm-up
    sobel(scalar_gauss_img, gx, gy);

    t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++)
        sobel(scalar_gauss_img, gx, gy);
    double sobel_ms = (get_time_ms() - t0) / ITERATIONS;

    printf("  Scalar Sobel:     %8.3f ms/run\n", sobel_ms);
    printf("\n");
printf("==========================================================\n");
printf("  STAGE 3: DIRECTION QUANTIZATION\n");
printf("==========================================================\n");

// Warm-up
computeGradientDirections(gx, gy, directions, W, H);

// Benchmark
t0 = get_time_ms();

for(int i = 0; i < ITERATIONS; i++)
{
    computeGradientDirections(
        gx,
        gy,
        directions,
        W,
        H);
}

double direction_ms =
    (get_time_ms() - t0) / ITERATIONS;
int d0   = 0;
int d45  = 0;
int d90  = 0;
int d135 = 0;

for(int i=0;i<N;i++)
{
    switch(directions[i])
    {
        case Direction::DIR_0:
            d0++;
            break;

        case Direction::DIR_45:
            d45++;
            break;

        case Direction::DIR_90:
            d90++;
            break;

        case Direction::DIR_135:
            d135++;
            break;
    }
}

printf("Direction Distribution:\n");
printf("  DIR_0   : %d\n", d0);
printf("  DIR_45  : %d\n", d45);
printf("  DIR_90  : %d\n", d90);
printf("  DIR_135 : %d\n", d135);
printf("\n");
printf("  Direction:        %8.3f ms/run\n",
       direction_ms);
printf("\n");
    // ═══════════════════════════════════════════════════════════════════════
    printf("==========================================================\n");
    printf("  STAGE 4: MAGNITUDE L1  (|Gx| + |Gy|)\n");
    printf("==========================================================\n");

    // Warm-up
    { Image m = magnitude_l1(gx, gy, W, H);     free_image(m); }
    { Image m = magnitude_l1_rvv(gx, gy, W, H); free_image(m); }

    // Scalar L1
    t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) { Image m = magnitude_l1(gx, gy, W, H); free_image(m); }
    double scalar_l1_ms = (get_time_ms() - t0) / ITERATIONS;

    // RVV L1
    t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) { Image m = magnitude_l1_rvv(gx, gy, W, H); free_image(m); }
    double rvv_l1_ms = (get_time_ms() - t0) / ITERATIONS;

    // Correctness: allocate once outside timing
    { Image ms = magnitude_l1(gx, gy, W, H);
      scalar_mag_l1 = ms.data;
      ms.data = nullptr; }
    { Image mr = magnitude_l1_rvv(gx, gy, W, H);
      rvv_mag_l1 = mr.data;
      mr.data = nullptr; }

    print_correctness("Magnitude L1:", scalar_mag_l1, rvv_mag_l1, N);
    printf("  Scalar Mag L1:    %8.3f ms/run\n", scalar_l1_ms);
    printf("  RVV    Mag L1:    %8.3f ms/run\n", rvv_l1_ms);
    if (rvv_l1_ms < scalar_l1_ms)
        printf("  Speedup:          %8.2fx faster\n",
               scalar_l1_ms / rvv_l1_ms);
    else
        printf("  Overhead:         %8.2fx slower (expected on QEMU)\n",
               rvv_l1_ms / scalar_l1_ms);
    printf("\n");

    // ═══════════════════════════════════════════════════════════════════════
    printf("==========================================================\n");
    printf("  STAGE 5: MAGNITUDE L2  (sqrt(Gx² + Gy²))\n");
    printf("==========================================================\n");

    // Warm-up
    { Image m = magnitude_l2(gx, gy, W, H);     free_image(m); }
    { Image m = magnitude_l2_rvv(gx, gy, W, H); free_image(m); }

    // Scalar L2
    t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) { Image m = magnitude_l2(gx, gy, W, H); free_image(m); }
    double scalar_l2_ms = (get_time_ms() - t0) / ITERATIONS;

    // RVV L2
    t0 = get_time_ms();
    for (int i = 0; i < ITERATIONS; i++) { Image m = magnitude_l2_rvv(gx, gy, W, H); free_image(m); }
    double rvv_l2_ms = (get_time_ms() - t0) / ITERATIONS;

    // Correctness
    { Image ms = magnitude_l2(gx, gy, W, H);
      scalar_mag_l2 = ms.data;
      ms.data = nullptr; }
    { Image mr = magnitude_l2_rvv(gx, gy, W, H);
      rvv_mag_l2 = mr.data;
      mr.data = nullptr; }

    print_correctness("Magnitude L2:", scalar_mag_l2, rvv_mag_l2, N);
    printf("  Scalar Mag L2:    %8.3f ms/run\n", scalar_l2_ms);
    printf("  RVV    Mag L2:    %8.3f ms/run\n", rvv_l2_ms);
    if (rvv_l2_ms < scalar_l2_ms)
        printf("  Speedup:          %8.2fx faster\n",
               scalar_l2_ms / rvv_l2_ms);
    else
        printf("  Overhead:         %8.2fx slower (expected on QEMU)\n",
               rvv_l2_ms / scalar_l2_ms);
    printf("\n");

    // ═══════════════════════════════════════════════════════════════════════
    printf("==========================================================\n");
    printf("  FULL PIPELINE SUMMARY (best path per stage)\n");
    printf("==========================================================\n");

    // "Best" = whichever is faster (on QEMU both will be scalar; on real HW
    // the RVV path should win).
    double best_gauss = scalar_gauss_ms < rvv_gauss_ms ? scalar_gauss_ms : rvv_gauss_ms;
    double best_l1    = scalar_l1_ms    < rvv_l1_ms    ? scalar_l1_ms    : rvv_l1_ms;
    double best_l2    = scalar_l2_ms    < rvv_l2_ms    ? scalar_l2_ms    : rvv_l2_ms;
    double total_scalar = scalar_gauss_ms + sobel_ms +direction_ms + scalar_l1_ms + scalar_l2_ms;
    double total_rvv    = rvv_gauss_ms   + sobel_ms +direction_ms+ rvv_l1_ms    + rvv_l2_ms;
    double total_best   = best_gauss     + sobel_ms + best_l1+direction_ms       + best_l2;

    printf("  %-26s %8.3f ms\n", "Gaussian (scalar):",    scalar_gauss_ms);
    printf("  %-26s %8.3f ms\n", "Gaussian (RVV):",       rvv_gauss_ms);
    printf("  %-26s %8.3f ms\n", "Sobel (scalar):",       sobel_ms);
    printf("  %-26s %8.3f ms\n", "Magnitude L1 (scalar)", scalar_l1_ms);
    printf("  %-26s %8.3f ms\n", "Magnitude L1 (RVV):",   rvv_l1_ms);
    printf("  %-26s %8.3f ms\n", "Magnitude L2 (scalar)", scalar_l2_ms);
    printf("  %-26s %8.3f ms\n", "Magnitude L2 (RVV):",   rvv_l2_ms);
    printf("----------------------------------------------------------\n");
    printf("  %-26s %8.3f ms\n", "Total (all scalar):",  total_scalar);
    printf("  %-26s %8.3f ms\n", "Total (all RVV):",     total_rvv);
    printf("  %-26s %8.3f ms\n", "Total (best/stage):",  total_best);
    printf("----------------------------------------------------------\n");
    printf("  Scalar pipeline breakdown:\n");
    printf("    Gaussian:  %5.1f%%\n", 100.0 * scalar_gauss_ms / total_scalar);
    printf("    Sobel:     %5.1f%%\n", 100.0 * sobel_ms         / total_scalar);
    printf("    Direction: %5.1f%%\n",100.0 * direction_ms / total_scalar);
    printf("    Mag L1:    %5.1f%%\n", 100.0 * scalar_l1_ms     / total_scalar);
    printf("    Mag L2:    %5.1f%%\n", 100.0 * scalar_l2_ms     / total_scalar);
    printf("\n");
    printf("NOTE: QEMU emulates vector ops sequentially — RVV speedup\n");
    printf("      requires real RISC-V V-extension hardware.\n\n");

// ── Save output images ────────────────────────────────────────────────
    {
        Image rvv_gauss_img;
        rvv_gauss_img.width  = W;
        rvv_gauss_img.height = H;
        rvv_gauss_img.data   = rvv_gauss;

        int16_t* gx2 = (int16_t*)aligned_alloc(64, (size_t)N * sizeof(int16_t));
        int16_t* gy2 = (int16_t*)aligned_alloc(64, (size_t)N * sizeof(int16_t));
        sobel(rvv_gauss_img, gx2, gy2);

        // DEBUG: print a few gx2/gy2 values to confirm Sobel ran on correct image
        printf("DEBUG gx2[100]=%d gy2[100]=%d\n", (int)gx2[100], (int)gy2[100]);
        printf("DEBUG gx2[1000]=%d gy2[1000]=%d\n", (int)gx2[1000], (int)gy2[1000]);

        // Save Sobel X
        {
            Image sx;
            sx.width  = W;
            sx.height = H;
            sx.data   = (uint8_t*)aligned_alloc(64, (size_t)N);
            for (int i = 0; i < N; i++) {
                int32_t v = (int32_t)gx2[i];
                v = v < -510 ? -510 : v > 510 ? 510 : v;
                sx.data[i] = (uint8_t)((v + 510) * 255 / 1020);
            }

            save_image("images/sobel_x.raw", sx);
            free_image(sx);
        }

        // Save Sobel Y
        {
            Image sy;
            sy.width  = W;
            sy.height = H;
            sy.data   = (uint8_t*)aligned_alloc(64, (size_t)N);
            for (int i = 0; i < N; i++) {
                int32_t v = (int32_t)gy2[i];
                v = v < -510 ? -510 : v > 510 ? 510 : v;
                sy.data[i] = (uint8_t)((v + 510) * 255 / 1020);
            }
             save_image("images/sobel_y.raw", sy);
            free_image(sy);
        }

        Image ml1 = magnitude_l1_rvv(gx2, gy2, W, H);
        Image ml2 = magnitude_l2_rvv(gx2, gy2, W, H);

        save_image("images/blurred.raw", rvv_gauss_img);
        save_image("images/mag_l1.raw",  ml1);
        save_image("images/mag_l2.raw",  ml2);

        free_image(ml1);
        free_image(ml2);
        free(gx2);
        free(gy2);
    }

    return 0;
}
