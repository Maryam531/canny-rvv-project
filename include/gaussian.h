#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include "image.h"
#include <cstdint>

// ── Scalar reference implementation ──────────────────────────────────────
// One-shot call: allocates its own output buffer (and internal scratch),
// returns a new Image. Use for correctness tests / convenience. Not ideal
// for benchmarking because every call performs internal aligned_alloc/free.
Image gaussian_blur(const Image& img);

// ── Step-4 benchmark API (scalar, allocation-free) ────────────────────────
// Mirrors the RVV init/into/free pattern below so the scalar and vector
// paths can be benchmarked on equal footing (no malloc/free inside the
// timed region for either one).

// Call ONCE before the benchmark loop to pre-allocate internal scratch.
void gaussian_blur_init(int W, int H);

// Call inside the timed loop: no heap allocation, writes into out_data
// (W×H bytes, must already be allocated by the caller).
void gaussian_blur_into(const Image& img, uint8_t* out_data);

// Call ONCE after the benchmark loop to release internal scratch.
void gaussian_blur_free();

// ── RVV-optimised implementation ───────────────────────────────────────────
// One-shot call: allocates its own ring buffer, returns a new Image.
// Use this for correctness tests. Not ideal for benchmarking because every
// call performs an internal aligned_alloc/free.
Image gaussian_blur_rvv(const Image& img);

// ── Step-4 benchmark API (eliminates allocation from the timed region) ────
// Call ONCE before your benchmark loop to pre-allocate internal ring buffer.
void gaussian_blur_rvv_init(int W, int H);

// Call inside the timed loop: no heap allocation, writes into out_data
// (W×H bytes, must already be allocated by the caller).
void gaussian_blur_rvv_into(const Image& img, uint8_t* out_data);

// Call ONCE after the benchmark loop to release the ring buffer.
void gaussian_blur_rvv_free();
// ── LMUL sweep entry points (Phase 6, §6.2) ───────────────────────────────
// Same algorithm and same output as gaussian_blur_rvv_into() above, just
// hardcoded to a specific LMUL instead of going through the
// GAUSSIAN_RVV_LMUL compile-time macro. This lets one binary call all
// three variants back to back and print a real sweep table, instead of
// needing three separate builds.
void gaussian_blur_rvv_into_lmul1(const Image& img, uint8_t* out_data);
void gaussian_blur_rvv_into_lmul2(const Image& img, uint8_t* out_data);
void gaussian_blur_rvv_into_lmul4(const Image& img, uint8_t* out_data);
#endif // GAUSSIAN_H
