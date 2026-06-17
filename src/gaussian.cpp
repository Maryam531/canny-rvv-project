#include "gaussian.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ── Kernel constants (must match rvv_gaussian.cpp exactly) ────────────────
static constexpr int32_t K1D[5]     = {1, 4, 7, 4, 1};
static constexpr int32_t KERNEL_SUM = 273;
static constexpr int     RADIUS     = 2;
static constexpr int     KSIZE      = 2 * RADIUS + 1;   // 5

// ── Reciprocal-multiply replacement for /273 ─────────────────────────────
// floor(x / 273) == floor(x * M >> S) for M = 122911, S = 25.
//
// This MUST match rvv_gaussian.cpp exactly so that both implementations
// produce identical results for border pixels (which both handle via the
// scalar fallback path).
//
// Why not M=30724, S=23?
//   That pair is INCORRECT: e.g. x=273 → (273*30724)>>23 = 0, but 273/273 = 1.
//   It under-counts at every exact multiple of 273 up to 69615 (1137 errors).
//
// M=122911, S=25 is exhaustively verified correct for x in [0, 73695]:
//   max scalar acc = 255 * 273 = 69615 < 73695 — well within range.
//   69615 * 122911 = 8,557,513,665 < 2^63 — safe as int64 intermediate.
static constexpr int64_t RECIP_M = 122911;
static constexpr int     RECIP_S = 25;

static inline int32_t div273_scalar(int32_t x) {
    return (int32_t)(((int64_t)x * RECIP_M) >> RECIP_S);
}

static inline size_t round_up64(size_t n) {
    return (n + 63u) & ~size_t(63);
}

// ── 2D kernel, used directly by the scalar path ───────────────────────────
static const int16_t KERNEL2D[5][5] = {
    { 1,  4,  7,  4, 1},
    { 4, 16, 26, 16, 4},
    { 7, 26, 41, 26, 7},
    { 4, 16, 26, 16, 4},
    { 1,  4,  7,  4, 1}
};

static inline uint8_t scalar_pixel(const uint8_t* in, int W, int H, int x, int y) {
    int32_t acc = 0;
    for (int ky = -RADIUS; ky <= RADIUS; ky++) {
        const int sy = y + ky;
        if (sy < 0 || sy >= H) continue;
        const uint8_t* row = in + sy * W;
        for (int kx = -RADIUS; kx <= RADIUS; kx++) {
            const int sx = x + kx;
            if (sx < 0 || sx >= W) continue;
            acc += (int32_t)row[sx] * (int32_t)KERNEL2D[ky + RADIUS][kx + RADIUS];
        }
    }
    const int32_t r = div273_scalar(acc);
    return (uint8_t)(r < 0 ? 0 : r > 255 ? 255 : r);
}

// ── Core scalar convolution (no allocation) ────────────────────────────────
// Pure brute-force 5×5, zero-padded. No ring buffer needed for the scalar
// path since there's no vector-width chunking to amortize row reuse over;
// keeping it simple keeps this a meaningful "naive baseline" for comparison.
static void gaussian_blur_scalar_core(const uint8_t* __restrict__ in,
                                       uint8_t* __restrict__ out,
                                       int W, int H)
{
    for (int y = 0; y < H; y++) {
        uint8_t* out_row = out + y * W;
        for (int x = 0; x < W; x++) {
            out_row[x] = scalar_pixel(in, W, H, x, y);
        }
    }
}

// ── Step-4 benchmark API (scalar, allocation-free) ─────────────────────────
// No persistent scratch is actually needed for the brute-force scalar path
// (unlike the RVV path's ring buffer), but we keep init/free as no-ops so
// main.cpp can call a symmetric API for both benchmarks.

void gaussian_blur_init(int /*W*/, int /*H*/) {
    // Nothing to allocate — scalar_pixel() reads directly from the input
    // image with zero scratch state.
}

void gaussian_blur_free() {
    // Nothing to release.
}

void gaussian_blur_into(const Image& img, uint8_t* out_data) {
    assert(img.data != nullptr && out_data != nullptr);
    gaussian_blur_scalar_core(img.data, out_data, img.width, img.height);
}

// ── One-shot convenience version (allocates its own output buffer) ───────
Image gaussian_blur(const Image& img) {
    assert(img.data   != nullptr);
    assert(img.width  >  0);
    assert(img.height >  0);

    const int W = img.width;
    const int H = img.height;

    Image out;
    out.width  = W;
    out.height = H;
    out.data   = (uint8_t*)aligned_alloc(64, round_up64((size_t)W * H));
    assert(out.data != nullptr);

    gaussian_blur_scalar_core(img.data, out.data, W, H);
    return out;
}
