// gradient_direction.cpp
// Section 2.5 — Gradient Direction Quantisation
//
// Quantises the 2-D gradient vector (Gx, Gy) at every pixel into one of
// four canonical directions used by the Canny non-maximum suppression step:
//
//   DIR_0    (0°)   — horizontal edge,  gradient points left/right
//   DIR_45   (45°)  — diagonal edge,    gradient points NE/SW
//   DIR_90   (90°)  — vertical edge,    gradient points up/down
//   DIR_135  (135°) — anti-diagonal,    gradient points NW/SE
//
// No atan2(), no division, no floating-point arithmetic.
// All comparisons use integer cross-multiplication.
//
// Build:
//   g++ -std=c++20 -O2 -Wall -Wextra -o gradient_direction gradient_direction.cpp
//
// Run self-tests:
//   ./gradient_direction
//
// ============================================================================
// Mathematical basis
// ============================================================================
//
// The full [0°, 180°) half-plane is divided into four equal sectors by
// decision boundaries at 22.5°, 67.5°, 112.5°, and 157.5°.
//
// Working with absolute values ax = |Gx|, ay = |Gy|, the true angle
// θ = atan2(ay, ax) lies in [0°, 90°].  The two interior boundaries are:
//
//   θ < 22.5°  ⟺  ay/ax < tan(22.5°) ≈ 0.4142 ≈ 2/5
//              ⟺  ay * 5 < ax * 2      (cross-multiply, no division)
//
//   θ < 67.5°  ⟺  ay/ax < tan(67.5°) ≈ 2.4142 ≈ 12/5
//              ⟺  ay * 5 < ax * 12     (cross-multiply, no division)
//
// The rational approximations 2/5 and 12/5 introduce an absolute angular
// error of < 0.03° relative to the true tangent values — negligible for
// edge-direction quantisation.
//
// Decision table (ax = |Gx|, ay = |Gy|):
// ─────────────────────────────────────────────────────────────────────
//   ay*5 < ax*2    → θ near  0°  → DIR_0   (horizontal)
//   ay*5 < ax*12   → θ near 45°  → DIR_45 or DIR_135 (sign check below)
//   otherwise      → θ near 90°  → DIR_90  (vertical)
//
// The 45°/135° ambiguity is resolved by checking sign agreement between
// the original signed Gx and Gy:
//   same sign  (NE or SW quadrant) → DIR_45
//   diff sign  (NW or SE quadrant) → DIR_135
//
// Overflow analysis:
//   Sobel output fits in int16_t (max ±32 767).
//   Largest cross-product: 32 767 × 12 = 393 204 — well within int32_t.
//
// ============================================================================

#include <cassert>
#include <cstdint>     // int16_t, int32_t, uint8_t
#include <cstdlib>     // std::abs
#include <iostream>

// ============================================================================
// Direction — four canonical edge orientations.
//
// Stored as uint8_t so a direction map has the same memory footprint as a
// grayscale image (one byte per pixel).
//
// The comment on each enumerator shows which neighbour pair the non-maximum
// suppression stage should compare for that direction.
// ============================================================================
enum class Direction : uint8_t {
    DIR_0   = 0,   // horizontal  — compare (x-1,y)   vs (x+1,y)
    DIR_45  = 1,   // diagonal    — compare (x+1,y-1) vs (x-1,y+1)
    DIR_90  = 2,   // vertical    — compare (x,y-1)   vs (x,y+1)
    DIR_135 = 3,   // anti-diag   — compare (x-1,y-1) vs (x+1,y+1)
};

// ============================================================================
// quantiseDirection — classify one gradient vector into a Direction.
//
// Parameters:
//   gx — horizontal Sobel response (signed, typically int16_t range)
//   gy — vertical   Sobel response (signed, typically int16_t range)
//
// Returns one of { DIR_0, DIR_45, DIR_90, DIR_135 }.
// Pure function — no side effects, safe to call in parallel across pixels.
// ============================================================================
Direction quantiseDirection(int gx, int gy)
{
    // Reduce to the first octant — absolute magnitudes only.
    const int ax = std::abs(gx);
    const int ay = std::abs(gy);

    // ── Sector test 1: gradient within 22.5° of horizontal? ─────────────
    // Equivalent to: ay / ax < tan(22.5°)  ⟺  ay * 5 < ax * 2
    if (ay * 5 < ax * 2)
        return Direction::DIR_0;

    // ── Sector test 2: gradient within 67.5° of horizontal? ─────────────
    // Equivalent to: ay / ax < tan(67.5°)  ⟺  ay * 5 < ax * 12
    if (ay * 5 < ax * 12) {
        // Gradient is near 45°; resolve to 45° or 135° via sign agreement.
        // Same sign  → gradient points NE or SW → edge runs NW–SE → DIR_45
        // Diff sign  → gradient points NW or SE → edge runs NE–SW → DIR_135
        const bool sameSign = (gx >= 0) == (gy >= 0);
        return sameSign ? Direction::DIR_45 : Direction::DIR_135;
    }

    // ── Fallthrough: gradient within 22.5° of vertical ───────────────────
    return Direction::DIR_90;
}

// ============================================================================
// computeGradientDirections — apply quantiseDirection over a full image.
//
// Parameters:
//   gx     — horizontal gradient image, int16_t, row-major, width*height elems
//   gy     — vertical   gradient image, int16_t, row-major, width*height elems
//   dirs   — output Direction map, caller-allocated, width*height elems
//   width  — image width  in pixels
//   height — image height in pixels
//
// Boundary pixels are handled naturally: Sobel values at the border produce
// small absolute magnitudes, and quantiseDirection returns a valid Direction
// without any special-casing.
// ============================================================================
void computeGradientDirections(
    const int16_t* gx,
    const int16_t* gy,
    Direction*     dirs,
    int            width,
    int            height)
{
    assert(gx    != nullptr);
    assert(gy    != nullptr);
    assert(dirs  != nullptr);
    assert(width  > 0);
    assert(height > 0);

    const int N = width * height;

    for (int i = 0; i < N; ++i)
        dirs[i] = quantiseDirection(gx[i], gy[i]);
}

// ============================================================================
// Self-tests
// ============================================================================

static void selfTest()
{
    // ── quantiseDirection unit tests ─────────────────────────────────────

    // Pure horizontal / near-horizontal → DIR_0
    assert(quantiseDirection( 100,   0) == Direction::DIR_0);
    assert(quantiseDirection(-100,   0) == Direction::DIR_0);

    // Pure vertical / near-vertical → DIR_90
    assert(quantiseDirection(  0,  100) == Direction::DIR_90);
    assert(quantiseDirection(  0, -100) == Direction::DIR_90);

    // Equal magnitudes, same sign → DIR_45
    assert(quantiseDirection( 100,  100) == Direction::DIR_45);
    assert(quantiseDirection(-100, -100) == Direction::DIR_45);

    // Equal magnitudes, opposite sign → DIR_135
    assert(quantiseDirection( 100, -100) == Direction::DIR_135);
    assert(quantiseDirection(-100,  100) == Direction::DIR_135);

    // Boundary just below 22.5°: gx=100, gy=39 → ay*5=195 < ax*2=200 → DIR_0
    assert(quantiseDirection(100,  39) == Direction::DIR_0);
    // Boundary just above 22.5°: gx=100, gy=41 → ay*5=205 > ax*2=200 → not DIR_0
    assert(quantiseDirection(100,  41) != Direction::DIR_0);

    // Boundary just below 67.5°: gx=100, gy=239 → ay*5=1195 < ax*12=1200 → DIR_45
    assert(quantiseDirection(100, 239) == Direction::DIR_45);
    // Boundary just above 67.5°: gx=100, gy=241 → ay*5=1205 > ax*12=1200 → DIR_90
    assert(quantiseDirection(100, 241) == Direction::DIR_90);

    // ── computeGradientDirections smoke test ─────────────────────────────
    constexpr int    N  = 4;
    const int16_t   gx[N] = {  100,   0,  100, -100 };
    const int16_t   gy[N] = {    0, 100,  100,  100  };
    Direction       dirs[N];

    computeGradientDirections(gx, gy, dirs, N, 1);

    assert(dirs[0] == Direction::DIR_0);
    assert(dirs[1] == Direction::DIR_90);
    assert(dirs[2] == Direction::DIR_45);
    assert(dirs[3] == Direction::DIR_135);

    std::cout << "All gradient direction tests PASSED.\n";
}

// ============================================================================
// main
// ============================================================================

int main()
{
    selfTest();
    return 0;
}
