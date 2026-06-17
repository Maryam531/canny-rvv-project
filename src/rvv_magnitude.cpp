#include "magnitude.h"
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <riscv_vector.h>

static inline size_t round_up64(size_t n) { return (n + 63u) & ~size_t(63); }

// ── Portable i16 abs: max(v, -v) ─────────────────────────────────────────
static inline vint16m2_t vabs_i16m2(vint16m2_t v, size_t vl) {
    // (1) computes absolute value of each int16 element by negating
    //           then taking the element-wise maximum
    // (2) LMUL=m2: input gradients are int16, m2 gives enough lanes
    //                  and pairs correctly with m4 after widening
    // (3) VLEN EFFECT: at VLEN=128 processes 8 int16 at once,
    //                  VLEN=256 → 16, VLEN=512 → 32. Code unchanged.
    return __riscv_vmax_vv_i16m2(v, __riscv_vneg_v_i16m2(v, vl), vl);
}

// ── Narrow helpers ────────────────────────────────────────────────────────
static inline vuint16m2_t narrow_u32_to_u16(vuint32m4_t v, size_t vl) {
    // (1) narrows uint32 to uint16 by logical right shift of 0
    //           (truncates upper bits, safe because values are clamped 0-255)
    // (2) LMUL=m4→m2: input is m4 (uint32), output is m2 (uint16),
    //                      narrowing always halves the LMUL
    // (3) VLEN EFFECT: same number of elements processed, just smaller type
    return __riscv_vnsrl_wx_u16m2(v, 0, vl);
}
static inline vuint8m1_t narrow_u16_to_u8(vuint16m2_t v, size_t vl) {
    // (1) narrows uint16 to uint8 by logical right shift of 0
    //           (truncates upper bits, safe because values are clamped 0-255)
    // (2) LMUL=m2→m1: input is m2 (uint16), output is m1 (uint8),
    //                      narrowing always halves the LMUL
    // (3) VLEN EFFECT: no change in behavior, more elements at higher VLEN
    return __riscv_vnsrl_wx_u8m1(v, 0, vl);
}

// ── L1 RVV: |Gx| + |Gy|, normalised to [0, 255] ─────────────────────────
Image magnitude_l1_rvv(const int16_t* gx, const int16_t* gy,
                        int width, int height)
{
    assert(gx && gy && width > 0 && height > 0);

    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, round_up64((size_t)width * height));
    assert(out.data);

    const int N = width * height;

    // ── Pass 1: find maximum magnitude ───────────────────────────────────
    int32_t max_mag = 1;
    size_t  vl_last = 1;

    {
        // (1) sets initial vector length for int32 accumulator
        // (2) LMUL=m4: accumulator holds int32 values after widening
        //                   from int16 m2 → int32 m4
        // (3) VLEN EFFECT: at VLEN=128 vl_init=4, VLEN=256→8, VLEN=512→16
        size_t vl_init = __riscv_vsetvl_e32m4((size_t)N);

        // (1) initializes all int32 accumulator lanes to 0
        // (2) LMUL=m4: matches the int32 magnitude vectors computed later
        // (3) VLEN EFFECT: more lanes initialized at higher VLEN, all zero
        vint32m4_t vmax_acc = __riscv_vmv_v_x_i32m4(0, vl_init);

        int i = 0;
        while (i < N) {
            // (1) sets vector length for int16 gradient loads
            // (2) LMUL=m2: gx/gy are int16, m2 pairs with m4 for widening
            // (3) VLEN EFFECT: VLEN=128→8 elements, VLEN=256→16, VLEN=512→32
            size_t vl = __riscv_vsetvl_e16m2((size_t)(N - i));

            // (1) loads vl int16 values from gx array
            // (2) LMUL=m2: matches int16 source type
            // (3) VLEN EFFECT: more pixels loaded per call at higher VLEN
            vint16m2_t vgx_v = __riscv_vle16_v_i16m2(gx + i, vl);

            // (1) loads vl int16 values from gy array
            // (2) LMUL=m2: matches int16 source type
            // (3) VLEN EFFECT: more pixels loaded per call at higher VLEN
            vint16m2_t vgy_v = __riscv_vle16_v_i16m2(gy + i, vl);

            // (1) takes abs of gx then widens int16→int32
            // (2) LMUL=m2→m4: |Gx|+|Gy| can reach 65534, overflows int16
            //                      so must widen to int32 before adding
            // (3) VLEN EFFECT: same element count, output register doubles
            vint32m4_t vax32 = __riscv_vwcvt_x_x_v_i32m4(vabs_i16m2(vgx_v, vl), vl);

            // (1) takes abs of gy then widens int16→int32
            // (2) LMUL=m4: must match vax32 for the addition below
            // (3) VLEN EFFECT: same as vax32
            vint32m4_t vay32 = __riscv_vwcvt_x_x_v_i32m4(vabs_i16m2(vgy_v, vl), vl);

            // (1) computes |Gx| + |Gy| for all vl pixels at once
            // (2) LMUL=m4: both inputs are int32 m4
            // (3) VLEN EFFECT: more additions per call at higher VLEN
            vint32m4_t vmag = __riscv_vadd_vv_i32m4(vax32, vay32, vl);

            // (1) resets vl for int32 operations after int16 vsetvl
            // (2) LMUL=m4: vmax_acc is int32 m4
            // (3) VLEN EFFECT: vl32 may differ from vl if N-i is small
            size_t vl32 = __riscv_vsetvl_e32m4((size_t)(N - i));

            // (1) element-wise max to track running maximum magnitude
            // (2) LMUL=m4: both vmag and vmax_acc are int32 m4
            // (3) VLEN EFFECT: more comparisons per call at higher VLEN
            vmax_acc = __riscv_vmax_vv_i32m4(vmax_acc, vmag, vl32);
            vl_last  = vl32;
            i       += (int)vl;
        }

        // (1) creates scalar seed value of 0 for reduction
        // (2) LMUL=m1: reduction output is always a single scalar
        // (3) VLEN EFFECT: always 1 element, unaffected by VLEN
        vint32m1_t red_init = __riscv_vmv_s_x_i32m1(0, 1);

        // (1) reduces all lanes of vmax_acc to single maximum value
        // (2) LMUL=m4→m1: input is m4 vector, output is m1 scalar
        // (3) VLEN EFFECT: result is same regardless of VLEN
        vint32m1_t red = __riscv_vredmax_vs_i32m4_i32m1(vmax_acc, red_init, vl_last);

        // (1) extracts scalar int32 from vector register
        // (2) LMUL=m1: red is a single-element vector
        // (3) VLEN EFFECT: no effect, always extracts element 0
        max_mag = __riscv_vmv_x_s_i32m1_i32(red);
        if (max_mag < 1) max_mag = 1;
    }

    // ── Pass 2: compute, normalise, and store ─────────────────────────────
    {
        int i = 0;
        while (i < N) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(N - i));

            vint16m2_t vgx_v = __riscv_vle16_v_i16m2(gx + i, vl);
            vint16m2_t vgy_v = __riscv_vle16_v_i16m2(gy + i, vl);

            vint32m4_t vax32 = __riscv_vwcvt_x_x_v_i32m4(vabs_i16m2(vgx_v, vl), vl);
            vint32m4_t vay32 = __riscv_vwcvt_x_x_v_i32m4(vabs_i16m2(vgy_v, vl), vl);
            vint32m4_t vmag  = __riscv_vadd_vv_i32m4(vax32, vay32, vl);

            // (1) multiplies all magnitudes by 255 for normalization
            // (2) LMUL=m4: vmag is int32 m4
            // (3) VLEN EFFECT: more multiplications per call at higher VLEN
            vmag = __riscv_vmul_vx_i32m4(vmag, 255, vl);

            // (1) divides all magnitudes by max_mag to normalize 0-255
            // (2) LMUL=m4: vmag is int32 m4
            // (3) VLEN EFFECT: more divisions per call at higher VLEN
            vmag = __riscv_vdiv_vx_i32m4(vmag, max_mag, vl);

            // (1) clamps all values to minimum of 0
            // (2) LMUL=m4: vmag is int32 m4
            // (3) VLEN EFFECT: more clamps per call at higher VLEN
            vmag = __riscv_vmax_vx_i32m4(vmag, 0, vl);

            // (1) clamps all values to maximum of 255
            // (2) LMUL=m4: vmag is int32 m4
            // (3) VLEN EFFECT: more clamps per call at higher VLEN
            vmag = __riscv_vmin_vx_i32m4(vmag, 255, vl);

            // (1) reinterprets signed int32 as unsigned uint32
            //           no bits changed, just type change for narrowing
            // (2) vnsrl requires unsigned input
            // (3) VLEN EFFECT: no change in behavior
            vuint32m4_t vmag_u   = __riscv_vreinterpret_v_i32m4_u32m4(vmag);

            // narrow uint32→uint16→uint8 (see helper comments above)
            vuint16m2_t narrow16 = narrow_u32_to_u16(vmag_u, vl);
            vuint8m1_t  narrow8  = narrow_u16_to_u8(narrow16, vl);

            // (1) stores vl uint8 results to output image
            // (2) LMUL=m1: output pixels are uint8, m1 is sufficient
            // (3) VLEN EFFECT: more pixels stored per call at higher VLEN
            __riscv_vse8_v_u8m1(out.data + i, narrow8, vl);
            i += (int)vl;
        }
    }
    return out;
}

// ── L2 RVV: sqrt(Gx² + Gy²), normalised to [0, 255] ─────────────────────
Image magnitude_l2_rvv(const int16_t* gx, const int16_t* gy,
                        int width, int height)
{
    assert(gx && gy && width > 0 && height > 0);

    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, round_up64((size_t)width * height));
    assert(out.data);

    const int N = width * height;

    // ── Pass 1: find maximum L2 magnitude ────────────────────────────────
    float  max_mag = 1.0f;
    size_t vl_last = 1;

    {
        // (1) sets initial vector length for float32 accumulator
        // (2) LMUL=m4: float32 vectors after widening from int16 need m4
        // (3) VLEN EFFECT: VLEN=128→4 floats, VLEN=256→8, VLEN=512→16
        size_t vl_init = __riscv_vsetvl_e32m4((size_t)N);

        // (1) initializes float32 max accumulator lanes to 0.0
        // (2) LMUL=m4: matches float32 magnitude vectors computed later
        // (3) VLEN EFFECT: more lanes initialized at higher VLEN
        vfloat32m4_t vfmax = __riscv_vfmv_v_f_f32m4(0.0f, vl_init);

        int i = 0;
        while (i < N) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(N - i));

            vint16m2_t vgx16 = __riscv_vle16_v_i16m2(gx + i, vl);
            vint16m2_t vgy16 = __riscv_vle16_v_i16m2(gy + i, vl);

            // (1) widens int16→int32 then converts to float32
            // (2) sqrt requires float. int16→int32 prevents overflow
            //          before float conversion. m2→m4 from widening
            // (3) VLEN EFFECT: same element count, wider registers used
            vfloat32m4_t vfx = __riscv_vfcvt_f_x_v_f32m4(
                                    __riscv_vwcvt_x_x_v_i32m4(vgx16, vl), vl);
            vfloat32m4_t vfy = __riscv_vfcvt_f_x_v_f32m4(
                                    __riscv_vwcvt_x_x_v_i32m4(vgy16, vl), vl);

            // (1) computes gx*gx then accumulates gy*gy using
            //           fused multiply-add: result = gx²+gy²
            // (2) LMUL=m4: float32 vectors require m4
            // (3) VLEN EFFECT: more pixels computed per call at higher VLEN
            vfloat32m4_t vsq = __riscv_vfmacc_vv_f32m4(
                                    __riscv_vfmul_vv_f32m4(vfx, vfx, vl),
                                    vfy, vfy, vl);

            // (1) computes square root of each element: sqrt(gx²+gy²)
            // (2) LMUL=m4: input vsq is float32 m4
            // (3) VLEN EFFECT: more square roots per call at higher VLEN
            vfloat32m4_t vmag = __riscv_vfsqrt_v_f32m4(vsq, vl);

            size_t vl32 = __riscv_vsetvl_e32m4((size_t)(N - i));

            // (1) element-wise float max to track running maximum
            // (2) LMUL=m4: both vfmax and vmag are float32 m4
            // (3) VLEN EFFECT: more comparisons per call at higher VLEN
            vfmax   = __riscv_vfmax_vv_f32m4(vfmax, vmag, vl32);
            vl_last = vl32;
            i      += (int)vl;
        }

        // (1) creates float32 scalar seed of 0.0 for reduction
        // (2) LMUL=m1: reduction output is always a single scalar
        // (3) VLEN EFFECT: always 1 element, unaffected by VLEN
        vfloat32m1_t red_init = __riscv_vfmv_s_f_f32m1(0.0f, 1);

        // (1) reduces all float32 lanes to single maximum value
        // (2) LMUL=m4→m1: input is m4 vector, output is m1 scalar
        // (3) VLEN EFFECT: result is same regardless of VLEN
        vfloat32m1_t red = __riscv_vfredmax_vs_f32m4_f32m1(vfmax, red_init, vl_last);

        // (1) extracts scalar float from vector register element 0
        // (2) LMUL=m1: red is single-element vector
        // (3) VLEN EFFECT: no effect
        max_mag = __riscv_vfmv_f_s_f32m1_f32(red);
        if (max_mag < 1.0f) max_mag = 1.0f;
    }

    // ── Pass 2: compute, normalise, and store ─────────────────────────────
    {
        const float scale = 255.0f / max_mag;
        int i = 0;
        while (i < N) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(N - i));

            vint16m2_t vgx16 = __riscv_vle16_v_i16m2(gx + i, vl);
            vint16m2_t vgy16 = __riscv_vle16_v_i16m2(gy + i, vl);

            vfloat32m4_t vfx = __riscv_vfcvt_f_x_v_f32m4(
                                    __riscv_vwcvt_x_x_v_i32m4(vgx16, vl), vl);
            vfloat32m4_t vfy = __riscv_vfcvt_f_x_v_f32m4(
                                    __riscv_vwcvt_x_x_v_i32m4(vgy16, vl), vl);

            vfloat32m4_t vsq  = __riscv_vfmacc_vv_f32m4(
                                     __riscv_vfmul_vv_f32m4(vfx, vfx, vl),
                                     vfy, vfy, vl);
            vfloat32m4_t vmag = __riscv_vfsqrt_v_f32m4(vsq, vl);

            // (1) multiplies each magnitude by scale (255/max_mag)
            //           to normalize into [0, 255] range
            // (2) LMUL=m4: vmag is float32 m4
            // (3) VLEN EFFECT: more multiplications per call at higher VLEN
            vmag = __riscv_vfmul_vf_f32m4(vmag, scale, vl);

            // (1) converts float32 to int32 truncating toward zero
            //           matches scalar (int32_t) cast behavior
            // (2) LMUL=m4: input float32 m4, output int32 m4
            // (3) VLEN EFFECT: more conversions per call at higher VLEN
            vint32m4_t vmag_i = __riscv_vfcvt_rtz_x_f_v_i32m4(vmag, vl);

            // (1) clamps to minimum 0
            // (2) LMUL=m4: vmag_i is int32 m4
            // (3) VLEN EFFECT: more clamps per call at higher VLEN
            vmag_i = __riscv_vmax_vx_i32m4(vmag_i, 0,   vl);

            // (1) clamps to maximum 255
            // (2) LMUL=m4: vmag_i is int32 m4
            // (3) VLEN EFFECT: more clamps per call at higher VLEN
            vmag_i = __riscv_vmin_vx_i32m4(vmag_i, 255, vl);

            // (1) reinterprets signed int32 as unsigned uint32
            //           no bits changed, required for vnsrl narrowing
            // (2) vnsrl requires unsigned input type
            // (3) VLEN EFFECT: no change in behavior
            vuint32m4_t vmag_u   = __riscv_vreinterpret_v_i32m4_u32m4(vmag_i);
            vuint16m2_t narrow16 = narrow_u32_to_u16(vmag_u, vl);
            vuint8m1_t  narrow8  = narrow_u16_to_u8(narrow16, vl);

            // (1) stores vl uint8 pixels to output image
            // (2) LMUL=m1: output is uint8, m1 is sufficient
            // (3) VLEN EFFECT: more pixels stored per call at higher VLEN
            __riscv_vse8_v_u8m1(out.data + i, narrow8, vl);
            i += (int)vl;
        }
    }
    return out;
}
