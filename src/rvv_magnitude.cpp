#include "magnitude.h"
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <riscv_vector.h>

static inline size_t round_up64(size_t n) { return (n + 63u) & ~size_t(63); }

// ── Portable i16 abs: max(v, -v) ─────────────────────────────────────────
// GCC 13 RVV headers do not expose __riscv_vabs_v_i16m2.
static inline vint16m2_t vabs_i16m2(vint16m2_t v, size_t vl) {
    return __riscv_vmax_vv_i16m2(v, __riscv_vneg_v_i16m2(v, vl), vl);
}

// ── Narrow i32 → u8 without vnclipu (GCC 13 API has no vxrm argument) ───
// Values are already clamped to [0,255] before calling these, so we just
// need a lossless truncating narrow.  We use vnsrl (logical right-shift by 0)
// which truncates and zero-extends — correct for non-negative clamped values.
static inline vuint16m2_t narrow_u32_to_u16(vuint32m4_t v, size_t vl) {
    // vnsrl_wx shifts right by 0 and narrows u32→u16
    return __riscv_vnsrl_wx_u16m2(v, 0, vl);
}
static inline vuint8m1_t narrow_u16_to_u8(vuint16m2_t v, size_t vl) {
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
        size_t     vl_init  = __riscv_vsetvl_e32m4((size_t)N);
        vint32m4_t vmax_acc = __riscv_vmv_v_x_i32m4(0, vl_init);

        int i = 0;
        while (i < N) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(N - i));

            vint16m2_t vgx_v = __riscv_vle16_v_i16m2(gx + i, vl);
            vint16m2_t vgy_v = __riscv_vle16_v_i16m2(gy + i, vl);

            vint32m4_t vax32 = __riscv_vwcvt_x_x_v_i32m4(vabs_i16m2(vgx_v, vl), vl);
            vint32m4_t vay32 = __riscv_vwcvt_x_x_v_i32m4(vabs_i16m2(vgy_v, vl), vl);
            vint32m4_t vmag  = __riscv_vadd_vv_i32m4(vax32, vay32, vl);

            size_t vl32 = __riscv_vsetvl_e32m4((size_t)(N - i));
            vmax_acc = __riscv_vmax_vv_i32m4(vmax_acc, vmag, vl32);
            vl_last  = vl32;
            i       += (int)vl;
        }

        vint32m1_t red_init = __riscv_vmv_s_x_i32m1(0, 1);
        vint32m1_t red = __riscv_vredmax_vs_i32m4_i32m1(vmax_acc, red_init, vl_last);
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

            // Scale: mag * 255 / max_mag (integer division, matches scalar)
            vmag = __riscv_vmul_vx_i32m4(vmag, 255, vl);
            vmag = __riscv_vdiv_vx_i32m4(vmag, max_mag, vl);

            // Clamp [0, 255] — after this, values fit in u8
            vmag = __riscv_vmax_vx_i32m4(vmag, 0,   vl);
            vmag = __riscv_vmin_vx_i32m4(vmag, 255, vl);

            // Narrow i32 → u8 via two truncating shifts (no vnclipu needed)
            vuint32m4_t vmag_u   = __riscv_vreinterpret_v_i32m4_u32m4(vmag);
            vuint16m2_t narrow16 = narrow_u32_to_u16(vmag_u, vl);
            vuint8m1_t  narrow8  = narrow_u16_to_u8(narrow16, vl);

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
        size_t       vl_init = __riscv_vsetvl_e32m4((size_t)N);
        vfloat32m4_t vfmax   = __riscv_vfmv_v_f_f32m4(0.0f, vl_init);

        int i = 0;
        while (i < N) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(N - i));

            vint16m2_t   vgx16 = __riscv_vle16_v_i16m2(gx + i, vl);
            vint16m2_t   vgy16 = __riscv_vle16_v_i16m2(gy + i, vl);
            vfloat32m4_t vfx   = __riscv_vfcvt_f_x_v_f32m4(
                                      __riscv_vwcvt_x_x_v_i32m4(vgx16, vl), vl);
            vfloat32m4_t vfy   = __riscv_vfcvt_f_x_v_f32m4(
                                      __riscv_vwcvt_x_x_v_i32m4(vgy16, vl), vl);

            // sqrt(gx² + gy²)
            vfloat32m4_t vsq  = __riscv_vfmacc_vv_f32m4(
                                     __riscv_vfmul_vv_f32m4(vfx, vfx, vl),
                                     vfy, vfy, vl);
            vfloat32m4_t vmag = __riscv_vfsqrt_v_f32m4(vsq, vl);

            size_t vl32 = __riscv_vsetvl_e32m4((size_t)(N - i));
            vfmax   = __riscv_vfmax_vv_f32m4(vfmax, vmag, vl32);
            vl_last = vl32;
            i      += (int)vl;
        }

        vfloat32m1_t red_init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t red = __riscv_vfredmax_vs_f32m4_f32m1(vfmax, red_init, vl_last);
        max_mag = __riscv_vfmv_f_s_f32m1_f32(red);
        if (max_mag < 1.0f) max_mag = 1.0f;
    }

    // ── Pass 2: compute, normalise, and store ─────────────────────────────
    {
        const float scale = 255.0f / max_mag;
        int i = 0;
        while (i < N) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(N - i));

            vint16m2_t   vgx16 = __riscv_vle16_v_i16m2(gx + i, vl);
            vint16m2_t   vgy16 = __riscv_vle16_v_i16m2(gy + i, vl);
            vfloat32m4_t vfx   = __riscv_vfcvt_f_x_v_f32m4(
                                      __riscv_vwcvt_x_x_v_i32m4(vgx16, vl), vl);
            vfloat32m4_t vfy   = __riscv_vfcvt_f_x_v_f32m4(
                                      __riscv_vwcvt_x_x_v_i32m4(vgy16, vl), vl);

            vfloat32m4_t vsq  = __riscv_vfmacc_vv_f32m4(
                                     __riscv_vfmul_vv_f32m4(vfx, vfx, vl),
                                     vfy, vfy, vl);
            vfloat32m4_t vmag = __riscv_vfsqrt_v_f32m4(vsq, vl);

            // Scale, truncate to i32 (matches scalar (int32_t) cast)
            vmag = __riscv_vfmul_vf_f32m4(vmag, scale, vl);
            vint32m4_t vmag_i = __riscv_vfcvt_rtz_x_f_v_i32m4(vmag, vl);

            // Clamp [0, 255]
            vmag_i = __riscv_vmax_vx_i32m4(vmag_i, 0,   vl);
            vmag_i = __riscv_vmin_vx_i32m4(vmag_i, 255, vl);

            // Narrow i32 → u8 via two truncating shifts
            vuint32m4_t vmag_u   = __riscv_vreinterpret_v_i32m4_u32m4(vmag_i);
            vuint16m2_t narrow16 = narrow_u32_to_u16(vmag_u, vl);
            vuint8m1_t  narrow8  = narrow_u16_to_u8(narrow16, vl);

            __riscv_vse8_v_u8m1(out.data + i, narrow8, vl);
            i += (int)vl;
        }
    }

    return out;
}
