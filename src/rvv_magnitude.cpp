#include "magnitude.h"
#include <cstdlib>
#include <cmath>
#include <riscv_vector.h>

// L1 RVV: |Gx| + |Gy|
Image magnitude_l1_rvv(const int16_t* gx, const int16_t* gy,
                        int width, int height) {
    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, width * height);

    const int N = width * height;

    // -------------------------------------------------------
    // First pass: find maximum magnitude using RVV
    // -------------------------------------------------------
    int32_t max_mag = 1;
    int i = 0;

    // vector max accumulator, starts at 0
    size_t vl0 = __riscv_vsetvl_e32m4(N);
    vint32m4_t vmax = __riscv_vmv_v_x_i32m4(0, vl0);

    while (i < N) {
        // Step 1: how many elements to process
        size_t vl = __riscv_vsetvl_e16m2(N - i);

        // Step 2: load gx and gy as 16-bit
        vint16m2_t vgx = __riscv_vle16_v_i16m2(gx + i, vl);
        vint16m2_t vgy = __riscv_vle16_v_i16m2(gy + i, vl);

        // Step 3: absolute value of each
        vint16m2_t vax = __riscv_vabs_v_i16m2(vgx, vl);
        vint16m2_t vay = __riscv_vabs_v_i16m2(vgy, vl);

        // Step 4: widen to 32-bit and add
        vint32m4_t vax32 = __riscv_vwcvt_x_x_v_i32m4(vax, vl);
        vint32m4_t vay32 = __riscv_vwcvt_x_x_v_i32m4(vay, vl);
        vint32m4_t vmag  = __riscv_vadd_vv_i32m4(vax32, vay32, vl);

        // Step 5: update vector max
        vl0 = __riscv_vsetvl_e32m4(N - i);
        vmax = __riscv_vmax_vv_i32m4(vmax, vmag, vl);

        i += vl;
    }

    // reduce vector max to scalar
    vint32m1_t scalar_max = __riscv_vmv_s_x_i32m1(0, 1);
    scalar_max = __riscv_vredmax_vs_i32m4_i32m1(vmax, scalar_max, vl0);
    max_mag = __riscv_vmv_x_s_i32m1_i32(scalar_max);
    if (max_mag < 1) max_mag = 1;

    // -------------------------------------------------------
    // Second pass: compute magnitude and normalize using RVV
    // -------------------------------------------------------
    i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e16m2(N - i);

        // load gx, gy
        vint16m2_t vgx = __riscv_vle16_v_i16m2(gx + i, vl);
        vint16m2_t vgy = __riscv_vle16_v_i16m2(gy + i, vl);

        // abs
        vint16m2_t vax = __riscv_vabs_v_i16m2(vgx, vl);
        vint16m2_t vay = __riscv_vabs_v_i16m2(vgy, vl);

        // widen and add
        vint32m4_t vax32 = __riscv_vwcvt_x_x_v_i32m4(vax, vl);
        vint32m4_t vay32 = __riscv_vwcvt_x_x_v_i32m4(vay, vl);
        vint32m4_t vmag  = __riscv_vadd_vv_i32m4(vax32, vay32, vl);

        // normalize: mag * 255 / max_mag
        vmag = __riscv_vmul_vx_i32m4(vmag, 255, vl);
        vmag = __riscv_vdiv_vx_i32m4(vmag, max_mag, vl);

        // clamp to 0-255
        vmag = __riscv_vmax_vx_i32m4(vmag, 0, vl);
        vmag = __riscv_vmin_vx_i32m4(vmag, 255, vl);

        // narrow 32 -> 16 -> 8
        vuint32m4_t vmag_u   = __riscv_vreinterpret_v_i32m4_u32m4(vmag);
        vuint16m2_t narrow16 = __riscv_vnclipu_wx_u16m2(vmag_u, 0, 0, vl);
        vuint8m1_t  narrow8  = __riscv_vnclipu_wx_u8m1(narrow16, 0, 0, vl);

        // store
        __riscv_vse8_v_u8m1(out.data + i, narrow8, vl);

        i += vl;
    }

    return out;
}

// L2 RVV: sqrt(Gx^2 + Gy^2)
Image magnitude_l2_rvv(const int16_t* gx, const int16_t* gy,
                        int width, int height) {
    Image out;
    out.width  = width;
    out.height = height;
    out.data   = (uint8_t*)aligned_alloc(64, width * height);

    const int N = width * height;

    // -------------------------------------------------------
    // First pass: find max using RVV (float)
    // -------------------------------------------------------
    float max_mag = 1.0f;
    int i = 0;

    size_t vl0 = __riscv_vsetvl_e32m4(N);
    vfloat32m4_t vfmax = __riscv_vfmv_v_f_f32m4(0.0f, vl0);

    while (i < N) {
        size_t vl = __riscv_vsetvl_e16m2(N - i);

        // load gx, gy as 16-bit
        vint16m2_t vgx16 = __riscv_vle16_v_i16m2(gx + i, vl);
        vint16m2_t vgy16 = __riscv_vle16_v_i16m2(gy + i, vl);

        // widen to 32-bit int
        vint32m4_t vgx32 = __riscv_vwcvt_x_x_v_i32m4(vgx16, vl);
        vint32m4_t vgy32 = __riscv_vwcvt_x_x_v_i32m4(vgy16, vl);

        // convert to float
        vfloat32m4_t vfx = __riscv_vfcvt_f_x_v_f32m4(vgx32, vl);
        vfloat32m4_t vfy = __riscv_vfcvt_f_x_v_f32m4(vgy32, vl);

        // gx^2 + gy^2
        vfloat32m4_t vmag = __riscv_vfmacc_vv_f32m4(
            __riscv_vfmul_vv_f32m4(vfx, vfx, vl),
            vfy, vfy, vl);

        // sqrt
        vmag = __riscv_vfsqrt_v_f32m4(vmag, vl);

        // update max
        vl0 = __riscv_vsetvl_e32m4(N - i);
        vfmax = __riscv_vfmax_vv_f32m4(vfmax, vmag, vl);

        i += vl;
    }

    // reduce to scalar
    vfloat32m1_t scalar_fmax = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    scalar_fmax = __riscv_vfredmax_vs_f32m4_f32m1(vfmax, scalar_fmax, vl0);
    max_mag = __riscv_vfmv_f_s_f32m1_f32(scalar_fmax);
    if (max_mag < 1.0f) max_mag = 1.0f;

    // -------------------------------------------------------
    // Second pass: compute and normalize using RVV
    // -------------------------------------------------------
    i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e16m2(N - i);

        vint16m2_t vgx16 = __riscv_vle16_v_i16m2(gx + i, vl);
        vint16m2_t vgy16 = __riscv_vle16_v_i16m2(gy + i, vl);

        vint32m4_t vgx32 = __riscv_vwcvt_x_x_v_i32m4(vgx16, vl);
        vint32m4_t vgy32 = __riscv_vwcvt_x_x_v_i32m4(vgy16, vl);

        vfloat32m4_t vfx = __riscv_vfcvt_f_x_v_f32m4(vgx32, vl);
        vfloat32m4_t vfy = __riscv_vfcvt_f_x_v_f32m4(vgy32, vl);

        // sqrt(gx^2 + gy^2)
        vfloat32m4_t vmag = __riscv_vfmacc_vv_f32m4(
            __riscv_vfmul_vv_f32m4(vfx, vfx, vl),
            vfy, vfy, vl);
        vmag = __riscv_vfsqrt_v_f32m4(vmag, vl);

        // normalize: mag * 255 / max_mag
        vmag = __riscv_vfmul_vf_f32m4(vmag, 255.0f / max_mag, vl);

        // convert float -> int32
        vint32m4_t vmag_i = __riscv_vfcvt_x_f_v_i32m4(vmag, vl);

        // clamp
        vmag_i = __riscv_vmax_vx_i32m4(vmag_i, 0, vl);
        vmag_i = __riscv_vmin_vx_i32m4(vmag_i, 255, vl);

        // narrow to u8
        vuint32m4_t vmag_u   = __riscv_vreinterpret_v_i32m4_u32m4(vmag_i);
        vuint16m2_t narrow16 = __riscv_vnclipu_wx_u16m2(vmag_u, 0, 0, vl);
        vuint8m1_t  narrow8  = __riscv_vnclipu_wx_u8m1(narrow16, 0, 0, vl);

        __riscv_vse8_v_u8m1(out.data + i, narrow8, vl);

        i += vl;
    }

    return out;
}
