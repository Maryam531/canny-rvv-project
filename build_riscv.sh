#!/bin/bash
set -e

mkdir -p build-riscv

riscv64-linux-gnu-g++ \
    -march=rv64gcv \
    -mabi=lp64d \
    -mcmodel=medany \
    -static \
    src/main.cpp \
    src/gaussian.cpp \
    src/sobel.cpp \
    src/magnitude.cpp \
    src/direction.cpp \
    src/rvv_gaussian.cpp \
    src/rvv_magnitude.cpp \
    src/image_io.cpp \
    src/syscall.cpp \
    src/nms.cpp \
    src/threshold.cpp \
    src/hysteresis.cpp \
    -Iinclude \
    -std=c++17 \
    ${OPT_LEVEL:--O3} \
    -ftree-vectorize \
    -lm \
    -o build-riscv/canny_rv

echo "Build done: ${OPT_LEVEL:--O3}"
