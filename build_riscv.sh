#!/bin/bash
set -e

mkdir -p build-riscv

riscv64-unknown-elf-g++ \
    -static \
    -no-pie \
    -Wl,-Ttext=0x80000000 \
    src/main.cpp \
    src/gaussian.cpp \
    src/sobel.cpp \
    src/magnitude.cpp \
    src/direction.cpp \
    src/nms.cpp \
    src/threshold.cpp \
    src/rvv_gaussian.cpp \
    src/rvv_magnitude.cpp \
    src/image_io.cpp \
    src/syscalls.cpp \
    -Iinclude \
    -march=rv64gcv \
    -mabi=lp64d \
    -std=c++17 \
    ${OPT_LEVEL:--O3} \
    -ftree-vectorize \
    -lm \
    -o build-riscv/canny_rv

echo "Build done: ${OPT_LEVEL:--O3}"
