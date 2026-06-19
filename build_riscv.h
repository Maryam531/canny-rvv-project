#!/bin/bash
set -e

mkdir -p build-riscv

# Build the RISC-V bare-metal binary
riscv64-unknown-elf-g++ \
    -march=rv64gcv \
    -mabi=lp64d \
    -mcmodel=medany \
    -static \
    -nostdlib \
    -nostartfiles \
    -T linker.ld \
    crt0.s \
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
    -std=c++17 \
    ${OPT_LEVEL:--O3} \
    -ftree-vectorize \
    -o build-riscv/canny_rv

echo "Build done: ${OPT_LEVEL:--O3}"
