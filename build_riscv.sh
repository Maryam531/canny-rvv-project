#!/bin/bash
set -e

mkdir -p build-riscv

riscv64-unknown-elf-g++ -static \
    src/main.cpp \
    src/image_io.cpp \
    src/gaussian.cpp \
    src/sobel.cpp \
    src/magnitude.cpp \
    src/direction.cpp \
    src/nms.cpp \
    src/nmh.cpp \
    src/threshold.cpp \
    src/rvv_gaussian.cpp \
    src/rvv_magnitude.cpp \
    src/syscalls.cpp \
    -Iinclude \
    -march=rv64gcv \
    -mabi=lp64d \
    ${OPT_LEVEL:--O0} \
    -o build-riscv/canny_rv

echo "Build done: ${OPT_LEVEL:--O0}"
