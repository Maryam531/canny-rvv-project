# Canny Edge Detection on RISC-V with RVV

A Canny edge detection pipeline implemented in C++, cross-compiled for RISC-V 
(rv64gcv), and accelerated using RISC-V Vector (RVV) intrinsics. The project 
demonstrates the full optimization journey: scalar C++ baseline → compiler 
optimization sweep → profiling → hand-written RVV vectorized kernels.

## Team
- Menna — Magnitude (L1/L2) implementation and RVV optimization, unit testing
- Maryam531 — Infrastructure, toolchain, Sobel, Gaussian
- Logine-Ahmed — Testing, direction module
- Mariam mohmmed — Infrastructure, toolchain, Sobel, Gaussian
- yousra — Testing, direction module


## Pipeline Stages Implemented
- [x] Gaussian Blur (5x5 kernel, zero-padding)
- [x] Sobel Gradient (Gx, Gy)
- [x] Gradient Magnitude (L1 and L2 norm)
- [x] Gradient Direction (quantized to 0/45/90/135°)
- [x] Non-Maximum Suppression
- [x] Double Thresholding
- [x] RVV-optimized Gaussian Blur
- [x] RVV-optimized Magnitude (L1 and L2)

## Prerequisites

- Linux or WSL2 (Ubuntu 24.04 recommended)
- RISC-V GNU toolchain built with RVV support (`riscv64-linux-gnu-g++`)
- QEMU built for `riscv64-linux-user` with vector extension support
- GoogleTest installed at `$HOME/gtest` (include and lib directories)


## Toolchain Setup

This project initially used the system `riscv64-linux-gnu-g++` (via apt) during 
early development. As required by the project specification, we later built the 
RISC-V GNU toolchain from source (from `riscv-collab/riscv-gnu-toolchain`) with 
`--with-arch=rv64gcv`, producing `riscv64-unknown-elf-g++`. 

Build the toolchain from source:
```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv64gcv
make -j$(nproc)
```

Bare-metal binaries are run using `qemu-riscv64` in user-mode emulation, with 
syscall translation handled by our minimal `syscalls.cpp` support file.


## Project Structure
src/          - implementation source files (scalar + RVV)

include/      - header files

tests/        - GoogleTest unit tests (host-side)

images/       - test images used for pipeline verification

app/          - application entry point support files

build/        - build output (not committed, see .gitignore)

Makefile      - dual-target build system (host tests + RISC-V cross-compile)


## Build Instructions

Clone the repository:
```bash
git clone https://github.com/Maryam531/canny-rvv-project.git
cd canny-rvv-project
```

Build the RISC-V cross-compiled binary:
```bash
make
```
This produces `build/canny`, compiled with `-march=rv64gcv -mabi=lp64d -O3 -ftree-vectorize`.


## Running the Pipeline on QEMU

Run at the default vector length (VLEN=256):
```bash
make run
```

Run at specific VLEN values to confirm vector-length-agnostic correctness:
```bash
make run128
make run256
make run512
```

Or run manually with a custom VLEN:
```bash
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./build/canny
```

## Running Tests

Run the full GoogleTest suite (host-side, native compilation):
```bash
make test
```

Run individual test groups:
```bash
make test_direction
make test_nms
make test_threshold
```

## Compiler Optimization Sweep

Build the pipeline at multiple optimization levels for performance comparison:
```bash
make sweep
```
This produces `build/canny_O0`, `build/canny_O2`, `build/canny_O3`, and 
`build/canny_O3_vec` (the last one also generates an auto-vectorization report 
at `build/vec_report.txt`).

## Cleaning Build Artifacts

s```bash
make clean
```

## RVV Implementation Notes

The hotspot kernels identified through profiling — Gaussian blur convolution 
(`src/rvv_gaussian.cpp`) and Sobel magnitude computation (`src/rvv_magnitude.cpp`) — 
were rewritten using RISC-V Vector intrinsics from `<riscv_vector.h>`. Each 
intrinsic call is annotated with: (1) what operation it performs, (2) why that 
specific LMUL was chosen, and (3) how the behavior adapts across different VLEN 
values. Output is verified bit-identical to the scalar baseline at VLEN=128, 
256, and 512.

## AI Usage Log

See [AI_USAGE.md](./AI_USAGE.md) for documented examples of AI tool usage 
during development.
