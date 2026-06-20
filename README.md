# Canny Edge Detection on RISC-V with RVV

A high-performance implementation of the Canny Edge Detection pipeline in C++, targeting the **RISC-V RV64GCV architecture** and accelerated using the **RISC-V Vector Extension (RVV)**. This proje>

---
# Project Objectives

The primary objectives of this project are:

* Implement a complete Canny Edge Detection pipeline in C++
* Cross-compile the application for the RISC-V RV64GCV architecture
* Execute the application using QEMU user-mode emulation
* Analyze compiler-generated optimizations
* Profile the pipeline to identify performance bottlenecks
* Accelerate critical kernels using RVV intrinsics
* Verify correctness across multiple vector lengths (VLEN)
* Compare scalar and vectorized implementations

---


# Pipeline Overview

The image processing workflow follows the standard Canny Edge Detection sequence:

```text
Input Image
     ↓
Gaussian Blur
     ↓
Sobel Gradient (Gx, Gy)
     ↓
Magnitude & Direction
     ↓
Non-Maximum Suppression
     ↓
Double Thresholding
     ↓
Final Edge Map
```

---
# Pipeline Stage Descriptions

## 1. Gaussian Blur

The first stage reduces image noise before edge detection. Without smoothing, small intensity variations and random noise may crea>

A 5×5 Gaussian kernel is convolved with the image, replacing each pixel with a weighted average of its neighbors.

### Purpose

* Remove high-frequency noise
* Smooth the image
* Improve edge detection accuracy

### Output

A blurred version of the original image with reduced noise.

---

## 2. Sobel Gradient Computation

After smoothing, the Sobel operator estimates intensity changes in both horizontal and vertical directions.

### Horizontal Gradient (Gx)

```text
-1  0  1
-2  0  2
-1  0  1
```

### Vertical Gradient (Gy)

```text
-1 -2 -1
 0  0  0
 1  2  1
```

### Purpose

* Detect potential edges
* Determine edge orientation
* Calculate gradient information

### Output

Two gradient images:

* Gx (horizontal gradient)
* Gy (vertical gradient)

---

## 3. Gradient Magnitude

The gradient magnitude represents edge strength.

Two methods were implemented:

### L1 Magnitude

```text
|Gx| + |Gy|
```

Advantages:

* Faster computation
* No square root operation
* More suitable for embedded systems

### L2 Magnitude

```text
sqrt(Gx² + Gy²)
```

Advantages:

* More accurate edge strength estimation
* Closer to the true Euclidean gradient

### Purpose
* Measure edge strength
* Distinguish strong edges from weak edges

### Output

A grayscale image where brighter pixels represent stronger edges.

---

## 4. Gradient Direction

The gradient direction describes edge orientation.

The angle is computed from Gx and Gy and quantized into four directions:

* 0°
* 45°
* 90°
* 135°

### Purpose

* Determine edge orientation
* Support Non-Maximum Suppression

### Output

A direction map containing one of the four quantized directions for each pixel.

---

## 5. Non-Maximum Suppression (NMS)

After magnitude computation, edges often appear thick and blurry.

Non-Maximum Suppression thins the edges by keeping only local maxima along the gradient direction and suppressing neighboring pixe>

### Purpose

* Remove thick edges
* Improve edge localization
* Produce single-pixel-wide edges

### Output

A thinner edge map containing only the strongest edge responses.

---
## 6. Double Thresholding

The final stage classifies pixels according to edge strength using two thresholds:

* High Threshold
* Low Threshold

Pixels are classified as:

* Strong Edge
* Weak Edge
* Non-Edge

### Purpose

* Remove remaining noise
* Separate meaningful edges from weak responses
* Improve final edge quality

### Output

The final edge-detected image.

---

# Additional Components

## Hysteresis Edge Tracking

Hysteresis is the final stage of the complete Canny Edge Detection algorithm. After double thresholding, pixels are classified as >

Weak edge pixels are preserved only if they are connected to a strong edge through neighboring pixels. Isolated weak edges are rem>

### Purpose

- Eliminate false edges caused by noise
- Preserve continuous object boundaries
- Improve the quality of the final edge map

### Output

A clean binary edge image containing only meaningful connected edges.

---

## Linker Script (linker.ld)

The linker script defines how the program is arranged in memory during the linking stage. It specifies where code, data, stack, an>

In this project, the linker script is required for bare-metal RISC-V execution and works together with the startup assembly file (>

### Purpose

- Define memory layout
- Place program sections correctly
- Support bare-metal execution

---

## Python Utility Scripts

Several Python scripts were created to assist development, debugging, and visualization.

### check_boundary.py

Verifies that boundary pixels are processed correctly, especially for convolution operations such as Gaussian blur and Sobel filte>

### view_image.py

Loads and displays raw grayscale image files, allowing visual inspection of intermediate and final outputs.

### view_pipeline.py

Visualizes multiple stages of the Canny pipeline side-by-side, making it easier to verify the correctness of each processing step >

---

# Implemented Features

## Pipeline Stages Implemented

- [x] Gaussian Blur (5x5 kernel, zero-padding)
- [x] Sobel Gradient (Gx, Gy)
- [x] Gradient Magnitude (L1 and L2 norm)
- [x] Gradient Direction (quantized to 0/45/90/135°)
- [x] Non-Maximum Suppression
- [x] Double Thresholding

## RVV Accelerated Kernels

- [x] RVV-optimized Gaussian Blur
- [x] RVV-optimized Magnitude (L1 and L2)

---


# Repository Structure

```text
canny-rvv-project/
│
├── include/            Header files
├── src/                Source code
├── tests/              GoogleTest test suite
├── images/             Test and output images
│
├── Makefile            Build configuration
├── README.md           Project documentation
├── .gitignore          Git ignore rules
│
├── build_riscv.sh      RISC-V build script
├── crt0.s              Bare-metal startup assembly
├── linker.ld           Memory layout and linker configuration for bare-metal builds
│
├── check_boundary.py   Boundary validation tool
├── view_image.py       Raw image viewer
├── view_pipeline.py    Visualizes intermediate stages of the pipeline
│
├── collage_triangle.png  Sample test image
├── vec_report_saved.txt  Auto-vectorization report
│
├── app                 Generated application binary
├── blur                Generated blur output
│
└── build/              Build artifacts
```

---

# Prerequisites

* Linux or WSL2 (Ubuntu 24.04 recommended)
* RISC-V GNU Toolchain with RVV support
* QEMU user-mode emulator
* GoogleTest

---

# Toolchain Setup

This project initially used the system `riscv64-linux-gnu-g++` (via apt) during
early development. As required by the project specification, we later built the
RISC-V GNU toolchain from source (from `riscv-collab/riscv-gnu-toolchain`) with
`--with-arch=rv64gcv`, producing `riscv64-unknown-elf-g++`.

Build the RISC-V GNU Toolchain:

```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git

cd riscv-gnu-toolchain

./configure \
    --prefix=/opt/riscv \
    --with-arch=rv64gcv

make -j$(nproc)
```
Bare-metal binaries are run using `qemu-riscv64` in user-mode emulation, with
syscall translation handled by our minimal `syscalls.cpp` support file.

The resulting compiler:

```bash
riscv64-unknown-elf-g++
```

supports RVV intrinsics through:

```cpp
#include <riscv_vector.h>
```

---

# Build Instructions

Clone the repository:

```bash
git clone https://github.com/Maryam531/canny-rvv-project.git

cd canny-rvv-project
```

Build the RISC-V executable:

```bash
make
```

Generated executable:

```text
build/canny
```

---

# Running the Application

Run with the default VLEN:

```bash
make run
```

Run specific VLEN configurations:

```bash
make run128
make run256
make run512
```

Manual execution:

```bash
qemu-riscv64 \
-cpu rv64,v=true,vlen=128 \
./build/canny
```

---

# Unit Testing

Run all tests:

```bash
make test
```

Run individual suites:

```bash
make test_direction
make test_nms
make test_threshold
```

Implemented tests include:

* Gaussian blur validation
* Sobel verification
* Magnitude computation
* Direction quantization
* NMS verification
* Threshold verification
* Scalar vs RVV equivalence

---
# Compiler Optimization Sweep

Build and compare optimization levels:

```bash
make sweep
```

Generated binaries:

```text
build/canny_O0
build/canny_O2
build/canny_O3
build/canny_O3_vec
```

Auto-vectorization report:

```text
build/vec_report.txt
```

---

# RVV Optimization

Profiling identified the following hotspots:

* Gaussian Blur
* Magnitude Computation

These kernels were rewritten using RVV intrinsics:

```text
src/rvv_gaussian.cpp
src/rvv_magnitude.cpp
```

Key RVV concepts demonstrated:

* Vector-length agnostic programming
* Dynamic VL configuration using vsetvl
* Strip-mining loops
* Widening arithmetic
* LMUL selection
* Cross-VLEN correctness

Correctness was verified for:

* VLEN = 128
* VLEN = 256
* VLEN = 512

---
# Performance Results

| Stage | -O0 | -O2 | -O3 | Auto-Vec (-O3) | RVV 128 | RVV 256 |
|---------|---------:|---------:|---------:|---------:|---------:|---------:|
| Gaussian 5×5 | 57.109 ms | 22.433 ms | 6.932 ms | 6.914 ms | 73.183 ms | 62.042 ms |
| Sobel Gx/Gy | 11.805 ms | 1.397 ms | 1.360 ms | 1.365 ms | – | – |
| Magnitude L1 (&#124;Gx&#124;+&#124;Gy&#124;) | 3.706 ms | 1.937 ms | 1.963 ms | 1.925 ms | 13.034 ms | 11.568 ms |
| Magnitude L2 (√(Gx²+Gy²)) | 57.065 ms | 14.676 ms | 14.379 ms | 14.486 ms | 28.268 ms | 26.474 ms |
| Direction | 6.811 ms | 1.808 ms | 1.808 ms | 1.387 ms | – | – |
| Binary Size | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB | 64 KB |

---

# AI Usage Log

Examples of AI-assisted development are documented in:

```text
AI_USAGE.md
```

Each entry includes:

1. Question asked
2. Suggested solution
3. Modifications made
4. Lessons learned

---

# Cleaning Build Artifacts

```bash
make clean
```

---

# Conclusion

This project demonstrates the implementation and optimization of a complete Canny Edge Detection pipeline on the RISC-V RV64GCV ar>
