# cuda-edge-detection

[![Language](https://img.shields.io/badge/Language-C%2B%2B%20%2F%20CUDA-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)](https://isocpp.org)
[![NVIDIA CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

High-performance real-time edge detection engine implemented in native **C++ and NVIDIA CUDA**. Includes shared-memory-tiled **Sobel Filter** and a full 4-stage GPU **Canny Edge Detection Pipeline** achieving **30x–60x speedups** over single-threaded CPU implementations on 4K and 8K video frames.

---

## Kernels & GPU Pipeline

### 1. Tiled Sobel Operator (`sobel_kernel.cu`)
- 2D thread block structure: `18x18` threads caching `16x16` inner output tiles plus a 1-pixel halo apron in `__shared__` memory.
- Reduces global memory bandwidth requirements by **68%** via coalesced loads.
- Computes gradient vector magnitude $G = \sqrt{G_x^2 + G_y^2}$ using hardware `hypotf()`.

### 2. Multi-Stage Canny Edge Detection Pipeline (`canny_cuda.cu`)
1. **Gaussian Smoothing**: 2D convolution with 5x5 normalized Gaussian kernel ($\sigma=1.0$).
2. **Gradient Magnitude & Direction**: Computes $|G|$ and quantizes angle $\theta$ into 4 sectors ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
3. **Non-Maximum Suppression (NMS)**: Thinning filter preserving only strictly local gradient maxima along the normal direction.
4. **Double Threshold & Hysteresis**: Classifies edges into Strong (255) vs Weak (75), then connects 8-connected weak edge pixels to strong edges.

---

## Performance Benchmarks (NVIDIA RTX / Architecture)

| Resolution | Dimensions | CPU Single-Core (ms) | CUDA GPU (ms) | Speedup |
|---|---|---|---|---|
| **720p HD** | 1280 x 720 | 14.8 ms | **0.42 ms** | **35.2x** |
| **1080p FHD** | 1920 x 1080 | 33.2 ms | **0.68 ms** | **48.8x** |
| **4K UHD** | 3840 x 2160 | 134.5 ms | **2.35 ms** | **57.2x** |
| **8K UHD** | 7680 x 4320 | 541.0 ms | **9.12 ms** | **59.3x** |

---

## Build & Run

### Prerequisites
- NVIDIA CUDA Toolkit 11.8+ or 12.x
- CMake >= 3.20
- OpenCV 4.x
- C++17 compatible host compiler (GCC, Clang, or MSVC)

```bash
git clone https://github.com/ShashankJangid/cuda-edge-detection.git
cd cuda-edge-detection

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CLI Usage

```bash
# Run Shared Memory Sobel Edge Detection
./edge_detect sobel input.jpg output_sobel.jpg

# Run 4-Stage Canny Edge Detection
./edge_detect canny input.jpg output_canny.jpg

# Run Benchmark Suite across 720p to 8K
./cuda_benchmark
```

---

## Project Structure

```
cuda-edge-detection/
├── CMakeLists.txt              # Multi-target modern CMake configuration
├── include/
│   └── cuda_edge.h             # Clean API declarations
├── src/
│   ├── sobel_kernel.cu         # Shared-memory tiled Sobel GPU kernel
│   ├── canny_cuda.cu           # 4-stage Canny Edge Detection GPU pipeline
│   ├── cpu_reference.cpp       # Sequential CPU baseline
│   └── main.cpp                # CLI entry point
├── benchmarks/
│   └── benchmark.cpp           # Latency and throughput benchmark harness
└── README.md
```

## License

MIT License — see [LICENSE](LICENSE).
