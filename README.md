# High-Performance CUDA Edge Detection Engine

[![Language](https://img.shields.io/badge/Language-C%2B%2B17%20%2F%20CUDA-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)](https://isocpp.org)
[![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA_12.x-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![Profiling](https://img.shields.io/badge/Profiling-Nsight_Compute_%26_Systems-76B900?style=flat-square)](https://developer.nvidia.com/tools-overview)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

High-throughput, real-time edge detection engine implemented in native **C++17 and NVIDIA CUDA**. Includes multiple kernel optimizations (Global Memory Naive, Constant Memory Mask, and 2D Shared Memory Tiling with Halo Apron), a full 4-stage GPU **Canny Edge Detection Pipeline**, detailed **Roofline Model analysis**, and **NVIDIA Nsight** profiling workflows.

Achieves **up to 59.3× speedup** over optimized single-core CPU baselines, processing 4K frames in **2.35 ms** (~425 FPS) and 8K video in **9.12 ms** (>100 FPS).

---

## 📁 Repository Structure

```
cuda-edge-detection/
├── CMakeLists.txt              # CMake configuration supporting sm_75 to sm_90
├── include/
│   └── cuda_edge.h             # Clean C++/CUDA API declarations
├── src/
│   ├── kernels/
│   │   ├── sobel_naive.cu      # Baseline global-memory kernel
│   │   ├── sobel_constant.cu   # Constant-memory mask broadcast kernel
│   │   └── sobel_shared.cu     # 2D Shared-memory tiled kernel (optimal)
│   ├── canny_cuda.cu           # 4-Stage Canny Edge Detection GPU pipeline
│   ├── cpu_reference.cpp       # Single-threaded CPU baseline implementation
│   ├── main.cpp                # Unified CLI runner (Sobel & Canny)
│   └── utils/
│       └── plot_results.py     # Benchmark plotting script (FPS, throughput, latency)
├── benchmarks/
│   ├── benchmark.cpp           # Multi-resolution timing and speedup benchmark harness
│   └── results/
│       └── cpu_gpu_comparison.csv
├── profiling/
│   └── profile_nsight.sh       # Nsight Systems & Nsight Compute automation
└── README.md
```

---

## 🔬 Kernel Architectures & Memory Hierarchy

### 1. Naive Global Memory Kernel (`sobel_naive.cu`)
- Each thread loads $3 \times 3 = 9$ pixels directly from uncached global memory.
- Highly memory-bandwidth bound due to redundant uncoalesced neighbor reads.

### 2. Constant Memory Mask Kernel (`sobel_constant.cu`)
- Places the horizontal and vertical $3 \times 3$ convolution matrices into `__constant__` memory (`c_sobel_x`, `c_sobel_y`).
- Leverages the dedicated 64 KB constant cache, broadcasting filter weights across the entire warp in a single cycle.

### 3. 2D Shared-Memory Tiling with Halo Apron (`sobel_shared.cu` - Optimal)
- Thread blocks configured as `18x18` threads to compute `16x16` internal output pixels.
- The 1-pixel boundary halo is cooperatively loaded into `__shared__ uint8_t s_data[18][18]` followed by `__syncthreads()`.
- **Reduces global memory transactions by 68%**, achieving near-roofline memory throughput.

### 4. 4-Stage GPU Canny Edge Detection Pipeline (`canny_cuda.cu`)
1. **Gaussian Smoothing Filter**: 2D convolution using a 5x5 normalized Gaussian kernel ($\sigma = 1.0$).
2. **Gradient Magnitude & Direction**: Evaluates $G = \sqrt{G_x^2 + G_y^2}$ using fast hardware `hypotf()` and quantizes gradient angles into 4 directional sectors ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
3. **Non-Maximum Suppression (NMS)**: Suppresses all non-peak gradient pixels along the local gradient vector.
4. **Double Threshold & Hysteresis**: Classifies edges into Strong (255) vs Weak (75), resolving 8-connected weak pixels to strong edges in parallel.

---

## 📊 Performance Benchmarks (CPU vs. NVIDIA CUDA GPU)

Evaluated across standard resolutions on an NVIDIA Ampere architecture:

| Resolution | Dimensions | CPU Single-Core (ms) | CUDA Naive (ms) | CUDA Shared Mem (ms) | GPU FPS | Speedup | Throughput (MPix/s) |
|---|---|---|---|---|---|---|---|
| **512×512** | 512 × 512 | 2.45 ms | 0.35 ms | **0.12 ms** | 8,333 FPS | **20.4×** | 2,184.5 |
| **1080p FHD** | 1920 × 1080 | 33.20 ms | 1.85 ms | **0.68 ms** | 1,470 FPS | **48.8×** | 3,049.4 |
| **4K UHD** | 3840 × 2160 | 134.50 ms | 7.20 ms | **2.35 ms** | 425 FPS | **57.2×** | 3,529.5 |
| **6K Cinema** | 5824 × 3264 | 312.40 ms | 16.50 ms | **5.40 ms** | 185 FPS | **57.9×** | 3,520.3 |
| **8K UHD** | 7680 × 4320 | 541.00 ms | 28.40 ms | **9.12 ms** | 109 FPS | **59.3×** | 3,637.9 |

---

## 📈 Roofline Model & Profiling (Nsight)

### Roofline Analysis
- **Operational Intensity**: $\approx 0.28 \text{ FLOP/byte}$ (Memory-bound workload).
- **Shared Memory Tiling** shifts the operational point significantly toward the compute ceiling by eliminating repetitive DRAM transactions.

### Nsight Tools Integration
Run the automated profiling script in `profiling/`:

```bash
# 1. Timeline & Memory Transfer Analysis via Nsight Systems
nsys profile --trace=cuda,nvtx --stats=true ./build/edge_detect sobel input.jpg

# 2. Kernel Occupancy & Roofline Analysis via Nsight Compute
ncu --section SpeedOfLight --section Occupancy ./build/edge_detect sobel input.jpg
```

---

## 🛠️ Build & Execution

### Requirements
- **OS**: Linux (Ubuntu 20.04 / 22.04) or Windows / WSL2
- **Compiler**: GCC >= 9.0 / Clang with C++17 support
- **CUDA Toolkit**: CUDA 11.8+ or 12.x (`nvcc --version`)
- **Libraries**: OpenCV 4.x, CMake >= 3.20

### Build Instructions
```bash
git clone https://github.com/ShashankJangid/cuda-edge-detection.git
cd cuda-edge-detection

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Usage

```bash
# Run Shared-Memory Sobel on an Image
./edge_detect sobel path/to/input.jpg output_sobel.jpg

# Run 4-Stage Canny Edge Detection Pipeline
./edge_detect canny path/to/input.jpg output_canny.jpg

# Run Multi-Resolution Benchmark Suite
./cuda_benchmark
```

---

## 🚀 Advanced Optimization Roadmaps

- [x] 2D Shared Memory Tiling with halo boundary clamping
- [x] Constant Memory convolution mask broadcasting
- [x] 4-Stage Canny Edge Detection with NMS & hysteresis
- [x] Nsight Systems & Nsight Compute profiling workflows
- [ ] Multi-GPU stream pipelining via `cudaStream_t` for concurrent 8K video decoding/filtering
- [ ] Warp-shuffle intrinsics (`__shfl_down_sync`) for register-level reductions

---

## 👤 Author

**Shashank Jangid** — Founder @ [Orange Future Tech](https://orangefuturetech.com)  
Portfolio: [shashankjangid.vercel.app](https://shashankjangid.vercel.app) · GitHub: [@ShashankJangid](https://github.com/ShashankJangid)

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
