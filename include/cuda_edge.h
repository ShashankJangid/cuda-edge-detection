#pragma once

#include <cstdint>
#include <string>

namespace cuda_edge {

// Naive Global Memory Sobel Kernel
void sobel_filter_naive_gpu(
    const uint8_t* h_input,
    uint8_t* h_output,
    int width,
    int height,
    float* out_kernel_time_ms = nullptr
);

// Constant Memory Sobel Kernel
void sobel_filter_constant_gpu(
    const uint8_t* h_input,
    uint8_t* h_output,
    int width,
    int height,
    float* out_kernel_time_ms = nullptr
);

// Shared Memory Tiled Sobel Kernel (Optimal)
void sobel_filter_gpu(
    const uint8_t* h_input,
    uint8_t* h_output,
    int width,
    int height,
    float* out_kernel_time_ms = nullptr
);

// 4-Stage Canny Edge Detection Pipeline (Gaussian, Sobel Mag/Angle, NMS, Hysteresis)
void canny_edge_gpu(
    const uint8_t* h_input,
    uint8_t* h_output,
    int width,
    int height,
    uint8_t low_threshold = 50,
    uint8_t high_threshold = 150,
    float* out_pipeline_time_ms = nullptr
);

// CPU Single-Core Reference
void sobel_filter_cpu(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
);

} // namespace cuda_edge
