#pragma once

#include <cstdint>
#include <vector>

namespace cuda_edge {

// Error checking helper
void check_cuda_error(const char* msg);

// GPU Sobel Edge Detection (Shared Memory optimized)
void sobel_filter_gpu(
    const uint8_t* h_input,
    uint8_t* h_output,
    int width,
    int height,
    float* out_kernel_time_ms = nullptr
);

// GPU Canny Edge Detection Pipeline
void canny_edge_gpu(
    const uint8_t* h_input,
    uint8_t* h_output,
    int width,
    int height,
    uint8_t low_threshold = 50,
    uint8_t high_threshold = 150,
    float* out_pipeline_time_ms = nullptr
);

// CPU Baseline for speedup benchmarking
void sobel_filter_cpu(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
);

void canny_filter_cpu(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    uint8_t low_threshold = 50,
    uint8_t high_threshold = 150
);

} // namespace cuda_edge
