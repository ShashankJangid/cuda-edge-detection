#include <cuda_runtime.h>
#include <cmath>
#include "cuda_edge.h"

namespace cuda_edge {

// Stage 1: 5x5 Gaussian Smoothing Filter
__global__ void gaussian_blur_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 5x5 Normalized Gaussian Kernel (sigma = 1.0)
    const float kernel[5][5] = {
        { 2/159.f,  4/159.f,  5/159.f,  4/159.f, 2/159.f },
        { 4/159.f,  9/159.f, 12/159.f,  9/159.f, 4/159.f },
        { 5/159.f, 12/159.f, 15/159.f, 12/159.f, 5/159.f },
        { 4/159.f,  9/159.f, 12/159.f,  9/159.f, 4/159.f },
        { 2/159.f,  4/159.f,  5/159.f,  4/159.f, 2/159.f }
    };

    float sum = 0.0f;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);
            sum += input[py * width + px] * kernel[ky + 2][kx + 2];
        }
    }
    output[y * width + x] = static_cast<uint8_t>(sum);
}

// Stage 2: Gradient Magnitude & Quantized Sector Direction (0, 45, 90, 135 deg)
__global__ void sobel_magnitude_direction_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ magnitude,
    uint8_t* __restrict__ direction,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int gx = -1 * input[(y - 1)*width + (x - 1)] + 1 * input[(y - 1)*width + (x + 1)]
             -2 * input[y * width + (x - 1)]     + 2 * input[y * width + (x + 1)]
             -1 * input[(y + 1)*width + (x - 1)] + 1 * input[(y + 1)*width + (x + 1)];

    int gy = -1 * input[(y - 1)*width + (x - 1)] - 2 * input[(y - 1)*width + x] - 1 * input[(y - 1)*width + (x + 1)]
             +1 * input[(y + 1)*width + (x - 1)] + 2 * input[(y + 1)*width + x] + 1 * input[(y + 1)*width + (x + 1)];

    float mag = hypotf(static_cast<float>(gx), static_cast<float>(gy));
    magnitude[y * width + x] = mag;

    // Angle in degrees [0, 180]
    float angle = atan2f(static_cast<float>(gy), static_cast<float>(gx)) * 180.0f / 3.14159265f;
    if (angle < 0) angle += 180.0f;

    // Quantize into 4 sectors: 0, 45, 90, 135
    if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) {
        direction[y * width + x] = 0;   // Horizontal
    } else if (angle >= 22.5f && angle < 67.5f) {
        direction[y * width + x] = 45;  // Diagonal /
    } else if (angle >= 67.5f && angle < 112.5f) {
        direction[y * width + x] = 90;  // Vertical |
    } else {
        direction[y * width + x] = 135; // Diagonal \
    }
}

// Stage 3: Non-Maximum Suppression (NMS)
__global__ void non_maximum_suppression_kernel(
    const float* __restrict__ magnitude,
    const uint8_t* __restrict__ direction,
    uint8_t* __restrict__ output,
    int width,
    int height,
    uint8_t low_thresh,
    uint8_t high_thresh
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    float current = magnitude[y * width + x];
    float p1 = 0.0f, p2 = 0.0f;
    uint8_t dir = direction[y * width + x];

    if (dir == 0) {
        p1 = magnitude[y * width + (x - 1)];
        p2 = magnitude[y * width + (x + 1)];
    } else if (dir == 45) {
        p1 = magnitude[(y - 1) * width + (x + 1)];
        p2 = magnitude[(y + 1) * width + (x - 1)];
    } else if (dir == 90) {
        p1 = magnitude[(y - 1) * width + x];
        p2 = magnitude[(y + 1) * width + x];
    } else if (dir == 135) {
        p1 = magnitude[(y - 1) * width + (x - 1)];
        p2 = magnitude[(y + 1) * width + (x + 1)];
    }

    // Preserve edge only if strictly local maximum
    if (current >= p1 && current >= p2) {
        if (current >= high_thresh) {
            output[y * width + x] = 255; // Strong edge
        } else if (current >= low_thresh) {
            output[y * width + x] = 75;  // Weak edge
        } else {
            output[y * width + x] = 0;
        }
    } else {
        output[y * width + x] = 0;
    }
}

// Stage 4: Double-Threshold Hysteresis Edge Tracking
__global__ void hysteresis_kernel(
    uint8_t* __restrict__ image,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    if (image[y * width + x] == 75) {
        // Check 8-connected neighborhood for strong edges (255)
        bool has_strong_neighbor = false;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (image[(y + dy) * width + (x + dx)] == 255) {
                    has_strong_neighbor = true;
                    break;
                }
            }
        }
        image[y * width + x] = has_strong_neighbor ? 255 : 0;
    }
}

void canny_edge_gpu(
    const uint8_t* h_input,
    uint8_t* h_output,
    int width,
    int height,
    uint8_t low_threshold,
    uint8_t high_threshold,
    float* out_pipeline_time_ms
) {
    size_t byte_size = width * height * sizeof(uint8_t);
    size_t float_size = width * height * sizeof(float);

    uint8_t *d_input = nullptr, *d_blurred = nullptr, *d_direction = nullptr, *d_output = nullptr;
    float *d_magnitude = nullptr;

    cudaMalloc(&d_input, byte_size);
    cudaMalloc(&d_blurred, byte_size);
    cudaMalloc(&d_direction, byte_size);
    cudaMalloc(&d_output, byte_size);
    cudaMalloc(&d_magnitude, float_size);

    cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 1. Gaussian Blur
    gaussian_blur_kernel<<<grid, block>>>(d_input, d_blurred, width, height);

    // 2. Sobel Magnitude & Direction
    sobel_magnitude_direction_kernel<<<grid, block>>>(d_blurred, d_magnitude, d_direction, width, height);

    // 3. Non-Maximum Suppression
    non_maximum_suppression_kernel<<<grid, block>>>(d_magnitude, d_direction, d_output, width, height, low_threshold, high_threshold);

    // 4. Hysteresis
    hysteresis_kernel<<<grid, block>>>(d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (out_pipeline_time_ms) {
        cudaEventElapsedTime(out_pipeline_time_ms, start, stop);
    }

    cudaMemcpy(h_output, d_output, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_blurred);
    cudaFree(d_direction);
    cudaFree(d_output);
    cudaFree(d_magnitude);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace cuda_edge
