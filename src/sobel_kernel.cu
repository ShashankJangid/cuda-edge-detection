#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "cuda_edge.h"

#define TILE_W 16
#define TILE_H 16
#define BLOCK_W (TILE_W + 2)
#define BLOCK_H (TILE_H + 2)

namespace cuda_edge {

// Device kernel for Sobel gradient computation using shared memory apron
__global__ void sobel_shared_mem_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height
) {
    __shared__ uint8_t s_data[BLOCK_H][BLOCK_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx - 1;
    int gy = blockIdx.y * TILE_H + ty - 1;

    // Load halo into shared memory
    int clamped_x = min(max(gx, 0), width - 1);
    int clamped_y = min(max(gy, 0), height - 1);
    s_data[ty][tx] = input[clamped_y * width + clamped_x];

    __syncthreads();

    // Only compute for internal tile threads
    if (tx > 0 && tx <= TILE_W && ty > 0 && ty <= TILE_H) {
        int x = blockIdx.x * TILE_W + tx - 1;
        int y = blockIdx.y * TILE_H + ty - 1;

        if (x < width && y < height) {
            // Sobel X kernel: [-1 0 1; -2 0 2; -1 0 1]
            int gx_val = -1 * s_data[ty - 1][tx - 1] + 1 * s_data[ty - 1][tx + 1]
                         -2 * s_data[ty][tx - 1]     + 2 * s_data[ty][tx + 1]
                         -1 * s_data[ty + 1][tx - 1] + 1 * s_data[ty + 1][tx + 1];

            // Sobel Y kernel: [-1 -2 -1; 0 0 0; 1 2 1]
            int gy_val = -1 * s_data[ty - 1][tx - 1] - 2 * s_data[ty - 1][tx] - 1 * s_data[ty - 1][tx + 1]
                         +1 * s_data[ty + 1][tx - 1] + 2 * s_data[ty + 1][tx] + 1 * s_data[ty + 1][tx + 1];

            int magnitude = static_cast<int>(hypotf(static_cast<float>(gx_val), static_cast<float>(gy_val)));
            output[y * width + x] = static_cast<uint8_t>(min(255, magnitude));
        }
    }
}

void sobel_filter_gpu(
    const uint8_t* h_input,
    uint8_t* h_output,
    int width,
    int height,
    float* out_kernel_time_ms
) {
    size_t size = width * height * sizeof(uint8_t);

    uint8_t *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_W, BLOCK_H);
    dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobel_shared_mem_kernel<<<grid, block>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (out_kernel_time_ms) {
        cudaEventElapsedTime(out_kernel_time_ms, start, stop);
    }

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace cuda_edge
