#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace cuda_edge {

// Store 3x3 Sobel kernels in __constant__ memory for high-speed warp broadcast
__constant__ int8_t c_sobel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int8_t c_sobel_y[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

__global__ void sobel_constant_mem_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int gx = 0;
    int gy = 0;

    #pragma unroll
    for (int ky = -1; ky <= 1; ++ky) {
        #pragma unroll
        for (int kx = -1; kx <= 1; ++kx) {
            uint8_t pixel = input[(y + ky) * width + (x + kx)];
            gx += pixel * c_sobel_x[ky + 1][kx + 1];
            gy += pixel * c_sobel_y[ky + 1][kx + 1];
        }
    }

    int mag = static_cast<int>(hypotf(static_cast<float>(gx), static_cast<float>(gy)));
    output[y * width + x] = static_cast<uint8_t>(min(255, mag));
}

void sobel_filter_constant_gpu(
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

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobel_constant_mem_kernel<<<grid, block>>>(d_input, d_output, width, height);
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
