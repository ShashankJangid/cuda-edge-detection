#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "cuda_edge.h"

struct Resolution {
    std::string name;
    int width;
    int height;
};

int main() {
    std::cout << "========================================================\n";
    std::cout << "       CUDA vs CPU Edge Detection Performance Benchmark\n";
    std::cout << "========================================================\n\n";

    std::vector<Resolution> resolutions = {
        {"720p HD",  1280,  720},
        {"1080p FHD", 1920, 1080},
        {"4K UHD",   3840, 2160},
        {"8K UHD",   7680, 4320}
    };

    std::cout << std::left << std::setw(12) << "Resolution"
              << std::setw(15) << "Dimensions"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "CUDA GPU (ms)"
              << std::setw(12) << "Speedup" << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    for (const auto& res : resolutions) {
        size_t num_pixels = res.width * res.height;
        std::vector<uint8_t> h_input(num_pixels);
        std::vector<uint8_t> h_output_cpu(num_pixels, 0);
        std::vector<uint8_t> h_output_gpu(num_pixels, 0);

        // Fill with mock image data
        for (size_t i = 0; i < num_pixels; ++i) {
            h_input[i] = static_cast<uint8_t>(i % 256);
        }

        // Benchmark CPU
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cuda_edge::sobel_filter_cpu(h_input.data(), h_output_cpu.data(), res.width, res.height);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

        // Benchmark GPU (warmup + execute)
        float gpu_ms = 0.0f;
        cuda_edge::sobel_filter_gpu(h_input.data(), h_output_gpu.data(), res.width, res.height, &gpu_ms);

        double speedup = cpu_ms / (gpu_ms > 0 ? gpu_ms : 0.001);

        std::cout << std::left << std::setw(12) << res.name
                  << std::setw(15) << (std::to_string(res.width) + "x" + std::to_string(res.height))
                  << std::setw(15) << std::fixed << std::setprecision(2) << cpu_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << gpu_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << (std::to_string(speedup) + "x")
                  << "\n";
    }

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
