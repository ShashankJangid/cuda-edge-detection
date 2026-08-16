#include <cmath>
#include <algorithm>
#include "cuda_edge.h"

namespace cuda_edge {

void sobel_filter_cpu(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = -1 * input[(y - 1)*width + (x - 1)] + 1 * input[(y - 1)*width + (x + 1)]
                     -2 * input[y * width + (x - 1)]     + 2 * input[y * width + (x + 1)]
                     -1 * input[(y + 1)*width + (x - 1)] + 1 * input[(y + 1)*width + (x + 1)];

            int gy = -1 * input[(y - 1)*width + (x - 1)] - 2 * input[(y - 1)*width + x] - 1 * input[(y - 1)*width + (x + 1)]
                     +1 * input[(y + 1)*width + (x - 1)] + 2 * input[(y + 1)*width + x] + 1 * input[(y + 1)*width + (x + 1)];

            int mag = static_cast<int>(std::hypot(static_cast<double>(gx), static_cast<double>(gy)));
            output[y * width + x] = static_cast<uint8_t>(std::min(255, mag));
        }
    }
}

} // namespace cuda_edge
