#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_edge.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./edge_detect <mode: sobel|canny> <input_image_path> [output_image_path]\n";
        return 1;
    }

    std::string mode = argv[1];
    std::string input_path = argv[2];
    std::string output_path = (argc >= 4) ? argv[3] : "output_edge.jpg";

    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << "\n";
        return 1;
    }

    cv::Mat out_img(img.rows, img.cols, CV_8UC1);
    float kernel_time_ms = 0.0f;

    if (mode == "sobel") {
        cuda_edge::sobel_filter_gpu(img.data, out_img.data, img.cols, img.rows, &kernel_time_ms);
        std::cout << "Executed CUDA Shared Memory Sobel in " << kernel_time_ms << " ms\n";
    } else if (mode == "canny") {
        cuda_edge::canny_edge_gpu(img.data, out_img.data, img.cols, img.rows, 50, 150, &kernel_time_ms);
        std::cout << "Executed CUDA 4-Stage Canny Pipeline in " << kernel_time_ms << " ms\n";
    } else {
        std::cerr << "Unknown mode: " << mode << " (use 'sobel' or 'canny')\n";
        return 1;
    }

    cv::imwrite(output_path, out_img);
    std::cout << "Saved processed edge map to " << output_path << "\n";
    return 0;
}
