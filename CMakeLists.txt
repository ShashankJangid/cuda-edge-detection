cmake_minimum_required(VERSION 3.20)
project(CUDA_Edge_Detection LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architectures: Turing (sm_75), Ampere (sm_80, sm_86), Ada (sm_89), Hopper (sm_90)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

add_library(cuda_edge_lib STATIC
    src/sobel_kernel.cu
    src/kernels/sobel_naive.cu
    src/kernels/sobel_constant.cu
    src/canny_cuda.cu
    src/cpu_reference.cpp
)
target_link_libraries(cuda_edge_lib PUBLIC ${OpenCV_LIBS})

# Main CLI application
add_executable(edge_detect src/main.cpp)
target_link_libraries(edge_detect PRIVATE cuda_edge_lib)

# Benchmark Suite
add_executable(cuda_benchmark benchmarks/benchmark.cpp)
target_link_libraries(cuda_benchmark PRIVATE cuda_edge_lib)
