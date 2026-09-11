cmake_minimum_required(VERSION 3.22)
project(gpu_batch_inference_scheduler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 70 80 89)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(gemm_tiled STATIC src/kernels/gemm.cu)
target_include_directories(gemm_tiled PUBLIC src)
target_link_libraries(gemm_tiled PUBLIC CUDA::cudart)

add_library(gemm_cublas STATIC src/kernels/gemm_cublas.cu)
target_include_directories(gemm_cublas PUBLIC src)
target_link_libraries(gemm_cublas PUBLIC CUDA::cudart CUDA::cublas)

add_executable(bench_gemm bench/bench_gemm.cpp)
target_include_directories(bench_gemm PRIVATE src)
target_link_libraries(bench_gemm PRIVATE gemm_tiled gemm_cublas CUDA::cudart)
