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

# Batched GEMM kernels
add_library(gemm_tiled STATIC src/kernels/gemm.cu)
target_include_directories(gemm_tiled PUBLIC src)
target_link_libraries(gemm_tiled PUBLIC CUDA::cudart)

add_library(gemm_cublas STATIC src/kernels/gemm_cublas.cu)
target_include_directories(gemm_cublas PUBLIC src)
target_link_libraries(gemm_cublas PUBLIC CUDA::cudart CUDA::cublas)

# Scheduler core
add_library(gbis_sched STATIC
  src/sched/scheduler.cpp
  src/sched/gpu_monitor.cpp
)
target_include_directories(gbis_sched PUBLIC src)
target_link_libraries(gbis_sched PUBLIC
  gemm_tiled
  gemm_cublas
  CUDA::cudart
  CUDA::cublas
  CUDA::nvml
)

# Benchmarks and daemon
add_executable(bench_gemm bench/bench_gemm.cpp)
target_include_directories(bench_gemm PRIVATE src)
target_link_libraries(bench_gemm PRIVATE gemm_tiled gemm_cublas CUDA::cudart)

add_executable(scheduler_daemon src/main.cpp)
target_link_libraries(scheduler_daemon PRIVATE gbis_sched)

# Tests
include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(test_scheduler tests/test_scheduler.cpp)
  target_link_libraries(test_scheduler PRIVATE gbis_sched GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(test_scheduler)
endif()
