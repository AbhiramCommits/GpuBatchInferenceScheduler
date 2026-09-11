#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace gbis {

inline void throw_cuda_error(cudaError_t err, const char* file, int line,
                             const char* expr) {
  if (err != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " [" << expr
        << "]: " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

inline void throw_cublas_error(cublasStatus_t status, const char* file, int line,
                               const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << "cuBLAS error at " << file << ":" << line << " [" << expr
        << "]: status = " << static_cast<int>(status);
    throw std::runtime_error(oss.str());
  }
}

}  // namespace gbis

#define CUDA_CHECK(expr) ::gbis::throw_cuda_error((expr), __FILE__, __LINE__, #expr)
#define CUBLAS_CHECK(expr) \
  ::gbis::throw_cublas_error((expr), __FILE__, __LINE__, #expr)
