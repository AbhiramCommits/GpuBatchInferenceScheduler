#pragma once

#include <cuda_runtime.h>

namespace gbis {

// C[b] = A[b] * B[b]; A is (M, K), B is (K, N), C is (M, N), all row-major,
// float32, batched over `batch` elements with contiguous per-batch strides.
void gemm_naive_batched(
    const float* A, const float* B, float* C, int M, int N, int K, int batch, cudaStream_t stream);
void gemm_tiled_batched(
    const float* A, const float* B, float* C, int M, int N, int K, int batch, cudaStream_t stream);

}  // namespace gbis
