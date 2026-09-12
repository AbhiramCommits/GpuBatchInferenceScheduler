#pragma once

#include <cuda_runtime.h>

namespace gbis {

// Same batched GEMM semantics as gemm.hpp, computed with
// cublasSgemmStridedBatched (row-major handled via the C^T = B^T * A^T
// column-major trick).
void gemm_cublas_batched(
    const float* A, const float* B, float* C, int M, int N, int K, int batch, cudaStream_t stream);

}  // namespace gbis
