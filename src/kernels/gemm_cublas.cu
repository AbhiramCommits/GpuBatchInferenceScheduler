#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdint>

#include "core/cuda_check.hpp"
#include "kernels/gemm_cublas.hpp"

namespace gbis {

namespace {

// Process-lifetime cuBLAS handle, destroyed at exit. Owned by RAII.
class CublasHandle {
 public:
  CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
  ~CublasHandle() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

}  // namespace

// C[b] = A[b] * B[b]; A is (M, K), B is (K, N), C is (M, N), all row-major.
//
// cuBLAS expects column-major storage, so the row-major product C = A * B is
// computed as the column-major product C^T = B^T * A^T (both operands
// non-transposed in cuBLAS terms), reusing the same underlying buffers with
// swapped leading dimensions and per-batch strides.
void gemm_cublas_batched(const float* A, const float* B, float* C, int M,
                         int N, int K, int batch, cudaStream_t stream) {
  static CublasHandle handle;
  CUBLAS_CHECK(cublasSetStream(handle.get(), stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle.get(),
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, N, static_cast<std::int64_t>(K) * N,  // B^T viewed as (N, K)
      A, K, static_cast<std::int64_t>(M) * K,  // A^T viewed as (K, M)
      &beta,
      C, N, static_cast<std::int64_t>(M) * N,  // C^T viewed as (N, M)
      batch));
}

}  // namespace gbis
