#include <cuda_runtime.h>

#include <cstddef>

namespace gbis {

namespace {

// Naive batched GEMM: one thread per (row, col) output element, K-length
// inner product. Row-major, float32. C[b] = A[b] * B[b].
__global__ void gemm_naive_batched_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C, int M, int N,
                                          int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch = blockIdx.z;

  if (row >= M || col >= N) return;

  const std::size_t stride_a = static_cast<std::size_t>(M) * K;
  const std::size_t stride_b = static_cast<std::size_t>(K) * N;
  const std::size_t stride_c = static_cast<std::size_t>(M) * N;

  const float* A_b = A + batch * stride_a;
  const float* B_b = B + batch * stride_b;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += A_b[row * K + k] * B_b[k * N + col];
  }
  C[batch * stride_c + row * N + col] = acc;
}

// Tiled batched GEMM with 32x32x32 shared-memory tiles. Each block computes
// a 32x32 tile of C for one batch element; gridDim.z indexes the batch.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_tiled_batched_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C, int M, int N,
                                          int K) {
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;
  const int batch = blockIdx.z;

  const std::size_t stride_a = static_cast<std::size_t>(M) * K;
  const std::size_t stride_b = static_cast<std::size_t>(K) * N;
  const std::size_t stride_c = static_cast<std::size_t>(M) * N;

  const float* A_b = A + batch * stride_a;
  const float* B_b = B + batch * stride_b;
  float* C_b = C + batch * stride_c;

  __shared__ float As[BLOCK_M][BLOCK_K];
  __shared__ float Bs[BLOCK_K][BLOCK_N];

  const int row = block_row * BLOCK_M + threadIdx.y;
  const int col = block_col * BLOCK_N + threadIdx.x;

  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    if (row < M && (k0 + static_cast<int>(threadIdx.x)) < K) {
      As[threadIdx.y][threadIdx.x] = A_b[row * K + k0 + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if ((k0 + static_cast<int>(threadIdx.y)) < K && col < N) {
      Bs[threadIdx.y][threadIdx.x] = B_b[(k0 + threadIdx.y) * N + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK_K; ++kk) {
      acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C_b[row * N + col] = acc;
  }
}

}  // namespace

// C[b] = A[b] * B[b]; A is (M, K), B is (K, N), C is (M, N), all row-major.
void gemm_naive_batched(const float* A, const float* B, float* C, int M, int N,
                        int K, int batch, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16, batch);
  gemm_naive_batched_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void gemm_tiled_batched(const float* A, const float* B, float* C, int M, int N,
                        int K, int batch, cudaStream_t stream) {
  constexpr int BLOCK = 32;
  dim3 block(BLOCK, BLOCK);
  dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK, batch);
  gemm_tiled_batched_kernel<BLOCK, BLOCK, BLOCK>
      <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

}  // namespace gbis
