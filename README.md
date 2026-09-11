# gpu-batch-inference-scheduler

C++17 + CUDA project for benchmarking batched GEMM on GPU, targeting a
GPU batch inference scheduler. Implements batched GEMM `C[b] = A[b] * B[b]`
(A: MxK, B: KxN, row-major, float32) three ways:

- `naive` — one thread per output element
- `tiled` — custom 32x32 shared-memory tiled kernel (`src/kernels/gemm.cu`)
- `cublas` — `cublasSgemmStridedBatched` (`src/kernels/gemm_cublas.cu`)

plus a single-threaded CPU triple-loop reference used for both correctness
verification and as the performance baseline.

## Layout

```
src/core/cuda_check.hpp      CUDA_CHECK / CUBLAS_CHECK macros (throw std::runtime_error)
src/core/device_buffer.hpp   RAII device memory wrapper
src/kernels/gemm.cu          naive + tiled batched GEMM kernels
src/kernels/gemm_cublas.cu   batched GEMM via cublasSgemmStridedBatched
bench/bench_gemm.cpp         CLI benchmark harness
```

## Requirements

- CMake >= 3.22
- NVIDIA CUDA Toolkit (nvcc), CUDA architectures 70 / 80 / 89
- C++17 host compiler

## Build

```sh
cmake -B build
cmake --build build -j
```

## Run

```sh
./build/bench_gemm --m 512 --n 512 --k 512 --batch 8 --iters 10 --impl tiled
./build/bench_gemm --m 1024 --n 1024 --k 1024 --batch 4 --iters 5 --impl cublas
```

Options: `--m --n --k --batch --iters --impl {naive,tiled,cublas,cpu}`.

The harness reports H2D transfer, kernel, D2H transfer, and total wall time
(all ms), GFLOP/s, speedup vs the CPU baseline, and the max absolute error vs
the CPU reference as a CSV row on stdout. It exits nonzero if the max abs
error exceeds `1e-3` (absolute, float32 — very large `--k` may exceed
float32 rounding bounds).
