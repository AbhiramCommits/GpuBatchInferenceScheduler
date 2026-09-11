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
src/sched/job.hpp            InferenceJob (id, m, n, k, batch, priority, bytes_required)
src/sched/scheduler.hpp/.cpp BatchScheduler: priority queue, packing, worker pool
src/sched/gpu_monitor.hpp/.cpp GPU telemetry via NVML (cudaMemGetInfo fallback)
src/main.cpp                 scheduler daemon entrypoint
bench/bench_gemm.cpp         CLI benchmark harness
tests/test_scheduler.cpp     GoogleTest unit tests
```

## Scheduler

`BatchScheduler` keeps a thread-safe priority queue (higher priority first,
FIFO within equal priority). A dispatcher thread packs queued jobs against the
free GPU memory budget (NVML `nvmlDeviceGetMemoryInfo` free memory minus a
configurable safety margin), greedily maximizing the number of jobs whose
summed `bytes_required()` fits, capped by `--max-batch`. A pool of worker
threads (one per CUDA stream pair: a copy stream and a compute stream)
executes each packed batch with the tiled or cuBLAS batched GEMM kernels,
overlapping H2D copies with compute using `cudaMemcpyAsync` + pinned host
memory (`cudaHostAlloc`).

Per-job metrics (queue wait ms, execute ms, end-to-end latency ms) are
recorded; on shutdown a CSV is written and p50/p95/p99 latency, throughput
(jobs/sec) and mean GPU utilization (sampled from NVML during the run) are
printed.

## Requirements

- CMake >= 3.22
- NVIDIA CUDA Toolkit (nvcc + NVML), CUDA architectures 70 / 80 / 89
- C++17 host compiler
- Network access for FetchContent (GoogleTest) on first build

## Build

```sh
cmake -B build
cmake --build build -j
```

## Run

```sh
# kernel benchmark
./build/bench_gemm --m 512 --n 512 --k 512 --batch 8 --iters 10 --impl tiled

# scheduler daemon: 100 synthetic jobs, poisson arrivals at 20 jobs/s
./build/scheduler_daemon --jobs 100 --arrival-rate 20 --impl tiled

# replay a job list from CSV (columns: id,m,n,k,batch,priority)
./build/scheduler_daemon --replay jobs.csv --workers 4 --max-batch 16
```

## Test

```sh
ctest --test-dir build --output-on-failure
```

GPU-only tests skip cleanly when no CUDA device is present.

Options: `--m --n --k --batch --iters --impl {naive,tiled,cublas,cpu}`.

The harness reports H2D transfer, kernel, D2H transfer, and total wall time
(all ms), GFLOP/s, speedup vs the CPU baseline, and the max absolute error vs
the CPU reference as a CSV row on stdout. It exits nonzero if the max abs
error exceeds `1e-3` (absolute, float32 — very large `--k` may exceed
float32 rounding bounds).
