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
src/server/http_server.*     /healthz + Prometheus /metrics (cpp-httplib)
src/bindings/py_module.cpp   pybind11 module `gpuinfer`
bench/bench_gemm.cpp         CLI benchmark harness
tests/test_scheduler.cpp     GoogleTest unit tests
python/benchmark.py          NumPy vs gpuinfer benchmark
python/test_bindings.py      pytest suite for the bindings
deploy/                      Dockerfile, k8s manifests, Slurm scripts
scripts/profile.sh           nsys + ncu profiling runner
scripts/parse_nsys.py        nsys sqlite -> Markdown breakdown
docs/PROFILING.md            profiling write-up and next optimization
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

## Python bindings

Install the `gpuinfer` extension (compiled via scikit-build-core, needs a
CUDA toolchain):

```sh
pip install -e .
```

API:

```python
import numpy as np
import gpuinfer

# batched GEMM C[b] = A[b] @ B[b], float32; C-contiguous inputs are zero-copy
a = np.random.randn(4, 512, 512).astype(np.float32)
b = np.random.randn(4, 512, 512).astype(np.float32)
c = gpuinfer.batched_gemm(a, b)

# scheduler
job_id = gpuinfer.submit_job(m=512, n=512, k=512, batch=4, priority=1)
sched = gpuinfer.Scheduler(max_batch=8, workers=2, impl="tiled")
sched.start()
stats = sched.stats()   # submitted/completed/queued/failed, p50/p95/p99, ...
sched.stop()
```

Benchmark and tests (GPU tests skip automatically without a CUDA device):

```sh
python python/benchmark.py            # -> results/benchmark.csv, results/speedup.png
pytest python/test_bindings.py
```

## Test

```sh
ctest --test-dir build --output-on-failure
```

GPU-only tests skip cleanly when no CUDA device is present.

## Profiling

Requires a CUDA GPU with `nsys` / `ncu` on PATH:

```sh
scripts/profile.sh            # -> results/nsys_report.sqlite, ncu_tiled.ncu-rep
python3 scripts/parse_nsys.py results/nsys_report.sqlite
```

The transfer-vs-compute breakdown, pinned-memory/stream-overlap numbers, ncu
occupancy and memory throughput, and the identified bottleneck are written up
in `docs/PROFILING.md`.

## CI and code style

`.github/workflows/ci.yml` runs: a compile-only CUDA job (apt toolkit, build,
`clang-format --dry-run --Werror`, clang-tidy, CPU-only GoogleTest subset,
`pytest -k "not gpu"`, black + ruff), a buildx Docker build pushed to ghcr.io
on main, and kubeconform + shellcheck for the deploy manifests. See
`.pre-commit-config.yaml` (clang-format, black, ruff) for the local hooks.

## Deployment

Docker image, Kubernetes manifests, and Slurm scripts that run the identical
benchmark workload three ways (bare binary, container, cluster) live in
`deploy/` — see `deploy/README.md`, `deploy/k8s/README.md`, and
`deploy/slurm/`. The daemon serves `GET /healthz` and Prometheus-format
`GET /metrics` (queue depth, in-flight jobs, p95 latency, GPU utilization) on
`--http-port` (default 8080); `--jobs 0` runs until SIGINT, `--force-cpu`
runs without a GPU.

Options: `--m --n --k --batch --iters --impl {naive,tiled,cublas,cpu}`.

The harness reports H2D transfer, kernel, D2H transfer, and total wall time
(all ms), GFLOP/s, speedup vs the CPU baseline, and the max absolute error vs
the CPU reference as a CSV row on stdout. It exits nonzero if the max abs
error exceeds `1e-3` (absolute, float32 — very large `--k` may exceed
float32 rounding bounds).
