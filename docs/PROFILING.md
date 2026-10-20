# Profiling the batched GEMM

This document summarizes Nsight Systems (nsys) and Nsight Compute (ncu)
profiling of the batched GEMM kernels, and records the transfer-vs-compute
analysis workflow.

> [!NOTE]
> **Partial GPU measurements.** The nsys transfer/breakdown tables and the
> overlap table were measured on a Tesla T4 (sm_75, CUDA 13.0); the ncu
> kernel-level table still holds placeholders and should be filled on the
> target host.

## How to profile

```sh
# GEMM (default 1024^3, batch 8):
scripts/profile.sh

# Scheduler:
WORKLOAD=scheduler JOBS=1000 ARRIVAL_RATE=20 WORKERS=2 scripts/profile.sh

# Custom GEMM parameters / skip the CPU reference:
M=512 N=512 K=512 BATCH=16 ITERS=50 SKIP_CHECK=1 scripts/profile.sh
```

Each run produces `nsys_report.nsys-rep`, `nsys_report.sqlite`,
`nsys_stats.txt`, and `nsys_breakdown.md` under `OUTDIR`
(default `results/profiling/`). Set `NCU=0` to skip Nsight Compute.

## Transfer vs compute breakdown (nsys)

Measured on a Tesla T4 (sm_75, CUDA 13.0) with Nsight Systems 2025.3.1 via
`scripts/profile.sh`. Percentages are relative to the nsys wall time, which
spans the first to the last GPU activity.

### 8 x 1024 x 1024 x 1024 GEMM

Workload: `bench_gemm --m 1024 --n 1024 --k 1024 --batch 8 --iters 10
--warmup 10 --impl tiled --skip-check` (10 warmup + 10 timed sequences,
20 kernel launches). Artifacts: `results/profiling/gemm_8x1024.nsys-rep`,
`results/profiling/gemm_8x1024_summary.txt` (key tables) and
`results/profiling/gemm_8x1024_stats.txt` (full nsys output).

| Category | Time (ms) | % of wall time |
|---|---|---|
| GPU kernels | 375.571 | 44.86 |
| Memcpy HtoD | 294.585 | 35.19 |
| Memcpy DtoH | 150.633 | 17.99 |
| Other (CPU / gaps) | 16.441 | 1.96 |
| **Total** | **837.230** | **100.00** |

Kernel average was 18.78 ms (median 16.79 ms, min 16.44 ms, max 26.86 ms).
HtoD averaged 7.36 ms per copy (40 copies) and DtoH 7.53 ms (20 copies);
this is the pageable-memory path.

### Scheduler: 1000 jobs, arrival rate 20/s, 2 workers

Workload: `scheduler_daemon --jobs 1000 --arrival-rate 20 --workers 2
--http-port 0`. Artifacts: `results/profiling/scheduler_1000j.nsys-rep`,
`results/profiling/scheduler_1000j_summary.txt` (key tables) and
`results/profiling/scheduler_1000j_stats.txt` (full nsys output).

| Category | Time (ms) | % of wall time |
|---|---|---|
| GPU kernels | 3987.311 | 7.78 |
| Memcpy HtoD | 947.632 | 1.85 |
| Memcpy DtoH | 440.205 | 0.86 |
| Other (CPU / gaps) | 45865.095 | 89.51 |
| **Total** | **51240.242** | **100.00** |

The scheduler profile completed 1000/1000 jobs at 19.264 jobs/s, with p50 /
p95 / p99 e2e latency of 33.777 / 153.945 / 233.907 ms and 11.9% mean NVML
utilization. There were 1000 kernel launches (one per job; median 2.27 ms,
max 37.56 ms), 2000 HtoD copies and 1000 DtoH copies.

### Timeline findings

- **GEMM:** HtoD and DtoH overlap the kernels for **0.0 ms** — all three
  categories are serialized on one stream. The GPU is busy 98.04% of the
  wall; only 16.44 ms (1.96%) is idle, with no gap above 1 ms (max 0.759 ms).
- **Scheduler:** HtoD overlaps kernels for only **57.2 ms of 947.6 ms
  (6.0%)**; DtoH overlaps **46.1 ms of 440.2 ms (10.5%)**. GPU busy time is
  5255.7 ms (10.26%) and idle time is 45984.5 ms (89.74%). There are 936
  gaps above 1 ms and 801 above 10 ms; 45.11 s of the 45.98 s idle time is
  in gaps above 10 ms, with the largest gaps around 200–330 ms. Each compute
  stream is idle about 96% of its own span (gap sums 49.1 s and 49.3 s).
  Throughput was 19.264 jobs/s against an offered 20 jobs/s, so this
  particular run is **arrival-rate-limited, not GPU-bound**: the large idle
  gaps mostly reflect waiting for the next Poisson arrival.
- The scheduler’s top CUDA API times confirm substantial per-job host churn:
  `cudaHostAlloc` 8.52 s, `cudaFreeHost` 3.29 s, `cudaEventCreate` +
  `cudaEventDestroy` 1.58 s, `cudaMalloc` + `cudaFree` 1.56 s, and
  `cudaEventSynchronize` 5.70 s across 1000 jobs.

### Biggest bottleneck and proposed fix

For the **8x1024 GEMM**, the kernel is the largest single category (44.9%),
but the serialized HtoD + DtoH path is larger combined (53.2%) and overlaps
the kernel for **0%** of its time. For the **scheduler**, the GPU is idle
89.7% because the workload is arrival-limited; within the active pipeline,
HtoD overlaps only 6% of its time, and each job also pays pinned-buffer and
event churn.

**Proposed fix (not implemented):** use a double-buffered copy pipeline —
keep two pinned/device buffer sets per worker and issue `H2D_{i+1}` before
waiting on `K_i`, with `D2H_i` on a second copy stream. That lets H2D and D2H
overlap the current kernel instead of being serialized behind it. It directly
targets the measured 0% GEMM overlap and 6% scheduler overlap. At 20 jobs/s
it will not raise scheduler throughput because the workload is
arrival-limited, but it will reduce per-job latency and remove the transfer
path as the limiter when the queue is backlogged.

## What pinned memory + stream overlap change

Measured with `bench_overlap --jobs 20 --reps 5 --warmup 2 --impl tiled` on a
Tesla T4 (sm_75), using the same tiled kernel in all rows. Times are per job:
median of five timed sequences of 20 jobs after two warmup sequences.

| Size (B x M x N x K) | Configuration | H2D+D2H (ms) | Kernel (ms) | Total (ms) | Speedup |
|---|---|---|---|---|---|
| 8 x 512 x 512 x 512 | Pageable host, single stream, sync | 5.782 | 2.052 | 7.989 | 1.00x |
| 8 x 512 x 512 x 512 | Pinned + async, single stream | 2.037 | 2.050 | 4.098 | 1.95x |
| 8 x 512 x 512 x 512 | Pinned + copy/compute streams | 2.036 | 2.047 | 4.104 | 1.95x |
| 8 x 1024 x 1024 x 1024 | Pageable host, single stream, sync | 22.688 | 17.012 | 39.691 | 1.00x |
| 8 x 1024 x 1024 x 1024 | Pinned + async, single stream | 8.094 | 18.144 | 26.250 | 1.51x |
| 8 x 1024 x 1024 x 1024 | Pinned + copy/compute streams | 8.044 | 18.215 | 26.295 | 1.51x |

Measured conclusions:

- **Pageable → pinned** removes most of the transfer cost: H2D+D2H drops
  5.78 → 2.04 ms on the 512 case and 22.69 → 8.09 ms on the 1024 case.
  End-to-end improves 1.95x and 1.51x respectively.
- **The split copy/compute streams do not overlap transfers with compute in
  the current implementation.** `pinned_copy_compute` is within 0.2% of
  `pinned_single_stream` (slightly slower from event overhead). The issue is
  the dependency order in `BatchScheduler::execute_batch`: it enqueues
  `H2D_i`, waits for `K_i`, enqueues `D2H_i`, and only then enqueues
  `H2D_{i+1}` on the same copy stream. Since `D2H_i` waits for `K_i`, and
  `H2D_{i+1}` is behind `D2H_i`, the next H2D cannot run during `K_i`.
- **To get real overlap**, double-buffer the copy stream: enqueue
  `H2D_{i+1}` before waiting on `K_i`, or use two copy streams so a D2H and
  the next H2D can proceed independently. Until then, the scheduler's
  pinned single-stream and split-stream configurations have effectively the
  same critical path.

## Kernel-level analysis (ncu)

`ncu --set full` on the tiled kernel (32x32x32 tiles, 1024 threads/block):

| Counter | Value |
|---|---|
| Achieved occupancy | — % (of 64 warps/SM max) |
| Occupancy limiter | — (registers / shared memory / blocks) |
| DRAM throughput | — % of peak |
| Shared memory throughput | — % of peak |
| Top warp stall reason | — |

## Identified bottleneck (expected)

**Synchronous shared-memory tile loads with no pipelining.** Each k0
iteration loads its A/B tiles into shared memory behind a full
`__syncthreads` before any FMA can run, so the SM stalls on global-load
latency every tile. Expect ncu's long-scoreboard stall counter to dominate
and shared-memory throughput to be far below peak. Fill in the measured
evidence from the ncu table above.

## Next optimization

**cp.async double buffering** (`cuda::memcpy_async`/libcudacxx or inline
`cp.async.cg.shared.global`):

1. Split the k-loop into a prologue plus a main loop with two shared-memory
   tile buffers per operand.
2. Issue the tile *k0+1* copy before computing tile *k0*, synchronized by
   `cp.async.commit_group`/`wait_group` instead of a per-tile
   `__syncthreads`.
3. Swizzle the shared-memory layout to stay bank-conflict-free.

Goal: hide global-load latency behind FMAs and push the kernel toward its
DRAM-bandwidth bound. After that, closing the remaining gap to cuBLAS
requires 128x128 tiles with warp-level outer products (CUTLASS-style SGEMM)
or TF32 tensor cores.
