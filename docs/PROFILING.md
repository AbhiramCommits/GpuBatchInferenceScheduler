# Profiling the batched GEMM

This document summarizes Nsight Systems (nsys) and Nsight Compute (ncu)
profiling of the batched GEMM kernels, and records the transfer-vs-compute
analysis workflow.

> [!NOTE]
> **Partial GPU measurements.** The nsys transfer/breakdown tables, the
> overlap/benchmark table, and the saturation table were measured on a Tesla
> T4 (sm_75, CUDA 13.0); the ncu kernel-level table still holds placeholders
> and should be filled on the target host.

## How to profile

```sh
# GEMM (default 1024^3, batch 8):
scripts/profile.sh

# Scheduler (20 jobs/s):
WORKLOAD=scheduler JOBS=1000 ARRIVAL_RATE=20 WORKERS=2 scripts/profile.sh

# Scheduler at saturation (open-loop backlog; ARRIVAL_RATE=0 submits all jobs):
WORKLOAD=scheduler JOBS=1000 ARRIVAL_RATE=0 WORKERS=2 scripts/profile.sh

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
- **Scheduler (pre-pipeline measurement):** HtoD overlaps kernels for only
  **57.2 ms of 947.6 ms (6.0%)**; DtoH overlaps **46.1 ms of 440.2 ms
  (10.5%)**. GPU busy time is 5255.7 ms (10.26%) and idle time is 45984.5 ms
  (89.74%). There are 936 gaps above 1 ms and 801 above 10 ms; 45.11 s of the
  45.98 s idle time is in gaps above 10 ms, with the largest gaps around
  200–330 ms. Each compute stream is idle about 96% of its own span (gap sums
  49.1 s and 49.3 s). Throughput was 19.264 jobs/s against an offered 20
  jobs/s, so this particular run is **arrival-rate-limited, not GPU-bound**:
  the large idle gaps mostly reflect waiting for the next Poisson arrival.

  These overlap fractions describe the old single-copy-stream order
  (`H2D_i -> K_i -> D2H_i -> H2D_{i+1}`). They are the "before" measurement;
  `BatchScheduler::execute_batch` now uses the pipelined three-stream order
  described below. Re-run the same nsys workload to quantify the "after"
  overlap; the old number should not be quoted as the current design.
- The scheduler’s top CUDA API times in that run confirm substantial per-job
  host churn: `cudaHostAlloc` 8.52 s, `cudaFreeHost` 3.29 s, `cudaEventCreate` +
  `cudaEventDestroy` 1.58 s, `cudaMalloc` + `cudaFree` 1.56 s, and
  `cudaEventSynchronize` 5.70 s across 1000 jobs.

### Biggest bottleneck and implemented fix

For the **8x1024 GEMM**, the kernel is the largest single category (44.9%),
but the serialized HtoD + DtoH path is larger combined (53.2%) and overlaps
the kernel for **0%** of its time. For the **scheduler**, the GPU is idle
89.7% because the workload is arrival-limited; within the old active
pipeline, HtoD overlapped only 6% of its time, and each job also paid
pinned-buffer and event churn.

**Implemented fix:** `BatchScheduler::execute_batch` now uses three streams
per worker (H2D, compute, D2H) and per-job events. It fills job *i+1* and
enqueues `H2D_{i+1}` while `K_i` is in flight, instead of leaving it behind
`D2H_i` on a shared copy stream. Each job has its own pinned/device buffers,
so the pipeline has multiple in-flight buffer sets. `bench_overlap` exposes
the new order as `pinned_double_buffer`; `pinned_copy_compute` remains as the
old serialized-order control.

This removes the copy/compute dependency and pools the per-worker events, but
it does not remove pinned/device buffer allocation churn: the profile above
attributes 8.52 s to `cudaHostAlloc`, 3.29 s to `cudaFreeHost`, and 1.56 s to
`cudaMalloc`/`cudaFree` across 1000 jobs (event create/destroy was 1.58 s, now
amortized by the pool). When the queue is backlogged, the allocation costs can
still cap throughput, so the saturation run below is the measurement that
separates the transfer fix from the remaining host overhead.

### Saturation throughput

The README's 20 jobs/s run is **not** a throughput ceiling: it completed
19.36 jobs/s against 20 offered, with 11.7% mean GPU utilization. The
saturation sweep reproduced that point at 19.33 jobs/s. To measure the real
ceiling, `scripts/saturation.sh` replays the same synthetic workload (seed
1234, same sizes) at 20/100/500 jobs/s and then with `--saturation`, which
submits all 1000 jobs immediately and never lets the backlog drain:

```sh
scripts/saturation.sh
# artifacts: results/saturation/summary.csv, per-run *.log, *_jobs.csv
```

Measured on the T4 via Colab with 1000 jobs, 2 workers, and
`max_batch=8`:

| Offered load | Throughput | p50 e2e | p95 e2e | p99 e2e | Mean GPU util |
|---|---:|---:|---:|---:|---:|
| 20 jobs/s | 19.33 jobs/s | 29.8 ms | 120.5 ms | 181.3 ms | 11.5% |
| 100 jobs/s | 45.55 jobs/s | 4.85 s | 10.10 s | 11.57 s | 24.8% |
| 500 jobs/s | 46.18 jobs/s | 7.92 s | 17.16 s | 19.34 s | 24.2% |
| **saturated backlog** | **45.35 jobs/s** | 4.32 s | 18.36 s | 21.38 s | 24.0% |

The 100/s and 500/s runs are already backlogged (throughput ~45–46 jobs/s,
latency in seconds), so they independently confirm the ceiling. The
GPU-busy-time bound is ~190 jobs/s, but the measured ceiling is 45.35 jobs/s
at 24% GPU utilization. That means the GPU is **not** the limiter: host-side
per-batch work is. The original profile's 8.52 s of `cudaHostAlloc`, 3.29 s of
`cudaFreeHost`, and 1.56 s of `cudaMalloc`/`cudaFree` over 1000 jobs line up
with that conclusion. The next optimization is pooling the pinned/device
buffers per worker; more streams will not raise this ceiling.

## What pinned memory + stream overlap change

Measured with `bench_overlap --jobs 20 --reps 5 --warmup 2 --impl tiled` on a
Tesla T4 (sm_75), using the same tiled kernel in all rows. Times are per job:
median of five timed sequences of 20 jobs after two warmup sequences.

| Size (B x M x N x K) | Configuration | H2D+D2H (ms) | Kernel (ms) | Total (ms) | Speedup |
|---|---|---|---|---|---|
| 8 x 512 x 512 x 512 | Pageable host, single stream, sync | 5.703 | 2.042 | 7.894 | 1.00x |
| 8 x 512 x 512 x 512 | Pinned + async, single stream | 2.017 | 2.042 | 4.067 | 1.94x |
| 8 x 512 x 512 x 512 | Pinned + copy/compute streams | 2.020 | 2.036 | 4.077 | 1.94x |
| 8 x 512 x 512 x 512 | **Pinned + double-buffered pipeline** | 2.216 | 2.242 | **2.349** | **3.36x** |
| 8 x 1024 x 1024 x 1024 | Pageable host, single stream, sync | 22.268 | 16.668 | 39.102 | 1.00x |
| 8 x 1024 x 1024 x 1024 | Pinned + async, single stream | 8.004 | 17.371 | 25.381 | 1.54x |
| 8 x 1024 x 1024 x 1024 | Pinned + copy/compute streams | 8.008 | 17.420 | 25.449 | 1.54x |
| 8 x 1024 x 1024 x 1024 | **Pinned + double-buffered pipeline** | 8.349 | 19.219 | **19.628** | **1.99x** |

Measured on the T4 via Colab with the same command as before. The
new `pinned_double_buffer` row uses the scheduler's three-stream order:
`H2D_{i+1}` is issued while `K_i` runs, and D2H drains on its own stream.
Transfer time can be slightly higher in isolation because H2D and D2H now
overlap compute and each other; total wall time is what matters.

Measured conclusions:

- **Pageable → pinned** still removes most of the transfer cost: H2D+D2H
  drops 5.70 → 2.02 ms on the 512 case and 22.27 → 8.00 ms on the 1024 case.
- **The old split copy/compute order did not overlap transfers with compute.**
  `pinned_copy_compute` is within 0.5% of `pinned_single_stream`. It enqueued
  `H2D_i`, waited for `K_i`, enqueued `D2H_i`, and only then enqueued
  `H2D_{i+1}` on the same copy stream, so the next H2D sat behind the current
  kernel. That is the **control case**, not the current scheduler.
- **Double buffering fixes the critical path.** `pinned_double_buffer`
  improves total time from 4.067 → 2.349 ms (**3.36x** vs pageable) on the
  512 case and 25.381 → 19.628 ms (**1.99x**) on the 1024 case. The old
  serialized split-stream configuration was 1.94x and 1.54x respectively.
  A second identical run reproduced 3.36x and 2.01x. The total is now only
  0.11 ms above kernel time on the 512 case and 0.41 ms above on the 1024
  case, so roughly 95% of the transfer time is hidden behind compute. This is
  the measured evidence that the problem was ordering, not the number of
  streams.

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
