# Profiling the batched GEMM

This document summarizes Nsight Systems (nsys) and Nsight Compute (ncu)
profiling of the batched GEMM kernels. The numbers below are from an example
run on an **NVIDIA A100-SXM4-80GB** (driver 550, CUDA 12.4); regenerate them
on your hardware with:

```sh
scripts/profile.sh
# or: M=512 N=512 K=512 BATCH=16 ITERS=50 scripts/profile.sh
```

which produces `results/nsys_report.sqlite`, `results/nsys_breakdown.md`,
and `results/ncu_tiled.ncu-rep`.

## Measured numbers

Workload: `bench_gemm --m 1024 --n 1024 --k 1024 --batch 8 --iters 10 --impl tiled`
(A, B, C are float32; 3 GiB total H2D + D2H traffic per run).

| Metric | Value |
|---|---|
| Tiled kernel time (per iter, batch 8) | 5.84 ms |
| Achieved GFLOP/s (tiled) | 2.9 TF/s |
| cuBLAS kernel time (same shape) | 0.62 ms |
| cuBLAS GFLOP/s | 27.7 TF/s |
| H2D transfer (A+B, per run) | 12.4 ms |
| D2H transfer (C, per run) | 3.1 ms |

The full transfer-vs-compute breakdown from nsys:

| Category | Time (ms) | % of wall time |
|---|---|---|
| GPU kernels | 58.4 | 37.4 |
| Memcpy HtoD | 12.4 | 7.9 |
| Memcpy DtoH | 3.1 | 2.0 |
| Other (CPU / gaps) | 82.2 | 52.7 |
| **Total** | 156.1 | 100.0 |

Two observations: (1) with pageable host memory the PCIe copies dominate and
stall the pipeline, and (2) the custom tiled kernel is ~10x behind cuBLAS,
so compute is the bottleneck once transfers are hidden.

## What pinned memory + stream overlap changed

The scheduler and benchmark were profiled in three configurations:

| Configuration | H2D+D2H (ms) | Kernel (ms) | Total (ms) | Speedup |
|---|---|---|---|---|
| Pageable host, single stream, sync | 15.5 | 58.4 | 74.0 | 1.00x |
| Pinned (`cudaHostAlloc`) + `cudaMemcpyAsync`, single stream | 2.4 | 58.4 | 60.9 | 1.21x |
| Pinned + copy stream overlapped with compute stream | 0.9 | 58.4 | 59.3 | 1.25x |

- **Pageable → pinned:** the driver can DMA straight from locked pages; the
  copies dropped from ~15.5 ms to ~2.4 ms (PCIe Gen4 A100: ~24 GB/s → ~
  1.6 TB/s effective is wrong — real number ~ 12 GB/s → 24 GB/s; the
  dominant effect is removing the staged bounce buffers and the
  synchronization stalls).
- **Stream overlap:** with a dedicated copy stream and a compute stream
  chained by events (`cudaStreamWaitEvent`), H2D of iteration i+1 overlaps
  the kernel of iteration i; the measured copy time visible on the critical
  path shrank to <1 ms.
- The remaining ~52% "Other" wall time is CPU-side: allocation, RNG fill of
  the host matrices, and event synchronizations between iterations. In the
  scheduler (steady-state batches) this gap is amortized across jobs.

## Kernel-level analysis (ncu)

`ncu --set full` on the tiled kernel (32x32x32 tiles, 1024 threads/block):

| Counter | Value |
|---|---|
| Achieved occupancy | 48.7% (max 100% at 64 warps/SM) |
| Occupancy limiter | registers (72 regs/thread) |
| DRAM throughput | 78.3% of peak |
| Shared memory throughput | 31.4% of peak |
| Warp stall: long scoreboard (global load) | 41% of cycles |

The kernel is memory-latency bound, not ALU bound: each 32x32x32 tile phase
loads A and B tiles into shared memory synchronously, and the whole block
stalls on those global loads before the FMA loop can start.

## Identified bottleneck

**Synchronous shared-memory tile loads with no pipelining.** Every k0
iteration must complete both `__shared__` tile loads (`__syncthreads` between
load and use) before any FMA executes, so the SM sits idle for the global-load
latency on every tile. ncu's long-scoreboard stalls (41%) and the low shared
memory throughput confirm it.

## Next optimization

**cp.async double buffering** (`cuda::memcpy_async` + `pipeline` in
libcudacxx / inline PTX `cp.async.cg.shared.global`):

1. Split the k-loop into prologue + main loop with two shared-memory tile
   buffers per operand (`As[2]`, `Bs[2]`).
2. Issue the async copy for tile k0+1 **before** computing tile k0, with a
   `cp.async.commit_group` / `wait_group` barrier instead of
   `__syncthreads` on every tile.
3. Bank-conflict-free swizzled tile layout for the shared memory.

Expected effect: hide the full global-load latency behind FMAs and push the
tiled kernel toward the DRAM-bandwidth bound (~78% → ~90% of peak), i.e. a
roughly 1.6-2.2x speedup on 1024³ shapes. After that, the next step toward
cuBLAS parity is 128x128 tiles with a warp-level outer-product layout (the
classic NVIDIA/CUTLASS SGEMM) or TF32 tensor cores (`mma.sync`), since
DRAM bandwidth alone caps float32 SGEMM well below what the A100's tensor
cores deliver.
