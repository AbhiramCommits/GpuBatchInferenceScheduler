# Design notes

## The packing algorithm

The dispatcher thread repeatedly drains the priority queue and runs
`pack_jobs` (see `src/sched/scheduler.cpp`):

1. **Sort** the candidate jobs by `bytes_required()` ascending, with
   priority descending and FIFO order (`seq` ascending) as tie-breaks.
2. **Greedy sweep**: walk the sorted list, adding each job while the
   running sum fits the budget and fewer than `--max-batch` jobs have been
   taken; everything else is re-enqueued for the next round.

Sorting by size first makes the greedy sweep take the smallest jobs first,
which is the classic approximation for the *maximum-cardinality subset* of
items under a capacity constraint (0/1 knapsack optimizing count, not
volume). It is not optimal in the worst case, but it is exact whenever all
remaining jobs are equal-sized, and it never exceeds the budget.

Complexity: one round is `O(n log n)` for the sort plus `O(n)` for the
sweep. The dispatcher re-packs only when the queue is non-empty, and sleeps
5 ms when nothing fit (waiting for in-flight batches to free memory), so
steady-state cost is negligible relative to kernel execution.

The budget itself is `free_mem_bytes − safety_margin_bytes` (saturating at
zero), sampled live from NVML each round, or a fixed value injected for
tests. The dispatcher also subtracts the bytes reserved by batches it has
packed but which have not finished yet. Without that reservation, it could
pack many batches against the same NVML snapshot before workers had allocated
their buffers; the reservation closes that window and prevents the scheduler
from overcommitting against itself.

NVML is only a snapshot: another process can take memory between the sample
and `cudaMalloc`. Workers therefore allocate with bounded retry/backoff on
`cudaErrorMemoryAllocation` before failing the batch. Jobs whose
`bytes_required()` exceeds `total_memory − safety_margin` can never fit and
are rejected once instead of being re-enqueued forever.

## The stream-overlap model

Each worker thread owns three CUDA streams:

- an **H2D stream** (non-blocking) for host-to-device `cudaMemcpyAsync` calls
  from pinned host buffers (`cudaHostAlloc`),
- a **compute stream** for the GEMM kernel launches, and
- a **D2H stream** (non-blocking) for device-to-host result copies.

Each job has its own pinned/device buffer set, so job *i+1* can be filled and
copied while job *i* is still using its buffers. Per job in a packed batch the
worker issues:

```text
h2d:     H2D(A_i), H2D(B_i) ── record(h2d_done_i)
compute: wait(h2d_done_i) ── K_i ── record(kern_done_i)
d2h:     wait(kern_done_i) ── D2H(C_i) ── record(done_i)
h2d:     H2D(A_{i+1}), H2D(B_{i+1}) ── record(h2d_done_{i+1})
```

The host enqueues `H2D_{i+1}` immediately after `D2H_i`, before it waits for
`K_i` to finish. Because H2D and D2H are separate streams, `D2H_i` does not
block `H2D_{i+1}`, and `H2D_{i+1}` can run during `K_i`. The previous
single-copy-stream version enqueued `H2D_{i+1}` behind `D2H_i`; since `D2H_i`
waited on `K_i`, the next transfer could not overlap the kernel, which is why
the measured single-stream and split-stream configurations were identical.

The host thread never blocks on a synchronous copy — pinned memory makes every
transfer DMA-capable and asynchronous. Host-side `fill_matrices` for job *i+1*
also runs while `K_i` is in flight, so its cost is hidden when the kernel is
long enough. Per-job completion is observed with one
`cudaEventSynchronize(done_i)` per job; the D2H stream serializes completions,
so this is O(1) each. Event objects are allocated once per worker (sized to
the largest batch seen) and reused, so the pipeline does not add per-job event
churn. The pipeline depth is bounded by the packed batch and by the host
fill/allocation rate, not by a fixed two-slot ring.

## The NVML safety margin

`GpuMonitor::sample()` reads `nvmlDeviceGetMemoryInfo` (and
`nvmlDeviceGetUtilizationRates` on a background sampler). The scheduler
subtracts a configurable margin (`--margin-mb`, default 256 MiB) before
packing, for three reasons:

1. **Fragmention/allocation granularity** — `cudaMalloc` rounds requests
   and can fail well before the reported free bytes are exhausted.
2. **Context and workspace overhead** — the CUDA context, cuBLAS
   workspaces, and the runtime's internal allocations are not counted in
   the job's own `bytes_required()`.
3. **Other tenants** — on shared nodes, other processes can grab memory
   between our sample and our allocation.

The margin is deliberately one conservative scalar rather than a model; see
"Limitations + Future Work" in the README for the planned feedback
controller. NVML is loaded via `dlopen` so the process still starts (and
falls back to `cudaMemGetInfo`) on hosts without the driver.

## Thread-safety argument for the queue

`std::priority_queue` has no internal locking, so all access goes through
`queue_mutex_`:

- **Submit** (`BatchScheduler::submit`): takes `queue_mutex_`, assigns
  `job.seq = next_seq_++` *inside* the lock (this is what makes equal
  priorities strictly FIFO — seq is a total order that agrees with
  arrival order), pushes, and notifies `queue_cv_`.
- **Dispatch**: the dispatcher holds `queue_mutex_` while draining into a
  local vector and again while re-enqueuing leftovers; it never executes
  kernels under the lock.
- **Shutdown**: `stop()` sets `shutdown_` under `lifecycle_mutex_` and
  broadcasts both condition variables; the dispatcher's wait predicate
  returns on `shutdown_` regardless of queue contents, so shutdown cannot
  deadlock with pending work.

Locking rules that keep it deadlock-free:

- no lock is held while waiting on a condition variable's `wait()` (it
  releases the mutex atomically with sleeping), and
- no lock is held across a kernel launch, a stream synchronization, or a
  join — the lock order is `lifecycle → queue → batch → metrics/results`,
  always acquired top-down.

Counters (`submitted_`, `completed_`) are `std::atomic`, the metrics vector
and result map each have their own mutex, and all cross-thread state is
either atomic, mutex-guarded, or read-only after `start()`.
