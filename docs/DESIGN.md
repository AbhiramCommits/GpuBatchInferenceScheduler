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
tests. Because only the dispatcher packs — never the workers — two batches
can never be simultaneously packed against the same free-memory snapshot,
which is what guarantees the sum of all packed jobs stays within one
budget's worth of headroom.

## The stream-overlap model

Each worker thread owns exactly two CUDA streams and one pair of events:

- a **copy stream** (non-blocking) for H2D and D2H `cudaMemcpyAsync` calls
  from pinned host buffers (`cudaHostAlloc`), and
- a **compute stream** for the GEMM kernel launches.

Per job in a packed batch the worker issues:

```text
copy:    H2D(A), H2D(B) ── record(ev_h2d)
compute: wait(ev_h2d) ── kernel ── record(ev_kern)
copy:    wait(ev_kern) ── D2H(C) ── record(ev_done[i])
```

Because the streams are different, the H2D of job *i+1* (copy stream) runs
concurrently with the kernel of job *i* (compute stream); the events form
the producer/consumer edges. The host thread never blocks on a synchronous
copy — pinned memory makes every transfer DMA-capable and asynchronous.
Per-job completion is observed with one `cudaEventSynchronize(ev_done[i])`
per job (events complete in order, so this is O(1) each).

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
