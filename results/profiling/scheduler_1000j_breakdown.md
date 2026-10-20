# Nsight Systems breakdown (/content/GpuBatchInferenceScheduler/results/profiling/scheduler_1000j/nsys_report.sqlite)

| Category | Time (ms) | % of wall time |
|---|---|---|
| GPU kernels | 3987.311 | 7.78 |
| Memcpy HtoD | 947.632 | 1.85 |
| Memcpy DtoH | 440.205 | 0.86 |
| Other (CPU / gaps) | 45865.095 | 89.51 |
| **Total** | 51240.242 | 100.00 |

_Kernel time is the summed duration of all GPU kernels; wall time spans the first to the last recorded activity._
