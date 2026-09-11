#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

namespace gbis {

// Snapshot of GPU telemetry.
struct GpuStats {
  std::size_t total_mem_bytes = 0;
  std::size_t free_mem_bytes = 0;
  unsigned utilization_pct = 0;  // 0..100
};

// GPU telemetry via NVML (nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates).
// NVML is loaded at runtime with dlopen so the process still starts on hosts
// without the NVIDIA driver; the monitor then falls back to cudaMemGetInfo
// for memory (utilization reported as 0). Optionally runs a background
// sampling thread accumulating the mean utilization over the run.
class GpuMonitor {
 public:
  GpuMonitor();
  ~GpuMonitor();

  GpuMonitor(const GpuMonitor&) = delete;
  GpuMonitor& operator=(const GpuMonitor&) = delete;

  GpuStats sample() const;

  bool gpu_present() const { return gpu_present_; }
  bool nvml_available() const { return nvml_available_; }

  void start_sampling(int interval_ms = 100);
  void stop_sampling();
  double mean_utilization() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool nvml_available_ = false;
  bool gpu_present_ = false;

  std::thread sampler_;
  std::atomic<bool> sampling_{false};

  mutable std::mutex stat_mutex_;
  double util_sum_ = 0.0;
  std::uint64_t util_count_ = 0;
};

}  // namespace gbis
