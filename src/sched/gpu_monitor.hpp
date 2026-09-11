#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

#include <nvml.h>

namespace gbis {

// Snapshot of GPU telemetry.
struct GpuStats {
  std::size_t total_mem_bytes = 0;
  std::size_t free_mem_bytes = 0;
  unsigned utilization_pct = 0;  // 0..100
};

// GPU telemetry via NVML (nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates).
// Falls back to cudaMemGetInfo for memory when NVML is unavailable; utilization
// is then reported as 0. Optionally runs a background sampling thread that
// accumulates the mean utilization over the run.
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
  nvmlDevice_t device_ = nullptr;
  bool nvml_initialized_ = false;
  bool nvml_available_ = false;
  bool gpu_present_ = false;

  std::thread sampler_;
  std::atomic<bool> sampling_{false};

  mutable std::mutex stat_mutex_;
  double util_sum_ = 0.0;
  std::uint64_t util_count_ = 0;
};

}  // namespace gbis
