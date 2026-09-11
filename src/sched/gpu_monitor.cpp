#include "sched/gpu_monitor.hpp"

#include <chrono>

#include <cuda_runtime.h>

namespace gbis {

GpuMonitor::GpuMonitor() {
  if (nvmlInit() == NVML_SUCCESS) {
    nvml_initialized_ = true;
    nvmlDevice_t dev = nullptr;
    if (nvmlDeviceGetHandleByIndex(0, &dev) == NVML_SUCCESS) {
      device_ = dev;
      nvml_available_ = true;
      gpu_present_ = true;
    }
  }
  if (!gpu_present_) {
    // NVML unavailable or no usable device: fall back to the CUDA runtime for
    // memory information (utilization will be reported as 0).
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
      gpu_present_ = true;
    }
  }
}

GpuMonitor::~GpuMonitor() {
  stop_sampling();
  if (nvml_initialized_) nvmlShutdown();
}

GpuStats GpuMonitor::sample() const {
  GpuStats stats;
  if (nvml_available_ && device_ != nullptr) {
    nvmlMemory_t mem{};
    if (nvmlDeviceGetMemoryInfo(device_, &mem) == NVML_SUCCESS) {
      stats.total_mem_bytes = static_cast<std::size_t>(mem.total);
      stats.free_mem_bytes = static_cast<std::size_t>(mem.free);
    }
    nvmlUtilization_t util{};
    if (nvmlDeviceGetUtilizationRates(device_, &util) == NVML_SUCCESS) {
      stats.utilization_pct = static_cast<unsigned>(util.gpu);
    }
    return stats;
  }
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    stats.total_mem_bytes = total_bytes;
    stats.free_mem_bytes = free_bytes;
  }
  return stats;
}

void GpuMonitor::start_sampling(int interval_ms) {
  if (!gpu_present_ || sampling_.exchange(true)) return;
  sampler_ = std::thread([this, interval_ms] {
    while (sampling_.load()) {
      const GpuStats st = sample();
      {
        std::lock_guard<std::mutex> lk(stat_mutex_);
        util_sum_ += static_cast<double>(st.utilization_pct);
        ++util_count_;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
  });
}

void GpuMonitor::stop_sampling() {
  sampling_.store(false);
  if (sampler_.joinable()) sampler_.join();
}

double GpuMonitor::mean_utilization() const {
  std::lock_guard<std::mutex> lk(stat_mutex_);
  return util_count_ > 0 ? util_sum_ / static_cast<double>(util_count_) : 0.0;
}

}  // namespace gbis
