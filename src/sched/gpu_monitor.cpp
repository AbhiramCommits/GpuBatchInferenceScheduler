#include "sched/gpu_monitor.hpp"

#include <chrono>

#include <cuda_runtime.h>
#include <dlfcn.h>

namespace gbis {

namespace {

// Minimal ABI-compatible definitions of the NVML structs we use (kept out of
// the header; NVML is resolved at runtime via dlopen).
using NvmlDevice = struct NvmlDevice_st*;
struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};
struct NvmlUtilization {
  unsigned int gpu;
  unsigned int memory;
};

}  // namespace

struct GpuMonitor::Impl {
  void* handle = nullptr;

  int (*init)() = nullptr;
  int (*shutdown)() = nullptr;
  int (*device_get_handle_by_index)(unsigned int, NvmlDevice*) = nullptr;
  int (*device_get_memory_info)(NvmlDevice, NvmlMemory*) = nullptr;
  int (*device_get_utilization_rates)(NvmlDevice, NvmlUtilization*) = nullptr;

  NvmlDevice device = nullptr;

  ~Impl() {
    if (handle != nullptr) {
      if (shutdown != nullptr) shutdown();
      dlclose(handle);
    }
  }
};

GpuMonitor::GpuMonitor() : impl_(std::make_unique<Impl>()) {
  // NVML is a driver library: load it at runtime so the process still links
  // and runs on hosts without the NVIDIA driver (e.g. kind/minikube).
  impl_->handle = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (impl_->handle != nullptr) {
    impl_->init = reinterpret_cast<int (*)()>(dlsym(impl_->handle, "nvmlInit"));
    impl_->shutdown =
        reinterpret_cast<int (*)()>(dlsym(impl_->handle, "nvmlShutdown"));
    impl_->device_get_handle_by_index =
        reinterpret_cast<int (*)(unsigned int, NvmlDevice*)>(
            dlsym(impl_->handle, "nvmlDeviceGetHandleByIndex"));
    impl_->device_get_memory_info =
        reinterpret_cast<int (*)(NvmlDevice, NvmlMemory*)>(
            dlsym(impl_->handle, "nvmlDeviceGetMemoryInfo"));
    impl_->device_get_utilization_rates =
        reinterpret_cast<int (*)(NvmlDevice, NvmlUtilization*)>(
            dlsym(impl_->handle, "nvmlDeviceGetUtilizationRates"));
    if (impl_->init != nullptr && impl_->shutdown != nullptr &&
        impl_->device_get_handle_by_index != nullptr &&
        impl_->device_get_memory_info != nullptr &&
        impl_->device_get_utilization_rates != nullptr &&
        impl_->init() == 0 &&
        impl_->device_get_handle_by_index(0, &impl_->device) == 0) {
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

GpuMonitor::~GpuMonitor() { stop_sampling(); }

GpuStats GpuMonitor::sample() const {
  GpuStats stats;
  if (nvml_available_ && impl_->device != nullptr) {
    NvmlMemory mem{};
    if (impl_->device_get_memory_info(impl_->device, &mem) == 0) {
      stats.total_mem_bytes = static_cast<std::size_t>(mem.total);
      stats.free_mem_bytes = static_cast<std::size_t>(mem.free);
    }
    NvmlUtilization util{};
    if (impl_->device_get_utilization_rates(impl_->device, &util) == 0) {
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
