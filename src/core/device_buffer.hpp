#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "core/cuda_check.hpp"

namespace gbis {

// RAII wrapper around cudaMalloc/cudaFree. Copying is disabled; moving
// transfers ownership. No raw cudaMalloc/cudaFree calls leak: the destructor
// always frees the underlying allocation exactly once.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) {
    allocate(count);
  }

  ~DeviceBuffer() {
    release();
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  // Non-throwing allocation used by the scheduler's OOM retry path. Returns
  // the cudaMalloc status; on failure the buffer is left empty.
  cudaError_t try_allocate(std::size_t count) {
    release();
    if (count == 0)
      return cudaSuccess;
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
    if (err != cudaSuccess) {
      data_ = nullptr;
      count_ = 0;
      return err;
    }
    count_ = count;
    return cudaSuccess;
  }

  void allocate(std::size_t count) {
    CUDA_CHECK(try_allocate(count));
  }

  void release() noexcept {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
    }
    count_ = 0;
  }

  T* data() noexcept {
    return data_;
  }
  const T* data() const noexcept {
    return data_;
  }
  std::size_t size() const noexcept {
    return count_;
  }
  std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }
  explicit operator bool() const noexcept {
    return data_ != nullptr;
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace gbis
