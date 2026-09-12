#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace gbis {

// A single batched-GEMM inference job. Jobs are queued into the scheduler and
// executed as C[b] = A[b] * B[b] with A:(m,k), B:(k,n), row-major, float32.
struct InferenceJob {
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  std::uint64_t id = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  int batch = 1;
  int priority = 0;  // higher runs first
  TimePoint submit_time{};
  std::uint64_t seq = 0;  // assigned by the scheduler; FIFO tie-break

  // Device bytes needed for A, B and C across the batch.
  std::size_t bytes_required() const {
    const std::size_t a = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
    const std::size_t b = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
    const std::size_t c = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    return (a + b + c) * static_cast<std::size_t>(batch) * sizeof(float);
  }
};

// std::priority_queue comparator: highest priority first, FIFO (lower seq)
// within equal priority.
struct JobPriorityCompare {
  bool operator()(const InferenceJob& a, const InferenceJob& b) const {
    if (a.priority != b.priority)
      return a.priority < b.priority;
    return a.seq > b.seq;
  }
};

}  // namespace gbis
