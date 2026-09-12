#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "sched/gpu_monitor.hpp"
#include "sched/job.hpp"

namespace gbis {

struct WorkerContext;

// Greedy packing: given the memory budget and a batch cap, select the largest
// set of jobs whose summed bytes_required() fits. Candidates are sorted by
// bytes ascending (priority descending, then FIFO as tie-breaks) so the
// greedy sweep maximizes the number of packed jobs. Returns true if anything
// was selected.
bool pack_jobs(const std::vector<InferenceJob>& candidates,
               std::size_t budget_bytes,
               int max_batch,
               std::vector<InferenceJob>& selected,
               std::vector<InferenceJob>& remaining);

// Deterministic synthetic inputs for a job (seeded by job id). Used by the
// worker threads and by tests that recompute the CPU reference.
void fill_matrices(const InferenceJob& job, float* a, float* b);

// Thread-safe batch scheduler: a priority queue (std::priority_queue +
// mutex + condition_variable) fed by submit(), a dispatcher that packs
// queued jobs against the free-GPU-memory budget (NVML free memory minus a
// safety margin), and a pool of worker threads, each with its own copy and
// compute cudaStream_t, executing packed batches with the tiled or cuBLAS
// batched GEMM kernels.
class BatchScheduler {
 public:
  struct Options {
    int max_batch = 8;                               // jobs per packed batch
    std::size_t safety_margin_bytes = 256ull << 20;  // 256 MiB
    int num_workers = 2;                             // threads (and streams)
    std::string impl = "tiled";                      // "tiled" | "cublas"
    bool retain_results = false;                     // keep C matrices
    std::size_t fixed_budget_bytes = 0;              // 0 => use GPU monitor
    int util_sample_ms = 100;                        // NVML sampling interval
  };

  explicit BatchScheduler(const Options& opts);
  ~BatchScheduler();

  BatchScheduler(const BatchScheduler&) = delete;
  BatchScheduler& operator=(const BatchScheduler&) = delete;

  void start();
  void stop(const std::string& metrics_csv = std::string());

  // Thread-safe. Returns the seq number identifying the submission.
  std::uint64_t submit(InferenceJob job);

  std::uint64_t submitted() const {
    return submitted_.load();
  }
  std::uint64_t completed() const {
    return completed_.load();
  }
  std::size_t queued() const;

  struct JobMetrics {
    std::uint64_t job_id = 0;
    std::uint64_t seq = 0;
    int priority = 0;
    double queue_wait_ms = 0.0;
    double exec_ms = 0.0;
    double e2e_ms = 0.0;
    bool ok = false;
  };

  std::vector<JobMetrics> metrics() const;
  bool get_result(std::uint64_t seq, std::vector<float>& out) const;
  void dump_metrics_csv(const std::string& path) const;
  void print_summary() const;

  double wall_seconds() const {
    return wall_seconds_;
  }
  double mean_gpu_utilization() const {
    return monitor_.mean_utilization();
  }

 private:
  void dispatcher_loop();
  void worker_loop();
  void execute_batch(WorkerContext& ctx, const std::vector<InferenceJob>& batch);
  void fail_batch(const std::vector<InferenceJob>& batch, const std::string& reason);
  std::size_t budget_bytes() const;

  Options opts_;
  GpuMonitor monitor_;

  std::priority_queue<InferenceJob, std::vector<InferenceJob>, JobPriorityCompare> queue_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::uint64_t next_seq_ = 0;

  std::deque<std::vector<InferenceJob>> batch_queue_;
  std::mutex batch_mutex_;
  std::condition_variable batch_cv_;

  std::thread dispatcher_;
  std::vector<std::thread> workers_;

  std::mutex lifecycle_mutex_;
  bool started_ = false;
  bool shutdown_ = false;
  bool gpu_absent_ = false;

  std::atomic<std::uint64_t> submitted_{0};
  std::atomic<std::uint64_t> completed_{0};

  mutable std::mutex metrics_mutex_;
  std::vector<JobMetrics> metrics_;

  mutable std::mutex results_mutex_;
  std::map<std::uint64_t, std::vector<float>> results_;

  InferenceJob::Clock::time_point start_time_{};
  double wall_seconds_ = 0.0;
};

}  // namespace gbis
