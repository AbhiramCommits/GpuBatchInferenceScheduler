#include "sched/scheduler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include "core/cuda_check.hpp"
#include "core/device_buffer.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemm_cublas.hpp"

namespace gbis {

namespace {

class CudaEvent {
 public:
  CudaEvent() {
    CUDA_CHECK(cudaEventCreate(&event_));
  }
  ~CudaEvent() {
    if (event_ != nullptr)
      cudaEventDestroy(event_);
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }
  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (event_ != nullptr)
        cudaEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }
  cudaEvent_t get() const {
    return event_;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

template <typename T>
class PinnedHostBuffer {
 public:
  PinnedHostBuffer() = default;
  explicit PinnedHostBuffer(std::size_t count) {
    allocate(count);
  }
  ~PinnedHostBuffer() {
    release();
  }
  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }
  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(std::size_t count) {
    release();
    if (count == 0)
      return;
    CUDA_CHECK(
        cudaHostAlloc(reinterpret_cast<void**>(&data_), count * sizeof(T), cudaHostAllocDefault));
    count_ = count;
  }

  void release() noexcept {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
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

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

double milliseconds(InferenceJob::TimePoint from, InferenceJob::TimePoint to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

// Multi-level wait-time aging: group jobs into 1-second wait buckets and pack
// older buckets first. Inside a bucket the existing smallest-first order is
// preserved, so jobs of similar age still pack for maximum count. Once a job
// has waited one interval it outranks every freshly arrived job, bounding
// starvation at roughly one interval plus one packing round. A stepwise
// bucket (rather than a continuous priority boost) avoids reordering peers of
// the same age and keeps the packing heuristic stable.
constexpr std::int64_t kAgingIntervalMs = 1000;

}  // namespace

struct WorkerContext {
  // Transfers are split so the next job's H2D can run while the current
  // kernel executes and the previous job's D2H drains on its own stream.
  cudaStream_t h2d_stream = nullptr;
  cudaStream_t compute_stream = nullptr;
  cudaStream_t d2h_stream = nullptr;

  // Event pool sized to the largest batch this worker has seen. Reusing
  // events across batches avoids per-job cudaEventCreate/Destroy churn.
  std::vector<CudaEvent> h2d_done;
  std::vector<CudaEvent> kern_done;
  std::vector<CudaEvent> done;
};

bool pack_jobs(const std::vector<InferenceJob>& candidates,
               std::size_t budget_bytes,
               int max_batch,
               std::vector<InferenceJob>& selected,
               std::vector<InferenceJob>& remaining) {
  selected.clear();
  remaining.clear();

  const auto now = InferenceJob::Clock::now();
  const auto age_bucket = [now](const InferenceJob& job) -> std::int64_t {
    if (job.submit_time == InferenceJob::TimePoint{})
      return 0;  // synthetic/test job without a submission timestamp
    const auto wait_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - job.submit_time).count();
    return wait_ms > 0 ? wait_ms / kAgingIntervalMs : 0;
  };

  std::vector<InferenceJob> sorted = candidates;
  std::stable_sort(
      sorted.begin(), sorted.end(), [&age_bucket](const InferenceJob& a, const InferenceJob& b) {
        const std::int64_t bucket_a = age_bucket(a);
        const std::int64_t bucket_b = age_bucket(b);
        if (bucket_a != bucket_b)
          return bucket_a > bucket_b;
        const std::size_t bytes_a = a.bytes_required();
        const std::size_t bytes_b = b.bytes_required();
        if (bytes_a != bytes_b)
          return bytes_a < bytes_b;
        if (a.priority != b.priority)
          return a.priority > b.priority;
        return a.seq < b.seq;
      });

  std::size_t used = 0;
  for (const auto& job : sorted) {
    const std::size_t need = job.bytes_required();
    if (max_batch > 0 && static_cast<int>(selected.size()) >= max_batch) {
      remaining.push_back(job);
      continue;
    }
    if (need > budget_bytes || used + need > budget_bytes) {
      remaining.push_back(job);
      continue;
    }
    selected.push_back(job);
    used += need;
  }
  return !selected.empty();
}

void fill_matrices(const InferenceJob& job, float* a, float* b) {
  const std::uint64_t seed = (job.id ^ (job.id >> 32)) * 0x9e3779b97f4a7c15ull;
  std::mt19937 rng(static_cast<std::uint32_t>(seed & 0xffffffffu));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const std::size_t sa = static_cast<std::size_t>(job.m) * job.k * job.batch;
  const std::size_t sb = static_cast<std::size_t>(job.k) * job.n * job.batch;
  for (std::size_t i = 0; i < sa; ++i)
    a[i] = dist(rng);
  for (std::size_t i = 0; i < sb; ++i)
    b[i] = dist(rng);
}

BatchScheduler::BatchScheduler(const Options& opts) : opts_(opts) {}

BatchScheduler::~BatchScheduler() {
  stop();
}

void BatchScheduler::start() {
  {
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    if (started_)
      return;
    started_ = true;
    shutdown_ = false;
  }
  start_time_ = InferenceJob::Clock::now();

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess)
    ndev = 0;
  gpu_absent_ = (ndev == 0);

  const GpuStats stats = monitor_.sample();
  total_mem_bytes_ = stats.total_mem_bytes;
  if (total_mem_bytes_ == 0) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess)
      total_mem_bytes_ = total_bytes;
  }

  monitor_.start_sampling(opts_.util_sample_ms);

  dispatcher_ = std::thread(&BatchScheduler::dispatcher_loop, this);
  workers_.reserve(opts_.num_workers);
  for (int i = 0; i < opts_.num_workers; ++i) {
    workers_.emplace_back(&BatchScheduler::worker_loop, this);
  }
}

void BatchScheduler::stop(const std::string& metrics_csv) {
  {
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    if (!started_)
      return;
    shutdown_ = true;
  }
  queue_cv_.notify_all();
  batch_cv_.notify_all();
  batch_space_cv_.notify_all();
  if (dispatcher_.joinable())
    dispatcher_.join();
  for (auto& w : workers_) {
    if (w.joinable())
      w.join();
  }
  monitor_.stop_sampling();
  wall_seconds_ = std::chrono::duration<double>(InferenceJob::Clock::now() - start_time_).count();
  {
    std::lock_guard<std::mutex> lk(lifecycle_mutex_);
    started_ = false;
  }
  if (!metrics_csv.empty())
    dump_metrics_csv(metrics_csv);
  print_summary();
}

std::uint64_t BatchScheduler::submit(InferenceJob job) {
  job.submit_time = InferenceJob::Clock::now();
  {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    job.seq = next_seq_++;
    queue_.push(job);
  }
  submitted_.fetch_add(1, std::memory_order_relaxed);
  queue_cv_.notify_one();
  return job.seq;
}

std::size_t BatchScheduler::queued() const {
  std::lock_guard<std::mutex> lk(queue_mutex_);
  return queue_.size();
}

std::vector<BatchScheduler::JobMetrics> BatchScheduler::metrics() const {
  std::lock_guard<std::mutex> lk(metrics_mutex_);
  return metrics_;
}

bool BatchScheduler::get_result(std::uint64_t seq, std::vector<float>& out) const {
  std::lock_guard<std::mutex> lk(results_mutex_);
  const auto it = results_.find(seq);
  if (it == results_.end())
    return false;
  out = it->second;
  return true;
}

void BatchScheduler::dump_metrics_csv(const std::string& path) const {
  std::ofstream f(path);
  if (!f) {
    std::cerr << "[scheduler] failed to open metrics CSV: " << path << "\n";
    return;
  }
  f << "job_id,seq,priority,queue_wait_ms,exec_ms,e2e_ms,ok\n";
  std::lock_guard<std::mutex> lk(metrics_mutex_);
  for (const auto& m : metrics_) {
    f << m.job_id << ',' << m.seq << ',' << m.priority << ',' << m.queue_wait_ms << ',' << m.exec_ms
      << ',' << m.e2e_ms << ',' << (m.ok ? 1 : 0) << '\n';
  }
}

void BatchScheduler::print_summary() const {
  std::vector<double> e2e;
  std::uint64_t failed = 0;
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    for (const auto& m : metrics_) {
      if (m.ok) {
        e2e.push_back(m.e2e_ms);
      } else {
        ++failed;
      }
    }
  }
  std::sort(e2e.begin(), e2e.end());
  const auto percentile = [&e2e](double p) -> double {
    if (e2e.empty())
      return 0.0;
    const std::size_t rank = static_cast<std::size_t>(std::ceil(p / 100.0 * e2e.size()));
    return e2e[std::max<std::size_t>(1, rank) - 1];
  };
  const double throughput =
      wall_seconds_ > 0.0 ? static_cast<double>(completed_.load()) / wall_seconds_ : 0.0;
  std::printf("[scheduler] submitted=%llu completed=%llu failed=%llu wall=%.3fs\n",
              static_cast<unsigned long long>(submitted_.load()),
              static_cast<unsigned long long>(completed_.load()),
              static_cast<unsigned long long>(failed),
              wall_seconds_);
  std::printf(
      "[scheduler] rejected_oversized=%llu cudaMalloc_retries=%llu "
      "cudaMalloc_failures=%llu\n",
      static_cast<unsigned long long>(rejected_jobs_.load()),
      static_cast<unsigned long long>(alloc_retries_.load()),
      static_cast<unsigned long long>(oom_failures_.load()));
  std::printf("[scheduler] throughput=%.3f jobs/s\n", throughput);
  std::printf("[scheduler] e2e latency ms: p50=%.3f p95=%.3f p99=%.3f\n",
              percentile(50.0),
              percentile(95.0),
              percentile(99.0));
  std::printf("[scheduler] mean gpu utilization: %.1f%%\n", monitor_.mean_utilization());
}

std::size_t BatchScheduler::max_job_bytes() const {
  if (total_mem_bytes_ == 0)
    return 0;
  return total_mem_bytes_ > opts_.safety_margin_bytes ? total_mem_bytes_ - opts_.safety_margin_bytes
                                                      : 0;
}

std::size_t BatchScheduler::reserved_bytes() const {
  std::lock_guard<std::mutex> lk(budget_mutex_);
  return reserved_bytes_;
}

void BatchScheduler::reserve_bytes(std::size_t bytes) {
  std::lock_guard<std::mutex> lk(budget_mutex_);
  reserved_bytes_ += bytes;
}

void BatchScheduler::release_bytes(std::size_t bytes) {
  std::lock_guard<std::mutex> lk(budget_mutex_);
  reserved_bytes_ = bytes > reserved_bytes_ ? 0 : reserved_bytes_ - bytes;
}

std::size_t BatchScheduler::budget_bytes() const {
  std::size_t base = 0;
  if (opts_.fixed_budget_bytes > 0) {
    base = opts_.fixed_budget_bytes;
  } else {
    const GpuStats stats = monitor_.sample();
    if (stats.free_mem_bytes <= opts_.safety_margin_bytes)
      return 0;
    base = stats.free_mem_bytes - opts_.safety_margin_bytes;
  }
  const std::size_t reserved = reserved_bytes();
  return base > reserved ? base - reserved : 0;
}

void BatchScheduler::dispatcher_loop() {
  // Keep only a small batch lookahead so jobs waiting in the priority queue
  // get re-evaluated (and can age) instead of being committed to a batch at
  // the back of a long batch_queue_. Two batches per worker is enough to keep
  // workers fed without turning the queue into an unbounded pre-pack buffer.
  const std::size_t max_queue_depth =
      std::max<std::size_t>(2, static_cast<std::size_t>(opts_.num_workers) * 2);

  while (true) {
    {
      std::unique_lock<std::mutex> lk(batch_mutex_);
      batch_space_cv_.wait(lk, [this, max_queue_depth] {
        return shutdown_ || batch_queue_.size() < max_queue_depth;
      });
      if (shutdown_)
        break;
    }

    std::vector<InferenceJob> candidates;
    {
      std::unique_lock<std::mutex> lk(queue_mutex_);
      queue_cv_.wait(lk, [this] { return shutdown_ || !queue_.empty(); });
      if (shutdown_)
        break;  // drop queued work; in-flight batches complete
      while (!queue_.empty()) {
        candidates.push_back(queue_.top());
        queue_.pop();
      }
    }

    if (gpu_absent_) {
      // No GPU: jobs can never execute; fail them so the daemon keeps
      // draining (e.g. --force-cpu on a GPU-less host or kind/minikube).
      fail_batch(candidates, "no gpu");
      continue;
    }

    // A job larger than the largest budget this device can ever offer will
    // never fit. Reject it once instead of re-enqueuing it forever.
    const std::size_t max_job = max_job_bytes();
    std::vector<InferenceJob> oversized;
    if (max_job > 0) {
      auto it = candidates.begin();
      while (it != candidates.end()) {
        if (it->bytes_required() > max_job) {
          oversized.push_back(*it);
          it = candidates.erase(it);
        } else {
          ++it;
        }
      }
    }
    if (!oversized.empty()) {
      rejected_jobs_.fetch_add(oversized.size(), std::memory_order_relaxed);
      for (const auto& job : oversized) {
        std::cerr << "[scheduler] rejecting oversized job id=" << job.id
                  << " bytes=" << job.bytes_required() << " max_job_bytes=" << max_job << "\n";
      }
      fail_batch(oversized, "job exceeds device memory");
    }

    std::vector<InferenceJob> selected;
    std::vector<InferenceJob> remaining;
    pack_jobs(candidates, budget_bytes(), opts_.max_batch, selected, remaining);

    {
      std::lock_guard<std::mutex> lk(queue_mutex_);
      for (const auto& job : remaining)
        queue_.push(job);
    }

    if (!selected.empty()) {
      std::size_t batch_bytes = 0;
      for (const auto& job : selected)
        batch_bytes += job.bytes_required();
      reserve_bytes(batch_bytes);
      {
        std::lock_guard<std::mutex> lk(batch_mutex_);
        batch_queue_.push_back(std::move(selected));
      }
      batch_cv_.notify_one();
    } else {
      // Nothing fit the budget: give in-flight batches time to free memory.
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
}

void BatchScheduler::worker_loop() {
  std::unique_ptr<WorkerContext> ctx;
  std::atomic<bool> warned{false};
  while (true) {
    std::vector<InferenceJob> batch;
    {
      std::unique_lock<std::mutex> lk(batch_mutex_);
      batch_cv_.wait(lk, [this] { return shutdown_ || !batch_queue_.empty(); });
      if (shutdown_ && batch_queue_.empty())
        break;
      batch = std::move(batch_queue_.front());
      batch_queue_.pop_front();
    }
    batch_space_cv_.notify_one();
    std::size_t batch_bytes = 0;
    for (const auto& job : batch)
      batch_bytes += job.bytes_required();

    if (gpu_absent_) {
      if (!warned.exchange(true)) {
        std::cerr << "[scheduler] no CUDA GPU available; dropping batches\n";
      }
      fail_batch(batch, "no gpu");
      release_bytes(batch_bytes);
      continue;
    }
    if (!ctx)
      ctx = std::make_unique<WorkerContext>();
    try {
      execute_batch(*ctx, batch);
    } catch (...) {
      release_bytes(batch_bytes);
      throw;
    }
    release_bytes(batch_bytes);
  }
  if (ctx) {
    if (ctx->h2d_stream != nullptr)
      cudaStreamDestroy(ctx->h2d_stream);
    if (ctx->compute_stream != nullptr)
      cudaStreamDestroy(ctx->compute_stream);
    if (ctx->d2h_stream != nullptr)
      cudaStreamDestroy(ctx->d2h_stream);
  }
}

void BatchScheduler::execute_batch(WorkerContext& ctx, const std::vector<InferenceJob>& batch) {
  struct JobBuffers {
    DeviceBuffer<float> da;
    DeviceBuffer<float> db;
    DeviceBuffer<float> dc;
    PinnedHostBuffer<float> ha;
    PinnedHostBuffer<float> hb;
    PinnedHostBuffer<float> hc;
    InferenceJob::TimePoint pickup{};
  };

  if (ctx.h2d_stream == nullptr) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.h2d_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreate(&ctx.compute_stream));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.d2h_stream, cudaStreamNonBlocking));
  }

  // Events are distinct within a batch (the host records and awaits them out
  // of order) but recycled once the previous batch has completed.
  if (ctx.h2d_done.size() < batch.size()) {
    ctx.h2d_done.resize(batch.size());
    ctx.kern_done.resize(batch.size());
    ctx.done.resize(batch.size());
  }

  std::vector<JobBuffers> execs(batch.size());

  try {
    // Allocate every job's buffers before launching anything. If an external
    // tenant grabs memory between the NVML sample and cudaMalloc, retry with
    // backoff instead of failing the jobs; the reservation accounting keeps
    // our own batches from overcommitting.
    constexpr int kMaxAllocRetries = 200;
    cudaError_t alloc_err = cudaSuccess;
    bool allocated = false;
    for (int attempt = 0; attempt <= kMaxAllocRetries; ++attempt) {
      alloc_err = cudaSuccess;
      for (std::size_t i = 0; i < batch.size(); ++i) {
        const auto& job = batch[i];
        auto& e = execs[i];
        const std::size_t sa = static_cast<std::size_t>(job.m) * job.k * job.batch;
        const std::size_t sb = static_cast<std::size_t>(job.k) * job.n * job.batch;
        const std::size_t sc = static_cast<std::size_t>(job.m) * job.n * job.batch;
        alloc_err = e.da.try_allocate(sa);
        if (alloc_err != cudaSuccess)
          break;
        alloc_err = e.db.try_allocate(sb);
        if (alloc_err != cudaSuccess)
          break;
        alloc_err = e.dc.try_allocate(sc);
        if (alloc_err != cudaSuccess)
          break;
        e.ha.allocate(sa);
        e.hb.allocate(sb);
        e.hc.allocate(sc);
      }
      if (alloc_err == cudaSuccess) {
        allocated = true;
        break;
      }
      for (auto& e : execs) {
        e.da.release();
        e.db.release();
        e.dc.release();
        e.ha.release();
        e.hb.release();
        e.hc.release();
      }
      cudaGetLastError();
      if (alloc_err != cudaErrorMemoryAllocation || attempt == kMaxAllocRetries)
        break;
      alloc_retries_.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "[scheduler] cudaMalloc out of memory; retry " << (attempt + 1) << "/"
                << kMaxAllocRetries << "\n";
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if (!allocated) {
      if (alloc_err == cudaErrorMemoryAllocation)
        oom_failures_.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error(std::string("buffer allocation failed: ") +
                               cudaGetErrorString(alloc_err));
    }

    // Double-buffered pipeline: H2D for job i+1 is issued on its own stream
    // while job i is running, so it overlaps K_i instead of waiting behind
    // D2H_i. D2H has a third stream, so it does not block the next H2D
    // either. Each job's buffers are distinct, giving the pipeline multiple
    // in-flight buffer sets.
    const auto enqueue_h2d = [&](std::size_t i) {
      const auto& job = batch[i];
      auto& e = execs[i];
      fill_matrices(job, e.ha.data(), e.hb.data());
      e.pickup = InferenceJob::Clock::now();
      CUDA_CHECK(cudaMemcpyAsync(
          e.da.data(), e.ha.data(), e.da.bytes(), cudaMemcpyHostToDevice, ctx.h2d_stream));
      CUDA_CHECK(cudaMemcpyAsync(
          e.db.data(), e.hb.data(), e.db.bytes(), cudaMemcpyHostToDevice, ctx.h2d_stream));
      CUDA_CHECK(cudaEventRecord(ctx.h2d_done[i].get(), ctx.h2d_stream));
    };

    enqueue_h2d(0);
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const auto& job = batch[i];
      auto& e = execs[i];

      CUDA_CHECK(cudaStreamWaitEvent(ctx.compute_stream, ctx.h2d_done[i].get(), 0));
      if (opts_.impl == "cublas") {
        gemm_cublas_batched(e.da.data(),
                            e.db.data(),
                            e.dc.data(),
                            job.m,
                            job.n,
                            job.k,
                            job.batch,
                            ctx.compute_stream);
      } else {
        gemm_tiled_batched(e.da.data(),
                           e.db.data(),
                           e.dc.data(),
                           job.m,
                           job.n,
                           job.k,
                           job.batch,
                           ctx.compute_stream);
      }
      CUDA_CHECK(cudaEventRecord(ctx.kern_done[i].get(), ctx.compute_stream));

      CUDA_CHECK(cudaStreamWaitEvent(ctx.d2h_stream, ctx.kern_done[i].get(), 0));
      CUDA_CHECK(cudaMemcpyAsync(
          e.hc.data(), e.dc.data(), e.dc.bytes(), cudaMemcpyDeviceToHost, ctx.d2h_stream));
      CUDA_CHECK(cudaEventRecord(ctx.done[i].get(), ctx.d2h_stream));

      // Queue the next H2D after D2H_i is on the wire, so a slow host fill
      // cannot delay the current job's result copy.
      if (i + 1 < batch.size())
        enqueue_h2d(i + 1);
    }

    std::vector<JobMetrics> out;
    out.reserve(batch.size());
    for (std::size_t i = 0; i < batch.size(); ++i) {
      const auto& job = batch[i];
      auto& e = execs[i];
      CUDA_CHECK(cudaEventSynchronize(ctx.done[i].get()));
      const auto finish = InferenceJob::Clock::now();
      JobMetrics m;
      m.job_id = job.id;
      m.seq = job.seq;
      m.priority = job.priority;
      m.queue_wait_ms = milliseconds(job.submit_time, e.pickup);
      m.exec_ms = milliseconds(e.pickup, finish);
      m.e2e_ms = milliseconds(job.submit_time, finish);
      m.ok = true;
      out.push_back(m);
      if (opts_.retain_results) {
        std::vector<float> c(e.hc.data(), e.hc.data() + e.hc.size());
        std::lock_guard<std::mutex> lk(results_mutex_);
        results_[job.seq] = std::move(c);
      }
    }
    {
      std::lock_guard<std::mutex> lk(metrics_mutex_);
      metrics_.insert(metrics_.end(), out.begin(), out.end());
    }
    completed_.fetch_add(out.size(), std::memory_order_relaxed);
  } catch (const std::exception& ex) {
    std::cerr << "[scheduler] batch execution failed: " << ex.what() << "\n";
    // The worker reuses its streams and event pool for the next batch, so
    // drain any work from this failed batch before returning. Ignore the
    // status here: the original exception is the useful error.
    cudaDeviceSynchronize();
    fail_batch(batch, ex.what());
  }
}

void BatchScheduler::fail_batch(const std::vector<InferenceJob>& batch, const std::string& reason) {
  (void)reason;
  std::lock_guard<std::mutex> lk(metrics_mutex_);
  for (const auto& job : batch) {
    JobMetrics m;
    m.job_id = job.id;
    m.seq = job.seq;
    m.priority = job.priority;
    m.ok = false;
    metrics_.push_back(m);
  }
  completed_.fetch_add(batch.size(), std::memory_order_relaxed);
}

}  // namespace gbis
