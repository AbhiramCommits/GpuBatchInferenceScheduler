#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/cuda_check.hpp"
#include "core/device_buffer.hpp"
#include "kernels/gemm.hpp"
#include "sched/job.hpp"
#include "sched/scheduler.hpp"

namespace py = pybind11;

namespace {

// Process-global default scheduler, shared by submit_job() and the Scheduler
// class. Guarded by g_mutex.
std::mutex g_mutex;
std::unique_ptr<gbis::BatchScheduler> g_sched;
gbis::BatchScheduler::Options g_opts;
bool g_running = false;

gbis::BatchScheduler& ensure_scheduler() {
  if (!g_sched) {
    g_sched = std::make_unique<gbis::BatchScheduler>(g_opts);
    g_sched->start();
    g_running = true;
  }
  return *g_sched;
}

void start_scheduler(int max_batch, int workers, const std::string& impl) {
  if (impl != "tiled" && impl != "cublas") {
    throw py::value_error("impl must be 'tiled' or 'cublas'");
  }
  if (max_batch <= 0)
    throw py::value_error("max_batch must be positive");
  if (workers <= 0)
    throw py::value_error("workers must be positive");
  py::gil_scoped_release release;
  std::lock_guard<std::mutex> lk(g_mutex);
  g_opts.max_batch = max_batch;
  g_opts.num_workers = workers;
  g_opts.impl = impl;
  if (g_running)
    return;  // already running; keep the current instance
  g_sched.reset();
  g_sched = std::make_unique<gbis::BatchScheduler>(g_opts);
  g_sched->start();
  g_running = true;
}

void stop_scheduler() {
  py::gil_scoped_release release;
  std::lock_guard<std::mutex> lk(g_mutex);
  if (!g_running)
    return;
  g_sched->stop();
  g_running = false;
}

double percentile(std::vector<double> v, double p) {
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t rank = static_cast<std::size_t>(std::ceil(p / 100.0 * v.size()));
  return v[std::max<std::size_t>(1, rank) - 1];
}

py::dict scheduler_stats() {
  std::lock_guard<std::mutex> lk(g_mutex);
  py::dict d;
  d["running"] = g_running;
  if (!g_sched) {
    d["submitted"] = std::uint64_t{0};
    d["completed"] = std::uint64_t{0};
    d["queued"] = std::size_t{0};
    d["failed"] = std::uint64_t{0};
    d["p50_ms"] = 0.0;
    d["p95_ms"] = 0.0;
    d["p99_ms"] = 0.0;
    d["throughput_jobs_per_s"] = 0.0;
    d["wall_seconds"] = 0.0;
    d["mean_gpu_utilization"] = 0.0;
    return d;
  }
  const auto metrics = g_sched->metrics();
  std::vector<double> e2e;
  std::uint64_t failed = 0;
  for (const auto& mm : metrics) {
    if (mm.ok) {
      e2e.push_back(mm.e2e_ms);
    } else {
      ++failed;
    }
  }
  const double wall = g_sched->wall_seconds();
  d["submitted"] = g_sched->submitted();
  d["completed"] = g_sched->completed();
  d["queued"] = g_sched->queued();
  d["failed"] = failed;
  d["p50_ms"] = percentile(e2e, 50.0);
  d["p95_ms"] = percentile(e2e, 95.0);
  d["p99_ms"] = percentile(e2e, 99.0);
  d["throughput_jobs_per_s"] = wall > 0.0 ? static_cast<double>(g_sched->completed()) / wall : 0.0;
  d["wall_seconds"] = wall;
  d["mean_gpu_utilization"] = g_sched->mean_gpu_utilization();
  return d;
}

struct SchedulerHandle {
  int max_batch = 8;
  int workers = 2;
  std::string impl = "tiled";
};

}  // namespace

PYBIND11_MODULE(gpuinfer, m) {
  m.doc() = "GPU batch inference scheduler bindings: batched GEMM + scheduler";

  m.def("cuda_available", [] {
    int ndev = 0;
    return cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0;
  });

  // Batched GEMM: C[b] = A[b] * B[b], A:(B, M, K), B:(B, K, N), float32.
  // C-contiguous float32 inputs are used zero-copy.
  m.def(
      "batched_gemm",
      [](py::array a, py::array b) -> py::array {
        if (!a.dtype().is(py::dtype::of<float>())) {
          throw py::value_error("a must be a float32 array");
        }
        if (!b.dtype().is(py::dtype::of<float>())) {
          throw py::value_error("b must be a float32 array");
        }
        if (a.ndim() != 3)
          throw py::value_error("a must be 3-D (B, M, K)");
        if (b.ndim() != 3)
          throw py::value_error("b must be 3-D (B, K, N)");
        const auto b_batch = a.shape(0);
        const auto m_dim = a.shape(1);
        const auto k_dim = a.shape(2);
        if (b.shape(0) != b_batch) {
          throw py::value_error("batch size mismatch between a and b");
        }
        if (b.shape(1) != k_dim) {
          throw py::value_error("inner dimension mismatch between a and b");
        }
        const auto n_dim = b.shape(2);
        if (m_dim <= 0 || n_dim <= 0 || k_dim <= 0) {
          throw py::value_error("matrix dimensions must be positive");
        }
        if (b_batch <= 0 || b_batch > 65535) {
          throw py::value_error("batch size must be in [1, 65535]");
        }

        // Zero-copy when the input is already C-contiguous float32;
        // otherwise materialize a contiguous copy.
        auto a_c = py::array_t<float, py::array::c_style>::ensure(a);
        auto b_c = py::array_t<float, py::array::c_style>::ensure(b);

        py::array_t<float, py::array::c_style> result(
            std::vector<py::ssize_t>{b_batch, m_dim, n_dim});
        const py::buffer_info a_info = a_c.request();
        const py::buffer_info b_info = b_c.request();
        const py::buffer_info r_info = result.request();

        {
          py::gil_scoped_release release;
          const int batch = static_cast<int>(b_batch);
          const int m_i = static_cast<int>(m_dim);
          const int n_i = static_cast<int>(n_dim);
          const int k_i = static_cast<int>(k_dim);
          gbis::DeviceBuffer<float> da(static_cast<std::size_t>(a_info.size));
          gbis::DeviceBuffer<float> db(static_cast<std::size_t>(b_info.size));
          gbis::DeviceBuffer<float> dc(static_cast<std::size_t>(r_info.size));
          CUDA_CHECK(cudaMemcpyAsync(da.data(), a_info.ptr, da.bytes(), cudaMemcpyHostToDevice, 0));
          CUDA_CHECK(cudaMemcpyAsync(db.data(), b_info.ptr, db.bytes(), cudaMemcpyHostToDevice, 0));
          gbis::gemm_tiled_batched(da.data(), db.data(), dc.data(), m_i, n_i, k_i, batch, 0);
          CUDA_CHECK(cudaMemcpyAsync(r_info.ptr, dc.data(), dc.bytes(), cudaMemcpyDeviceToHost, 0));
          CUDA_CHECK(cudaDeviceSynchronize());
        }
        return result;
      },
      py::arg("a"),
      py::arg("b"));

  // Submit a job to the default scheduler; returns the job id (seq).
  m.def(
      "submit_job",
      [](int m_dim, int n_dim, int k_dim, int batch, int priority) -> std::uint64_t {
        if (m_dim <= 0 || n_dim <= 0 || k_dim <= 0) {
          throw py::value_error("m, n and k must be positive");
        }
        if (batch <= 0)
          throw py::value_error("batch must be positive");
        gbis::InferenceJob job;
        job.m = m_dim;
        job.n = n_dim;
        job.k = k_dim;
        job.batch = batch;
        job.priority = priority;
        py::gil_scoped_release release;
        return ensure_scheduler().submit(job);
      },
      py::arg("m"),
      py::arg("n"),
      py::arg("k"),
      py::arg("batch") = 1,
      py::arg("priority") = 0);

  py::class_<SchedulerHandle>(m, "Scheduler")
      .def(py::init([](int max_batch, int workers, const std::string& impl) {
             if (impl != "tiled" && impl != "cublas") {
               throw py::value_error("impl must be 'tiled' or 'cublas'");
             }
             if (max_batch <= 0) {
               throw py::value_error("max_batch must be positive");
             }
             if (workers <= 0) {
               throw py::value_error("workers must be positive");
             }
             SchedulerHandle h;
             h.max_batch = max_batch;
             h.workers = workers;
             h.impl = impl;
             return h;
           }),
           py::arg("max_batch") = 8,
           py::arg("workers") = 2,
           py::arg("impl") = "tiled")
      .def("start", [](SchedulerHandle& h) { start_scheduler(h.max_batch, h.workers, h.impl); })
      .def("stop", [](SchedulerHandle&) { stop_scheduler(); })
      .def("stats", [](SchedulerHandle&) { return scheduler_stats(); });
}
