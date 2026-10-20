// Transfer/compute overlap microbenchmark.
//
// Compares, for the same jobs and the same tiled/cuBLAS kernel:
//   pageable  : pageable host memory, one stream, synchronous cudaMemcpy
//   pinned1   : pinned host memory, cudaMemcpyAsync, one stream (serialized)
//   pinned2   : pinned host memory, copy stream + compute stream with events,
//               using the same dependency order as BatchScheduler::execute_batch
//
// Reports per-job H2D+D2H, kernel, and end-to-end time (median of --reps
// timed sequences after --warmup warmups).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cuda_check.hpp"
#include "core/device_buffer.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemm_cublas.hpp"

namespace {

using gbis::DeviceBuffer;

struct Shape {
  int batch;
  int m;
  int n;
  int k;
};

struct Options {
  std::vector<Shape> shapes;
  int jobs = 20;
  int reps = 5;
  int warmup = 2;
  std::string impl = "tiled";
  bool help = false;
};

struct Result {
  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
};

class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  ~PinnedBuffer() {
    release();
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  PinnedBuffer(PinnedBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
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
        cudaHostAlloc(reinterpret_cast<void**>(&data_), count * sizeof(float),
                      cudaHostAllocDefault));
    count_ = count;
  }
  void release() noexcept {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
      data_ = nullptr;
    }
    count_ = 0;
  }
  float* data() {
    return data_;
  }
  std::size_t size() const {
    return count_;
  }

 private:
  float* data_ = nullptr;
  std::size_t count_ = 0;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  return values.size() % 2 == 1 ? values[mid]
                                : 0.5 * (values[mid - 1] + values[mid]);
}

double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

Shape parse_shape(const std::string& text) {
  std::vector<int> v;
  std::size_t pos = 0;
  while (pos < text.size()) {
    const std::size_t next = text.find('x', pos);
    v.push_back(std::stoi(text.substr(pos, next - pos)));
    if (next == std::string::npos)
      break;
    pos = next + 1;
  }
  if (v.size() != 4)
    throw std::runtime_error("shape must be BxMxNxK: " + text);
  return Shape{v[0], v[1], v[2], v[3]};
}

std::string shape_label(const Shape& s) {
  return std::to_string(s.batch) + "x" + std::to_string(s.m) + "x" +
         std::to_string(s.n) + "x" + std::to_string(s.k);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  opt.shapes = {Shape{8, 512, 512, 512}, Shape{8, 1024, 1024, 1024}};
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](const std::string& name) -> std::string {
      const std::string prefix = name + "=";
      if (arg.compare(0, prefix.size(), prefix) == 0)
        return arg.substr(prefix.size());
      if (i + 1 < argc)
        return argv[++i];
      throw std::runtime_error("missing value for " + name);
    };
    if (arg == "--help" || arg == "-h") {
      opt.help = true;
    } else if (arg == "--shape" || arg.rfind("--shape=", 0) == 0) {
      if (arg == "--shape")
        opt.shapes.clear();
      opt.shapes.push_back(parse_shape(value("--shape")));
    } else if (arg == "--jobs" || arg.rfind("--jobs=", 0) == 0) {
      opt.jobs = std::stoi(value("--jobs"));
    } else if (arg == "--reps" || arg.rfind("--reps=", 0) == 0) {
      opt.reps = std::stoi(value("--reps"));
    } else if (arg == "--warmup" || arg.rfind("--warmup=", 0) == 0) {
      opt.warmup = std::stoi(value("--warmup"));
    } else if (arg == "--impl" || arg.rfind("--impl=", 0) == 0) {
      opt.impl = value("--impl");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opt.jobs <= 0 || opt.reps <= 0 || opt.warmup < 0)
    throw std::runtime_error("invalid --jobs/--reps/--warmup");
  if (opt.impl != "tiled" && opt.impl != "cublas")
    throw std::runtime_error("--impl must be tiled|cublas");
  return opt;
}

void print_usage(std::ostream& os) {
  os << "Usage: bench_overlap [options]\n"
     << "  --shape BxMxNxK  shape to test (repeatable; default 8x512^3, 8x1024^3)\n"
     << "  --jobs <N>       jobs per timed sequence (default 20)\n"
     << "  --reps <N>       timed sequences, median reported (default 5)\n"
     << "  --warmup <N>     warmup sequences (default 2)\n"
     << "  --impl <name>    tiled|cublas (default tiled)\n"
     << "  --help           show this message\n";
}

using GemmFn = void (*)(const float*, const float*, float*, int, int, int, int, cudaStream_t);

GemmFn select_impl(const std::string& impl) {
  return impl == "cublas" ? gbis::gemm_cublas_batched : gbis::gemm_tiled_batched;
}

class Event {
 public:
  Event() {
    CUDA_CHECK(cudaEventCreate(&e_));
  }
  ~Event() {
    if (e_ != nullptr)
      cudaEventDestroy(e_);
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept : e_(other.e_) {
    other.e_ = nullptr;
  }
  Event& operator=(Event&& other) noexcept {
    if (this != &other) {
      if (e_ != nullptr)
        cudaEventDestroy(e_);
      e_ = other.e_;
      other.e_ = nullptr;
    }
    return *this;
  }
  cudaEvent_t get() const {
    return e_;
  }

 private:
  cudaEvent_t e_ = nullptr;
};

struct Workspace {
  std::vector<std::vector<float>> page_a, page_b, page_c;
  std::vector<PinnedBuffer> pin_a, pin_b, pin_c;
  std::vector<DeviceBuffer<float>> dev_a, dev_b, dev_c;
  std::vector<std::size_t> sa, sb, sc;
  std::vector<Event> c0, c1, k0, k1, d0, d1;
  Event start, end, ev_h2d, ev_kern;
};

Result run_pageable(const Options& opt,
                    const Shape& shape,
                    const GemmFn gemm,
                    Workspace& w) {
  const std::size_t n = static_cast<std::size_t>(opt.jobs);
  std::vector<double> totals, h2d, kern, d2h;
  for (int rep = 0; rep < opt.warmup + opt.reps; ++rep) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto wall0 = std::chrono::steady_clock::now();
    double h = 0.0, ksum = 0.0, d = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaMemcpy(
          w.dev_a[i].data(), w.page_a[i].data(), w.sa[i] * sizeof(float),
          cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(
          w.dev_b[i].data(), w.page_b[i].data(), w.sb[i] * sizeof(float),
          cudaMemcpyHostToDevice));
      auto t1 = std::chrono::steady_clock::now();
      h += std::chrono::duration<double, std::milli>(t1 - t0).count();

      CUDA_CHECK(cudaEventRecord(w.k0[i].get(), 0));
      gemm(w.dev_a[i].data(), w.dev_b[i].data(), w.dev_c[i].data(), shape.m, shape.n, shape.k,
           shape.batch, 0);
      CUDA_CHECK(cudaEventRecord(w.k1[i].get(), 0));
      CUDA_CHECK(cudaEventSynchronize(w.k1[i].get()));
      ksum += elapsed_ms(w.k0[i].get(), w.k1[i].get());

      t0 = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaMemcpy(
          w.page_c[i].data(), w.dev_c[i].data(), w.sc[i] * sizeof(float),
          cudaMemcpyDeviceToHost));
      t1 = std::chrono::steady_clock::now();
      d += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const auto wall1 = std::chrono::steady_clock::now();
    if (rep >= opt.warmup) {
      totals.push_back(std::chrono::duration<double, std::milli>(wall1 - wall0).count());
      h2d.push_back(h);
      kern.push_back(ksum);
      d2h.push_back(d);
    }
  }
  return Result{median(h2d) / static_cast<double>(n), median(kern) / static_cast<double>(n),
                median(d2h) / static_cast<double>(n),
                median(totals) / static_cast<double>(n)};
}

Result run_pinned_single(const Options& opt,
                         const Shape& shape,
                         const GemmFn gemm,
                         Workspace& w) {
  const std::size_t n = static_cast<std::size_t>(opt.jobs);
  std::vector<double> totals, h2d, kern, d2h;
  for (int rep = 0; rep < opt.warmup + opt.reps; ++rep) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(w.start.get(), 0));
    for (std::size_t i = 0; i < n; ++i) {
      CUDA_CHECK(cudaEventRecord(w.c0[i].get(), 0));
      CUDA_CHECK(cudaMemcpyAsync(
          w.dev_a[i].data(), w.pin_a[i].data(), w.sa[i] * sizeof(float),
          cudaMemcpyHostToDevice, 0));
      CUDA_CHECK(cudaMemcpyAsync(
          w.dev_b[i].data(), w.pin_b[i].data(), w.sb[i] * sizeof(float),
          cudaMemcpyHostToDevice, 0));
      CUDA_CHECK(cudaEventRecord(w.c1[i].get(), 0));
      CUDA_CHECK(cudaEventRecord(w.k0[i].get(), 0));
      gemm(w.dev_a[i].data(), w.dev_b[i].data(), w.dev_c[i].data(), shape.m, shape.n, shape.k,
           shape.batch, 0);
      CUDA_CHECK(cudaEventRecord(w.k1[i].get(), 0));
      CUDA_CHECK(cudaEventRecord(w.d0[i].get(), 0));
      CUDA_CHECK(cudaMemcpyAsync(
          w.pin_c[i].data(), w.dev_c[i].data(), w.sc[i] * sizeof(float),
          cudaMemcpyDeviceToHost, 0));
      CUDA_CHECK(cudaEventRecord(w.d1[i].get(), 0));
    }
    CUDA_CHECK(cudaEventRecord(w.end.get(), 0));
    CUDA_CHECK(cudaEventSynchronize(w.end.get()));
    double h = 0.0, ksum = 0.0, d = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      h += elapsed_ms(w.c0[i].get(), w.c1[i].get());
      ksum += elapsed_ms(w.k0[i].get(), w.k1[i].get());
      d += elapsed_ms(w.d0[i].get(), w.d1[i].get());
    }
    if (rep >= opt.warmup) {
      totals.push_back(elapsed_ms(w.start.get(), w.end.get()));
      h2d.push_back(h);
      kern.push_back(ksum);
      d2h.push_back(d);
    }
  }
  return Result{median(h2d) / static_cast<double>(n), median(kern) / static_cast<double>(n),
                median(d2h) / static_cast<double>(n),
                median(totals) / static_cast<double>(n)};
}

Result run_pinned_overlap(const Options& opt,
                          const Shape& shape,
                          const GemmFn gemm,
                          Workspace& w) {
  const std::size_t n = static_cast<std::size_t>(opt.jobs);
  cudaStream_t copy = nullptr;
  cudaStream_t compute = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreate(&compute));

  std::vector<double> totals, h2d, kern, d2h;
  for (int rep = 0; rep < opt.warmup + opt.reps; ++rep) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(w.start.get(), copy));
    for (std::size_t i = 0; i < n; ++i) {
      CUDA_CHECK(cudaEventRecord(w.c0[i].get(), copy));
      CUDA_CHECK(cudaMemcpyAsync(
          w.dev_a[i].data(), w.pin_a[i].data(), w.sa[i] * sizeof(float),
          cudaMemcpyHostToDevice, copy));
      CUDA_CHECK(cudaMemcpyAsync(
          w.dev_b[i].data(), w.pin_b[i].data(), w.sb[i] * sizeof(float),
          cudaMemcpyHostToDevice, copy));
      CUDA_CHECK(cudaEventRecord(w.c1[i].get(), copy));
      CUDA_CHECK(cudaEventRecord(w.ev_h2d.get(), copy));
      CUDA_CHECK(cudaStreamWaitEvent(compute, w.ev_h2d.get(), 0));

      CUDA_CHECK(cudaEventRecord(w.k0[i].get(), compute));
      gemm(w.dev_a[i].data(), w.dev_b[i].data(), w.dev_c[i].data(), shape.m, shape.n, shape.k,
           shape.batch, compute);
      CUDA_CHECK(cudaEventRecord(w.k1[i].get(), compute));
      CUDA_CHECK(cudaEventRecord(w.ev_kern.get(), compute));
      CUDA_CHECK(cudaStreamWaitEvent(copy, w.ev_kern.get(), 0));

      CUDA_CHECK(cudaEventRecord(w.d0[i].get(), copy));
      CUDA_CHECK(cudaMemcpyAsync(
          w.pin_c[i].data(), w.dev_c[i].data(), w.sc[i] * sizeof(float),
          cudaMemcpyDeviceToHost, copy));
      CUDA_CHECK(cudaEventRecord(w.d1[i].get(), copy));
    }
    CUDA_CHECK(cudaEventRecord(w.end.get(), copy));
    CUDA_CHECK(cudaEventSynchronize(w.end.get()));
    double h = 0.0, ksum = 0.0, d = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      h += elapsed_ms(w.c0[i].get(), w.c1[i].get());
      ksum += elapsed_ms(w.k0[i].get(), w.k1[i].get());
      d += elapsed_ms(w.d0[i].get(), w.d1[i].get());
    }
    if (rep >= opt.warmup) {
      totals.push_back(elapsed_ms(w.start.get(), w.end.get()));
      h2d.push_back(h);
      kern.push_back(ksum);
      d2h.push_back(d);
    }
  }
  CUDA_CHECK(cudaStreamDestroy(copy));
  CUDA_CHECK(cudaStreamDestroy(compute));
  return Result{median(h2d) / static_cast<double>(n), median(kern) / static_cast<double>(n),
                median(d2h) / static_cast<double>(n),
                median(totals) / static_cast<double>(n)};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    if (opt.help) {
      print_usage(std::cout);
      return EXIT_SUCCESS;
    }

    const GemmFn gemm = select_impl(opt.impl);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "impl,shape,jobs,config,transfer_ms,kernel_ms,total_ms,speedup_vs_pageable\n";

    for (const Shape& shape : opt.shapes) {
      const std::size_t n = static_cast<std::size_t>(opt.jobs);
      Workspace w;
      w.page_a.resize(n);
      w.page_b.resize(n);
      w.page_c.resize(n);
      w.pin_a.resize(n);
      w.pin_b.resize(n);
      w.pin_c.resize(n);
      w.dev_a.resize(n);
      w.dev_b.resize(n);
      w.dev_c.resize(n);
      w.sa.resize(n);
      w.sb.resize(n);
      w.sc.resize(n);
      w.c0.resize(n);
      w.c1.resize(n);
      w.k0.resize(n);
      w.k1.resize(n);
      w.d0.resize(n);
      w.d1.resize(n);

      const std::size_t sa = static_cast<std::size_t>(shape.m) * shape.k * shape.batch;
      const std::size_t sb = static_cast<std::size_t>(shape.k) * shape.n * shape.batch;
      const std::size_t sc = static_cast<std::size_t>(shape.m) * shape.n * shape.batch;

      std::mt19937 rng(42);
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      for (std::size_t i = 0; i < n; ++i) {
        w.sa[i] = sa;
        w.sb[i] = sb;
        w.sc[i] = sc;
        w.page_a[i].resize(sa);
        w.page_b[i].resize(sb);
        w.page_c[i].resize(sc);
        w.pin_a[i].allocate(sa);
        w.pin_b[i].allocate(sb);
        w.pin_c[i].allocate(sc);
        w.dev_a[i].allocate(sa);
        w.dev_b[i].allocate(sb);
        w.dev_c[i].allocate(sc);
        for (std::size_t j = 0; j < sa; ++j) {
          const float v = dist(rng);
          w.page_a[i][j] = v;
          w.pin_a[i].data()[j] = v;
        }
        for (std::size_t j = 0; j < sb; ++j) {
          const float v = dist(rng);
          w.page_b[i][j] = v;
          w.pin_b[i].data()[j] = v;
        }
      }

      const Result pageable = run_pageable(opt, shape, gemm, w);
      const Result pinned1 = run_pinned_single(opt, shape, gemm, w);
      const Result pinned2 = run_pinned_overlap(opt, shape, gemm, w);

      const Result results[3] = {pageable, pinned1, pinned2};
      const char* names[3] = {"pageable_sync", "pinned_single_stream", "pinned_copy_compute"};
      double base_total = pageable.total_ms;
      for (int c = 0; c < 3; ++c) {
        const double speedup = base_total > 0.0 ? base_total / results[c].total_ms : 0.0;
        std::cout << opt.impl << ',' << shape_label(shape) << ',' << opt.jobs << ','
                  << names[c] << ',' << results[c].h2d_ms + results[c].d2h_ms << ','
                  << results[c].kernel_ms << ',' << results[c].total_ms << ',' << speedup
                  << '\n';
      }
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
