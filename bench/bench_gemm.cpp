#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cuda_check.hpp"
#include "core/device_buffer.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemm_cublas.hpp"

namespace {

constexpr double kTolerance = 1e-3;

struct Options {
  int m = 512;
  int n = 512;
  int k = 512;
  int batch = 8;
  int iters = 50;   // timed runs (median reported)
  int warmup = 10;  // untimed warmup runs
  std::string impl = "tiled";
  bool skip_check = false;
  bool help = false;
};

using GemmFn = void (*)(const float*, const float*, float*, int, int, int, int, cudaStream_t);

void print_usage(std::ostream& os) {
  os << "Usage: bench_gemm [options]\n"
     << "Batched GEMM benchmark: C[b] = A[b] * B[b], A:(M,K), B:(K,N), "
        "row-major, float32.\n\n"
     << "Options:\n"
     << "  --m <int>       rows of A/C        (default 512)\n"
     << "  --n <int>       cols of B/C        (default 512)\n"
     << "  --k <int>       inner dimension    (default 512)\n"
     << "  --batch <int>   batch size B       (default 8)\n"
     << "  --iters <int>   timed runs, median reported (default 50)\n"
     << "  --warmup <int>  untimed warmup runs (default 10)\n"
     << "  --impl <name>   naive|tiled|cublas|cpu (default tiled)\n"
     << "  --skip-check    skip the CPU reference (benchmark-only mode)\n"
     << "  --help          show this message\n";
}

int parse_int(const std::string& name, const std::string& value) {
  std::size_t pos = 0;
  int v;
  try {
    v = std::stoi(value, &pos);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for " + name + ": " + value);
  }
  if (pos != value.size()) {
    throw std::runtime_error("invalid integer for " + name + ": " + value);
  }
  return v;
}

std::string option_value(int argc, char** argv, int& i, const std::string& name) {
  const std::string arg = argv[i];
  const std::string prefix = name + "=";
  if (arg.compare(0, prefix.size(), prefix) == 0)
    return arg.substr(prefix.size());
  if (i + 1 < argc)
    return argv[++i];
  throw std::runtime_error("missing value for " + name);
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      opt.help = true;
    } else if (arg == "--m" || arg.rfind("--m=", 0) == 0) {
      opt.m = parse_int("--m", option_value(argc, argv, i, "--m"));
    } else if (arg == "--n" || arg.rfind("--n=", 0) == 0) {
      opt.n = parse_int("--n", option_value(argc, argv, i, "--n"));
    } else if (arg == "--k" || arg.rfind("--k=", 0) == 0) {
      opt.k = parse_int("--k", option_value(argc, argv, i, "--k"));
    } else if (arg == "--batch" || arg.rfind("--batch=", 0) == 0) {
      opt.batch = parse_int("--batch", option_value(argc, argv, i, "--batch"));
    } else if (arg == "--iters" || arg.rfind("--iters=", 0) == 0) {
      opt.iters = parse_int("--iters", option_value(argc, argv, i, "--iters"));
    } else if (arg == "--warmup" || arg.rfind("--warmup=", 0) == 0) {
      opt.warmup = parse_int("--warmup", option_value(argc, argv, i, "--warmup"));
    } else if (arg == "--skip-check") {
      opt.skip_check = true;
    } else if (arg == "--impl" || arg.rfind("--impl=", 0) == 0) {
      opt.impl = option_value(argc, argv, i, "--impl");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0) {
    throw std::runtime_error("--m, --n and --k must be positive");
  }
  if (opt.batch <= 0)
    throw std::runtime_error("--batch must be positive");
  if (opt.iters <= 0)
    throw std::runtime_error("--iters must be positive");
  if (opt.warmup < 0)
    throw std::runtime_error("--warmup must be >= 0");
  if (opt.impl != "naive" && opt.impl != "tiled" && opt.impl != "cublas" && opt.impl != "cpu") {
    throw std::runtime_error("--impl must be one of naive|tiled|cublas|cpu");
  }
  return opt;
}

void gemm_cpu_reference(const float* A, const float* B, float* C, int M, int N, int K, int batch) {
  for (int b = 0; b < batch; ++b) {
    const float* Ab = A + static_cast<std::size_t>(b) * M * K;
    const float* Bb = B + static_cast<std::size_t>(b) * K * N;
    float* Cb = C + static_cast<std::size_t>(b) * M * N;
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += static_cast<double>(Ab[m * K + k]) * static_cast<double>(Bb[k * N + n]);
        }
        Cb[m * N + n] = static_cast<float>(acc);
      }
    }
  }
}

double max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
  double max_err = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return max_err;
}

double median(std::vector<double> values) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if (values.size() % 2 == 1)
    return values[mid];
  return 0.5 * (values[mid - 1] + values[mid]);
}

class CudaEvent {
 public:
  CudaEvent() {
    CUDA_CHECK(cudaEventCreate(&event_));
  }
  ~CudaEvent() {
    cudaEventDestroy(event_);
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  cudaEvent_t get() const {
    return event_;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

GemmFn select_impl(const std::string& impl) {
  if (impl == "naive")
    return gbis::gemm_naive_batched;
  if (impl == "tiled")
    return gbis::gemm_tiled_batched;
  if (impl == "cublas")
    return gbis::gemm_cublas_batched;
  return nullptr;
}

int run_benchmark(const Options& opt) {
  const std::size_t size_a = static_cast<std::size_t>(opt.m) * opt.k * opt.batch;
  const std::size_t size_b = static_cast<std::size_t>(opt.k) * opt.n * opt.batch;
  const std::size_t size_c = static_cast<std::size_t>(opt.m) * opt.n * opt.batch;

  std::vector<float> h_a(size_a), h_b(size_b), h_c(size_c), h_ref(size_c);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : h_a)
    v = dist(rng);
  for (float& v : h_b)
    v = dist(rng);

  double cpu_ms = 0.0;
  if (!opt.skip_check || opt.impl == "cpu") {
    const auto cpu_start = std::chrono::steady_clock::now();
    gemm_cpu_reference(h_a.data(), h_b.data(), h_ref.data(), opt.m, opt.n, opt.k, opt.batch);
    const auto cpu_stop = std::chrono::steady_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();
  }

  double h2d_ms = 0.0;
  double kernel_ms = 0.0;
  double d2h_ms = 0.0;
  double total_ms = 0.0;
  double max_err = std::numeric_limits<double>::quiet_NaN();

  if (opt.impl == "cpu") {
    h_c = h_ref;
    kernel_ms = cpu_ms;
    total_ms = cpu_ms;
    max_err = 0.0;
  } else {
    if (opt.batch > 65535) {
      throw std::runtime_error("--batch exceeds gridDim.z limit of 65535");
    }
    const GemmFn gemm = select_impl(opt.impl);
    gbis::DeviceBuffer<float> d_a(size_a), d_b(size_b), d_c(size_c);
    cudaStream_t stream = 0;

    CudaEvent e_start, e_h2d, e_k0, e_k1, e_end;

    const auto run_once = [&]() {
      CUDA_CHECK(
          cudaMemcpyAsync(d_a.data(), h_a.data(), d_a.bytes(), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(
          cudaMemcpyAsync(d_b.data(), h_b.data(), d_b.bytes(), cudaMemcpyHostToDevice, stream));
      gemm(d_a.data(), d_b.data(), d_c.data(), opt.m, opt.n, opt.k, opt.batch, stream);
      CUDA_CHECK(
          cudaMemcpyAsync(h_c.data(), d_c.data(), d_c.bytes(), cudaMemcpyDeviceToHost, stream));
    };

    // Warm up allocations, clocks, caches and the cuBLAS handle.
    for (int i = 0; i < opt.warmup; ++i)
      run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h2d_runs;
    std::vector<double> kernel_runs;
    std::vector<double> d2h_runs;
    std::vector<double> e2e_runs;
    h2d_runs.reserve(static_cast<std::size_t>(opt.iters));
    kernel_runs.reserve(static_cast<std::size_t>(opt.iters));
    d2h_runs.reserve(static_cast<std::size_t>(opt.iters));
    e2e_runs.reserve(static_cast<std::size_t>(opt.iters));

    for (int i = 0; i < opt.iters; ++i) {
      CUDA_CHECK(cudaEventRecord(e_start.get(), stream));
      CUDA_CHECK(
          cudaMemcpyAsync(d_a.data(), h_a.data(), d_a.bytes(), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(
          cudaMemcpyAsync(d_b.data(), h_b.data(), d_b.bytes(), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaEventRecord(e_h2d.get(), stream));

      CUDA_CHECK(cudaEventRecord(e_k0.get(), stream));
      gemm(d_a.data(), d_b.data(), d_c.data(), opt.m, opt.n, opt.k, opt.batch, stream);
      CUDA_CHECK(cudaEventRecord(e_k1.get(), stream));

      CUDA_CHECK(
          cudaMemcpyAsync(h_c.data(), d_c.data(), d_c.bytes(), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaEventRecord(e_end.get(), stream));
      CUDA_CHECK(cudaEventSynchronize(e_end.get()));

      h2d_runs.push_back(elapsed_ms(e_start.get(), e_h2d.get()));
      kernel_runs.push_back(elapsed_ms(e_k0.get(), e_k1.get()));
      d2h_runs.push_back(elapsed_ms(e_k1.get(), e_end.get()));
      e2e_runs.push_back(elapsed_ms(e_start.get(), e_end.get()));
    }

    h2d_ms = median(h2d_runs);
    kernel_ms = median(kernel_runs);
    d2h_ms = median(d2h_runs);
    total_ms = median(e2e_runs);

    if (!opt.skip_check)
      max_err = max_abs_error(h_c, h_ref);
  }

  const double flops = 2.0 * static_cast<double>(opt.m) * opt.n * opt.k * opt.batch;
  const double kernel_gflops = kernel_ms > 0.0 ? flops / kernel_ms / 1.0e6 : 0.0;
  const double e2e_gflops = total_ms > 0.0 ? flops / total_ms / 1.0e6 : 0.0;
  const double cpu_gflops = cpu_ms > 0.0 ? flops / cpu_ms / 1.0e6 : 0.0;
  const double speedup = kernel_ms > 0.0 ? cpu_ms / kernel_ms : 0.0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "impl,m,n,k,batch,iters,warmup,h2d_ms,kernel_ms,d2h_ms,total_ms,"
               "kernel_gflops,e2e_gflops,cpu_ms,cpu_gflops,speedup_vs_cpu,max_abs_err\n";
  std::cout << opt.impl << ',' << opt.m << ',' << opt.n << ',' << opt.k << ',' << opt.batch << ','
            << opt.iters << ',' << opt.warmup << ',' << h2d_ms << ',' << kernel_ms << ',' << d2h_ms
            << ',' << total_ms << ',' << kernel_gflops << ',' << e2e_gflops << ',' << cpu_ms << ','
            << cpu_gflops << ',' << speedup << ',' << max_err << '\n';

  if (!opt.skip_check && max_err >= kTolerance) {
    std::cerr << "error: max abs error " << max_err << " >= tolerance " << kTolerance << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    if (opt.help) {
      print_usage(std::cout);
      return EXIT_SUCCESS;
    }
    return run_benchmark(opt);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
