// GPU correctness tests for the three batched-GEMM implementations.
//
// Every kernel is compared against a float64 CPU reference computed from the
// same float32 inputs. The tolerance is derived from the forward error of a
// length-K float32 dot product, not tuned to the observed error:
//
//   |fl(sum) - sum| <= gamma_K * sum_i |a_i b_i|
//
// For a conservative, data-independent scale we bound each |a_i b_i| by
// max|A| * max|B|. With unit roundoff u = 2^-24 this gives
//
//   tol(K) = 8 * u * K * max|A| * max|B|
//
// The factor 8 leaves room for the reference's own float32 rounding and for
// a different (tree/blocked) summation order in cuBLAS; it is still O(K), so a
// wrong tile, a dropped K-slice or a missing batch element produces an error
// many orders of magnitude larger than the tolerance.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "core/cuda_check.hpp"
#include "core/device_buffer.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemm_cublas.hpp"

namespace {

using gbis::DeviceBuffer;

// Unit roundoff for binary32 round-to-nearest.
constexpr double kUnitRoundoff = 0.5 * static_cast<double>(std::numeric_limits<float>::epsilon());

struct Shape {
  int m;
  int n;
  int k;
  int batch;
};

std::string shape_string(const Shape& s) {
  return "m" + std::to_string(s.m) + "_n" + std::to_string(s.n) + "_k" + std::to_string(s.k) +
         "_b" + std::to_string(s.batch);
}

// Awkward sizes: every dimension that matters to tiling/vectorisation, plus
// very skinny matrices and both batch extremes. Large-K cases stay skinny so
// the float64 reference is affordable.
const std::vector<Shape>& shapes() {
  static const std::vector<Shape> kShapes = {
      // Degenerate and tiny shapes.
      {1, 1, 1, 1},
      {1, 1, 1, 64},
      {1, 31, 33, 1},
      {1, 31, 33, 64},
      {31, 1, 257, 1},
      {31, 1, 257, 64},
      {33, 31, 1, 1},
      {33, 31, 1, 64},
      // Just around the 32x32 tile boundary.
      {31, 33, 31, 1},
      {33, 31, 33, 64},
      {257, 33, 31, 1},
      {257, 33, 31, 64},
      {33, 257, 31, 1},
      {33, 257, 31, 64},
      // Larger-K, skinny.
      {1, 257, 1000, 1},
      {1, 257, 1000, 64},
      {1000, 1, 31, 1},
      {1000, 1, 31, 64},
      {1, 1, 1000, 1},
      {1, 1, 1000, 64},
      {1, 1000, 33, 1},
      {1, 1000, 1000, 1},
      {33, 33, 1000, 1},
      // Single row/column.
      {1000, 1, 1, 1},
      {1000, 1, 1, 64},
      {1, 1000, 1, 1},
      {1, 1000, 1, 64},
      {1000, 1000, 1, 1},
      // Full non-multiple square-ish shape.
      {257, 257, 257, 1},
  };
  return kShapes;
}

bool gpu_available() {
  static const bool available = [] {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  }();
  return available;
}

using GemmFn = void (*)(const float*, const float*, float*, int, int, int, int, cudaStream_t);

GemmFn select_impl(const std::string& impl) {
  if (impl == "naive")
    return gbis::gemm_naive_batched;
  if (impl == "tiled")
    return gbis::gemm_tiled_batched;
  return gbis::gemm_cublas_batched;
}

// Reference: exact products and summation in float64, rounded once to float32.
void cpu_reference(const std::vector<float>& a,
                   const std::vector<float>& b,
                   int m,
                   int n,
                   int k,
                   int batch,
                   std::vector<float>& c,
                   double& max_abs_a,
                   double& max_abs_b) {
  max_abs_a = 0.0;
  max_abs_b = 0.0;
  for (float v : a)
    max_abs_a = std::max(max_abs_a, std::fabs(static_cast<double>(v)));
  for (float v : b)
    max_abs_b = std::max(max_abs_b, std::fabs(static_cast<double>(v)));

  for (int bi = 0; bi < batch; ++bi) {
    const float* ab = a.data() + static_cast<std::size_t>(bi) * m * k;
    const float* bb = b.data() + static_cast<std::size_t>(bi) * k * n;
    float* cb = c.data() + static_cast<std::size_t>(bi) * m * n;
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        double acc = 0.0;
        for (int kk = 0; kk < k; ++kk) {
          acc += static_cast<double>(ab[row * k + kk]) * static_cast<double>(bb[kk * n + col]);
        }
        cb[row * n + col] = static_cast<float>(acc);
      }
    }
  }
}

class GemmCorrectnessTest : public ::testing::TestWithParam<std::tuple<std::string, Shape>> {};

TEST_P(GemmCorrectnessTest, MatchesFloat64Reference) {
  if (!gpu_available())
    GTEST_SKIP() << "no CUDA GPU available";

  const std::string& impl = std::get<0>(GetParam());
  const Shape& s = std::get<1>(GetParam());
  SCOPED_TRACE("impl=" + impl + " shape=" + shape_string(s));

  const std::size_t sa = static_cast<std::size_t>(s.m) * s.k * s.batch;
  const std::size_t sb = static_cast<std::size_t>(s.k) * s.n * s.batch;
  const std::size_t sc = static_cast<std::size_t>(s.m) * s.n * s.batch;

  std::vector<float> a(sa), b(sb);
  std::mt19937 rng(0x5eed1234u);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : a)
    v = dist(rng);
  for (float& v : b)
    v = dist(rng);

  std::vector<float> ref(sc);
  double max_abs_a = 0.0;
  double max_abs_b = 0.0;
  cpu_reference(a, b, s.m, s.n, s.k, s.batch, ref, max_abs_a, max_abs_b);

  const double tol = 8.0 * kUnitRoundoff * static_cast<double>(s.k) * max_abs_a * max_abs_b;

  DeviceBuffer<float> da(sa), db(sb), dc(sc);
  CUDA_CHECK(cudaMemcpy(da.data(), a.data(), da.bytes(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db.data(), b.data(), db.bytes(), cudaMemcpyHostToDevice));
  // 0xffffffff is a NaN bit pattern: any output element the kernel fails to
  // write stays non-finite and is caught below.
  CUDA_CHECK(cudaMemset(dc.data(), 0xFF, dc.bytes()));

  select_impl(impl)(da.data(), db.data(), dc.data(), s.m, s.n, s.k, s.batch, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> got(sc);
  CUDA_CHECK(cudaMemcpy(got.data(), dc.data(), dc.bytes(), cudaMemcpyDeviceToHost));

  double max_err = 0.0;
  std::size_t worst = 0;
  std::size_t non_finite = 0;
  for (std::size_t i = 0; i < sc; ++i) {
    if (!std::isfinite(got[i])) {
      ++non_finite;
      continue;
    }
    const double err = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    if (err > max_err) {
      max_err = err;
      worst = i;
    }
  }

  EXPECT_EQ(non_finite, 0u) << "unwritten/non-finite output elements";
  EXPECT_LE(max_err, tol) << "worst element " << worst << " got=" << got[worst]
                          << " ref=" << ref[worst] << " tol=" << tol;
}

INSTANTIATE_TEST_SUITE_P(Kernels,
                         GemmCorrectnessTest,
                         ::testing::Combine(::testing::Values(std::string("naive"),
                                                              std::string("tiled"),
                                                              std::string("cublas")),
                                            ::testing::ValuesIn(shapes())),
                         [](const ::testing::TestParamInfo<GemmCorrectnessTest::ParamType>& info) {
                           return std::get<0>(info.param) + "_" +
                                  shape_string(std::get<1>(info.param));
                         });

}  // namespace
