#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "sched/job.hpp"
#include "sched/scheduler.hpp"

namespace {

using gbis::BatchScheduler;
using gbis::InferenceJob;
using gbis::JobPriorityCompare;

InferenceJob make_job(std::uint64_t id, int m, int n, int k, int batch,
                      int priority) {
  InferenceJob j;
  j.id = id;
  j.m = m;
  j.n = n;
  j.k = k;
  j.batch = batch;
  j.priority = priority;
  return j;
}

std::size_t total_bytes(const std::vector<InferenceJob>& jobs) {
  std::size_t total = 0;
  for (const auto& j : jobs) total += j.bytes_required();
  return total;
}

void cpu_reference(const InferenceJob& job, const std::vector<float>& a,
                   const std::vector<float>& b, std::vector<float>& c) {
  for (int bi = 0; bi < job.batch; ++bi) {
    const float* ab = a.data() + static_cast<std::size_t>(bi) * job.m * job.k;
    const float* bb = b.data() + static_cast<std::size_t>(bi) * job.k * job.n;
    float* cb = c.data() + static_cast<std::size_t>(bi) * job.m * job.n;
    for (int m = 0; m < job.m; ++m) {
      for (int n = 0; n < job.n; ++n) {
        double acc = 0.0;
        for (int k = 0; k < job.k; ++k) {
          acc += static_cast<double>(ab[m * job.k + k]) *
                 static_cast<double>(bb[k * job.n + n]);
        }
        cb[m * job.n + n] = static_cast<float>(acc);
      }
    }
  }
}

TEST(PackingTest, NeverExceedsBudgetOrMaxBatch) {
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dim(1, 64);
  std::uniform_int_distribution<int> batch_d(1, 8);
  std::uniform_int_distribution<int> prio(0, 3);
  std::uniform_int_distribution<int> max_batch_d(1, 16);
  std::uniform_int_distribution<std::uint64_t> budget_d(0, 8ull << 20);
  for (int trial = 0; trial < 500; ++trial) {
    std::vector<InferenceJob> jobs;
    const int count = 1 + static_cast<int>(rng() % 64);
    for (int i = 0; i < count; ++i) {
      jobs.push_back(make_job(static_cast<std::uint64_t>(i), dim(rng), dim(rng),
                              dim(rng), batch_d(rng), prio(rng)));
    }
    const std::uint64_t budget = budget_d(rng);
    const int max_batch = max_batch_d(rng);
    std::vector<InferenceJob> selected;
    std::vector<InferenceJob> remaining;
    gbis::pack_jobs(jobs, budget, max_batch, selected, remaining);
    EXPECT_LE(total_bytes(selected), budget);
    EXPECT_LE(static_cast<int>(selected.size()), max_batch);
    EXPECT_EQ(selected.size() + remaining.size(), jobs.size());
  }
}

TEST(PackingTest, GreedySelectsLargestCount) {
  // Budget 200 with sizes {200, 100, 100}: the two 100-byte jobs fit where
  // the single 200-byte job would.
  auto j100a = make_job(0, 5, 5, 1, 1, 0);  // 100 elements
  auto j100b = make_job(1, 5, 5, 1, 1, 0);
  auto j200 = make_job(2, 5, 5, 1, 2, 0);
  std::vector<InferenceJob> selected;
  std::vector<InferenceJob> remaining;
  gbis::pack_jobs({j100a, j100b, j200}, j100a.bytes_required() * 2, 8,
                  selected, remaining);
  ASSERT_EQ(selected.size(), 2u);
  EXPECT_EQ(remaining.size(), 1u);
  EXPECT_EQ(remaining[0].id, 2u);
}

TEST(PackingTest, SelectsAllWhenBudgetSufficient) {
  std::vector<InferenceJob> jobs = {make_job(0, 8, 8, 8, 1, 1),
                                    make_job(1, 8, 8, 8, 1, 0),
                                    make_job(2, 8, 8, 8, 1, 2)};
  std::vector<InferenceJob> selected;
  std::vector<InferenceJob> remaining;
  gbis::pack_jobs(jobs, total_bytes(jobs), 8, selected, remaining);
  EXPECT_EQ(selected.size(), jobs.size());
  EXPECT_TRUE(remaining.empty());
}

TEST(PriorityTest, QueueOrdersHigherPriorityFirstThenFifo) {
  std::priority_queue<InferenceJob, std::vector<InferenceJob>, JobPriorityCompare>
      q;
  auto push = [&q](std::uint64_t seq, int priority) {
    auto j = make_job(seq, 1, 1, 1, 1, priority);
    j.seq = seq;
    q.push(j);
  };
  push(0, 1);
  push(1, 3);
  push(2, 1);
  push(3, 2);
  push(4, 3);
  const std::vector<std::uint64_t> expected = {1, 4, 3, 0, 2};
  for (std::uint64_t want : expected) {
    ASSERT_FALSE(q.empty());
    EXPECT_EQ(q.top().seq, want);
    q.pop();
  }
  EXPECT_TRUE(q.empty());
}

TEST(PriorityTest, PackPrefersHigherPriorityForEqualSize) {
  std::vector<InferenceJob> jobs = {
      make_job(0, 4, 4, 4, 1, 0),
      make_job(1, 4, 4, 4, 1, 2),
      make_job(2, 4, 4, 4, 1, 1),
  };
  std::vector<InferenceJob> selected;
  std::vector<InferenceJob> remaining;
  gbis::pack_jobs(jobs, jobs[0].bytes_required(), 1, selected, remaining);
  ASSERT_EQ(selected.size(), 1u);
  EXPECT_EQ(selected[0].id, 1u);  // highest priority wins
  EXPECT_EQ(remaining.size(), 2u);
}

TEST(SchedulerTest, EmptyQueueShutdownDoesNotDeadlock) {
  BatchScheduler::Options opts;
  opts.num_workers = 2;
  opts.fixed_budget_bytes = 1ull << 20;
  BatchScheduler sched(opts);
  sched.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  sched.stop();  // must return; would hang forever on deadlock
  EXPECT_EQ(sched.submitted(), 0u);
  EXPECT_EQ(sched.completed(), 0u);
  SUCCEED();
}

TEST(SchedulerTest, ShutdownWithPendingJobsDoesNotDeadlock) {
  // Budget too small to pack anything: jobs stay queued while the dispatcher
  // keeps polling. stop() must still shut everything down. (On GPU-less
  // hosts the dispatcher fails the jobs instead of queueing them; either way
  // shutdown must complete without deadlock.)
  BatchScheduler::Options opts;
  opts.num_workers = 2;
  opts.fixed_budget_bytes = 1;  // nothing fits
  BatchScheduler sched(opts);
  sched.start();
  for (std::uint64_t i = 1; i <= 4; ++i) {
    auto j = make_job(i, 64, 64, 64, 1, 0);
    sched.submit(j);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  sched.stop();  // must return; regression test for dispatcher shutdown
  EXPECT_EQ(sched.submitted(), 4u);
  EXPECT_LE(sched.completed(), sched.submitted());
  SUCCEED();
}

TEST(SchedulerTest, GpuResultsMatchCpuReference) {
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
    GTEST_SKIP() << "no CUDA GPU available";
  }
  for (const char* impl : {"tiled", "cublas"}) {
    SCOPED_TRACE(impl);
    BatchScheduler::Options opts;
    opts.impl = impl;
    opts.num_workers = 2;
    opts.retain_results = true;
    opts.fixed_budget_bytes = 1ull << 30;
    BatchScheduler sched(opts);
    sched.start();

    const std::vector<InferenceJob> jobs = {
        make_job(1, 64, 64, 32, 2, 1),
        make_job(2, 32, 48, 64, 1, 0),
        make_job(3, 17, 33, 25, 3, 2),  // non-multiples of the 32x32 tile
    };
    std::vector<std::uint64_t> seqs;
    for (const auto& j : jobs) seqs.push_back(sched.submit(j));

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (sched.completed() < jobs.size() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    sched.stop();
    ASSERT_EQ(sched.completed(), jobs.size())
        << "jobs did not complete within the deadline";

    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const auto& job = jobs[i];
      const std::size_t sa = static_cast<std::size_t>(job.m) * job.k * job.batch;
      const std::size_t sb = static_cast<std::size_t>(job.k) * job.n * job.batch;
      const std::size_t sc = static_cast<std::size_t>(job.m) * job.n * job.batch;
      std::vector<float> a(sa);
      std::vector<float> b(sb);
      gbis::fill_matrices(job, a.data(), b.data());
      std::vector<float> got;
      ASSERT_TRUE(sched.get_result(seqs[i], got))
          << "missing result for job " << job.id;
      ASSERT_EQ(got.size(), sc);
      std::vector<float> ref(sc);
      cpu_reference(job, a, b, ref);
      double max_err = 0.0;
      for (std::size_t e = 0; e < sc; ++e) {
        max_err = std::max(max_err,
                           std::fabs(static_cast<double>(got[e]) - ref[e]));
      }
      EXPECT_LT(max_err, 1e-3) << "job " << job.id << " impl " << impl;
    }
  }
}

}  // namespace
