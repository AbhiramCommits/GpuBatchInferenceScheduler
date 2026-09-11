#include "server/http_server.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>

namespace gbis {

namespace {

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t rank =
      static_cast<std::size_t>(std::ceil(p / 100.0 * v.size()));
  return v[std::max<std::size_t>(1, rank) - 1];
}

}  // namespace

struct HttpStatusServer::Impl {
  httplib::Server server;
  std::thread thread;
};

HttpStatusServer::HttpStatusServer(const BatchScheduler& sched, int port)
    : impl_(std::make_unique<Impl>()), sched_(sched), port_(port) {}

HttpStatusServer::~HttpStatusServer() { stop(); }

bool HttpStatusServer::start() {
  impl_->server.Get("/healthz",
                    [](const httplib::Request&, httplib::Response& res) {
                      res.set_content("ok\n", "text/plain");
                    });

  impl_->server.Get("/metrics", [this](const httplib::Request&,
                                       httplib::Response& res) {
    const std::uint64_t submitted = sched_.submitted();
    const std::uint64_t completed = sched_.completed();
    const std::uint64_t queued = sched_.queued();
    const auto metrics = sched_.metrics();

    std::uint64_t failed = 0;
    std::vector<double> e2e;
    for (const auto& m : metrics) {
      if (m.ok) {
        e2e.push_back(m.e2e_ms);
      } else {
        ++failed;
      }
    }
    const std::uint64_t dispatched = submitted >= completed + queued
                                         ? submitted - completed - queued
                                         : 0;

    std::ostringstream os;
    os << "# HELP gbis_queue_depth Current number of queued jobs.\n"
       << "# TYPE gbis_queue_depth gauge\n"
       << "gbis_queue_depth " << queued << "\n"
       << "# HELP gbis_in_flight_jobs Jobs currently executing on the GPU.\n"
       << "# TYPE gbis_in_flight_jobs gauge\n"
       << "gbis_in_flight_jobs " << dispatched << "\n"
       << "# HELP gbis_submitted_jobs_total Total jobs submitted.\n"
       << "# TYPE gbis_submitted_jobs_total counter\n"
       << "gbis_submitted_jobs_total " << submitted << "\n"
       << "# HELP gbis_completed_jobs_total Total jobs completed.\n"
       << "# TYPE gbis_completed_jobs_total counter\n"
       << "gbis_completed_jobs_total " << completed << "\n"
       << "# HELP gbis_failed_jobs_total Total jobs failed.\n"
       << "# TYPE gbis_failed_jobs_total counter\n"
       << "gbis_failed_jobs_total " << failed << "\n"
       << "# HELP gbis_e2e_latency_p95_ms End-to-end latency p95 in ms.\n"
       << "# TYPE gbis_e2e_latency_p95_ms gauge\n"
       << "gbis_e2e_latency_p95_ms " << percentile(e2e, 95.0) << "\n"
       << "# HELP gbis_gpu_utilization Mean GPU utilization percent (NVML).\n"
       << "# TYPE gbis_gpu_utilization gauge\n"
       << "gbis_gpu_utilization " << sched_.mean_gpu_utilization() << "\n";
    res.set_content(os.str(), "text/plain; version=0.0.4");
  });

  if (!impl_->server.bind_to_port("0.0.0.0", port_)) return false;
  impl_->thread = std::thread([this] { impl_->server.listen_after_bind(); });
  return true;
}

void HttpStatusServer::stop() {
  if (impl_->thread.joinable()) {
    impl_->server.stop();
    impl_->thread.join();
  }
}

}  // namespace gbis
