#pragma once

#include <memory>

#include "sched/scheduler.hpp"

namespace gbis {

// Minimal HTTP status server (cpp-httplib) exposing:
//   GET /healthz  -> liveness/readiness probe
//   GET /metrics  -> Prometheus text format: queue depth, in-flight jobs,
//                    submitted/completed/failed totals, p95 e2e latency,
//                    mean GPU utilization
class HttpStatusServer {
 public:
  HttpStatusServer(const BatchScheduler& sched, int port);
  ~HttpStatusServer();

  HttpStatusServer(const HttpStatusServer&) = delete;
  HttpStatusServer& operator=(const HttpStatusServer&) = delete;

  // Binds the port and starts the serving thread. Returns false if the port
  // cannot be bound.
  bool start();
  void stop();

  int port() const { return port_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  const BatchScheduler& sched_;
  int port_;
};

}  // namespace gbis
