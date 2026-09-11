#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "sched/gpu_monitor.hpp"
#include "sched/job.hpp"
#include "sched/scheduler.hpp"

namespace {

std::atomic<bool> g_interrupted{false};

extern "C" void on_sigint(int) { g_interrupted.store(true); }

struct Config {
  std::uint64_t jobs = 0;
  double arrival_rate = 0.0;  // jobs/sec, poisson (0 => submit all at once)
  std::string replay_csv;
  int max_batch = 8;
  int workers = 2;
  std::string impl = "tiled";
  std::uint64_t margin_mb = 256;
  std::string metrics_csv = "scheduler_metrics.csv";
  int min_size = 64;
  int max_size = 1024;
  int max_job_batch = 8;
  int priority_levels = 4;
  bool help = false;
};

void print_usage(std::ostream& os) {
  os << "Usage: scheduler_daemon [options]\n"
     << "GPU batch inference scheduler daemon.\n\n"
     << "Options:\n"
     << "  --jobs <N>          synthetic jobs to generate\n"
     << "  --arrival-rate <R>  poisson arrival rate in jobs/sec (default 0)\n"
     << "  --replay <csv>      replay jobs from CSV (id,m,n,k,batch,priority)\n"
     << "  --max-batch <N>     jobs per packed batch (default 8)\n"
     << "  --workers <N>       worker threads / streams (default 2)\n"
     << "  --impl <name>       tiled|cublas (default tiled)\n"
     << "  --margin-mb <MB>    GPU memory safety margin (default 256)\n"
     << "  --metrics-csv <p>   metrics CSV output path\n"
     << "  --min-size <N>      min m/n/k for synthetic jobs (default 64)\n"
     << "  --max-size <N>      max m/n/k for synthetic jobs (default 1024)\n"
     << "  --job-batch-max <N> max internal batch per job (default 8)\n"
     << "  --help              show this message\n";
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

std::uint64_t parse_u64(const std::string& name, const std::string& value) {
  std::size_t pos = 0;
  unsigned long long v;
  try {
    v = std::stoull(value, &pos);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for " + name + ": " + value);
  }
  if (pos != value.size()) {
    throw std::runtime_error("invalid integer for " + name + ": " + value);
  }
  return static_cast<std::uint64_t>(v);
}

double parse_double(const std::string& name, const std::string& value) {
  std::size_t pos = 0;
  double v;
  try {
    v = std::stod(value, &pos);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid number for " + name + ": " + value);
  }
  if (pos != value.size()) {
    throw std::runtime_error("invalid number for " + name + ": " + value);
  }
  return v;
}

std::string option_value(int argc, char** argv, int& i, const std::string& name) {
  const std::string arg = argv[i];
  const std::string prefix = name + "=";
  if (arg.compare(0, prefix.size(), prefix) == 0) return arg.substr(prefix.size());
  if (i + 1 < argc) return argv[++i];
  throw std::runtime_error("missing value for " + name);
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      cfg.help = true;
    } else if (arg == "--jobs" || arg.rfind("--jobs=", 0) == 0) {
      cfg.jobs = parse_u64("--jobs", option_value(argc, argv, i, "--jobs"));
    } else if (arg == "--arrival-rate" || arg.rfind("--arrival-rate=", 0) == 0) {
      cfg.arrival_rate =
          parse_double("--arrival-rate", option_value(argc, argv, i, "--arrival-rate"));
    } else if (arg == "--replay" || arg.rfind("--replay=", 0) == 0) {
      cfg.replay_csv = option_value(argc, argv, i, "--replay");
    } else if (arg == "--max-batch" || arg.rfind("--max-batch=", 0) == 0) {
      cfg.max_batch = parse_int("--max-batch", option_value(argc, argv, i, "--max-batch"));
    } else if (arg == "--workers" || arg.rfind("--workers=", 0) == 0) {
      cfg.workers = parse_int("--workers", option_value(argc, argv, i, "--workers"));
    } else if (arg == "--impl" || arg.rfind("--impl=", 0) == 0) {
      cfg.impl = option_value(argc, argv, i, "--impl");
    } else if (arg == "--margin-mb" || arg.rfind("--margin-mb=", 0) == 0) {
      cfg.margin_mb = parse_u64("--margin-mb", option_value(argc, argv, i, "--margin-mb"));
    } else if (arg == "--metrics-csv" || arg.rfind("--metrics-csv=", 0) == 0) {
      cfg.metrics_csv = option_value(argc, argv, i, "--metrics-csv");
    } else if (arg == "--min-size" || arg.rfind("--min-size=", 0) == 0) {
      cfg.min_size = parse_int("--min-size", option_value(argc, argv, i, "--min-size"));
    } else if (arg == "--max-size" || arg.rfind("--max-size=", 0) == 0) {
      cfg.max_size = parse_int("--max-size", option_value(argc, argv, i, "--max-size"));
    } else if (arg == "--job-batch-max" || arg.rfind("--job-batch-max=", 0) == 0) {
      cfg.max_job_batch = parse_int("--job-batch-max", option_value(argc, argv, i, "--job-batch-max"));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (cfg.max_batch <= 0) throw std::runtime_error("--max-batch must be positive");
  if (cfg.workers <= 0) throw std::runtime_error("--workers must be positive");
  if (cfg.impl != "tiled" && cfg.impl != "cublas") {
    throw std::runtime_error("--impl must be tiled|cublas");
  }
  if (cfg.arrival_rate < 0.0) throw std::runtime_error("--arrival-rate must be >= 0");
  if (cfg.min_size <= 0 || cfg.max_size < cfg.min_size) {
    throw std::runtime_error("invalid --min-size/--max-size");
  }
  if (cfg.jobs > 0 && !cfg.replay_csv.empty()) {
    throw std::runtime_error("use either --jobs or --replay, not both");
  }
  if (cfg.jobs == 0 && cfg.replay_csv.empty()) {
    throw std::runtime_error("specify --jobs <N> or --replay <csv>");
  }
  return cfg;
}

gbis::InferenceJob make_job(std::uint64_t id, const Config& cfg,
                            std::mt19937& rng) {
  gbis::InferenceJob job;
  job.id = id;
  std::uniform_int_distribution<int> size(cfg.min_size, cfg.max_size);
  job.m = size(rng);
  job.n = size(rng);
  job.k = size(rng);
  std::uniform_int_distribution<int> batch(1, cfg.max_job_batch);
  job.batch = batch(rng);
  std::uniform_int_distribution<int> prio(0, cfg.priority_levels - 1);
  job.priority = prio(rng);
  return job;
}

bool parse_replay(const std::string& path, std::vector<gbis::InferenceJob>& out) {
  std::ifstream f(path);
  if (!f) {
    std::cerr << "error: cannot open replay CSV: " << path << "\n";
    return false;
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    std::string tok[6];
    for (auto& t : tok) {
      if (!std::getline(ss, t, ',')) return false;
    }
    if (tok[0] == "id") continue;  // header row
    gbis::InferenceJob job;
    job.id = parse_u64("replay id", tok[0]);
    job.m = parse_int("replay m", tok[1]);
    job.n = parse_int("replay n", tok[2]);
    job.k = parse_int("replay k", tok[3]);
    job.batch = parse_int("replay batch", tok[4]);
    job.priority = parse_int("replay priority", tok[5]);
    out.push_back(job);
  }
  return !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    if (cfg.help) {
      print_usage(std::cout);
      return EXIT_SUCCESS;
    }

    gbis::GpuMonitor monitor;
    if (!monitor.gpu_present()) {
      std::cerr << "error: no CUDA GPU detected\n";
      return EXIT_FAILURE;
    }

    gbis::BatchScheduler::Options opts;
    opts.max_batch = cfg.max_batch;
    opts.num_workers = cfg.workers;
    opts.impl = cfg.impl;
    opts.safety_margin_bytes = cfg.margin_mb * 1024ull * 1024ull;
    gbis::BatchScheduler sched(opts);

    std::vector<gbis::InferenceJob> replay_jobs;
    if (!cfg.replay_csv.empty() && !parse_replay(cfg.replay_csv, replay_jobs)) {
      return EXIT_FAILURE;
    }

    std::signal(SIGINT, on_sigint);
    sched.start();

    std::thread generator;
    if (cfg.replay_csv.empty()) {
      generator = std::thread([&cfg, &sched] {
        std::mt19937 rng(1234);
        std::exponential_distribution<double> inter_arrival(
            cfg.arrival_rate > 0.0 ? cfg.arrival_rate : 1.0);
        for (std::uint64_t i = 1; i <= cfg.jobs && !g_interrupted.load(); ++i) {
          sched.submit(make_job(i, cfg, rng));
          if (cfg.arrival_rate > 0.0) {
            std::this_thread::sleep_for(
                std::chrono::duration<double>(inter_arrival(rng)));
          }
        }
      });
    } else {
      for (const auto& job : replay_jobs) {
        if (g_interrupted.load()) break;
        sched.submit(job);
      }
    }

    while (!g_interrupted.load()) {
      if (sched.submitted() > 0 && sched.queued() == 0 &&
          sched.completed() >= sched.submitted()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (generator.joinable()) generator.join();
    sched.stop(cfg.metrics_csv);
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
