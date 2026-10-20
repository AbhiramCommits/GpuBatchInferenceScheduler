// Adversarial GPU-memory tenant for scheduler tests.
//
// Repeatedly allocates a set of cudaMalloc chunks, holds them for a while,
// frees them, and sleeps. Used to make NVML free-memory readings change
// underneath scheduler_daemon. Prints one line per allocation cycle; a failed
// cudaMalloc is reported but does not abort the process.

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
  std::size_t chunk_mb = 1024;
  int chunks = 8;
  int hold_ms = 750;
  int sleep_ms = 500;
  int duration_s = 60;
  bool help = false;
};

void print_usage(std::ostream& os) {
  os << "Usage: gpu_memory_hog [options]\n"
     << "  --chunk-mb <N>   MiB per allocation chunk   (default 1024)\n"
     << "  --chunks <N>     chunks held simultaneously (default 8)\n"
     << "  --hold-ms <N>    hold time per cycle        (default 750)\n"
     << "  --sleep-ms <N>   idle time between cycles   (default 500)\n"
     << "  --duration-s <N> total run time             (default 60)\n"
     << "  --help           show this message\n";
}

int parse_int(const std::string& name, const std::string& value) {
  std::size_t pos = 0;
  int v = 0;
  try {
    v = std::stoi(value, &pos);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for " + name + ": " + value);
  }
  if (pos != value.size())
    throw std::runtime_error("invalid integer for " + name + ": " + value);
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
    } else if (arg == "--chunk-mb" || arg.rfind("--chunk-mb=", 0) == 0) {
      opt.chunk_mb = static_cast<std::size_t>(
          parse_int("--chunk-mb", option_value(argc, argv, i, "--chunk-mb")));
    } else if (arg == "--chunks" || arg.rfind("--chunks=", 0) == 0) {
      opt.chunks = parse_int("--chunks", option_value(argc, argv, i, "--chunks"));
    } else if (arg == "--hold-ms" || arg.rfind("--hold-ms=", 0) == 0) {
      opt.hold_ms = parse_int("--hold-ms", option_value(argc, argv, i, "--hold-ms"));
    } else if (arg == "--sleep-ms" || arg.rfind("--sleep-ms=", 0) == 0) {
      opt.sleep_ms = parse_int("--sleep-ms", option_value(argc, argv, i, "--sleep-ms"));
    } else if (arg == "--duration-s" || arg.rfind("--duration-s=", 0) == 0) {
      opt.duration_s = parse_int("--duration-s", option_value(argc, argv, i, "--duration-s"));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (opt.chunk_mb == 0 || opt.chunks <= 0 || opt.hold_ms < 0 || opt.sleep_ms < 0 ||
      opt.duration_s <= 0) {
    throw std::runtime_error("invalid chunk/hold/sleep/duration value");
  }
  return opt;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    if (opt.help) {
      print_usage(std::cout);
      return EXIT_SUCCESS;
    }

    const std::size_t chunk_bytes = opt.chunk_mb * 1024ull * 1024ull;
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::seconds(opt.duration_s);
    std::vector<void*> blocks(static_cast<std::size_t>(opt.chunks), nullptr);

    int cycle = 0;
    while (std::chrono::steady_clock::now() < deadline) {
      ++cycle;
      int allocated = 0;
      for (int i = 0; i < opt.chunks; ++i) {
        void* p = nullptr;
        const cudaError_t err = cudaMalloc(&p, chunk_bytes);
        if (err != cudaSuccess) {
          std::cerr << "[hog] cycle " << cycle << " chunk " << i
                    << " cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
          cudaGetLastError();
          break;
        }
        blocks[static_cast<std::size_t>(i)] = p;
        ++allocated;
      }
      const auto elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::cout << "[hog] t=" << elapsed << "s cycle=" << cycle
                << " allocated_mb=" << allocated * static_cast<int>(opt.chunk_mb) << "\n";
      std::cout.flush();

      std::this_thread::sleep_for(std::chrono::milliseconds(opt.hold_ms));
      for (void*& p : blocks) {
        if (p != nullptr) {
          cudaFree(p);
          p = nullptr;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(opt.sleep_ms));
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
