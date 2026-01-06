#include "core/BenchmarkRunner.h"
#include "benchmarks/CacheBench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/Fp4Bench.h"
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/Fp8Bench.h"
#include "benchmarks/Int4Bench.h"
#include "benchmarks/Int8Bench.h"
#include "benchmarks/MemBandwidthBench.h"
#include "benchmarks/SysMemBandwidthBench.h"
#include "benchmarks/SysMemLatencyBench.h"
#include "core/ComputeBackendFactory.h"
#include "core/ResultFormatter.h"
#include "utils/KernelPath.h"
// #include "benchmarks/Fp6Bench.h" // Temporarily disabled
#include <algorithm>
#include <chrono>
#include <iostream>
#include <locale>
#include <numeric>
#include <random>
#include <string>
#include <thread>

// Helper function to create a shuffled index array for pointer chasing
std::vector<uint32_t> create_shuffled_indices(size_t size) {
  std::vector<uint32_t> indices(size);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 g(1337); // Use a fixed seed for reproducibility
  std::shuffle(indices.begin(), indices.end(), g);
  return indices;
}

BenchmarkRunner::BenchmarkRunner(const std::vector<IComputeContext *> &contexts,
                                 bool verbose, bool debug)
    : contexts(contexts), verbose(verbose), debug(debug) {
  discoverBenchmarks();
  formatter = std::make_unique<ResultFormatter>();
}

BenchmarkRunner::~BenchmarkRunner() {}

std::vector<std::string> BenchmarkRunner::getAvailableBenchmarks() const {
  std::vector<std::string> names;
  for (const auto &bench : benchmarks) {
    std::string name = bench->GetName();
    if (name == "Performance") {
      name += " (" + std::string(bench->GetSubCategory()) + ")";
    }
    names.push_back(name);
  }
  return names;
}

void BenchmarkRunner::discoverBenchmarks() {
  benchmarks.push_back(std::make_unique<Fp64Bench>());
  benchmarks.push_back(std::make_unique<Fp32Bench>());
  benchmarks.push_back(std::make_unique<Fp16Bench>());
  benchmarks.push_back(std::make_unique<Fp8Bench>());
  // benchmarks.push_back(std::make_unique<Fp6Bench>()); // Temporarily disabled
  benchmarks.push_back(std::make_unique<Fp4Bench>());
  benchmarks.push_back(std::make_unique<Int8Bench>());
  benchmarks.push_back(std::make_unique<Int4Bench>());
  benchmarks.push_back(std::make_unique<MemBandwidthBench>());
  benchmarks.push_back(std::make_unique<SysMemBandwidthBench>());
  benchmarks.push_back(std::make_unique<SysMemLatencyBench>());

  // Cache Bandwidth
  const size_t l0_size = 16 * 1024; // 16KB L0 cache
  std::vector<uint32_t> l0_init(l0_size / sizeof(uint32_t));
  std::iota(l0_init.begin(), l0_init.end(), 0);

  benchmarks.push_back(std::make_unique<CacheBench>(
      "L0 Cache Bandwidth", "GB/s", l0_size, "l0_cache_bandwidth", l0_init,
      std::vector<std::string>{"l0b"}, 0));

  // Define target cache sizes for isolation
  const size_t l1_size = 128 * 1024;       // 128KB
  const size_t l2_size = 4 * 1024 * 1024;  // 4MB
  const size_t l3_size = 64 * 1024 * 1024; // 64MB

  // Cache bandwidth kernels use float4 arrays and access large index ranges
  // We need to allocate enough space based on the dispatch pattern (65536
  // workgroups * 256 threads) cachebw_l1: max index = 65536 * 2 + 1 = ~131K
  // float4 elements = ~2MB cachebw_l2: max index = 65536 * 256 + 255 = ~16.7M
  // float4 elements = ~268MB cachebw_l3: max index = 65536 * 8192 + 255*32+31 =
  // ~537M float4 elements = ~8.6GB (too large!)

  // For cachebw_l1 (L1 cache), allocate 2MB (enough for the access pattern)
  size_t cachebw_l1_size = 2 * 1024 * 1024;
  std::vector<uint32_t> l1_bw_init(cachebw_l1_size / sizeof(uint32_t), 1);

  // L1 Cache Bandwidth
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L1 Cache Bandwidth", "GB/s", l1_size, "cachebw_l1",
      std::vector<uint32_t>{}, std::vector<std::string>{"l1b"}, 1));

  // L2 Cache Bandwidth
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L2 Cache Bandwidth", "GB/s", l2_size, "cachebw_l2",
      std::vector<uint32_t>{}, std::vector<std::string>{"l2b"}, 2));

  // L3 Cache Bandwidth
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L3 Cache Bandwidth", "GB/s", l3_size, "cachebw_l3",
      std::vector<uint32_t>{}, std::vector<std::string>{"l3b"}, 3));

  // Cache Latency
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L0 Cache Latency", "ns", l0_size, "l0_cache_latency",
      create_shuffled_indices(l0_size / sizeof(uint32_t)),
      std::vector<std::string>{"l0l"}, 0));
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L1 Cache Latency", "ns", l1_size, "cache_latency",
      create_shuffled_indices(l1_size / sizeof(uint32_t)),
      std::vector<std::string>{"l1l"}));
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L2 Cache Latency", "ns", l2_size, "cache_latency",
      create_shuffled_indices(l2_size / sizeof(uint32_t)),
      std::vector<std::string>{"l2l"}));
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L3 Cache Latency", "ns", l3_size, "cache_latency",
      create_shuffled_indices(l3_size / sizeof(uint32_t)),
      std::vector<std::string>{"l3l"}));
}

struct BenchmarkResultRow {
  std::string testName;
  double performance;
  std::string unit;
};

// Helper to lowercase a string
static std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

void BenchmarkRunner::run(const std::vector<std::string> &benchmarks_to_run) {
  std::vector<std::string> lower_benchmarks_to_run;
  for (const auto &b : benchmarks_to_run) {
    lower_benchmarks_to_run.push_back(to_lower(b));
  }

  int totalSelected = contexts.size();
  int totalAvailable = 0;

  // Check if we need to run any device-dependent benchmarks
  bool hasDeviceBenchmarks = false;
  for (const auto &bench : benchmarks) {
    if (!bench->IsDeviceDependent())
      continue;

    bool should_run = false;
    if (benchmarks_to_run.empty()) {
      should_run = true;
    } else {
      std::string bench_name = bench->GetName();
      if (bench_name == "Performance") {
        bench_name += " (" + std::string(bench->GetSubCategory()) + ")";
      }
      std::string bench_name_lower = to_lower(bench_name);
      auto aliases = bench->GetAliases();
      for (const auto &run_name : lower_benchmarks_to_run) {
        if (bench_name_lower.find(run_name) != std::string::npos) {
          should_run = true;
          break;
        }
        for (const auto &alias : aliases) {
          if (to_lower(alias) == run_name) {
            should_run = true;
            break;
          }
        }
      }
    }

    if (should_run) {
      hasDeviceBenchmarks = true;
      break;
    }
  }

  std::vector<ComputeBackend> countedBackends;

  if (hasDeviceBenchmarks) {
    for (auto *context : contexts) {
      bool alreadyCounted = false;
      for (auto b : countedBackends) {
        if (b == context->getBackend()) {
          alreadyCounted = true;
          break;
        }
      }

      if (!alreadyCounted) {
        totalAvailable += context->getDevices().size();
        countedBackends.push_back(context->getBackend());
      }
    }

    std::cout
        << "==============================================================="
           "================="
        << std::endl;
    std::cout
        << "   ______ ______  _    _  ____   ______  _   _   _____  _    _"
        << std::endl;
    std::cout
        << "  |  ____|  __  || |  | ||  _ \\ |  ____|| \\ | | / ____|| |  | |"
        << std::endl;
    std::cout
        << "  | |  __| |__) || |  | || |_) || |____ |  \\| || |     | |__| |"
        << std::endl;
    std::cout
        << "  | | |_ |  ___/ | |  | ||  _ < |  ____|| . ` || |     |  __  |"
        << std::endl;
    std::cout
        << "  | |__| | |     | |__| || |_) || |____ | |\\  || |____ | |  | |"
        << std::endl;
    std::cout
        << "  \\______|_|      \\____/ |____/ |______||_| \\_| \\_____||_|  |_|"
        << std::endl;
    std::cout
        << "==============================================================="
           "================="
        << std::endl;
    std::cout << std::endl;

    if (hasDeviceBenchmarks) {
      std::cout << "Selected execution targets:" << std::endl;
    }

    for (auto *context : contexts) {
      try {
        DeviceInfo info = context->getCurrentDeviceInfo();
        std::cout << " [Device " << context->getSelectedDeviceIndex() << "] "
                  << info.name << " ("
                  << ComputeBackendFactory::getBackendName(
                         context->getBackend())
                  << ")" << std::endl;
        if (verbose) {
          std::cout << "  - VRAM:         "
                    << static_cast<int>(std::round(info.memorySize /
                                                   (1024.0 * 1024.0 * 1024.0)))
                    << " GB" << std::endl;
          std::cout << "  - Subgroup:     " << info.subgroupSize << " threads"
                    << std::endl;
          std::cout << "  - Shared Memory: "
                    << (info.maxComputeSharedMemorySize / 1024) << " KB"
                    << std::endl;
        }
        std::cout << std::endl;

        for (auto &bench : benchmarks) {
          bool should_run = false;
          if (benchmarks_to_run.empty()) {
            should_run = true;
          } else {
            std::string bench_name_lower = to_lower(bench->GetName());
            auto aliases = bench->GetAliases();
            for (const auto &run_name : lower_benchmarks_to_run) {
              if (bench_name_lower.find(run_name) != std::string::npos) {
                should_run = true;
                break;
              }
              for (const auto &alias : aliases) {
                if (to_lower(alias) == run_name) {
                  should_run = true;
                  break;
                }
              }
            }
          }

          if (should_run && bench->IsSupported(info, context)) {
            if (!bench->IsDeviceDependent())
              continue; // Run system benchmarks separately

            try {
              // Set debug flag for benchmarks
              if (auto *membw =
                      dynamic_cast<MemBandwidthBench *>(bench.get())) {
                membw->setDebug(debug);
              } else if (auto *cache =
                             dynamic_cast<CacheBench *>(bench.get())) {
                cache->setDebug(debug);
              }

              if (verbose) {
                std::cout << "Setting up " << bench->GetName() << "..."
                          << std::endl;
              }
              bench->Setup(*context, KernelPath::find());

              uint32_t num_configs = bench->GetNumConfigs();

              // For Memory Bandwidth tests in non-verbose mode, print a single
              // summary message For Memory Bandwidth tests in non-verbose mode,
              // we don't need a separate message as the individual configs will
              // print updates via \r
              bool is_membw =
                  (std::string(bench->GetName()) == "Memory Bandwidth");

              for (uint32_t i = 0; i < num_configs; ++i) {
                std::string bench_name = bench->GetName();
                std::string config_name = bench->GetConfigName(i);
                if (!config_name.empty()) {
                  bench_name += " (" + config_name + ")";
                }

                // Only print individual "Running..." messages in verbose mode
                if (verbose) {
                  std::cout << "[D" << context->getSelectedDeviceIndex()
                            << "] Running " << bench_name << "..." << std::endl;
                } else {
                  // Use \r to return to start of line, \033[K to clear until
                  // end of line
                  std::cout << "\r\033[K[D" << context->getSelectedDeviceIndex()
                            << "] Running [" << (i + 1) << "/" << num_configs
                            << "] " << bench_name << "..." << std::flush;
                }

                // Timed run
                double total_time_ms = 0;
                uint64_t total_invocations = 0;
                auto bench_start = std::chrono::high_resolution_clock::now();
                while (total_time_ms < 5000) {
                  bench->Run(i);
                  context->waitIdle();
                  total_invocations++;
                  auto now = std::chrono::high_resolution_clock::now();
                  total_time_ms =
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          now - bench_start)
                          .count() /
                      1e6;
                }

                BenchmarkResult bench_result = bench->GetResult(i);

                ResultData result_data;
                result_data.backendName = ComputeBackendFactory::getBackendName(
                    context->getBackend());
                result_data.deviceName = info.name;
                result_data.benchmarkName = bench_name;
                result_data.metric = bench->GetMetric();
                result_data.operations =
                    bench_result.operations * total_invocations;
                result_data.time_ms = total_time_ms;
                result_data.isEmulated = bench->IsEmulated();
                result_data.component = bench->GetComponent(i);
                result_data.subcategory = bench->GetSubCategory(i);
                result_data.maxWorkGroupSize = info.maxWorkGroupSize;
                result_data.deviceIndex = context->getSelectedDeviceIndex();
                result_data.sortWeight = bench->GetSortWeight();

                formatter->addResult(result_data);
              }

              bench->Teardown();
            } catch (const std::exception &e) {
              std::cerr << "Error running " << bench->GetName() << ": "
                        << e.what() << std::endl;
              // Make sure to clean up
              try {
                bench->Teardown();
              } catch (...) {
                // Ignore errors during cleanup
              }
            }
          }
        }
        if (!verbose)
          std::cout << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "Error processing device: " << e.what() << std::endl;
        continue;
      }
    }
  } // End hasDeviceBenchmarks

  // Run System/Host Benchmarks
  if (!contexts.empty()) {
    bool headerPrinted = false;
    IComputeContext *context = contexts[0]; // Reuse first context for utility

    for (auto &bench : benchmarks) {
      if (bench->IsDeviceDependent())
        continue; // Skip device benchmarks

      bool should_run = false;
      if (benchmarks_to_run.empty()) {
        should_run = true;
      } else {
        std::string bench_name_lower = to_lower(bench->GetName());
        auto aliases = bench->GetAliases();
        for (const auto &run_name : lower_benchmarks_to_run) {
          if (bench_name_lower.find(run_name) != std::string::npos) {
            should_run = true;
            break;
          }
          for (const auto &alias : aliases) {
            if (to_lower(alias) == run_name) {
              should_run = true;
              break;
            }
          }
        }
      }

      if (should_run) {
        if (!headerPrinted) {
          std::cout << " [System] Host CPU" << std::endl;
          if (verbose) {
            std::cout << "  - Threads:      "
                      << std::thread::hardware_concurrency() << std::endl;
          }
          std::cout << std::endl;
          headerPrinted = true;
        }

        try {
          if (verbose) {
            std::cout << "Setting up " << bench->GetName() << "..."
                      << std::endl;
          }
          bench->Setup(*context, KernelPath::find());

          uint32_t num_configs = bench->GetNumConfigs();

          for (uint32_t i = 0; i < num_configs; ++i) {
            std::string bench_name = bench->GetName();
            std::string config_name = bench->GetConfigName(i);
            if (!config_name.empty()) {
              bench_name += " (" + config_name + ")";
            }

            if (verbose) {
              std::cout << "[Sys] Running " << bench_name << "..." << std::endl;
            } else {
              std::cout << "\r\033[K[Sys] Running [" << (i + 1) << "/"
                        << num_configs << "] " << bench_name << "..."
                        << std::flush;
            }

            double total_time_ms = 0;
            uint64_t total_invocations = 0;
            auto bench_start = std::chrono::high_resolution_clock::now();
            while (total_time_ms < 5000) {
              bench->Run(i);
              // context->waitIdle(); // Not needed for system bench usually
              total_invocations++;
              auto now = std::chrono::high_resolution_clock::now();
              total_time_ms =
                  std::chrono::duration_cast<std::chrono::nanoseconds>(
                      now - bench_start)
                      .count() /
                  1e6;
            }

            BenchmarkResult bench_result = bench->GetResult(i);

            ResultData result_data;
            result_data.backendName = "System";
            result_data.deviceName = "Host CPU";
            result_data.benchmarkName = bench_name;
            result_data.metric = bench->GetMetric();
            result_data.operations =
                bench_result.operations * total_invocations;
            result_data.time_ms = total_time_ms;
            result_data.isEmulated = false;
            result_data.component = bench->GetComponent(i);
            result_data.subcategory = bench->GetSubCategory(i);
            result_data.component = bench->GetComponent(i);
            result_data.subcategory = bench->GetSubCategory(i);
            result_data.maxWorkGroupSize = 0;
            result_data.deviceIndex = 0xFFFFFFFF;
            result_data.sortWeight = bench->GetSortWeight();

            formatter->addResult(result_data);
          }
          bench->Teardown();
        } catch (const std::exception &e) {
          std::cerr << "Error running " << bench->GetName() << ": " << e.what()
                    << std::endl;
          try {
            bench->Teardown();
          } catch (...) {
          }
        }
      }
    }
  }
  std::cout << "\r\033[K"
            << std::endl; // Clear the last "Running..." line and move to next
  formatter->print();
}
