#include "core/BenchmarkRunner.h"
#include "benchmarks/CacheBench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Bf16Bench.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/Fp4Bench.h"
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/Fp8Bench.h"
#include "benchmarks/Int4Bench.h"
#include "benchmarks/Int8Bench.h"
#include "benchmarks/MemBandwidthBench.h"
#include "benchmarks/RayAnyHitBench.h"
#include "benchmarks/RayASBuildBench.h"
#include "benchmarks/RayDivergenceBench.h"
#include "benchmarks/RayIncoherentBench.h"
#include "benchmarks/RayMaterialDivergenceBench.h"
#include "benchmarks/RayPathTracingBench.h"
#include "benchmarks/RayPayloadBench.h"
#include "benchmarks/RayProceduralBench.h"
#include "benchmarks/RayTracingBench.h"
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
                                 bool verbose, bool debug, bool dumpGeometry)
    : contexts(contexts), verbose(verbose), debug(debug),
      dumpGeometry(dumpGeometry) {
  for (auto *context : contexts) {
    context->setVerbose(verbose);
  }
  discoverBenchmarks();
  formatter = std::make_unique<ResultFormatter>();
}

BenchmarkRunner::~BenchmarkRunner() {}

std::vector<std::string> BenchmarkRunner::getAvailableBenchmarks() const {
  std::vector<std::string> names;
  for (const auto &bench : benchmarks) {
    names.push_back(bench->GetName());
  }
  return names;
}

const std::vector<ResultData>& BenchmarkRunner::getResults() const {
  return formatter->getResults();
}

void BenchmarkRunner::discoverBenchmarks() {
  benchmarks.push_back(std::make_unique<Fp64Bench>());
  benchmarks.push_back(std::make_unique<Fp32Bench>());
  benchmarks.push_back(std::make_unique<Fp16Bench>());
  benchmarks.push_back(std::make_unique<Bf16Bench>());
  benchmarks.push_back(std::make_unique<Fp8Bench>());
  // benchmarks.push_back(std::make_unique<Fp6Bench>()); // Temporarily disabled
  benchmarks.push_back(std::make_unique<Fp4Bench>());
  benchmarks.push_back(std::make_unique<Int8Bench>());
  benchmarks.push_back(std::make_unique<Int4Bench>());
  benchmarks.push_back(std::make_unique<MemBandwidthBench>());
  benchmarks.push_back(std::make_unique<SysMemBandwidthBench>());
  benchmarks.push_back(std::make_unique<SysMemLatencyBench>());
  benchmarks.push_back(std::make_unique<RayTracingBench>());
  benchmarks.push_back(std::make_unique<RayDivergenceBench>());
  benchmarks.push_back(std::make_unique<RayAnyHitBench>());
  benchmarks.push_back(std::make_unique<RayIncoherentBench>());
  benchmarks.push_back(std::make_unique<RayPayloadBench>());
  benchmarks.push_back(std::make_unique<RayASBuildBench>());
  benchmarks.push_back(std::make_unique<RayProceduralBench>());
  benchmarks.push_back(std::make_unique<RayMaterialDivergenceBench>());
  benchmarks.push_back(std::make_unique<RayPathTracingBench>());

  // Cache Bandwidth
  const size_t l0_size = 16 * 1024; // 16KB L0 cache
  std::vector<uint32_t> l0_init(l0_size / sizeof(uint32_t));
  std::iota(l0_init.begin(), l0_init.end(), 0);

  // Cache Bandwidth is currently difficult to measure reliably because shader compilers
  // aggressively optimize out the memory reading loops via Dead-Code Elimination. 
  // We've disabled these by default until a more robust measurement technique is implemented.
  /*
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L0 Cache Bandwidth", "GB/s", l0_size, "cache_bw_robust", l0_init,
      std::vector<std::string>{"l0b"}, 0));

  // Define target cache sizes for isolation
  const size_t l1_size = 128 * 1024;       // 128KB
  const size_t l2_size = 4 * 1024 * 1024;  // 4MB
  const size_t l3_size = 64 * 1024 * 1024; // 64MB

  // For cachebw_l1 (L1 cache), allocate 2MB (enough for the access pattern)
  size_t cachebw_l1_size = 2 * 1024 * 1024;
  std::vector<uint32_t> l1_bw_init(cachebw_l1_size / sizeof(uint32_t), 1);

  // L1 Cache Bandwidth
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L1 Cache Bandwidth", "GB/s", l1_size, "cache_bw_robust",
      std::vector<uint32_t>{}, std::vector<std::string>{"l1b"}, 1));

  // L2 Cache Bandwidth
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L2 Cache Bandwidth", "GB/s", l2_size, "cache_bw_robust",
      std::vector<uint32_t>{}, std::vector<std::string>{"l2b"}, 2));

  // L3 Cache Bandwidth
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L3 Cache Bandwidth", "GB/s", l3_size, "cache_bw_robust",
      std::vector<uint32_t>{}, std::vector<std::string>{"l3b"}, 3));
  */

  // We still need the sizes for latency tests
  const size_t l1_size = 128 * 1024;       // 128KB
  const size_t l2_size = 4 * 1024 * 1024;  // 4MB
  const size_t l3_size = 64 * 1024 * 1024; // 64MB

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

  // Determine which requested names match at least one benchmark, using the
  // same matching rule as the run loops below (substring of the benchmark
  // name, case-insensitive, or exact alias match).
  unmatchedBenchmarks.clear();
  numBenchmarksRun = 0;
  for (size_t i = 0; i < lower_benchmarks_to_run.size(); ++i) {
    const std::string &run_name = lower_benchmarks_to_run[i];
    bool matched = false;
    for (const auto &bench : benchmarks) {
      // Replicate the name decoration used by the run loops below:
      // benchmarks named "Performance" are matched as
      // "Performance (<subcategory>)".
      std::string bench_name_lower = to_lower(bench->GetName());
      if (bench_name_lower == "performance") {
        bench_name_lower +=
            " (" + to_lower(std::string(bench->GetSubCategory())) + ")";
      }
      if (bench_name_lower.find(run_name) != std::string::npos) {
        matched = true;
        break;
      }
      for (const auto &alias : bench->GetAliases()) {
        if (to_lower(alias) == run_name) {
          matched = true;
          break;
        }
      }
      if (matched)
        break;
    }
    if (!matched) {
      unmatchedBenchmarks.push_back(benchmarks_to_run[i]);
    }
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

      if (!alreadyCounted && context->isAvailable()) {
        totalAvailable += context->getDevices().size();
        countedBackends.push_back(context->getBackend());
      }
    }

    if (verbose) {
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
    }

    if (hasDeviceBenchmarks && verbose) {
      std::cout << "Selected execution targets:" << std::endl;
    }

    for (auto *context : contexts) {
      if (!context->isAvailable())
        continue;
      try {
        DeviceInfo info = context->getCurrentDeviceInfo();
        if (verbose) {
            std::cout << " [Device " << context->getSelectedDeviceIndex() << "] "
                      << info.name << " ("
                      << ComputeBackendFactory::getBackendName(
                             context->getBackend())
                      << ")" << std::endl;
            std::cout << "  - VRAM:         "
                      << static_cast<int>(std::round(info.memorySize /
                                                     (1024.0 * 1024.0 * 1024.0)))
                      << " GB" << std::endl;
            std::cout << "  - Subgroup:     " << info.subgroupSize << " threads"
                      << std::endl;
            std::cout << "  - Shared Memory: "
                      << (info.maxComputeSharedMemorySize / 1024) << " KB"
                      << std::endl;
            std::cout << std::endl;
        }

        // Calculate total expected kernels for progress bar
        uint32_t totalKernels = 0;
        for (auto &bench : benchmarks) {
          bool should_run = false;
          if (benchmarks_to_run.empty()) {
            should_run = true;
          } else {
            std::string bench_name_lower = to_lower(bench->GetName());
            if (bench_name_lower == "performance") {
              bench_name_lower +=
                  " (" + to_lower(std::string(bench->GetSubCategory())) + ")";
            }
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
          if (should_run && bench->IsSupported(info, context) &&
              bench->IsDeviceDependent()) {
            totalKernels += bench->GetExpectedKernelCount();
          }
        }
        context->setExpectedKernelCount(totalKernels);

        // Shared selection predicate, applying the "Performance
        // (subcategory)" name decoration used by the pre-scan and the
        // kernel-count loop above.
        auto isSelected = [&](IBenchmark *b) {
          if (lower_benchmarks_to_run.empty())
            return true;
          std::string bench_name_lower = to_lower(b->GetName());
          if (bench_name_lower == "performance") {
            bench_name_lower +=
                " (" + to_lower(std::string(b->GetSubCategory())) + ")";
          }
          for (const auto &run_name : lower_benchmarks_to_run) {
            if (bench_name_lower.find(run_name) != std::string::npos)
              return true;
            for (const auto &alias : b->GetAliases()) {
              if (to_lower(alias) == run_name)
                return true;
            }
          }
          return false;
        };

        // Phase 1: set up (compile kernels, allocate buffers) ALL selected
        // benchmarks up front, so shader compilation is never interleaved
        // with timed runs.
        std::cout << "Preparing benchmarks (compiling kernels, uploading "
                     "data, building acceleration structures)..."
                  << std::endl;
        std::vector<IBenchmark *> runnable;
        for (auto &bench : benchmarks) {
          bool should_run = isSelected(bench.get());

          if (should_run && bench->IsSupported(info, context)) {
            if (dumpGeometry) {
              bench->DumpGeometry();
            }
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
              runnable.push_back(bench.get());
            } catch (const std::exception &e) {
              if (verbose) {
                  std::cerr << "Error setting up " << bench->GetName() << ": "
                            << e.what() << std::endl;
              }
              try {
                bench->Teardown();
              } catch (...) {
                // Ignore errors during cleanup
              }
            }
          } else if (should_run && bench->IsDeviceDependent()) {
            // Benchmark was selected but is not supported on this
            // device/backend (e.g. missing hardware capability). Report it
            // explicitly as UNSUPPORTED instead of skipping silently.
            ResultData result_data;
            result_data.backendName =
                ComputeBackendFactory::getBackendName(context->getBackend());
            result_data.deviceName = info.name;
            result_data.benchmarkName = bench->GetName();
            result_data.metric = "";
            result_data.operations = 0;
            result_data.time_ms = 0;
            result_data.isEmulated = false;
            result_data.isUnsupported = true;
            result_data.supportNote = bench->GetSupportNote();
            switch (bench->GetSupportLimitation()) {
            case IBenchmark::SupportLimitation::kHardware:
              result_data.supportCategory = "hardware";
              break;
            case IBenchmark::SupportLimitation::kApi:
              result_data.supportCategory = "api";
              break;
            case IBenchmark::SupportLimitation::kToolchain:
              result_data.supportCategory = "toolchain";
              break;
            default:
              break;
            }
            result_data.component = bench->GetComponent(0);
            result_data.subcategory = bench->GetSubCategory(0);
            result_data.maxWorkGroupSize = info.maxWorkGroupSize;
            result_data.deviceIndex = context->getSelectedDeviceIndex();
            result_data.configIndex = 0;
            result_data.sortWeight = bench->GetSortWeight();

            formatter->addResult(result_data);
            // Not counted in numBenchmarksRun: nothing was measured.
            if (onResult) {
                onResult(result_data);
            }
          }
        }

        // Phase 2: timed runs. All kernels are already compiled above, so
        // no compilation overlaps the measurements. The leading newline
        // terminates the \r progress-bar line from phase 1.
        std::cout << "\nPreparation complete. Running benchmarks..."
                  << std::endl;
        for (auto *bench : runnable) {
          try {
              uint32_t num_configs = bench->GetNumConfigs();

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
                }

                // Warmup (not counted): ramp GPU clocks and fill caches
                // before the measurement window starts. Skipped for
                // latency (ns) benchmarks: they are single-thread pointer
                // chases whose ns-per-step is extremely clock-sensitive,
                // and pre-heating the GPU measurably distorts them.
                if (std::string(bench->GetMetric(i)) != "ns") {
                  auto warmup_start =
                      std::chrono::high_resolution_clock::now();
                  double warmup_ms = 0;
                  while (warmup_ms < 250.0) {
                    bench->Run(i);
                    context->waitIdle();
                    auto now = std::chrono::high_resolution_clock::now();
                    warmup_ms =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            now - warmup_start)
                            .count() /
                        1e6;
                  }
                }

                // Timed run with adaptive dispatch batching: submit several
                // dispatches before each waitIdle so the GPU is never left
                // idle while the CPU records the next command buffer.
                double total_time_ms = 0;
                uint64_t total_invocations = 0;
                uint32_t batch = 1;
                auto bench_start = std::chrono::high_resolution_clock::now();
                while (total_time_ms < 2500) {
                  auto iter_start =
                      std::chrono::high_resolution_clock::now();
                  for (uint32_t j = 0; j < batch; ++j) {
                    bench->Run(i);
                  }
                  context->waitIdle();
                  auto iter_end =
                      std::chrono::high_resolution_clock::now();
                  double iter_ms =
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          iter_end - iter_start)
                          .count() /
                      1e6;
                  if (verbose && iter_ms > 500.0) {
                    std::cerr
                        << "\n[WARNING] Dispatch batch took " << iter_ms
                        << " ms — approaching amdgpu TDR timeout!" << std::endl;
                  }
                  if (iter_ms > 3000.0) {
                    std::cerr
                        << "\n[ABORT] Dispatch batch took " << iter_ms
                        << " ms — aborting benchmark to avoid system crash."
                        << std::endl;
                    break;
                  }
                  total_invocations += batch;
                  // Grow the batch until one batch occupies >= ~25 ms of GPU
                  // time (cap 64), keeping the pipeline fed for short kernels.
                  if (iter_ms < 25.0 && batch < 64 && !getenv("GPUBENCH_NO_BATCH")) {
                    batch = (batch * 2 > 64) ? 64 : batch * 2;
                  }
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
                result_data.metric = bench->GetMetric(i);
                result_data.operations =
                    bench_result.operations * total_invocations;
                result_data.time_ms = total_time_ms;
                result_data.isEmulated = bench->IsEmulated(i);
                result_data.component = bench->GetComponent(i);
                result_data.subcategory = bench->GetSubCategory(i);
                result_data.maxWorkGroupSize = info.maxWorkGroupSize;
                result_data.deviceIndex = context->getSelectedDeviceIndex();
                result_data.configIndex = i;
                result_data.sortWeight = bench->GetSortWeight();

                formatter->addResult(result_data);
                numBenchmarksRun++;
                if (onResult) {
                    onResult(result_data);
                }
              }

              bench->Teardown();
              // Let GPU clocks/power state settle before the next
              // benchmark. NOTE: the necessity of this delay is unproven —
              // the test that suggested it was confounded by a competing
              // GPU process. Keep it; removing it is untested.
              std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            } catch (const std::exception &e) {
              if (verbose) {
                  std::cerr << "Error running " << bench->GetName() << ": "
                            << e.what() << std::endl;
              }
              // Make sure to clean up
              try {
                bench->Teardown();
              } catch (...) {
                // Ignore errors during cleanup
              }
            }
        }
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
            result_data.metric = bench->GetMetric(i);
            result_data.operations =
                bench_result.operations * total_invocations;
            result_data.time_ms = total_time_ms;
            result_data.isEmulated = false;
            result_data.component = bench->GetComponent(i);
            result_data.subcategory = bench->GetSubCategory(i);
            result_data.maxWorkGroupSize = 0;
            result_data.deviceIndex = 0xFFFFFFFF;
            result_data.configIndex = i;
            result_data.sortWeight = bench->GetSortWeight();

            formatter->addResult(result_data);
            numBenchmarksRun++;
            if (onResult) {
                onResult(result_data);
            }
          }
          bench->Teardown();
        } catch (const std::exception &e) {
          if (verbose) {
              std::cerr << "Error running " << bench->GetName() << ": " << e.what()
                        << std::endl;
          }
          try {
            bench->Teardown();
          } catch (...) {
          }
        }
      }
    }
  }
  if (verbose) {
      std::cout << "\r\033[K" << std::flush;
  }
  if (!onResult) {
      formatter->print();
  }
}
