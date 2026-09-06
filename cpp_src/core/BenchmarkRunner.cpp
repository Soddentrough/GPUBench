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
#include "benchmarks/PixelFillRateBench.h"
#include "benchmarks/RayAnyHitBench.h"
#include "benchmarks/RayASBuildBench.h"
#include "benchmarks/RayDivergenceBench.h"
#include "benchmarks/RayIncoherentBench.h"
#include "benchmarks/RayMaterialDivergenceBench.h"
#include "benchmarks/RayPathTracingBench.h"
#include "benchmarks/RayPayloadBench.h"
#include "benchmarks/RayProceduralBench.h"
#include "benchmarks/RayTracingBench.h"
#include "benchmarks/RaySchedulingBench.h"
#include "benchmarks/SysMemBandwidthBench.h"
#include "benchmarks/SysMemLatencyBench.h"
#include "core/ComputeBackendFactory.h"
#include "core/ResultFormatter.h"
#include "utils/KernelPath.h"
#include "utils/SleepInhibitor.h"
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
                                 bool verbose, bool debug, bool dumpGeometry,
                                 bool dumpRenders, const std::string &scene)
    : contexts(contexts), verbose(verbose), debug(debug),
      dumpGeometry(dumpGeometry), dumpRenders(dumpRenders), sceneName(scene) {
  for (auto *context : contexts) {
    context->setVerbose(verbose);
  }
  discoverBenchmarks();
  formatter = std::make_unique<ResultFormatter>();
}

BenchmarkRunner::~BenchmarkRunner() {}

void BenchmarkRunner::setBounceDepth(uint32_t b) {
  bounceDepth = b;
  for (auto &bench : benchmarks) {
    if (auto *rs = dynamic_cast<RaySchedulingBench *>(bench.get())) {
      rs->SetBounceDepth(b);
    }
  }
}

void BenchmarkRunner::setSamplesPerPixel(uint32_t spp) {
  samplesPerPixel = std::clamp(spp, 1u, 256u);
  for (auto &bench : benchmarks) {
    if (auto *rs = dynamic_cast<RaySchedulingBench *>(bench.get())) {
      rs->SetSamplesPerPixel(samplesPerPixel);
    }
  }
}

std::vector<std::string> BenchmarkRunner::getAvailableBenchmarks() const {
  std::vector<std::string> names;
  for (const auto &bench : benchmarks) {
    std::string name = bench->GetName();
    if (dynamic_cast<RaySchedulingBench *>(bench.get())) {
      name = "RayScheduling";
    }
    if (std::find(names.begin(), names.end(), name) == names.end()) {
      names.push_back(name);
    }
  }
  return names;
}

std::vector<BenchmarkGroupInfo> BenchmarkRunner::getAvailableGroups() const {
  std::vector<BenchmarkGroupInfo> groups = {
    {"Compute", "compute", {"comp"}, "Vector and matrix compute arithmetic (FP64 down to INT4)", {}},
    {"Memory", "memory", {"mem", "cache"}, "Device memory bandwidth and cache latency", {}},
    {"Graphics", "graphics", {"gfx"}, "Complete 3D graphics rendering pipelines (combines Raster and Ray Tracing)", {}},
    {"Raster", "raster", {"rop", "rasterization"}, "Fixed-function rasterization and ROP pixel/blend fill rates (subset of Graphics)", {}},
    {"Ray Tracing", "raytracing", {"rt", "ray", "ray tracing", "ray_tracing"}, "Hardware BVH traversal, intersection, and scheduling architectures (subset of Graphics)", {}},
    {"System", "system", {"sys", "host"}, "Host system memory bandwidth and latency", {}}
  };

  for (const auto &bench : benchmarks) {
    std::string comp = bench->GetComponent();
    std::string name = bench->GetName();
    if (dynamic_cast<RaySchedulingBench *>(bench.get())) {
      name = "RayScheduling";
    }
    if (!bench->IsDeviceDependent() || comp == "System") {
      if (std::find(groups[5].benchmarks.begin(), groups[5].benchmarks.end(), name) == groups[5].benchmarks.end())
        groups[5].benchmarks.push_back(name);
    } else if (comp == "Compute") {
      if (std::find(groups[0].benchmarks.begin(), groups[0].benchmarks.end(), name) == groups[0].benchmarks.end())
        groups[0].benchmarks.push_back(name);
    } else if (comp == "Memory") {
      if (std::find(groups[1].benchmarks.begin(), groups[1].benchmarks.end(), name) == groups[1].benchmarks.end())
        groups[1].benchmarks.push_back(name);
    } else if (comp == "Ray Tracing") {
      if (std::find(groups[2].benchmarks.begin(), groups[2].benchmarks.end(), name) == groups[2].benchmarks.end())
        groups[2].benchmarks.push_back(name);
      if (std::find(groups[4].benchmarks.begin(), groups[4].benchmarks.end(), name) == groups[4].benchmarks.end())
        groups[4].benchmarks.push_back(name);
    } else if (comp == "Graphics" || comp == "Raster") {
      if (std::find(groups[2].benchmarks.begin(), groups[2].benchmarks.end(), name) == groups[2].benchmarks.end())
        groups[2].benchmarks.push_back(name);
      if (std::find(groups[3].benchmarks.begin(), groups[3].benchmarks.end(), name) == groups[3].benchmarks.end())
        groups[3].benchmarks.push_back(name);
    }
  }

  return groups;
}

std::vector<std::string> BenchmarkRunner::expandGroups(const std::vector<std::string> &inputs) const {
  auto normalize = [](const std::string &s) {
    std::string out;
    for (char c : s) {
      if (c != ' ' && c != '_' && c != '-') {
        out.push_back(std::tolower(static_cast<unsigned char>(c)));
      }
    }
    return out;
  };

  auto groups = getAvailableGroups();
  std::vector<std::string> expanded;

  for (const auto &input : inputs) {
    std::string normInput = normalize(input);
    if (normInput.empty()) continue;

    bool matchesExactBench = false;
    for (const auto &bench : benchmarks) {
      if (normalize(bench->GetName()) == normInput) {
        matchesExactBench = true;
        break;
      }
      if (dynamic_cast<RaySchedulingBench *>(bench.get()) && (normInput == "rayscheduling" || normInput == "rayexecutionparadigm")) {
        matchesExactBench = true;
        break;
      }
    }

    bool isGroup = false;
    if (!matchesExactBench) {
      for (const auto &grp : groups) {
        if (normalize(grp.id) == normInput || normalize(grp.name) == normInput) {
          isGroup = true;
        } else {
          for (const auto &alias : grp.aliases) {
            if (normalize(alias) == normInput) {
              isGroup = true;
              break;
            }
          }
        }

        if (isGroup) {
          for (const auto &benchName : grp.benchmarks) {
            if (std::find(expanded.begin(), expanded.end(), benchName) == expanded.end()) {
              expanded.push_back(benchName);
            }
          }
          break;
        }
      }
    }

    if (!isGroup) {
      if (std::find(expanded.begin(), expanded.end(), input) == expanded.end()) {
        expanded.push_back(input);
      }
    }
  }

  return expanded;
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
  benchmarks.push_back(std::make_unique<PixelFillRateBench>());
  benchmarks.push_back(std::make_unique<SysMemBandwidthBench>());
  benchmarks.push_back(std::make_unique<SysMemLatencyBench>());
  // Ray Tracing Acceleration (Real-World Pipeline Order: AS Build -> Primary Rays/Intersection -> Ray Scheduling -> Secondary Rays/Divergence -> Path Tracing -> Payload Pressure)
  benchmarks.push_back(std::make_unique<RayASBuildBench>());
  benchmarks.push_back(std::make_unique<RayTracingBench>());
  benchmarks.push_back(std::make_unique<RayAnyHitBench>());
  benchmarks.push_back(std::make_unique<RayProceduralBench>());
  if (sceneName == "all") {
    auto showroom = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::Showroom);
    showroom->SetBounceDepth(bounceDepth);
    showroom->SetSamplesPerPixel(samplesPerPixel);
    showroom->SetDumpRenders(dumpRenders);
    benchmarks.push_back(std::move(showroom));

    auto indoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::IndoorAtrium);
    indoor->SetBounceDepth(bounceDepth);
    indoor->SetSamplesPerPixel(samplesPerPixel);
    indoor->SetDumpRenders(dumpRenders);
    benchmarks.push_back(std::move(indoor));

    auto outdoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::OutdoorLandscape);
    outdoor->SetBounceDepth(bounceDepth);
    outdoor->SetSamplesPerPixel(samplesPerPixel);
    outdoor->SetDumpRenders(dumpRenders);
    benchmarks.push_back(std::move(outdoor));

    auto forest = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::AAAOutdoorForest);
    forest->SetBounceDepth(bounceDepth);
    forest->SetSamplesPerPixel(samplesPerPixel);
    forest->SetDumpRenders(dumpRenders);
    benchmarks.push_back(std::move(forest));
  } else if (sceneName == "outdoor") {
    auto outdoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::OutdoorLandscape);
    outdoor->SetBounceDepth(bounceDepth);
    outdoor->SetSamplesPerPixel(samplesPerPixel);
    outdoor->SetDumpRenders(dumpRenders);
    benchmarks.push_back(std::move(outdoor));
  } else if (sceneName == "forest" || sceneName == "aaa_forest") {
    auto forest = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::AAAOutdoorForest);
    forest->SetBounceDepth(bounceDepth);
    forest->SetSamplesPerPixel(samplesPerPixel);
    forest->SetDumpRenders(dumpRenders);
    benchmarks.push_back(std::move(forest));
  } else if (sceneName == "showroom") {
    auto showroom = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::Showroom);
    showroom->SetBounceDepth(bounceDepth);
    showroom->SetSamplesPerPixel(samplesPerPixel);
    showroom->SetDumpRenders(dumpRenders);
    benchmarks.push_back(std::move(showroom));
  } else {
    auto indoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::IndoorAtrium);
    indoor->SetBounceDepth(bounceDepth);
    indoor->SetSamplesPerPixel(samplesPerPixel);
    indoor->SetDumpRenders(dumpRenders);
    benchmarks.push_back(std::move(indoor));
  }
  benchmarks.push_back(std::make_unique<RayMaterialDivergenceBench>());
  benchmarks.push_back(std::make_unique<RayIncoherentBench>());
  benchmarks.push_back(std::make_unique<RayDivergenceBench>());
  // Note: Standalone synthetic 16k-triangle grid RayPathTracingBench is retired in favor of
  // Full Scene Path Tracing (Multi-Bounce) on real-world scenes in RaySchedulingBench.
  benchmarks.push_back(std::make_unique<RayPayloadBench>());

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

// Case-insensitive, delimiter-agnostic exact benchmark matching
static bool benchmarkMatches(const IBenchmark *bench, const std::string &run_name) {
  auto normalize = [](const std::string &s) {
    std::string out;
    for (char c : s) {
      if (c != ' ' && c != '_' && c != '-') {
        out.push_back(std::tolower(static_cast<unsigned char>(c)));
      }
    }
    return out;
  };

  std::string normRun = normalize(run_name);
  if (normRun.empty()) return false;

  std::string benchName = bench->GetName();
  if (normalize(benchName) == normRun) return true;

  if (benchName == "Performance") {
    std::string decorated = benchName + " (" + bench->GetSubCategory() + ")";
    if (normalize(decorated) == normRun) return true;
  }

  for (const auto &alias : bench->GetAliases()) {
    if (normalize(alias) == normRun) return true;
  }

  // Ray Scheduling benchmark alias & prefix matching
  if (dynamic_cast<const RaySchedulingBench *>(bench)) {
    if (normRun == "rayscheduling" || normRun == "rayexecutionparadigm") {
      return true;
    }
    // Also match legacy scene-qualified monikers (e.g. "rayschedulingindooratrium")
    if (normRun.rfind("rayscheduling", 0) == 0) {
      return true;
    }
  }

  return false;
}

void BenchmarkRunner::printBanner() {
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

void BenchmarkRunner::initRunConfig(const std::vector<std::string> &benchmarks_to_run) {
  if (runConfigInitialized)
    return;
  runConfigInitialized = true;

  effective_benchmarks = expandGroups(benchmarks_to_run);
  lower_benchmarks_to_run.clear();
  for (const auto &b : effective_benchmarks) {
    lower_benchmarks_to_run.push_back(to_lower(b));
  }

  unmatchedBenchmarks.clear();
  numBenchmarksRun = 0;
  for (size_t i = 0; i < lower_benchmarks_to_run.size(); ++i) {
    const std::string &run_name = lower_benchmarks_to_run[i];
    bool matched = false;
    for (const auto &bench : benchmarks) {
      if (benchmarkMatches(bench.get(), run_name)) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      unmatchedBenchmarks.push_back(effective_benchmarks[i]);
    }
  }
}

void BenchmarkRunner::runForContext(IComputeContext *context,
                                    const std::vector<std::string> &benchmarks_to_run) {
  if (!context || !context->isAvailable())
    return;

  initRunConfig(benchmarks_to_run);

  if (verbose && !bannerPrinted) {
    printBanner();
    bannerPrinted = true;
  }

  context->setVerbose(verbose);

  try {
    DeviceInfo info = context->getCurrentDeviceInfo();
    if (verbose) {
      std::cout << " [Device " << context->getSelectedDeviceIndex() << "] "
                << info.name << " ("
                << ComputeBackendFactory::getBackendName(context->getBackend())
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

    auto isSelected = [&](IBenchmark *b) {
      if (lower_benchmarks_to_run.empty())
        return true;
      for (const auto &run_name : lower_benchmarks_to_run) {
        if (benchmarkMatches(b, run_name))
          return true;
      }
      return false;
    };

    uint32_t totalKernels = 0;
    for (auto &bench : benchmarks) {
      if (isSelected(bench.get()) && bench->IsSupported(info, context) &&
          bench->IsDeviceDependent()) {
        totalKernels += bench->GetExpectedKernelCount();
      }
    }
    context->setExpectedKernelCount(totalKernels);

    std::cout << "Preparing benchmarks (compiling kernels, uploading "
                 "data, building acceleration structures)..."
              << std::endl;
    uint32_t effectiveWidth = renderWidth;
    uint32_t effectiveHeight = renderHeight;
    if (effectiveWidth == 0 || effectiveHeight == 0) {
      if (info.memorySize >= 15ULL * 1024 * 1024 * 1024) {
        effectiveWidth = 3840;
        effectiveHeight = 2160;
      } else if (info.memorySize >= 9ULL * 1024 * 1024 * 1024) {
        effectiveWidth = 2560;
        effectiveHeight = 1440;
      } else {
        effectiveWidth = 1920;
        effectiveHeight = 1080;
      }
      if (verbose) {
        std::cout << "Auto-selected resolution: " << effectiveWidth << "x" << effectiveHeight
                  << " for " << info.name << " (" << (info.memorySize / (1024 * 1024 * 1024))
                  << " GB VRAM)" << std::endl;
      }
    }

    std::vector<IBenchmark *> runnable;
    for (auto &bench : benchmarks) {
      bool should_run = isSelected(bench.get());

      if (should_run && bench->IsSupported(info, context)) {
        if (dumpGeometry) {
          bench->DumpGeometry();
        }
        if (!bench->IsDeviceDependent())
          continue;

        try {
          if (auto *membw = dynamic_cast<MemBandwidthBench *>(bench.get())) {
            membw->setDebug(debug);
          } else if (auto *cache = dynamic_cast<CacheBench *>(bench.get())) {
            cache->setDebug(debug);
          } else if (auto *rs = dynamic_cast<RaySchedulingBench *>(bench.get())) {
            rs->SetBounceDepth(bounceDepth);
            rs->SetSamplesPerPixel(samplesPerPixel);
          }

          if (verbose) {
            std::cout << "Setting up " << bench->GetName() << "..." << std::endl;
          }
          bench->SetResolution(effectiveWidth, effectiveHeight);
          bench->Setup(*context, KernelPath::find());
          runnable.push_back(bench.get());
        } catch (const std::exception &e) {
          std::cerr << "Error setting up " << bench->GetName() << ": "
                    << e.what() << std::endl;
          try {
            bench->Teardown();
          } catch (...) {
          }
        }
      } else if (should_run && bench->IsDeviceDependent()) {
        std::string bname = bench->GetName();
        uint32_t num_unsupported_configs = 1;
        if (bname == "FP8" || bname == "INT4" || bname == "FP16" || bname == "BF16" || bname == "INT8") {
          num_unsupported_configs = 2;
        }

        for (uint32_t ci = 0; ci < num_unsupported_configs; ++ci) {
          ResultData result_data;
          result_data.backendName =
              ComputeBackendFactory::getBackendName(context->getBackend());
          result_data.deviceName = info.name;
          result_data.benchmarkName = (num_unsupported_configs > 1)
              ? (bname + (ci == 0 ? " (Vector)" : " (Matrix)"))
              : bname;
          result_data.metric = "";
          result_data.operations = 0;
          result_data.time_ms = 0;
          result_data.isEmulated = false;
          result_data.isUnsupported = true;
          result_data.supportNote = bench->GetConfigSupportNote(ci, info, context);
          if (result_data.supportNote.empty()) {
            result_data.supportNote = bench->GetSupportNote(info, context);
          }
          switch (bench->GetConfigSupportLimitation(ci, info, context)) {
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
          result_data.component = bench->GetComponent(ci);
          result_data.subcategory = bench->GetSubCategory(ci);
          result_data.maxWorkGroupSize = info.maxWorkGroupSize;
          result_data.deviceIndex = context->getSelectedDeviceIndex();
          result_data.configIndex = ci;
          result_data.sortWeight = bench->GetSortWeight(ci);

          formatter->addResult(result_data);
          numBenchmarksRun++;
          if (onResult) {
            onResult(result_data);
          }
        }
      }
    }

    std::cout << "\nPreparation complete. Running benchmarks..." << std::endl;
    struct BenchmarkTask {
      IBenchmark *bench;
      uint32_t configIndex;
      int sortWeight;
    };

    std::vector<BenchmarkTask> tasks;
    for (auto *bench : runnable) {
      uint32_t num_configs = bench->GetNumConfigs();
      for (uint32_t i = 0; i < num_configs; ++i) {
        if (targetConfig >= 0 && static_cast<int>(i) != targetConfig) {
          continue;
        }
        tasks.push_back({bench, i, bench->GetSortWeight(i)});
      }
    }

    std::stable_sort(tasks.begin(), tasks.end(),
                     [](const BenchmarkTask &a, const BenchmarkTask &b) {
                       return a.sortWeight < b.sortWeight;
                     });

    IBenchmark *prevBench = nullptr;
    for (const auto &task : tasks) {
      auto *bench = task.bench;
      uint32_t i = task.configIndex;

      if (prevBench && prevBench != bench) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
      }
      prevBench = bench;

      std::string bench_name = bench->GetName();
      std::string config_name = bench->GetConfigName(i);
      if (!config_name.empty()) {
        bench_name += " (" + config_name + ")";
      }

      try {
        if (verbose) {
          std::cout << "[D" << context->getSelectedDeviceIndex()
                    << "] Running " << bench_name << "..." << std::endl;
        } else {
          std::cout << "  - [" << ComputeBackendFactory::getBackendName(context->getBackend())
                    << "] Running " << bench_name << "..." << std::flush;
        }

        if (onResult) {
          ResultData start_data;
          start_data.backendName = ComputeBackendFactory::getBackendName(context->getBackend());
          start_data.deviceName = info.name;
          start_data.benchmarkName = bench_name;
          start_data.component = bench->GetComponent(i);
          start_data.subcategory = bench->GetSubCategory(i);
          start_data.metric = bench->GetMetric(i);
          start_data.operations = 0;
          start_data.time_ms = -1.0;
          start_data.isEmulated = false;
          start_data.isUnsupported = false;
          start_data.maxWorkGroupSize = info.maxWorkGroupSize;
          start_data.deviceIndex = context->getSelectedDeviceIndex();
          start_data.configIndex = i;
          start_data.sortWeight = bench->GetSortWeight(i);
          start_data.width = effectiveWidth;
          start_data.height = effectiveHeight;
          onResult(start_data);
        }

        if (!bench->IsConfigSupported(i, info, context)) {
          std::string note = bench->GetConfigSupportNote(i, info, context);
          if (note.empty()) {
            note = bench->GetSupportNote(info, context);
          }
          if (!verbose) {
            if (!note.empty()) {
              std::cout << " Unsupported (" << note << ")." << std::endl;
            } else {
              std::cout << " Unsupported." << std::endl;
            }
          }
          ResultData result_data;
          result_data.backendName = ComputeBackendFactory::getBackendName(context->getBackend());
          result_data.deviceName = info.name;
          result_data.benchmarkName = bench_name;
          result_data.metric = bench->GetMetric(i);
          result_data.operations = 0;
          result_data.time_ms = 0;
          result_data.isEmulated = false;
          result_data.isUnsupported = true;
          result_data.supportNote = note;
          switch (bench->GetConfigSupportLimitation(i, info, context)) {
          case IBenchmark::SupportLimitation::kHardware:
            result_data.supportCategory = "hardware";
            break;
          case IBenchmark::SupportLimitation::kApi:
            result_data.supportCategory = "driver/api";
            break;
          case IBenchmark::SupportLimitation::kToolchain:
            result_data.supportCategory = "toolchain";
            break;
          default:
            result_data.supportCategory = "hardware";
            break;
          }
          result_data.component = bench->GetComponent(i);
          result_data.subcategory = bench->GetSubCategory(i);
          result_data.maxWorkGroupSize = info.maxWorkGroupSize;
          result_data.deviceIndex = context->getSelectedDeviceIndex();
          result_data.configIndex = i;
          result_data.sortWeight = bench->GetSortWeight(i);
          result_data.width = effectiveWidth;
          result_data.height = effectiveHeight;

          formatter->addResult(result_data);
          numBenchmarksRun++;
          if (onResult) {
            onResult(result_data);
          }
          continue;
        }

        double total_time_ms = 0;
        uint64_t total_invocations = 0;

        if (profileSnapshot) {
          bench->Run(i);
          context->waitIdle();

          auto start = std::chrono::high_resolution_clock::now();
          bench->Run(i);
          context->waitIdle();
          auto end = std::chrono::high_resolution_clock::now();
          total_time_ms =
              std::chrono::duration<double, std::milli>(end - start).count();
          total_invocations = 1;
        } else {
          auto start = std::chrono::high_resolution_clock::now();
          bench->Run(i);
          context->waitIdle();
          auto end = std::chrono::high_resolution_clock::now();
          double single_run_ms =
              std::chrono::duration<double, std::milli>(end - start).count();

          // Warmup: Run until the GPU clocks ramp up from idle/sleep to sustained boost clocks.
          // On modern GPUs with dynamic power management (DPM) governors, ramping takes ~250-400ms.
          const double min_warmup_duration_ms = 400.0;
          uint64_t warmup_iters = 3;
          if (single_run_ms > 0.0) {
            warmup_iters = static_cast<uint64_t>(
                std::max(3.0, std::ceil(min_warmup_duration_ms / single_run_ms)));
          }
          warmup_iters = std::min(warmup_iters, static_cast<uint64_t>(200));

          for (uint64_t w = 0; w < warmup_iters; ++w) {
            bench->Run(i);
          }
          context->waitIdle();

          // After warmup, re-measure single run latency at warmed-up clock speeds
          start = std::chrono::high_resolution_clock::now();
          bench->Run(i);
          context->waitIdle();
          end = std::chrono::high_resolution_clock::now();
          single_run_ms =
              std::chrono::duration<double, std::milli>(end - start).count();

          const double target_duration_ms = 250.0;
          uint64_t iterations = 1;
          if (single_run_ms > 0.0) {
            iterations = static_cast<uint64_t>(
                std::max(1.0, std::round(target_duration_ms / single_run_ms)));
          }
          iterations = std::min(iterations, static_cast<uint64_t>(10000));
          iterations = std::max(iterations, static_cast<uint64_t>(1));

          total_invocations = iterations;
          start = std::chrono::high_resolution_clock::now();
          for (uint64_t iter = 0; iter < iterations; ++iter) {
            bench->Run(i);
          }
          context->waitIdle();
          end = std::chrono::high_resolution_clock::now();
          total_time_ms =
              std::chrono::duration<double, std::milli>(end - start).count();
          if (verbose) {
            std::cout << "[TIMING " << bench_name << "] single_run_ms: " << single_run_ms
                      << ", iterations: " << iterations
                      << ", total_time_ms: " << total_time_ms
                      << ", avg_ms: " << (total_time_ms / iterations) << std::endl;
          }
        }

        if (!verbose) {
          std::cout << " Done." << std::endl;
        }

        bool isValid = bench->ValidateResults(i);
        if (!isValid && verbose) {
          std::cerr << " [WARNING] Result validation failed for "
                    << bench_name << std::endl;
        }

        BenchmarkResult bench_result = bench->GetResult(i);
        bench->RecordRunResult(i, total_invocations, total_time_ms);

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
        result_data.supportNote = bench->GetConfigCaveat(i, info, context);
        if (result_data.supportNote.empty() && result_data.isEmulated) {
          result_data.supportNote = "Emulated via software unpack";
        }
        result_data.component = bench->GetComponent(i);
        result_data.subcategory = bench->GetSubCategory(i);
        result_data.maxWorkGroupSize = info.maxWorkGroupSize;
        result_data.deviceIndex = context->getSelectedDeviceIndex();
        result_data.configIndex = i;
        result_data.sortWeight = bench->GetSortWeight(i);
        result_data.width = effectiveWidth;
        result_data.height = effectiveHeight;

        formatter->addResult(result_data);
        numBenchmarksRun++;
        if (onResult) {
          onResult(result_data);
        }
      } catch (const std::exception &e) {
        if (!verbose) {
          std::cout << " Failed (" << e.what() << ")" << std::endl;
        } else {
          std::cerr << "Error running task " << bench_name << ": " << e.what() << std::endl;
        }

        std::string errStr = e.what();
        bool isLost = (errStr.find("DEVICE_LOST") != std::string::npos ||
                       errStr.find("timed out") != std::string::npos ||
                       errStr.find("timeout") != std::string::npos ||
                       errStr.find("context lost") != std::string::npos ||
                       errStr.find("Device lost") != std::string::npos ||
                       errStr.find("result: -4") != std::string::npos);
        if (isLost) {
          std::cerr << "  [CRITICAL] GPU device hung or lost during " << bench_name
                    << ". Aborting remaining tasks on this device." << std::endl;
          break;
        }
      }
    }

    for (auto *bench : runnable) {
      try {
        bench->Teardown();
      } catch (const std::exception &e) {
        if (verbose) {
          std::cerr << "Error tearing down " << bench->GetName() << ": "
                    << e.what() << std::endl;
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Error processing device: " << e.what() << std::endl;
  }
}

void BenchmarkRunner::runHostBenchmarks(const std::vector<std::string> &benchmarks_to_run) {
  initRunConfig(benchmarks_to_run);

  bool headerPrinted = false;
  struct DummyHostContext : public IComputeContext {
    ComputeBackend getBackend() const override { return ComputeBackend::Vulkan; }
    bool isAvailable() const override { return true; }
    const std::vector<DeviceInfo> &getDevices() const override { static std::vector<DeviceInfo> d; return d; }
    DeviceInfo getCurrentDeviceInfo() const override { return {}; }
    uint32_t getSelectedDeviceIndex() const override { return 0; }
    void pickDevice(uint32_t) override {}
    ComputeBuffer createBuffer(size_t, const void *) override { return nullptr; }
    void releaseBuffer(ComputeBuffer) override {}
    void writeBuffer(ComputeBuffer, size_t, size_t, const void *) override {}
    void readBuffer(ComputeBuffer, size_t, size_t, void *) const override {}
    ComputeKernel createKernel(const std::string &, const std::string &, uint32_t) override { return nullptr; }
    void releaseKernel(ComputeKernel) override {}
    void setKernelArg(ComputeKernel, uint32_t, size_t, const void *) override {}
    void setKernelArg(ComputeKernel, uint32_t, ComputeBuffer) override {}
    void setKernelAS(ComputeKernel, uint32_t, AccelerationStructure) override {}
    void dispatch(ComputeKernel, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) override {}
    void waitIdle() override {}
  } dummy;
  IComputeContext *context = contexts.empty() ? &dummy : contexts[0];

  for (auto &bench : benchmarks) {
    if (bench->IsDeviceDependent())
      continue;

    bool should_run = false;
    if (effective_benchmarks.empty()) {
      should_run = true;
    } else {
      for (const auto &run_name : lower_benchmarks_to_run) {
        if (benchmarkMatches(bench.get(), run_name)) {
          should_run = true;
          break;
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
          std::cout << "Setting up " << bench->GetName() << "..." << std::endl;
        }
        bench->SetResolution(renderWidth, renderHeight);
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

          if (onResult) {
            ResultData start_data;
            start_data.backendName = "System";
            start_data.deviceName = "Host CPU";
            start_data.benchmarkName = bench_name;
            start_data.component = bench->GetComponent(i);
            start_data.subcategory = bench->GetSubCategory(i);
            start_data.metric = bench->GetMetric(i);
            start_data.operations = 0;
            start_data.time_ms = -1.0;
            start_data.isEmulated = false;
            start_data.isUnsupported = false;
            start_data.maxWorkGroupSize = 0;
            start_data.deviceIndex = 0xFFFFFFFF;
            start_data.configIndex = i;
            start_data.sortWeight = bench->GetSortWeight(i);
            onResult(start_data);
          }

          double total_time_ms = 0;
          uint64_t total_invocations = 0;
          auto bench_start = std::chrono::high_resolution_clock::now();
          while (total_time_ms < 5000) {
            bench->Run(i);
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
          result_data.sortWeight = bench->GetSortWeight(i);

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

void BenchmarkRunner::printReport() {
  if (verbose) {
    std::cout << "\r\033[K" << std::flush;
  }
  if (!onResult) {
    formatter->print();
  }
}

void BenchmarkRunner::run(const std::vector<std::string> &benchmarks_to_run) {
  utils::SleepInhibitor sleepInhibitor("Running GPU compute benchmarks");
  initRunConfig(benchmarks_to_run);

  for (auto *context : contexts) {
    runForContext(context, benchmarks_to_run);
  }

  runHostBenchmarks(benchmarks_to_run);

  if (!onResult) {
    printReport();
  }
}
