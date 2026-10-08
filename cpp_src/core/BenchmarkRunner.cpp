#include "core/BenchmarkRunner.h"
#include "benchmarks/CacheBench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Bf16Bench.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/DualIssueBench.h"
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
#include "benchmarks/RayPayloadBench.h"
#include "benchmarks/RayProceduralBench.h"
#include "benchmarks/RayIntersectBench.h"
#include "benchmarks/RayRawTraversalBench.h"
#include "benchmarks/RaySchedulingBench.h"
#include "benchmarks/SysMemBandwidthBench.h"
#include "benchmarks/SysMemLatencyBench.h"
#include "benchmarks/CacheLatencyCurveBench.h"
#include "benchmarks/LdsBankConflictBench.h"
#include "benchmarks/InShaderIndirectBench.h"
#include <iomanip>

static std::vector<int> g_targetConfigs;

void SetRunnerTargetConfigs(const std::vector<int> &configs) {
  g_targetConfigs = configs;
}
#include "core/ComputeBackendFactory.h"
#include "core/ResultFormatter.h"
#include "utils/KernelPath.h"
#include "utils/SleepInhibitor.h"
#include "utils/HardwareTelemetry.h"
#include "benchmarks/Fp6Bench.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <locale>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#ifdef _WIN32
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

// Helper function to create a shuffled index array forming a single Hamiltonian cycle for pointer chasing
std::vector<uint32_t> create_shuffled_indices(size_t size) {
  if (size == 0) return {};
  if (size == 1) return {0};
  std::vector<uint32_t> perm(size);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 g(1337); // Use a fixed seed for reproducibility
  std::shuffle(perm.begin(), perm.end(), g);
  std::vector<uint32_t> indices(size);
  for (size_t i = 0; i < size - 1; ++i) {
    indices[perm[i]] = perm[i + 1];
  }
  indices[perm[size - 1]] = perm[0];
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
  benchmarks.push_back(std::make_unique<DualIssueBench>());
  benchmarks.push_back(std::make_unique<LdsBankConflictBench>());
  benchmarks.push_back(std::make_unique<InShaderIndirectBench>());
  benchmarks.push_back(std::make_unique<Fp16Bench>());
  benchmarks.push_back(std::make_unique<Bf16Bench>());
  benchmarks.push_back(std::make_unique<Fp8Bench>());
  benchmarks.push_back(std::make_unique<Fp6Bench>());
  benchmarks.push_back(std::make_unique<Fp4Bench>());
  benchmarks.push_back(std::make_unique<Int8Bench>());
  benchmarks.push_back(std::make_unique<Int4Bench>());
  benchmarks.push_back(std::make_unique<MemBandwidthBench>());
  benchmarks.push_back(std::make_unique<PixelFillRateBench>());
  benchmarks.push_back(std::make_unique<SysMemBandwidthBench>());
  benchmarks.push_back(std::make_unique<SysMemLatencyBench>());
  // Ray Tracing Acceleration (Real-World Pipeline Order: AS Build -> Primary Rays/Intersection -> Ray Scheduling -> Secondary Rays/Divergence -> Path Tracing -> Payload Pressure)
  benchmarks.push_back(std::make_unique<RayRawTraversalBench>());
  benchmarks.push_back(std::make_unique<RayASBuildBench>());
  benchmarks.push_back(std::make_unique<RayIntersectBench>());
  benchmarks.push_back(std::make_unique<RayAnyHitBench>());
  benchmarks.push_back(std::make_unique<RayProceduralBench>());
  if (sceneName == "all") {
    auto showroom = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::Showroom);
    showroom->SetBounceDepth(bounceDepth);
    showroom->SetSamplesPerPixel(samplesPerPixel);
    showroom->SetDumpRenders(dumpRenders || verifyParity);
    showroom->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(showroom));

    auto indoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::IndoorAtrium);
    indoor->SetBounceDepth(bounceDepth);
    indoor->SetSamplesPerPixel(samplesPerPixel);
    indoor->SetDumpRenders(dumpRenders || verifyParity);
    indoor->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(indoor));

    auto outdoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::OutdoorLandscape);
    outdoor->SetBounceDepth(bounceDepth);
    outdoor->SetSamplesPerPixel(samplesPerPixel);
    outdoor->SetDumpRenders(dumpRenders || verifyParity);
    outdoor->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(outdoor));

    auto forest = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::AAAOutdoorForest);
    forest->SetBounceDepth(bounceDepth);
    forest->SetSamplesPerPixel(samplesPerPixel);
    forest->SetDumpRenders(dumpRenders || verifyParity);
    forest->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(forest));
  } else if (sceneName == "outdoor") {
    auto outdoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::OutdoorLandscape);
    outdoor->SetBounceDepth(bounceDepth);
    outdoor->SetSamplesPerPixel(samplesPerPixel);
    outdoor->SetDumpRenders(dumpRenders || verifyParity);
    outdoor->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(outdoor));
  } else if (sceneName == "forest" || sceneName == "aaa_forest") {
    auto forest = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::AAAOutdoorForest);
    forest->SetBounceDepth(bounceDepth);
    forest->SetSamplesPerPixel(samplesPerPixel);
    forest->SetDumpRenders(dumpRenders || verifyParity);
    forest->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(forest));
  } else if (sceneName == "showroom") {
    auto showroom = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::Showroom);
    showroom->SetBounceDepth(bounceDepth);
    showroom->SetSamplesPerPixel(samplesPerPixel);
    showroom->SetDumpRenders(dumpRenders || verifyParity);
    showroom->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(showroom));
  } else if (sceneName == "indoor") {
    auto indoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::IndoorAtrium);
    indoor->SetBounceDepth(bounceDepth);
    indoor->SetSamplesPerPixel(samplesPerPixel);
    indoor->SetDumpRenders(dumpRenders || verifyParity);
    indoor->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(indoor));
  } else {
    // Default fallback: all scenes
    auto showroom = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::Showroom);
    showroom->SetBounceDepth(bounceDepth);
    showroom->SetSamplesPerPixel(samplesPerPixel);
    showroom->SetDumpRenders(dumpRenders || verifyParity);
    showroom->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(showroom));

    auto indoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::IndoorAtrium);
    indoor->SetBounceDepth(bounceDepth);
    indoor->SetSamplesPerPixel(samplesPerPixel);
    indoor->SetDumpRenders(dumpRenders || verifyParity);
    indoor->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(indoor));

    auto outdoor = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::OutdoorLandscape);
    outdoor->SetBounceDepth(bounceDepth);
    outdoor->SetSamplesPerPixel(samplesPerPixel);
    outdoor->SetDumpRenders(dumpRenders || verifyParity);
    outdoor->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(outdoor));

    auto forest = std::make_unique<RaySchedulingBench>(RaySchedulingBench::SceneType::AAAOutdoorForest);
    forest->SetBounceDepth(bounceDepth);
    forest->SetSamplesPerPixel(samplesPerPixel);
    forest->SetDumpRenders(dumpRenders || verifyParity);
    forest->SetVerifyParity(verifyParity);
    benchmarks.push_back(std::move(forest));
  }
  benchmarks.push_back(std::make_unique<RayDivergenceBench>());
  // RayPayload benchmark disabled (payload size invariance on modern wide VGPR files)
  // benchmarks.push_back(std::make_unique<RayPayloadBench>());

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

  // We still need the sizes for latency tests (commented out while L1-L3 latency tests disabled)
  // const size_t l1_size = 128 * 1024;       // 128KB
  // const size_t l2_size = 4 * 1024 * 1024;  // 4MB
  // const size_t l3_size = 64 * 1024 * 1024; // 64MB

  // Cache Latency
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L0 Vector Cache Latency (16 KB)", "ns", l0_size, "l0_cache_latency",
      std::vector<uint32_t>{},
      std::vector<std::string>{"l0l", "l0", "l0_cache_latency", "l0-cache-latency", "l0-latency", "l0cache"}, 0));
  benchmarks.push_back(std::make_unique<CacheLatencyCurveBench>());
  // L1, L2, and L3 cache latency tests temporarily disabled due to memory prefetcher & measurement volatility
  /*
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L1 Cache Latency", "ns", l1_size, "cache_latency",
      create_shuffled_indices(l1_size / sizeof(uint32_t)),
      std::vector<std::string>{"l1l"}, 1));
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L2 Cache Latency", "ns", l2_size, "cache_latency",
      create_shuffled_indices(l2_size / sizeof(uint32_t)),
      std::vector<std::string>{"l2l"}, 2));
  benchmarks.push_back(std::make_unique<CacheBench>(
      "L3 Cache Latency", "ns", l3_size, "cache_latency",
      create_shuffled_indices(l3_size / sizeof(uint32_t)),
      std::vector<std::string>{"l3l"}, 3));
  */
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

  // Cache latency benchmark alias & general cache matching
  if (dynamic_cast<const CacheBench *>(bench)) {
    if (normRun == "cache" || normRun == "cachelatency" || normRun == "caches" || normRun == "cachelat") {
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
  context->setQuiet(quiet);

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

    bool hasVisualVerification = false;
    bool hasRayTracing = false;
    for (const auto &bench : benchmarks) {
      if (isSelected(bench.get()) && bench->IsSupported(info, context)) {
        if (bench->HasVisualVerification()) {
          hasVisualVerification = true;
        }
        if (dynamic_cast<RaySchedulingBench *>(bench.get()) != nullptr) {
          hasRayTracing = true;
        }
      }
    }

    uint32_t effectiveWidth = renderWidth;
    uint32_t effectiveHeight = renderHeight;
    const bool isAutoRes = (effectiveWidth == 0 || effectiveHeight == 0);
    if (isAutoRes) {
      // Adaptive tier-down based on device memory:
      // >= 16 GB: 4K UHD (3840x2160)
      // >= 8 GB:  1440p QHD (2560x1440)
      // < 8 GB:   1080p FHD (1920x1080)
      const uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
      if (info.memorySize >= 16ULL * gib) {
        effectiveWidth = 3840;
        effectiveHeight = 2160;
      } else if (info.memorySize >= 8ULL * gib) {
        effectiveWidth = 2560;
        effectiveHeight = 1440;
      } else {
        effectiveWidth = 1920;
        effectiveHeight = 1080;
      }
    }

    if (!quiet && !verbose && !onResult) {
      std::string backendStr = ComputeBackendFactory::getBackendName(context->getBackend());
      int vramGb = static_cast<int>(std::round(info.memorySize / (1024.0 * 1024.0 * 1024.0)));
      std::string memLabel = info.isApu ? "GB Unified Memory" : "GB Dedicated VRAM";
      std::string line1_plain = "Target Device : [GPU " + std::to_string(context->getSelectedDeviceIndex()) + "] " + info.name;
      std::string line2_plain = "Backend / API : " + backendStr + " | VRAM: " + std::to_string(vramGb) + " " + memLabel;

      std::string resPreset = "";
      if (isAutoRes) {
        if (effectiveWidth == 3840 && effectiveHeight == 2160) resPreset = " (Auto -> 4K UHD)";
        else if (effectiveWidth == 2560 && effectiveHeight == 1440) resPreset = " (Auto -> 1440p QHD)";
        else if (effectiveWidth == 1920 && effectiveHeight == 1080) resPreset = " (Auto -> 1080p FHD)";
      } else {
        if (effectiveWidth == 3840 && effectiveHeight == 2160) resPreset = " (4K UHD)";
        else if (effectiveWidth == 2560 && effectiveHeight == 1440) resPreset = " (1440p QHD)";
        else if (effectiveWidth == 1920 && effectiveHeight == 1080) resPreset = " (1080p FHD)";
        else if (effectiveWidth == 1280 && effectiveHeight == 720) resPreset = " (720p HD)";
        else if (effectiveWidth == 1024 && effectiveHeight == 1024) resPreset = " (1024x1024 Square)";
      }
      std::string line3_plain = "Resolution    : " + std::to_string(effectiveWidth) + "x" + std::to_string(effectiveHeight) + resPreset;

      std::string scLabel = sceneName;
      if (sceneName == "all") scLabel = "All Scenarios (Showroom, Atrium, Landscape, Forest)";
      else if (sceneName == "showroom") scLabel = "Showroom Studio";
      else if (sceneName == "indoor") scLabel = "Indoor Atrium";
      else if (sceneName == "outdoor") scLabel = "Outdoor Landscape";
      else if (sceneName == "forest" || sceneName == "aaa_forest") scLabel = "Open-World Forest";
      std::string line4_plain = "RT Scenario   : " + scLabel;

      size_t innerCardW = 74;
      if (line1_plain.length() > innerCardW) innerCardW = line1_plain.length();
      if (line2_plain.length() > innerCardW) innerCardW = line2_plain.length();
      if (line3_plain.length() > innerCardW) innerCardW = line3_plain.length();
      if (hasRayTracing && line4_plain.length() > innerCardW) innerCardW = line4_plain.length();

      size_t pad1 = (innerCardW > line1_plain.length()) ? (innerCardW - line1_plain.length()) : 0;
      size_t pad2 = (innerCardW > line2_plain.length()) ? (innerCardW - line2_plain.length()) : 0;
      size_t pad3 = (innerCardW > line3_plain.length()) ? (innerCardW - line3_plain.length()) : 0;
      size_t pad4 = (innerCardW > line4_plain.length()) ? (innerCardW - line4_plain.length()) : 0;

      std::string topTitle = "╭─ GPUBench v" + std::string(GPUBENCH_VERSION) + " ";
      size_t prefixLen = 14 + std::string(GPUBENCH_VERSION).length();
      size_t dashCount = (innerCardW + 2 > prefixLen) ? (innerCardW + 2 - prefixLen) : 10;

      std::cout << "\n\033[1m\033[36m" << topTitle;
      for (size_t d = 0; d < dashCount; ++d) std::cout << "─";
      std::cout << "╮\033[0m\n";

      std::cout << "\033[1m\033[36m│\033[0m \033[1mTarget Device\033[0m : [GPU " << context->getSelectedDeviceIndex() << "] \033[33m" << info.name << "\033[0m"
                << std::string(pad1, ' ') << " \033[1m\033[36m│\033[0m\n";
      std::cout << "\033[1m\033[36m│\033[0m \033[1mBackend / API\033[0m : \033[32m" << backendStr << "\033[0m | \033[1mVRAM\033[0m: \033[32m" << vramGb << " " << memLabel << "\033[0m"
                << std::string(pad2, ' ') << " \033[1m\033[36m│\033[0m\n";
      std::cout << "\033[1m\033[36m│\033[0m \033[1mResolution\033[0m    : \033[36m" << effectiveWidth << "x" << effectiveHeight << resPreset << "\033[0m"
                << std::string(pad3, ' ') << " \033[1m\033[36m│\033[0m\n";
      if (hasRayTracing) {
        std::cout << "\033[1m\033[36m│\033[0m \033[1mRT Scenario\033[0m   : \033[35m" << scLabel << "\033[0m"
                  << std::string(pad4, ' ') << " \033[1m\033[36m│\033[0m\n";
      }
      std::cout << "\033[1m\033[36m╰";
      for (size_t d = 0; d < innerCardW + 2; ++d) std::cout << "─";
      std::cout << "╯\033[0m\n\n";
    }

    GpuContentionInfo contention = HardwareTelemetry::checkContention(context->getSelectedDeviceIndex());
    if (contention.hasContention && !quiet) {
      size_t alertW = 76;
      for (const auto &reason : contention.reasons) {
        if (reason.length() + 4 > alertW) {
          alertW = reason.length() + 4;
        }
      }
      std::string headerPrefix = "╭─ ⚠ Background Hardware Contention Detected ";
      size_t headerPrefixCol = 44;
      size_t topDashes = (alertW > headerPrefixCol) ? (alertW - headerPrefixCol) : 4;
      std::cout << "\033[1;33m" << headerPrefix;
      for (size_t d = 0; d < topDashes; ++d) std::cout << "─";
      std::cout << "╮\n";
      std::string line1 = "Active background tasks or resident memory allocations were detected:";
      std::cout << "│ " << line1 << std::string(alertW > line1.length() + 1 ? alertW - line1.length() - 1 : 0, ' ') << "│\n";
      for (const auto &reason : contention.reasons) {
        std::string rContent = " • " + reason;
        std::cout << "│ " << rContent << std::string(alertW > rContent.length() + 1 ? alertW - rContent.length() - 1 : 0, ' ') << "│\n";
      }
      std::string line2 = "Benchmark throughput may be contaminated or throttled by background load.";
      std::string line3 = "For clean publication results, terminate active background LLMs or apps.";
      std::cout << "│ " << std::string(alertW - 1, ' ') << "│\n";
      std::cout << "│ " << line2 << std::string(alertW > line2.length() + 1 ? alertW - line2.length() - 1 : 0, ' ') << "│\n";
      std::cout << "│ " << line3 << std::string(alertW > line3.length() + 1 ? alertW - line3.length() - 1 : 0, ' ') << "│\n";
      std::cout << "╰";
      for (size_t d = 0; d < alertW; ++d) std::cout << "─";
      std::cout << "╯\033[0m\n\n";
    }

    if (strictMode && contention.isCritical && !forceExecution) {
      std::cerr << "\n\033[1;31mError: Benchmark aborted under --strict due to active background GPU contention ("
                << contention.gpuBusyPct << "% busy).\n"
                << "Use -f / --force to benchmark under contention, or terminate the active workload.\033[0m\n\n";
      executionFailure = true;
      return;
    }

    if (!quiet && !verbose && !onResult) {
      if (hasVisualVerification) {
        std::cout << "  \033[1m[1/3] Preparation Phase\033[0m (compiling kernels, uploading data, building BVHs)..." << std::endl;
      } else {
        std::cout << "  \033[1m[1/2] Preparation Phase\033[0m (compiling kernels, uploading data, building BVHs)..." << std::endl;
      }
    } else if (!quiet) {
      std::cout << "Preparing benchmarks (compiling kernels, uploading "
                   "data, building acceleration structures)..."
                << std::endl;
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
        // Skip logging unsupported Ray Tracing benchmarks under compute-only backends (ROCm / OpenCL)
        if (std::string_view(bench->GetComponent()) == "Ray Tracing" && context->getBackend() != ComputeBackend::Vulkan) {
          continue;
        }

        std::string bname = bench->GetName();
        uint32_t num_unsupported_configs = bench->GetNumConfigs();

        for (uint32_t ci = 0; ci < num_unsupported_configs; ++ci) {
          ResultData result_data;
          result_data.backendName =
              ComputeBackendFactory::getBackendName(context->getBackend());
          result_data.deviceName = info.name;
          result_data.vendorId = info.vendorID;
          result_data.deviceId = info.deviceID;
          std::string cname = bench->GetConfigName(ci);
          std::string fullName = bname;
          if (cname.rfind(bname, 0) == 0) {
            fullName = cname;
          } else if (num_unsupported_configs > 1 || !cname.empty()) {
            fullName = bname + " (" + cname + ")";
          }
          result_data.benchmarkName = fullName;
          result_data.metric = bench->GetMetric(ci);
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

    struct BenchmarkTask {
      IBenchmark *bench;
      uint32_t configIndex;
      int sortWeight;
    };

    std::vector<BenchmarkTask> tasks;
    for (auto *bench : runnable) {
      uint32_t num_configs = bench->GetNumConfigs();
      for (uint32_t i = 0; i < num_configs; ++i) {
        if (!g_targetConfigs.empty()) {
          if (std::find(g_targetConfigs.begin(), g_targetConfigs.end(), static_cast<int>(i)) == g_targetConfigs.end()) {
            continue;
          }
        } else if (targetConfig >= 0 && static_cast<int>(i) != targetConfig) {
          continue;
        }

        // Deduplicate algorithmic microbenchmarks across non-Showroom scenes
        if (auto *rs = dynamic_cast<RaySchedulingBench *>(bench)) {
          if (rs->GetSceneType() != RaySchedulingBench::SceneType::Showroom) {
            if (i == 12 || i == 13 || i == 14 || i == 15 || i == 16 || i == 26 || i == 28) {
              continue;
            }
          }
        }

        if (!workloadFilter.empty()) {
          bool matched = false;
          std::string bName = bench->GetName();
          std::string cName = bench->GetConfigName(i);
          std::string fullTaskName = bName;
          if (cName.rfind(bName, 0) == 0) {
            fullTaskName = cName;
          } else if (!cName.empty()) {
            fullTaskName += " (" + cName + ")";
          }
          std::string keyIndex = bName + "#" + std::to_string(i);

          // Strip any parenthesized scene name from RayScheduling (e.g. "RayScheduling (Indoor Atrium)" -> "RayScheduling")
          std::string baseBenchName = bName;
          size_t p = baseBenchName.find(" (");
          if (p != std::string::npos) {
            baseBenchName = baseBenchName.substr(0, p);
          }
          std::string baseKeyIndex = baseBenchName + "#" + std::to_string(i);

          for (const auto &filt : workloadFilter) {
            if (filt == baseBenchName || filt == bName) {
              matched = true;
              break;
            }
            if (filt == baseKeyIndex || filt == keyIndex) {
              matched = true;
              break;
            }
            if (!cName.empty() && filt == cName) {
              matched = true;
              break;
            }
            if (filt == fullTaskName) {
              matched = true;
              break;
            }
          }
          if (!matched) {
            continue;
          }
        }

        tasks.push_back({bench, i, bench->GetSortWeight(i)});
      }
    }

    std::stable_sort(tasks.begin(), tasks.end(),
                     [](const BenchmarkTask &a, const BenchmarkTask &b) {
                       return a.sortWeight < b.sortWeight;
                     });

    const bool isInteractive = !quiet && !verbose && !onResult && isatty(fileno(stdout));
    if (!quiet && !verbose && !onResult) {
      std::cout << "\r\033[K  \033[32m✔\033[0m Preparation complete.\n\n";
      if (!tasks.empty()) {
        if (hasVisualVerification) {
          std::cout << "  \033[1m[2/3] Running Benchmarks\033[0m ("
                    << tasks.size() << " workloads)..." << std::endl;
        } else {
          std::cout << "  \033[1m[2/2] Running Benchmarks\033[0m ("
                    << tasks.size() << " workloads)..." << std::endl;
        }
      }
    }

    IBenchmark *prevBench = nullptr;
    size_t taskIdx = 0;
    bool deviceLost = false;
    for (const auto &task : tasks) {
      if (cancelToken && cancelToken->load()) {
        if (!quiet && !verbose && !onResult) {
          std::cout << "\n  \033[33m⚠\033[0m Benchmark run cancelled by user." << std::endl;
        }
        break;
      }
      auto *bench = task.bench;
      uint32_t i = task.configIndex;

      if (prevBench) {
        if (prevBench != bench) {
          std::this_thread::sleep_for(std::chrono::milliseconds(150));
        } else if (dynamic_cast<MemBandwidthBench *>(bench)) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      }
      prevBench = bench;

      std::string bench_name = bench->GetName();
      std::string config_name = bench->GetConfigName(i);
      if (config_name.rfind(bench_name, 0) == 0) {
        bench_name = config_name;
      } else if (!config_name.empty()) {
        bench_name += " (" + config_name + ")";
      }

      try {
        if (verbose) {
          std::cout << "[D" << context->getSelectedDeviceIndex()
                    << "] Running " << bench_name << "..." << std::endl;
        } else if (isInteractive) {
          std::string disp = bench_name;
          if (disp.rfind("RayScheduling (", 0) == 0) {
            size_t secondOpen = disp.find(") (");
            if (secondOpen != std::string::npos && disp.back() == ')') {
              disp = disp.substr(secondOpen + 3, disp.length() - (secondOpen + 4));
            }
          } else if (disp.rfind("RayASBuild (", 0) == 0 && disp.back() == ')') {
            disp = disp.substr(12, disp.length() - 13);
          } else {
            size_t firstOpen = disp.find(" (");
            if (firstOpen != std::string::npos && disp.back() == ')') {
              disp = disp.substr(firstOpen + 2, disp.length() - (firstOpen + 3));
            }
          }
          if (disp.length() > 50) disp = disp.substr(0, 47) + "...";
          static const char* kSpinnerFrames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
          const char* spinner = kSpinnerFrames[taskIdx % 10];
          std::cout << "\r\033[K  \033[36m" << spinner << "\033[0m [" << (taskIdx + 1) << "/" << tasks.size() << "] "
                    << disp << "..." << std::flush;
        } else if (!quiet) {
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
          if (!quiet && !verbose && !isInteractive) {
            if (!note.empty()) {
              std::cout << " Unsupported (" << note << ")." << std::endl;
            } else {
              std::cout << " Unsupported." << std::endl;
            }
          }
          taskIdx++;
          ResultData result_data;
          result_data.backendName = ComputeBackendFactory::getBackendName(context->getBackend());
          result_data.deviceName = info.name;
          result_data.vendorId = info.vendorID;
          result_data.deviceId = info.deviceID;
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
        double stat_min_ms = 0.0;
        double stat_med_ms = 0.0;
        double stat_mean_ms = 0.0;
        double stat_p95_ms = 0.0;
        uint32_t stat_sample_count = 0;
        std::vector<double> stat_samples;
        uint32_t devIdx = context->getSelectedDeviceIndex();
        GpuTelemetryData powerStart = HardwareTelemetry::queryGpu(devIdx);
        float avgPowerWatts = 0.0f;
        HardwarePerformanceCounters measuredPerfCounters;

        if (profileSnapshot) {
          bench->Run(i);
          context->waitIdle();

          if (context->hasPerformanceQuery()) {
            context->startPerformanceQuery();
          }
          if (context->hasGpuTiming()) {
            context->startTiming();
          }
          auto start = std::chrono::high_resolution_clock::now();
          bench->Run(i);
          if (context->hasGpuTiming()) {
            double gpu_time = context->stopTiming();
            if (gpu_time > 0.0) {
              total_time_ms = gpu_time;
            } else {
              context->waitIdle();
              auto end = std::chrono::high_resolution_clock::now();
              total_time_ms =
                  std::chrono::duration<double, std::milli>(end - start).count();
            }
          } else {
            context->waitIdle();
            auto end = std::chrono::high_resolution_clock::now();
            total_time_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
          }
          if (context->hasPerformanceQuery()) {
            measuredPerfCounters = context->stopPerformanceQuery();
          }
          total_invocations = 1;
          stat_min_ms = total_time_ms;
          stat_med_ms = total_time_ms;
          stat_mean_ms = total_time_ms;
          stat_p95_ms = total_time_ms;
          stat_sample_count = 1;
          stat_samples.push_back(total_time_ms);
        } else {
          auto start = std::chrono::high_resolution_clock::now();
          bench->Run(i);
          context->waitIdle();
          auto end = std::chrono::high_resolution_clock::now();
          double single_run_ms =
              std::chrono::duration<double, std::milli>(end - start).count();

          // Warmup: Run until GPU clocks ramp up to sustained boost clocks.
          // If single_run_ms >= 200ms, GPU is already at boost clocks; skip warmup.
          const double min_warmup_duration_ms = 400.0;
          uint64_t warmup_iters = 0;
          if (single_run_ms > 0.0 && single_run_ms < 200.0) {
            if (single_run_ms < 50.0) {
              warmup_iters = static_cast<uint64_t>(
                  std::max(2.0, std::ceil(min_warmup_duration_ms / single_run_ms)));
            } else {
              warmup_iters = static_cast<uint64_t>(
                  std::ceil(min_warmup_duration_ms / single_run_ms));
            }
          }
          warmup_iters = std::min(warmup_iters, static_cast<uint64_t>(50));
          if (dynamic_cast<MemBandwidthBench *>(bench)) {
            warmup_iters = std::min(warmup_iters, static_cast<uint64_t>(2));
          }

          for (uint64_t w = 0; w < warmup_iters; ++w) {
            if (cancelToken && cancelToken->load()) break;
            bench->Run(i);
            if (single_run_ms >= 10.0) {
              context->waitIdle();
            }
          }
          context->waitIdle();

          // Re-measure single run latency only if warmup was actually performed
          if (warmup_iters > 0) {
            start = std::chrono::high_resolution_clock::now();
            bench->Run(i);
            context->waitIdle();
            end = std::chrono::high_resolution_clock::now();
            single_run_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
          }

          // Optimization O-3: Multi-sample distribution measurement (min / median / mean / p95)
          // Optimization O-2: Command buffer batching for short compute microbenchmarks
          uint32_t num_samples = 5;
          if (single_run_ms >= 1500.0) {
            num_samples = 1;
          } else if (single_run_ms >= 400.0) {
            num_samples = 3;
          } else if (single_run_ms < 0.5) {
            num_samples = 7;
          }

          const double target_sample_ms = (num_samples > 1) ? 50.0 : 250.0;
          uint64_t sample_iters = 1;
          if (single_run_ms > 0.0) {
            sample_iters = static_cast<uint64_t>(
                std::max(1.0, std::round(target_sample_ms / single_run_ms)));
          }
          sample_iters = std::min(sample_iters, static_cast<uint64_t>(500));
          sample_iters = std::max(sample_iters, static_cast<uint64_t>(1));

          // Hard clamp: ensure a single sample does not exceed 350ms
          if (single_run_ms > 0.0 && (sample_iters * single_run_ms > 350.0)) {
            sample_iters = static_cast<uint64_t>(std::max(1.0, 350.0 / single_run_ms));
          }

          if (dynamic_cast<MemBandwidthBench *>(bench)) {
            sample_iters = std::min(sample_iters, static_cast<uint64_t>(2));
            num_samples = std::min(num_samples, 3u);
          }

          std::vector<double> sample_time_per_invoc;
          total_invocations = 0;

          for (uint32_t s = 0; s < num_samples; ++s) {
            if (cancelToken && cancelToken->load()) break;

            if (s == 0 && context->hasPerformanceQuery()) {
              context->startPerformanceQuery();
            }

            if (context->hasGpuTiming()) {
              context->startTiming();
            }
            auto s_start = std::chrono::high_resolution_clock::now();

            if (sample_iters > 1) {
              context->beginBatch(sample_iters);
              for (uint64_t it = 0; it < sample_iters; ++it) {
                bench->Run(i);
              }
              context->endBatch();
            } else {
              bench->Run(i);
            }

            double sample_ms = 0.0;
            if (context->hasGpuTiming()) {
              double gpu_time = context->stopTiming();
              if (gpu_time > 0.0) {
                sample_ms = gpu_time;
              } else {
                context->waitIdle();
                auto s_end = std::chrono::high_resolution_clock::now();
                sample_ms =
                    std::chrono::duration<double, std::milli>(s_end - s_start).count();
              }
            } else {
              context->waitIdle();
              auto s_end = std::chrono::high_resolution_clock::now();
              sample_ms =
                  std::chrono::duration<double, std::milli>(s_end - s_start).count();
            }

            if (s == 0 && context->hasPerformanceQuery()) {
              measuredPerfCounters = context->stopPerformanceQuery();
            }

            if (sample_ms > 0.0) {
              sample_time_per_invoc.push_back(sample_ms / sample_iters);
              total_invocations += sample_iters;
            }
          }

          if (!sample_time_per_invoc.empty()) {
            std::vector<double> sorted_samples = sample_time_per_invoc;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            size_t S = sorted_samples.size();

            double min_per_inv = sorted_samples.front();
            double sum_per_inv = 0.0;
            for (double val : sorted_samples) sum_per_inv += val;
            double mean_per_inv = sum_per_inv / S;

            double med_per_inv = (S % 2 == 1)
                ? sorted_samples[S / 2]
                : 0.5 * (sorted_samples[S / 2 - 1] + sorted_samples[S / 2]);

            size_t p95_idx = std::min(static_cast<size_t>(std::ceil(0.95 * S)) - 1, S - 1);
            double p95_per_inv = sorted_samples[p95_idx];

            double raw_median_total_ms = med_per_inv * total_invocations;
            total_time_ms = bench->FilterDuration(i, total_invocations, raw_median_total_ms);

            stat_min_ms = min_per_inv * total_invocations;
            stat_med_ms = total_time_ms;
            stat_mean_ms = mean_per_inv * total_invocations;
            stat_p95_ms = p95_per_inv * total_invocations;
            stat_sample_count = static_cast<uint32_t>(S);
            stat_samples = sample_time_per_invoc;
          }

          if (total_invocations == 0) {
            ResultData abort_data;
            abort_data.backendName = ComputeBackendFactory::getBackendName(context->getBackend());
            abort_data.deviceName = info.name;
            abort_data.vendorId = info.vendorID;
            abort_data.deviceId = info.deviceID;
            abort_data.benchmarkName = bench_name;
            abort_data.time_ms = -3.0; // ABORTED
            abort_data.errorString = "Benchmark cancelled by user before execution";
            abort_data.isUnsupported = false;
            abort_data.isValid = false;
            abort_data.component = bench->GetComponent(i);
            abort_data.subcategory = bench->GetSubCategory(i);
            abort_data.configIndex = i;
            abort_data.sortWeight = bench->GetSortWeight(i);
            formatter->addResult(abort_data);
            executionFailure = true;
            continue;
          }
          if (verbose) {
            std::cout << "[STATS " << bench_name << "] " << stat_sample_count << " samples"
                      << ", invocations: " << total_invocations
                      << ", median_total_ms: " << total_time_ms
                      << " | median_inv: " << (stat_med_ms / (total_invocations ? total_invocations : 1)) << " ms"
                      << " | min_inv: " << (stat_min_ms / (total_invocations ? total_invocations : 1)) << " ms"
                      << " | mean_inv: " << (stat_mean_ms / (total_invocations ? total_invocations : 1)) << " ms"
                      << " | p95_inv: " << (stat_p95_ms / (total_invocations ? total_invocations : 1)) << " ms"
                      << std::endl;
          }
        }

        GpuTelemetryData powerEnd = HardwareTelemetry::queryGpu(devIdx);
        if (powerStart.powerWatts > 0.0f && powerEnd.powerWatts > 0.0f) {
          avgPowerWatts = (powerStart.powerWatts + powerEnd.powerWatts) * 0.5f;
        } else if (powerStart.powerWatts > 0.0f) {
          avgPowerWatts = powerStart.powerWatts;
        } else if (powerEnd.powerWatts > 0.0f) {
          avgPowerWatts = powerEnd.powerWatts;
        }

        bool isValid = bench->ValidateResults(i);
        if (!isValid) {
          validationFailure = true;
          if (verbose) {
            std::cerr << " [WARNING] Result validation failed for "
                      << bench_name << std::endl;
          }
        }

        BenchmarkResult bench_result = bench->GetResult(i);
        bench->RecordRunResult(i, total_invocations, total_time_ms);

        ResultData result_data;
        result_data.backendName = ComputeBackendFactory::getBackendName(
            context->getBackend());
        result_data.deviceName = info.name;
        result_data.vendorId = info.vendorID;
        result_data.deviceId = info.deviceID;
        result_data.benchmarkName = bench_name;
        result_data.metric = bench->GetMetric(i);
        result_data.operations =
            bench_result.operations * total_invocations;
        result_data.time_ms = total_time_ms;
        result_data.min_time_ms = stat_min_ms;
        result_data.median_time_ms = stat_med_ms;
        result_data.mean_time_ms = stat_mean_ms;
        result_data.p95_time_ms = stat_p95_ms;
        result_data.sample_count = stat_sample_count;
        result_data.sample_durations_ms = stat_samples;
        result_data.isValid = isValid;
        result_data.baselineConfigIndex = bench->GetBaselineConfigIndex(i);
        result_data.isEmulated = bench->IsEmulated(i);
        result_data.supportNote = bench->GetConfigCaveat(i, info, context);
        if (result_data.supportNote.empty()) {
          result_data.supportNote = bench->GetConfigSupportNote(i, info, context);
        }
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

        KernelResourceUsage kUsage = context->getLastKernelResourceUsage();
        if (kUsage.available) {
          result_data.hasRegisterTelemetry = true;
          result_data.vgprCount = kUsage.vgprCount;
          result_data.sgprCount = kUsage.sgprCount;
          result_data.ldsSizeBytes = kUsage.ldsSizeBytes;
          result_data.scratchSizeBytes = kUsage.scratchSizeBytes;
          result_data.codeSizeBytes = kUsage.codeSizeBytes;
          result_data.maxWavesPerSimd = kUsage.maxWavesPerSimd;
          result_data.compilerTarget = kUsage.compilerNotes;

          if (verbose) {
            std::cout << " [COMPILER TELEMETRY " << ComputeBackendFactory::getBackendName(context->getBackend())
                      << "] VGPRs: " << kUsage.vgprCount
                      << (kUsage.sgprCount > 0 ? (" | SGPRs: " + std::to_string(kUsage.sgprCount)) : "")
                      << " | LDS: " << kUsage.ldsSizeBytes << " B"
                      << " | Scratch/Spill: " << kUsage.scratchSizeBytes << " B"
                      << (kUsage.codeSizeBytes > 0 ? (" | Code: " + std::to_string(kUsage.codeSizeBytes) + " B") : "")
                      << (kUsage.maxWavesPerSimd > 0 ? (" | Occupancy: " + std::to_string(kUsage.maxWavesPerSimd) + "/16 waves (" + std::to_string(static_cast<int>(kUsage.maxWavesPerSimd * 100.0 / 16.0)) + "%)") : "")
                      << " (" << kUsage.compilerNotes << ")"
                      << std::endl;
          }
        }

        if (measuredPerfCounters.available) {
          result_data.hasPerfQueryTelemetry = true;
          result_data.perfCounters = measuredPerfCounters;

          if (verbose) {
            std::cout << " [HARDWARE TELEMETRY VK_KHR_performance_query] Active Cycles: "
                      << measuredPerfCounters.gpuActiveCycles
                      << " | Waves: " << measuredPerfCounters.waves
                      << " | VALU Insts: " << measuredPerfCounters.valuInstructions
                      << " | SALU Insts: " << measuredPerfCounters.saluInstructions
                      << " | VALU Busy: " << std::fixed << std::setprecision(1) << measuredPerfCounters.valuBusyPct << "%"
                      << " | SALU Busy: " << measuredPerfCounters.saluBusyPct << "%"
                      << " | VRAM Read: " << (static_cast<double>(measuredPerfCounters.vramReadBytes) / (1024.0 * 1024.0)) << " MB"
                      << " | L0 Hit: " << measuredPerfCounters.l0CacheHitRatio << "%"
                      << " | L1 Hit: " << measuredPerfCounters.l1CacheHitRatio << "%"
                      << std::defaultfloat
                      << std::endl;
          }
        }

        if (avgPowerWatts > 0.0f && total_time_ms > 0.0) {
          result_data.hasPowerTelemetry = true;
          result_data.powerWatts = avgPowerWatts;
          double durationSec = total_time_ms / 1000.0;
          result_data.energyJoules = avgPowerWatts * durationSec;

          if (result_data.component == "Compute" && result_data.operations > 0) {
            double ops12 = (static_cast<double>(result_data.operations) / durationSec) / 1e12;
            if (ops12 > 0.0) {
              result_data.joulesPerUnit = avgPowerWatts / ops12;
              result_data.unitPerWatt = (ops12 * 1000.0) / avgPowerWatts;
              result_data.efficiencyUnit = (result_data.metric == "TOPS") ? "J/TOP" : "J/TFLOP";
            }
          } else if (result_data.component == "Memory" && result_data.operations > 0) {
            double gbps = (static_cast<double>(result_data.operations) / durationSec) / 1e9;
            if (gbps > 0.0) {
              result_data.joulesPerUnit = avgPowerWatts / gbps;
              result_data.unitPerWatt = gbps / avgPowerWatts; // GB/Joule
              result_data.efficiencyUnit = "J/GB";
            }
          } else if (result_data.component == "Ray Tracing" && result_data.operations > 0) {
            if (result_data.metric == "GIS/s") {
              double gis = (static_cast<double>(result_data.operations) / durationSec) / 1e9;
              if (gis > 0.0) {
                result_data.joulesPerUnit = avgPowerWatts / gis;
                result_data.unitPerWatt = (gis * 1000.0) / avgPowerWatts; // MIS/Joule
                result_data.efficiencyUnit = "J/GIS";
              }
            } else {
              double mrays = (static_cast<double>(result_data.operations) / durationSec) / 1e6;
              if (mrays > 0.0) {
                result_data.joulesPerUnit = avgPowerWatts / mrays;
                result_data.unitPerWatt = (mrays * 1000.0) / avgPowerWatts; // kRays/Joule
                result_data.efficiencyUnit = "J/MRay";
              }
            }
          }

          if (verbose && result_data.hasPowerTelemetry) {
            std::cout << " [POWER & ENERGY] Measured: " << ResultFormatter::formatDouble(result_data.powerWatts, 1) << " W"
                      << " | Energy: " << ResultFormatter::formatDouble(result_data.energyJoules, 2) << " J";
            if (result_data.joulesPerUnit > 0.0) {
              std::cout << " | Efficiency: " << ResultFormatter::formatDouble(result_data.joulesPerUnit, 2) << " " << result_data.efficiencyUnit;
              if (result_data.efficiencyUnit == "J/TFLOP") {
                std::cout << " (" << ResultFormatter::formatDouble(result_data.unitPerWatt, 1) << " GFLOPS/W)";
              } else if (result_data.efficiencyUnit == "J/TOP") {
                std::cout << " (" << ResultFormatter::formatDouble(result_data.unitPerWatt, 1) << " GOPS/W)";
              } else if (result_data.efficiencyUnit == "J/GB") {
                std::cout << " (" << ResultFormatter::formatDouble(result_data.unitPerWatt, 2) << " GB/J)";
              } else if (result_data.efficiencyUnit == "J/MRay") {
                std::cout << " (" << ResultFormatter::formatDouble(result_data.unitPerWatt, 1) << " kRays/J)";
              } else if (result_data.efficiencyUnit == "J/GIS") {
                std::cout << " (" << ResultFormatter::formatDouble(result_data.unitPerWatt, 1) << " MIS/J)";
              }
            }
            std::cout << std::endl;
          }
        }

        formatter->addResult(result_data);
        double opsPerSec = (result_data.time_ms > 0.0 && result_data.operations > 0)
            ? (static_cast<double>(result_data.operations) / result_data.time_ms) * 1000.0
            : 0.0;
        char scoreBuf[64];
        if (result_data.metric.find("TFLOPS") != std::string::npos || result_data.metric.find("TOPS") != std::string::npos) {
          snprintf(scoreBuf, sizeof(scoreBuf), "%.2f %s", opsPerSec / 1e12, result_data.metric.c_str());
        } else if (result_data.metric.find("TB/s") != std::string::npos) {
          snprintf(scoreBuf, sizeof(scoreBuf), "%.2f TB/s", opsPerSec / 1e12);
        } else if (result_data.metric.find("GB/s") != std::string::npos) {
          snprintf(scoreBuf, sizeof(scoreBuf), "%.1f GB/s", opsPerSec / 1e9);
        } else if (result_data.metric.find("GIS/s") != std::string::npos) {
          snprintf(scoreBuf, sizeof(scoreBuf), "%.2f GIS/s", opsPerSec / 1e9);
        } else if (result_data.metric.find("MRays/s") != std::string::npos) {
          snprintf(scoreBuf, sizeof(scoreBuf), "%.1f MRays/s", opsPerSec / 1e6);
        } else if (result_data.metric.find("GPixels/s") != std::string::npos) {
          snprintf(scoreBuf, sizeof(scoreBuf), "%.2f GPixels/s", opsPerSec / 1e9);
        } else if (result_data.metric.find("ns") != std::string::npos) {
          double nsVal = (result_data.operations > 0) ? ((result_data.time_ms * 1e6) / result_data.operations) : result_data.time_ms;
          snprintf(scoreBuf, sizeof(scoreBuf), "%.1f ns", nsVal);
        } else if (result_data.metric.find("us") != std::string::npos) {
          double usVal = (result_data.operations > 0) ? ((result_data.time_ms * 1000.0) / result_data.operations) : (result_data.time_ms * 1000.0);
          snprintf(scoreBuf, sizeof(scoreBuf), "%.1f us", usVal);
        } else if (result_data.metric.find("M") != std::string::npos) {
          snprintf(scoreBuf, sizeof(scoreBuf), "%.1f %s", opsPerSec / 1e6, result_data.metric.c_str());
        } else {
          snprintf(scoreBuf, sizeof(scoreBuf), "%.1f %s", opsPerSec, result_data.metric.c_str());
        }

        if (isInteractive) {
          std::string disp = bench_name;
          if (disp.rfind("RayScheduling (", 0) == 0) {
            size_t secondOpen = disp.find(") (");
            if (secondOpen != std::string::npos && disp.back() == ')') {
              disp = disp.substr(secondOpen + 3, disp.length() - (secondOpen + 4));
            }
          } else if (disp.rfind("RayASBuild (", 0) == 0 && disp.back() == ')') {
            disp = disp.substr(12, disp.length() - 13);
          } else {
            size_t firstOpen = disp.find(" (");
            if (firstOpen != std::string::npos && disp.back() == ')') {
              disp = disp.substr(firstOpen + 2, disp.length() - (firstOpen + 3));
            }
          }
          if (disp.length() > 50) disp = disp.substr(0, 47) + "...";
          std::cout << "\r\033[K  \033[32m✓\033[0m [" << (taskIdx + 1) << "/" << tasks.size() << "] "
                    << disp << "  \033[33m" << scoreBuf << "\033[0m\n" << std::flush;
        } else if (!quiet && !verbose) {
          std::cout << " Done. [" << scoreBuf << "]" << std::endl;
        }
        numBenchmarksRun++;
        taskIdx++;
        if (onResult) {
          onResult(result_data);
        }
        context->waitIdle();
      } catch (const std::exception &e) {
        taskIdx++;
        executionFailure = true;
        if (!verbose) {
          if (!quiet) {
            std::cout << " Failed (" << e.what() << ")" << std::endl;
          } else {
            std::cerr << " [ERROR] Task " << bench_name << " failed: " << e.what() << std::endl;
          }
        } else {
          std::cerr << "Error running task " << bench_name << ": " << e.what() << std::endl;
        }

        std::string errStr = e.what();
        ResultData fail_data;
        fail_data.backendName = ComputeBackendFactory::getBackendName(context->getBackend());
        fail_data.deviceName = info.name;
        fail_data.benchmarkName = bench_name;
        fail_data.component = bench->GetComponent(i);
        fail_data.subcategory = bench->GetSubCategory(i);
        fail_data.metric = bench->GetMetric(i);
        fail_data.operations = 0;
        fail_data.time_ms = -2.0; // Signals error/failure
        fail_data.isEmulated = false;
        fail_data.isUnsupported = false;
        fail_data.maxWorkGroupSize = info.maxWorkGroupSize;
        fail_data.deviceIndex = context->getSelectedDeviceIndex();
        fail_data.configIndex = i;
        fail_data.sortWeight = bench->GetSortWeight(i);
        fail_data.width = effectiveWidth;
        fail_data.height = effectiveHeight;
        fail_data.errorString = errStr;

        formatter->addResult(fail_data);
        if (onResult) {
          onResult(fail_data);
        }

        bool isLost = (errStr.find("DEVICE_LOST") != std::string::npos ||
                       errStr.find("timed out") != std::string::npos ||
                       errStr.find("timeout") != std::string::npos ||
                       errStr.find("context lost") != std::string::npos ||
                       errStr.find("Device lost") != std::string::npos ||
                       errStr.find("result: -4") != std::string::npos);
        if (isLost) {
          deviceLost = true;
          std::cerr << "  [CRITICAL] GPU device hung or lost during " << bench_name
                    << ". Aborting remaining tasks on this device." << std::endl;
          for (size_t k = taskIdx; k < tasks.size(); ++k) {
            auto *abortedBench = tasks[k].bench;
            uint32_t abortedConfig = tasks[k].configIndex;
            std::string abortedName = abortedBench->GetConfigName(abortedConfig);
            if (abortedName.empty()) {
              abortedName = abortedBench->GetName();
            }
            ResultData abort_data;
            abort_data.backendName = ComputeBackendFactory::getBackendName(context->getBackend());
            abort_data.deviceName = info.name;
            abort_data.benchmarkName = abortedName;
            abort_data.component = abortedBench->GetComponent(abortedConfig);
            abort_data.subcategory = abortedBench->GetSubCategory(abortedConfig);
            abort_data.metric = abortedBench->GetMetric(abortedConfig);
            abort_data.operations = 0;
            abort_data.time_ms = -3.0; // Signals aborted
            abort_data.isEmulated = false;
            abort_data.isUnsupported = false;
            abort_data.maxWorkGroupSize = info.maxWorkGroupSize;
            abort_data.deviceIndex = context->getSelectedDeviceIndex();
            abort_data.configIndex = abortedConfig;
            abort_data.sortWeight = abortedBench->GetSortWeight(abortedConfig);
            abort_data.width = effectiveWidth;
            abort_data.height = effectiveHeight;
            abort_data.errorString = "Aborted: GPU device hung/lost";

            formatter->addResult(abort_data);
            if (onResult) {
              onResult(abort_data);
            }
          }
          break;
        }
      }
    }

    if (hasVisualVerification && !deviceLost) {
      if (!quiet && !verbose && !onResult) {
        std::cout << "\r\033[K  \033[32m✔\033[0m Benchmark execution complete ("
                  << tasks.size() << " workloads).\n\n  \033[1m[3/3] Visual Parity & Frame Export\033[0m..." << std::endl;
      }
      for (auto *bench : runnable) {
        if (bench->HasVisualVerification()) {
          try {
            bench->RunVisualVerification(isInteractive);
          } catch (const std::exception &e) {
            parityFailure = true;
            if (verbose || verifyParity) {
              std::cerr << "Visual verification error on " << bench->GetName() << ": " << e.what() << std::endl;
            }
          } catch (...) {
            parityFailure = true;
            if (verbose || verifyParity) {
              std::cerr << "Visual verification error on " << bench->GetName() << ": unknown exception" << std::endl;
            }
          }
        }
      }
      if (!quiet && !verbose && !onResult) {
        std::cout << "\r\033[K  \033[32m✔\033[0m Visual parity verification & frame export complete.\n" << std::endl;
      }
    } else {
      if (!quiet && isInteractive) {
        std::cout << "\r\033[K  \033[32m✔\033[0m Benchmark suite completed ("
                  << tasks.size() << " workloads).\n" << std::endl;
      }
    }

    for (auto *bench : runnable) {
      try {
        bench->Teardown();
      } catch (const std::exception &e) {
        if (bench->HasParityFailure()) {
          parityFailure = true;
        }
        std::string err = e.what();
        if (err.find("parity") != std::string::npos || err.find("Parity") != std::string::npos) {
          parityFailure = true;
        }
        if (verbose || verifyParity) {
          std::cerr << "Error tearing down " << bench->GetName() << ": "
                    << e.what() << std::endl;
        }
      } catch (...) {
        if (bench->HasParityFailure()) {
          parityFailure = true;
        }
        if (verbose || verifyParity) {
          std::cerr << "Error tearing down " << bench->GetName() << ": unknown exception"
                    << std::endl;
        }
      }
      if (bench->HasParityFailure()) {
        parityFailure = true;
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
        if (!quiet) {
          std::cout << " [System] Host CPU" << std::endl;
          if (verbose) {
            std::cout << "  - Threads:      "
                      << std::thread::hardware_concurrency() << std::endl;
          }
          std::cout << std::endl;
        }
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
          if (cancelToken && cancelToken->load()) {
            break;
          }
          std::string bName = bench->GetName();
          std::string config_name = bench->GetConfigName(i);
          std::string bench_name = bName;
          if (!config_name.empty()) {
            bench_name += " (" + config_name + ")";
          }
          std::string keyIndex = bName + "#" + std::to_string(i);

          if (!workloadFilter.empty()) {
            bool matched = false;
            for (const auto &filt : workloadFilter) {
              if (filt == bName || filt == keyIndex || filt == config_name || filt == bench_name) {
                matched = true;
                break;
              }
            }
            if (!matched) {
              continue;
            }
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
          while (total_time_ms < 500.0 && total_invocations < 1000) {
            if (cancelToken && cancelToken->load()) {
              break;
            }
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
        executionFailure = true;
        ResultData fail_data;
        fail_data.backendName = "System";
        fail_data.deviceName = "Host CPU";
        fail_data.benchmarkName = bench->GetName();
        fail_data.component = "Host System";
        fail_data.subcategory = bench->GetSubCategory(0);
        fail_data.metric = bench->GetMetric(0);
        fail_data.operations = 0;
        fail_data.time_ms = -2.0;
        fail_data.isEmulated = false;
        fail_data.isUnsupported = false;
        fail_data.maxWorkGroupSize = 0;
        fail_data.deviceIndex = 0xFFFFFFFF;
        fail_data.configIndex = 0;
        fail_data.sortWeight = bench->GetSortWeight(0);
        fail_data.errorString = e.what();
        formatter->addResult(fail_data);
        if (onResult) {
          onResult(fail_data);
        }
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
  if (onResult) {
    return;
  }
  if (verbose) {
    std::cout << "\r\033[K" << std::flush;
  }
  formatter->print();
}

void BenchmarkRunner::run(const std::vector<std::string> &benchmarks_to_run) {
  utils::SleepInhibitor sleepInhibitor("Running GPU compute benchmarks");
  initRunConfig(benchmarks_to_run);

  for (auto *context : contexts) {
    if (cancelToken && cancelToken->load()) {
      break;
    }
    runForContext(context, benchmarks_to_run);
  }

  if (!cancelToken || !cancelToken->load()) {
    runHostBenchmarks(benchmarks_to_run);
  }

  if (!onResult) {
    printReport();
  }
}
