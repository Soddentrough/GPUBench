#include "test_harness.h"

#include "benchmarks/IBenchmark.h"
#include "core/BenchmarkRunner.h"
#include "core/RunnerAPI.h"
#include "scene/GltfScene.h"
#include "utils/ShaderCache.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

class ScalingProbeBench : public IBenchmark {
public:
  const char *GetName() const override { return "ScalingProbe"; }
  bool IsSupported(const DeviceInfo &, IComputeContext *) const override { return true; }
  void Setup(IComputeContext &, const std::string &) override {}
  void Run(uint32_t) override {}
  void Teardown() override {}
  BenchmarkResult GetResult(uint32_t) const override { return {1, 0.0}; }
  bool SupportsWorkScaling(uint32_t) const override { return true; }

  uint32_t probeGroups(uint32_t baseGroups) const { return ScaledGroups(baseGroups); }
  uint64_t probeOps(uint64_t fullOps, uint32_t baseGroups) const {
    return ScaledOps(fullOps, baseGroups);
  }
};

struct DirCleanupGuard {
  std::filesystem::path dir;
  ~DirCleanupGuard() {
    if (!dir.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
    }
  }
};

struct FileCleanupGuard {
  std::filesystem::path file;
  ~FileCleanupGuard() {
    if (!file.empty()) {
      std::error_code ec;
      std::filesystem::remove(file, ec);
    }
  }
};

std::string findAssetModel(const std::string &name) {
  namespace fs = std::filesystem;
  for (const auto &base : {fs::path("assets/models"),
                           fs::path("../assets/models"),
                           fs::path("../../assets/models")}) {
    std::error_code ec;
    fs::path p = base / name;
    if (fs::exists(p, ec)) {
      return p.string();
    }
  }
  return "";
}

} // namespace

TEST_CASE(RunnerAndUtils, BenchmarkRunnerGroupAndAliasExpansion) {
  BenchmarkRunner runner({});
  const auto groups = runner.getAvailableGroups();
  ASSERT_GE(groups.size(), 7u);

  std::set<std::string> groupIds;
  for (const auto &g : groups) {
    groupIds.insert(g.id);
    ASSERT_FALSE(g.benchmarks.empty());
  }
  ASSERT_TRUE(groupIds.count("compute") == 1u);
  ASSERT_TRUE(groupIds.count("memory") == 1u);
  ASSERT_TRUE(groupIds.count("graphics") == 1u);
  ASSERT_TRUE(groupIds.count("raster") == 1u);
  ASSERT_TRUE(groupIds.count("raytracing") == 1u);
  ASSERT_TRUE(groupIds.count("scheduling") == 1u);
  ASSERT_TRUE(groupIds.count("system") == 1u);

  const auto sched = runner.expandGroups({"scheduling"});
  ASSERT_TRUE(std::find(sched.begin(), sched.end(), "GPU-Driven Compute Dispatch") != sched.end());
  ASSERT_TRUE(std::find(sched.begin(), sched.end(), "Async Compute & Copy Overlap") != sched.end());

  const auto gfx = runner.expandGroups({"gfx"});
  ASSERT_TRUE(std::find(gfx.begin(), gfx.end(), "Pixel Fill Rate") != gfx.end());
  ASSERT_TRUE(std::find(gfx.begin(), gfx.end(), "RayRawTraversal") != gfx.end());
  ASSERT_TRUE(std::find(gfx.begin(), gfx.end(), "RayScheduling") != gfx.end());

  const auto all = runner.expandGroups({"all"});
  std::set<std::string> uniqueAll(all.begin(), all.end());
  ASSERT_EQ(uniqueAll.size(), all.size());
  ASSERT_GE(all.size(), 25u);
}

TEST_CASE(RunnerAndUtils, AdaptiveWorkScalingMathInvariants) {
  ScalingProbeBench bench;
  constexpr uint32_t kBaseGroups = 4096u;
  constexpr uint64_t kFullOps = 4096ULL * 1000000ULL;

  bench.SetWorkScale(0, 1.0);
  ASSERT_EQ(bench.probeGroups(kBaseGroups), kBaseGroups);
  ASSERT_EQ(bench.probeOps(kFullOps, kBaseGroups), kFullOps);

  bench.SetWorkScale(0, 0.25);
  ASSERT_EQ(bench.probeGroups(kBaseGroups), 1024u);
  ASSERT_EQ(bench.probeOps(kFullOps, kBaseGroups), kFullOps / 4ULL);

  // Tiny scale clamps ScaledGroups to at least 1 and scales ops proportionally
  bench.SetWorkScale(0, 1e-9);
  ASSERT_EQ(bench.probeGroups(kBaseGroups), 1u);
  ASSERT_EQ(bench.probeOps(kFullOps, kBaseGroups), 1000000ULL);

  // Scale > 1.0 clamps to baseGroups
  bench.SetWorkScale(0, 5.0);
  ASSERT_EQ(bench.probeGroups(kBaseGroups), kBaseGroups);
  ASSERT_EQ(bench.probeOps(kFullOps, kBaseGroups), kFullOps);
}

TEST_CASE(RunnerAndUtils, ShaderCacheRoundTripAndCorruptRejection) {
  DeviceInfo dev{};
#ifdef _WIN32
  const unsigned long pid = static_cast<unsigned long>(_getpid());
#else
  const long pid = static_cast<long>(getpid());
#endif
  dev.name = "UnitTest_ShaderCache_" + std::to_string(pid);
  dev.vendorID = 0x1002;
  dev.deviceID = 0x7551;

  DirCleanupGuard cacheCleanup{utils::ShaderCache::getCacheDir(dev)};

  // 1. Vulkan SPIR-V round-trip
  const std::vector<uint32_t> spvPayload = {0x07230203u, 0x00010500u, 0x00000001u, 0xDEADBEEFu};
  utils::ShaderCache::saveVulkanCache("unit_test_kernel.comp", dev, spvPayload);
  std::vector<uint32_t> spvLoaded;
  ASSERT_TRUE(utils::ShaderCache::loadVulkanCache("unit_test_kernel.comp", dev, spvLoaded));
  ASSERT_EQ(spvLoaded.size(), spvPayload.size());
  for (size_t i = 0; i < spvPayload.size(); ++i) {
    ASSERT_EQ(spvLoaded[i], spvPayload[i]);
  }

  // 2. ROCm & OpenCL byte payload round-trip
  const std::vector<char> binPayload = {'E', 'L', 'F', 1, 2, 3, 4, 5};
  utils::ShaderCache::saveROCmCache("unit_test_kernel.hip", dev, binPayload);
  std::vector<char> rocmLoaded;
  ASSERT_TRUE(utils::ShaderCache::loadROCmCache("unit_test_kernel.hip", dev, rocmLoaded));
  ASSERT_EQ(rocmLoaded.size(), binPayload.size());

  utils::ShaderCache::saveOpenCLCache("unit_test_kernel.cl", dev, binPayload);
  std::vector<char> oclLoaded;
  ASSERT_TRUE(utils::ShaderCache::loadOpenCLCache("unit_test_kernel.cl", dev, oclLoaded));
  ASSERT_EQ(oclLoaded.size(), binPayload.size());

  // 3. Reject unaligned / truncated Vulkan cache binary (3 bytes != multiple of uint32_t)
  const std::filesystem::path badFile = cacheCleanup.dir / "corrupt_unaligned.spv.cache";
  {
    std::ofstream ofs(badFile, std::ios::binary | std::ios::trunc);
    const char bytes[3] = {'b', 'a', 'd'};
    ofs.write(bytes, sizeof(bytes));
  }
  std::vector<uint32_t> badLoaded;
  ASSERT_FALSE(utils::ShaderCache::loadVulkanCache("corrupt_unaligned", dev, badLoaded));

  // 4. Reject empty cache file
  const std::filesystem::path emptyFile = cacheCleanup.dir / "empty_zero.spv.cache";
  {
    std::ofstream ofs(emptyFile, std::ios::binary | std::ios::trunc);
  }
  ASSERT_FALSE(utils::ShaderCache::loadVulkanCache("empty_zero", dev, badLoaded));

  // 5. Reject stale cache when source kernel file is newer than cache file
  const std::filesystem::path srcFile = cacheCleanup.dir / "freshness_probe.comp";
  {
    std::ofstream ofs(srcFile);
    ofs << "// initial\n";
  }
  utils::ShaderCache::saveVulkanCache(srcFile.string(), dev, spvPayload);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  {
    std::ofstream ofs(srcFile, std::ios::app);
    ofs << "// updated\n";
  }
  std::error_code ec;
  std::filesystem::last_write_time(
      srcFile, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(5), ec);
  std::vector<uint32_t> staleLoaded;
  ASSERT_FALSE(utils::ShaderCache::loadVulkanCache(srcFile.string(), dev, staleLoaded));
}

TEST_CASE(RunnerAndUtils, GltfSceneLoadShippedAssetsAndRejectMissing) {
  const std::string toycarPath = findAssetModel("toycar.glb");
  ASSERT_FALSE(toycarPath.empty());

  GltfScene scene;
  std::string err;
  ASSERT_TRUE(scene.loadFromFile(toycarPath, err));
  ASSERT_GT(scene.getVertexCount(), 0u);
  ASSERT_GT(scene.getTriangleCount(), 0u);
  ASSERT_FALSE(scene.getMaterials().empty());
  ASSERT_EQ(scene.getTriangleMaterials().size(), static_cast<size_t>(scene.getTriangleCount()));

  float bmin[3]{}, bmax[3]{};
  scene.getBounds(bmin, bmax);
  for (int axis = 0; axis < 3; ++axis) {
    ASSERT_LT(bmin[axis], bmax[axis]);
  }

  // Missing file rejection
  GltfScene missing;
  std::string missingErr;
  ASSERT_FALSE(missing.loadFromFile("definitely_missing_scene_model_999.glb", missingErr));
  ASSERT_FALSE(missingErr.empty());

  // Corrupt truncated GLB rejection
  const std::filesystem::path badGlb =
      std::filesystem::temp_directory_path() / "gpubench_corrupt_test_scene.glb";
  FileCleanupGuard glbGuard{badGlb};
  {
    std::ofstream ofs(badGlb, std::ios::binary | std::ios::trunc);
    ofs << "glTF_CORRUPT_HEADER";
  }
  GltfScene corrupt;
  std::string corruptErr;
  ASSERT_FALSE(corrupt.loadFromFile(badGlb.string(), corruptErr));
}

TEST_CASE(RunnerAndUtils, BackendAwareCatalogExplanations) {
  for (const char *backend : {"vulkan", "rocm"}) {
    const auto catalog = GetBenchmarkMetadataCatalogAPI(backend, 0);
    if (catalog.empty()) continue;
    for (const auto &meta : catalog) {
      ASSERT_GE(meta.whatItTests.size(), 20u);
      ASSERT_GE(meta.pipelineImpact.size(), 20u);
      ASSERT_GE(meta.method.size(), 20u);
      if (std::string(backend) == "rocm" &&
          meta.benchmarkName == "GPU-Driven Compute Dispatch" &&
          meta.configIndex >= 2u && meta.configIndex <= 4u) {
        ASSERT_TRUE(meta.method.find("vkCmdDispatchIndirect") == std::string::npos);
        ASSERT_TRUE(meta.whatItTests.find("vkCmdDispatchIndirect") == std::string::npos);
      }
    }
  }
}
