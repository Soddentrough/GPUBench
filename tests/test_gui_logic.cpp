#include "test_harness.h"
#include "core/RunnerAPI.h"
#include "gui/GuiSuiteLogic.h"

#include <set>
#include <string>
#include <vector>

using namespace gpubench::gui;

TEST_CASE(GuiLogic, CatalogSubgroupsAndDescriptionsComplete) {
  const auto catalog = GetBenchmarkMetadataCatalogAPI();
  ASSERT_FALSE(catalog.empty());

  const std::set<std::string> validCategories = {
      "Compute", "Memory", "Ray Tracing", "Scheduling", "Raster", "Host System"};

  for (const auto &meta : catalog) {
    std::string category;
    std::string sgDesc;
    std::string sgTitle = subgroupTitleForCatalogEntry(meta, category, sgDesc);

    ASSERT_TRUE(validCategories.find(category) != validCategories.end());
    ASSERT_FALSE(sgTitle.empty());
    ASSERT_GE(sgDesc.size(), 20u);

    std::string fb = fallbackDescriptionForBenchmark(meta);
    ASSERT_GE(fb.size(), 20u);
    ASSERT_NE(std::string(shortCategoryTag(category)).size(), 0u);
  }
}

TEST_CASE(GuiLogic, CompactSlotLabelsShortAndDistinct) {
  const std::vector<std::string> dispatchNames = {
      "Compute Fixed Grid - CPU Direct (100% Active)",
      "Compute Fixed Grid - CPU Direct (10% Active)",
      "GPU Indirect Compute Dispatch (100% Active)",
      "GPU Indirect Compute Dispatch (10% Active)",
      "GPU Zero-Dispatch Pruning (0% Active)",
      "1024-Shader Sparse Buckets (CPU Fixed Grid)",
      "1024-Shader Sparse Buckets (1024x Indirect)",
      "1024-Shader Sparse Buckets (Single DGC Call)",
  };
  std::set<std::string> seenDispatch;
  for (const auto &nm : dispatchNames) {
    std::string compact =
        compactWorkloadSlotLabel(nm, "GPU-Driven Compute Dispatch (Indirect / DGC)");
    ASSERT_FALSE(compact.empty());
    ASSERT_LT(compact.size(), nm.size());
    ASSERT_TRUE(seenDispatch.insert(compact).second);
  }

  const std::vector<std::string> ilpNames = {
      "FP32 Serialized (1 Chain / Low ILP)",
      "FP32 Moderate ILP (8 Chains)",
      "FP32 Peak ILP / Dual-Issue (16 Chains)",
      "INT32 Serialized (1 Chain / Low ILP)",
      "INT32 Moderate ILP (8 Chains)",
      "INT32 Peak ILP / Dual-Issue (16 Chains)",
      "Concurrent Mixed (FP32 + INT32)",
  };
  std::set<std::string> seenIlp;
  for (const auto &nm : ilpNames) {
    std::string compact =
        compactWorkloadSlotLabel(nm, "Shader ILP & Dual-Issue ALU Scaling");
    ASSERT_FALSE(compact.empty());
    ASSERT_TRUE(seenIlp.insert(compact).second);
  }
}

TEST_CASE(GuiLogic, MatchesItemAcrossDomainsAndDevices) {
  // 1. Standard GPU compute/scheduling item matching by deviceIndex & configIndex
  ResultData gpu0Ind{};
  gpu0Ind.backendName = "Vulkan";
  gpu0Ind.deviceIndex = 0;
  gpu0Ind.deviceName = "GPU 0";
  gpu0Ind.component = "Scheduling";
  gpu0Ind.benchmarkName = "GPU-Driven Compute Dispatch (1024-Shader Sparse Buckets (Single DGC Call))";
  gpu0Ind.configIndex = 7;

  ASSERT_TRUE(matchesWorkloadItem(
      gpu0Ind, "GPU-Driven Compute Dispatch",
      "1024-Shader Sparse Buckets (Single DGC Call)", "Scheduling", 7, 0));
  // Wrong configIndex or wrong GPU device must be rejected
  ASSERT_FALSE(matchesWorkloadItem(
      gpu0Ind, "GPU-Driven Compute Dispatch",
      "1024-Shader Sparse Buckets (Single DGC Call)", "Scheduling", 6, 0));
  ASSERT_FALSE(matchesWorkloadItem(
      gpu0Ind, "GPU-Driven Compute Dispatch",
      "1024-Shader Sparse Buckets (Single DGC Call)", "Scheduling", 7, 1));
  // GPU result must never match a Host System slot
  ASSERT_FALSE(matchesWorkloadItem(
      gpu0Ind, "GPU-Driven Compute Dispatch",
      "1024-Shader Sparse Buckets (Single DGC Call)", "System", 7, 0));

  // 2. Multi-precision Vector/Matrix normalized records
  ResultData fp16Mat{};
  fp16Mat.backendName = "Vulkan";
  fp16Mat.deviceIndex = 1;
  fp16Mat.component = "Compute";
  fp16Mat.benchmarkName = "Matrix";
  fp16Mat.subcategory = "FP16";
  fp16Mat.configIndex = 1;

  ASSERT_TRUE(matchesWorkloadItem(fp16Mat, "FP16", "FP16 (Matrix)", "Compute", 1, 1));
  ASSERT_FALSE(matchesWorkloadItem(fp16Mat, "FP16", "FP16 (Vector)", "Compute", 0, 1));
  ASSERT_FALSE(matchesWorkloadItem(fp16Mat, "BF16", "BF16 (Matrix)", "Compute", 1, 1));

  // 3. RayScheduling across scenes
  ResultData rsShowroom{};
  rsShowroom.backendName = "Vulkan";
  rsShowroom.deviceIndex = 1;
  rsShowroom.component = "Ray Tracing";
  rsShowroom.benchmarkName = "RayScheduling (Showroom Studio) (Path Tracing (1 SPP) (DGC))";
  rsShowroom.configIndex = 5;

  ASSERT_TRUE(matchesWorkloadItem(
      rsShowroom, "RayScheduling", "Path Tracing (1 SPP) (DGC)", "Ray Tracing", 5, 1));
  ASSERT_FALSE(matchesWorkloadItem(
      rsShowroom, "RayScheduling", "Path Tracing (1 SPP) (Megakernel)", "Ray Tracing", 4, 1));

  // 4. Cache Latency Curve prefix matching
  ResultData curvePt{};
  curvePt.backendName = "Vulkan";
  curvePt.deviceIndex = 0;
  curvePt.component = "Memory";
  curvePt.benchmarkName = "Cache Latency Curve (64 KB)";
  curvePt.configIndex = 2;

  ASSERT_TRUE(matchesWorkloadItem(
      curvePt, "Cache Latency Curve", "Cache Latency Curve (16 KB -> 256 MB Sweep)",
      "Memory", -1, 0));

  // 5. Host System result matching
  ResultData sysBw{};
  sysBw.backendName = "System";
  sysBw.deviceIndex = UINT32_MAX;
  sysBw.deviceName = "Host CPU";
  sysBw.component = "System";
  sysBw.benchmarkName = "System Memory Bandwidth (Read)";
  sysBw.configIndex = 0;

  ASSERT_TRUE(matchesWorkloadItem(
      sysBw, "System Memory Bandwidth", "Read", "System", 0, 1));
  ASSERT_FALSE(matchesWorkloadItem(
      sysBw, "System Memory Bandwidth", "Read", "Memory", 0, 1));
}
