#include "test_harness.h"
#include "benchmarks/PixelFillRateBench.h"
#include "utils/KernelPath.h"
#include <algorithm>
#include <filesystem>

TEST_CASE(PixelFill, MetadataAndInvariants) {
  PixelFillRateBench bench;
  ASSERT_EQ(std::string(bench.GetName()), "Pixel Fill Rate");
  ASSERT_EQ(std::string(bench.GetMetric()), "GPixels/s");
  ASSERT_EQ(std::string(bench.GetComponent()), "Graphics");
  ASSERT_EQ(std::string(bench.GetSubCategory()), "ROP Throughput");
  ASSERT_EQ(bench.GetSortWeight(), 450);

  auto aliases = bench.GetAliases();
  ASSERT_TRUE(std::find(aliases.begin(), aliases.end(), "pixelfill") != aliases.end());
  ASSERT_TRUE(std::find(aliases.begin(), aliases.end(), "fillrate") != aliases.end());
  ASSERT_TRUE(std::find(aliases.begin(), aliases.end(), "rop") != aliases.end());

  ASSERT_EQ(bench.GetNumConfigs(), 3u);
  ASSERT_EQ(bench.GetConfigName(0), "RGBA8 Color Fill");
  ASSERT_EQ(bench.GetConfigName(1), "RGBA16F HDR Fill");
  ASSERT_EQ(bench.GetConfigName(2), "Alpha Blending Fill");
  ASSERT_EQ(bench.GetConfigName(3), "");
}

TEST_CASE(PixelFill, SupportAndLimitations) {
  PixelFillRateBench bench;
  DeviceInfo info{};

  ASSERT_FALSE(bench.IsSupported(info, nullptr));
  ASSERT_TRUE(bench.GetSupportLimitation() == IBenchmark::SupportLimitation::kApi);
  ASSERT_TRUE(bench.GetSupportLimitation(info, nullptr) == IBenchmark::SupportLimitation::kApi);

  std::string defaultNote = bench.GetSupportNote();
  ASSERT_TRUE(defaultNote.find("dynamic rendering") != std::string::npos);

  std::string contextNote = bench.GetSupportNote(info, nullptr);
  ASSERT_TRUE(contextNote.find("dynamic rendering") != std::string::npos);
}

TEST_CASE(PixelFill, ResultCalculations) {
  PixelFillRateBench bench;
  auto res0 = bench.GetResult(0);
  // Total pixels rendered = width (8192) * height (8192) * passesPerDispatch (8)
  constexpr uint64_t expectedPixels = 8192ULL * 8192ULL * 8ULL;
  ASSERT_EQ(res0.operations, expectedPixels);

  auto res1 = bench.GetResult(1);
  ASSERT_EQ(res1.operations, expectedPixels);

  auto res2 = bench.GetResult(2);
  ASSERT_EQ(res2.operations, expectedPixels);
}

TEST_CASE(PixelFill, ShaderArtifactsExist) {
  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());
  std::filesystem::path kp(kdir);
  bool vertExists = std::filesystem::exists(kp / "vulkan" / "pixel_fill.vert") ||
                    std::filesystem::exists(kp / "vulkan" / "pixel_fill.vert.spv");
  bool fragExists = std::filesystem::exists(kp / "vulkan" / "pixel_fill.frag") ||
                    std::filesystem::exists(kp / "vulkan" / "pixel_fill.frag.spv");
  ASSERT_TRUE(vertExists);
  ASSERT_TRUE(fragExists);
}
