#include "test_harness.h"
#include "benchmarks/RaySchedulingBench.h"
#include "core/ComputeBackendFactory.h"
#include "core/VulkanContext.h"
#include "utils/ImageExport.h"
#include "utils/KernelPath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace gpubench::test {

struct RaySchedulingParityHelperTag {};

template <>
struct TestBody<RaySchedulingParityHelperTag> {
  static std::unique_ptr<IComputeContext> createParityVulkanContext() {
    if (!ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
      return nullptr;
    }
    try {
      auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
      if (!ctx) {
        return nullptr;
      }
      const auto &devs = ctx->getDevices();
      if (devs.empty()) {
        return nullptr;
      }
      const uint32_t devIdx = preferredTestDeviceIndex(devs.size());
      ctx->pickDevice(devIdx);
      return ctx;
    } catch (...) {
      return nullptr;
    }
  }

  static void runSceneParity(RaySchedulingBench::SceneType scene,
                             const char *sceneName) {
    auto ctx = createParityVulkanContext();
    if (!ctx) {
      SKIP_TEST("Vulkan backend unavailable");
    }

    std::string kdir = KernelPath::find();
    ASSERT_FALSE(kdir.empty());

    RaySchedulingBench bench(scene);
    DeviceInfo info = ctx->getCurrentDeviceInfo();
    if (!bench.IsSupported(info, ctx.get())) {
      SKIP_TEST("RayScheduling unsupported on device");
    }

    constexpr uint32_t kW = 256u;
    constexpr uint32_t kH = 144u;
    constexpr size_t kPixels = static_cast<size_t>(kW) * kH;
    constexpr size_t kFbFloats = kPixels * 4u;
    constexpr size_t kFbBytes = kFbFloats * sizeof(float);

    bench.SetResolution(kW, kH);
    bench.Setup(*ctx, kdir);

    struct TeardownGuard {
      RaySchedulingBench &b;
      ~TeardownGuard() { b.Teardown(); }
    } guard{bench};

    struct ParityPair {
      uint32_t mkCfg;
      uint32_t dgcCfg;
      const char *label;
      bool hasFramebuffer;
    };
    static constexpr ParityPair kPairs[] = {
        {0u, 1u, "MaterialShading_LegacyIndirect", true},
        {0u, 2u, "MaterialShading_DGC", true},
        {4u, 5u, "FullFramePBR_LegacyIndirect", true},
        {4u, 6u, "FullFramePBR_DGC", true},
        {8u, 9u, "PathTracing_1SPP", true},
        {11u, 12u, "PathTracing_16SPP", true},
        {13u, 14u, "PrimaryGBuffer", true},
        {17u, 18u, "IncoherentGI", false},
        {19u, 20u, "DirectionalShadows", true},
        {21u, 22u, "MultiLight_128", true},
    };

    const uint32_t zeroStats[4] = {0u, 0u, 0u, 0u};
    const std::vector<uint32_t> zeroHeader(4096u, 0u);
    const std::vector<float> sentinelTrad(kFbFloats, -1.0f);
    const std::vector<float> sentinelWork(kFbFloats, -2.0f);

    std::vector<float> hdrTrad(kFbFloats, 0.0f);
    std::vector<float> hdrWork(kFbFloats, 0.0f);
    std::vector<uint8_t> ldrTrad, ldrWork, ldrDiff;

    uint32_t verifiedPairs = 0u;
    for (const auto &p : kPairs) {
      if (!bench.IsConfigSupported(p.dgcCfg, info, ctx.get()) ||
          bench.unsupportedConfig[p.dgcCfg]) {
        std::cout << "[SKIP " << p.label << ": "
                  << bench.GetConfigSupportNote(p.dgcCfg, info, ctx.get()) << "] ";
        continue;
      }

      // 1. Run Megakernel baseline
      ctx->writeBuffer(bench.resultBuffer, 0, sizeof(zeroStats), zeroStats);
      ctx->writeBuffer(bench.workListBuffer, 0,
                       zeroHeader.size() * sizeof(uint32_t), zeroHeader.data());
      if (p.hasFramebuffer) {
        ctx->writeBuffer(bench.fbTraditional, 0, kFbBytes, sentinelTrad.data());
        ctx->writeBuffer(bench.fbWorkList, 0, kFbBytes, sentinelWork.data());
      }
      bench.captureFramebuffer = true;
      bench.lastRunConfig = 0xFFFFFFFFu;
      bench.Run(p.mkCfg);
      ctx->waitIdle();

      uint32_t statsMK[4] = {0u, 0u, 0u, 0u};
      ctx->readBuffer(bench.resultBuffer, 0, sizeof(statsMK), statsMK);
      if (p.hasFramebuffer) {
        ctx->readBuffer(bench.fbTraditional, 0, kFbBytes, hdrTrad.data());
      }

      // 2. Run DGC counterpart
      ctx->writeBuffer(bench.resultBuffer, 0, sizeof(zeroStats), zeroStats);
      ctx->writeBuffer(bench.workListBuffer, 0,
                       zeroHeader.size() * sizeof(uint32_t), zeroHeader.data());
      bench.captureFramebuffer = true;
      bench.lastRunConfig = 0xFFFFFFFFu;
      bench.Run(p.dgcCfg);
      ctx->waitIdle();

      uint32_t statsDGC[4] = {0u, 0u, 0u, 0u};
      ctx->readBuffer(bench.resultBuffer, 0, sizeof(statsDGC), statsDGC);
      if (p.hasFramebuffer) {
        ctx->readBuffer(bench.fbWorkList, 0, kFbBytes, hdrWork.data());
      }

      if (statsMK[0] == 0u || statsMK[1] == 0u ||
          statsMK[0] != statsDGC[0] || statsMK[1] != statsDGC[1]) {
        std::cerr << "\n[PARITY COUNTER MISMATCH] scene=" << sceneName
                  << " pair=" << p.label << " (cfg " << p.mkCfg << " vs " << p.dgcCfg << ")"
                  << " MK=(rays=" << statsMK[0] << ", hits=" << statsMK[1] << ")"
                  << " DGC=(rays=" << statsDGC[0] << ", hits=" << statsDGC[1] << ")\n";
      }
      ASSERT_TRUE(statsMK[0] > 0u && statsMK[1] > 0u);
      ASSERT_EQ(statsMK[0], statsDGC[0]);
      ASSERT_EQ(statsMK[1], statsDGC[1]);

      if (p.hasFramebuffer) {
        for (size_t i = 0; i < kPixels; ++i) {
          const size_t base = i * 4u;
          for (size_t c = 0; c < 4u; ++c) {
            if (!std::isfinite(hdrTrad[base + c]) || hdrTrad[base + c] < 0.0f) {
              std::cerr << "\n[INVALID MK PIXEL] scene=" << sceneName
                        << " pair=" << p.label << " pixel=" << i << " ch=" << c
                        << " val=" << hdrTrad[base + c] << "\n";
            }
            if (!std::isfinite(hdrWork[base + c]) || hdrWork[base + c] < 0.0f) {
              std::cerr << "\n[INVALID DGC PIXEL] scene=" << sceneName
                        << " pair=" << p.label << " pixel=" << i << " ch=" << c
                        << " val=" << hdrWork[base + c] << "\n";
            }
            ASSERT_TRUE(std::isfinite(hdrTrad[base + c]) && hdrTrad[base + c] >= 0.0f);
            ASSERT_TRUE(std::isfinite(hdrWork[base + c]) && hdrWork[base + c] >= 0.0f);
          }
        }

        auto metrics = gpubench::ImageExport::compareAndTonemap(
            hdrTrad.data(), hdrWork.data(), kW, kH, ldrTrad, ldrWork, ldrDiff);
        const uint32_t maxDiffPixels =
            std::max(8u, static_cast<uint32_t>(metrics.totalPixels * 0.0005f));

        std::cout << "[" << p.label
                  << " psnr=" << std::fixed << std::setprecision(1) << metrics.psnr
                  << " hdrRmse=" << std::scientific << std::setprecision(2) << metrics.hdrRmse
                  << std::fixed
                  << " diff=" << metrics.diffPixels
                  << " rays=" << statsMK[0]
                  << " hits=" << statsMK[1] << "] ";

        if (metrics.psnr < 45.0f || metrics.diffPixels > maxDiffPixels ||
            metrics.hdrRmse > 0.01f) {
          std::cerr << "\n[PIXEL PARITY FAILURE] scene=" << sceneName
                    << " pair=" << p.label
                    << " psnr=" << metrics.psnr
                    << " diffPixels=" << metrics.diffPixels << " (max=" << maxDiffPixels << ")"
                    << " hdrRmse=" << metrics.hdrRmse
                    << " hdrMaxAbs=" << metrics.hdrMaxAbsDelta << "\n";
        }
        ASSERT_TRUE(metrics.psnr >= 45.0f);
        ASSERT_TRUE(metrics.diffPixels <= maxDiffPixels);
        ASSERT_TRUE(metrics.hdrRmse <= 0.01f);
      } else {
        std::cout << "[" << p.label
                  << " rays=" << statsMK[0]
                  << " hits=" << statsMK[1] << "] ";
      }

      ++verifiedPairs;
    }

    if (info.dgcSupported) {
      ASSERT_TRUE(verifiedPairs >= 8u);
    }
  }
};

} // namespace gpubench::test

TEST_CASE(RaySchedulingParity, Showroom_All9Pairs) {
  ::gpubench::test::TestBody<::gpubench::test::RaySchedulingParityHelperTag>::runSceneParity(
      RaySchedulingBench::SceneType::Showroom, "Showroom");
}

TEST_CASE(RaySchedulingParity, IndoorAtrium_All9Pairs) {
  ::gpubench::test::TestBody<::gpubench::test::RaySchedulingParityHelperTag>::runSceneParity(
      RaySchedulingBench::SceneType::IndoorAtrium, "IndoorAtrium");
}

TEST_CASE(RaySchedulingParity, OutdoorLandscape_All9Pairs) {
  ::gpubench::test::TestBody<::gpubench::test::RaySchedulingParityHelperTag>::runSceneParity(
      RaySchedulingBench::SceneType::OutdoorLandscape, "OutdoorLandscape");
}

TEST_CASE(RaySchedulingParity, AAAOutdoorForest_All9Pairs) {
  ::gpubench::test::TestBody<::gpubench::test::RaySchedulingParityHelperTag>::runSceneParity(
      RaySchedulingBench::SceneType::AAAOutdoorForest, "AAAOutdoorForest");
}
