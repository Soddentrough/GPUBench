#include "test_harness.h"

#include "core/ComputeBackendFactory.h"
#include "core/VulkanContext.h"
#include "showcase/ShowcaseRenderer.h"
#include "showcase/ShowcaseScene.h"
#include "utils/KernelPath.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<IComputeContext> createShowcaseVulkanContext() {
  if (!ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
    return nullptr;
  }
  try {
    auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
    if (!ctx) return nullptr;
    const auto &devs = ctx->getDevices();
    if (devs.empty()) return nullptr;
    const uint32_t devIdx = gpubench::test::preferredTestDeviceIndex(devs.size());
    ctx->pickDevice(devIdx);
    return ctx;
  } catch (...) {
    return nullptr;
  }
}

std::string findShowcaseKernelDir() {
  namespace fs = std::filesystem;
  const std::string kdir = KernelPath::find();
  for (const auto &p : {
           fs::path(kdir) / "vulkan",
           fs::path("build/kernels/vulkan"),
           fs::path("kernels/vulkan"),
           fs::path("../kernels/vulkan"),
       }) {
    std::error_code ec;
    if (fs::exists(p / "showcase_naive.vert.spv", ec)) {
      return p.string();
    }
  }
  return (fs::path(kdir) / "vulkan").string();
}

void computeImageParity(const std::vector<uint8_t> &a,
                        const std::vector<uint8_t> &b,
                        double &outPsnrDb,
                        double &outMeanLumaA,
                        double &outMeanLumaB) {
  const size_t n = std::min(a.size(), b.size());
  double sqSum = 0.0;
  double sumA = 0.0;
  double sumB = 0.0;
  const size_t pixels = n / 4u;
  for (size_t i = 0; i + 3 < n; i += 4u) {
    for (int ch = 0; ch < 3; ++ch) {
      const double da = static_cast<double>(a[i + ch]) / 255.0;
      const double db = static_cast<double>(b[i + ch]) / 255.0;
      const double diff = da - db;
      sqSum += diff * diff;
      sumA += da;
      sumB += db;
    }
  }
  const double denom = std::max<double>(1.0, static_cast<double>(pixels) * 3.0);
  const double mse = sqSum / denom;
  outMeanLumaA = (sumA / denom) * 255.0;
  outMeanLumaB = (sumB / denom) * 255.0;
  outPsnrDb = (mse < 1e-10) ? 100.0 : (-10.0 * std::log10(mse));
}

} // namespace

TEST_CASE(ShowcaseValidation, SceneEnrichmentAndMeshletIntegrity) {
  using namespace gpubench::showcase;

  ShowcaseSceneData scene;
  std::string err;
  ASSERT_TRUE(buildShowcaseScene(scene, err));

  ASSERT_FALSE(scene.vertices.empty());
  ASSERT_FALSE(scene.indices.empty());
  ASSERT_FALSE(scene.meshlets.empty());
  ASSERT_FALSE(scene.instances.empty());
  ASSERT_FALSE(scene.materials.empty());
  ASSERT_FALSE(scene.lights.empty());
  ASSERT_FALSE(scene.phase0Packets.empty());
  ASSERT_FALSE(scene.phase1Packets.empty());
  ASSERT_TRUE(scene.presets.size() >= 4u);

  for (const auto &m : scene.meshlets) {
    ASSERT_TRUE(m.vertexCount > 0u && m.vertexCount <= kMaxMeshletVertices);
    ASSERT_TRUE(m.triangleCount > 0u && m.triangleCount <= kMaxMeshletTriangles);
    ASSERT_TRUE(m.vertexOffset + m.vertexCount <= scene.meshletVertices.size());
    ASSERT_TRUE(m.triangleOffset + m.triangleCount <= scene.meshletTriangles.size());
    ASSERT_TRUE(m.materialId < scene.materials.size());
  }

  for (const auto &inst : scene.instances) {
    ASSERT_TRUE(inst.materialId < scene.materials.size());
    ASSERT_TRUE(inst.lodMeshletCount[0] > 0u);
    ASSERT_TRUE(inst.lodFirstMeshlet[0] + inst.lodMeshletCount[0] <= scene.meshlets.size());
  }

  std::cout << "      [Scene] " << scene.sceneSourceLabel
            << " | verts=" << scene.vertices.size()
            << " | meshlets=" << scene.meshlets.size()
            << " | instances=" << scene.instances.size() << "\n";
}

TEST_CASE(ShowcaseValidation, HeadlessThreeEraValidationAndParity) {
  using namespace gpubench::showcase;

  auto ctx = createShowcaseVulkanContext();
  if (!ctx) {
    SKIP_TEST("Vulkan backend unavailable");
  }
  auto *vkCtx = dynamic_cast<::VulkanContext *>(ctx.get());
  ASSERT_TRUE(vkCtx != nullptr);

  const uint32_t W = 960u;
  const uint32_t H = 540u;

  ShowcaseRenderer renderer;
  ASSERT_TRUE(renderer.init(vkCtx, findShowcaseKernelDir(), W, H));

  const auto &presets = renderer.getScene().presets;
  ASSERT_TRUE(presets.size() >= 4u);

  ShowcaseSettings cfg{};
  cfg.activeCam = presets[0].camera;
  cfg.frozenCam = presets[0].camera;
  cfg.overlay = OverlayMode::Beauty;

  FrameTelemetry tNaive{}, tLegacy{}, tModern{};
  std::vector<uint8_t> rgbaNaive, rgbaModern;

  cfg.mode = RenderMode::Naive;
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tNaive, nullptr)); // Warmup
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tNaive, &rgbaNaive));

  cfg.mode = RenderMode::Legacy;
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tLegacy, nullptr));
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tLegacy, nullptr));

  cfg.mode = RenderMode::Modern;
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tModern, nullptr));
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tModern, &rgbaModern));

  // Exercise all X-Ray diagnostic modes & Frozen Culling cleanly
  FrameTelemetry tDiag{};
  cfg.overlay = OverlayMode::Overdraw;
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tDiag, nullptr));
  cfg.overlay = OverlayMode::MicroTriangles;
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tDiag, nullptr));
  cfg.overlay = OverlayMode::MeshletClusters;
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tDiag, nullptr));
  cfg.overlay = OverlayMode::Beauty;
  cfg.freezeCulling = true;
  cfg.frozenCam = presets[0].camera;
  cfg.activeCam = presets[3].camera;
  ASSERT_TRUE(renderer.renderHeadlessFrame(cfg, W, H, tDiag, nullptr));

  double psnrDb = 0.0, meanA = 0.0, meanB = 0.0;
  computeImageParity(rgbaNaive, rgbaModern, psnrDb, meanA, meanB);

  std::cout << "      [3-Era GPU Telemetry @ " << W << "x" << H << "]\n"
            << std::fixed << std::setprecision(2)
            << "        Naive : " << tNaive.totalMs << " ms | prims=" << tNaive.clippedPrimitives
            << " | overdraw=" << tNaive.avgOverdraw << "x | subPx=" << tNaive.subPixelTriPct << "%\n"
            << "        Legacy: " << tLegacy.totalMs << " ms | prims=" << tLegacy.clippedPrimitives
            << " | overdraw=" << tLegacy.avgOverdraw << "x | subPx=" << tLegacy.subPixelTriPct << "%\n"
            << "        Modern: " << tModern.totalMs << " ms | prims=" << tModern.clippedPrimitives
            << " | overdraw=" << tModern.avgOverdraw << "x | culled=" << tModern.culledPct << "%"
            << " | mesh=" << (tModern.meshShaderActive ? 1 : 0)
            << " dgc=" << (tModern.dgcExecutionSetActive ? 1 : 0)
            << " bary=" << (tModern.barycentricActive ? 1 : 0) << "\n"
            << "        Beauty Parity: PSNR=" << psnrDb << " dB | meanLuma Naive=" << meanA
            << " Modern=" << meanB << "\n";

  // Verify strict architectural wins & non-black visual parity
  ASSERT_TRUE(meanA > 5.0 && meanB > 5.0);
  ASSERT_TRUE(psnrDb >= 18.0);
  if (vkCtx->isDGCExecutionSetSupported()) {
    ASSERT_TRUE(tModern.dgcExecutionSetActive);
  }
  ASSERT_TRUE(tModern.avgOverdraw <= 1.05);
  ASSERT_TRUE(tModern.totalMs < tLegacy.totalMs);
  ASSERT_TRUE(tLegacy.totalMs < tNaive.totalMs);
  ASSERT_TRUE(tNaive.totalMs > tModern.totalMs * 1.5);
  ASSERT_TRUE(tModern.culledPct > 30.0);
  ASSERT_TRUE(tModern.avgOverdraw < tNaive.avgOverdraw);
  ASSERT_TRUE(tModern.clippedPrimitives < tNaive.clippedPrimitives);

  const ThreeWayResult bench3 = renderer.runThreeWayBenchmark(presets[0].camera, W, H);
  ASSERT_TRUE(bench3.valid);
  ASSERT_TRUE(bench3.modernVsNaiveSpeedup > 1.5);

  renderer.shutdown();
}

