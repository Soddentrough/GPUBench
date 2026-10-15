#include "test_harness.h"
#include "core/ComputeBackendFactory.h"
#include "core/VulkanContext.h"
#include "core/ResultFormatter.h"
#include "utils/KernelPath.h"

#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <array>
#include <cmath>

#define private public
#include "benchmarks/PixelFillRateBench.h"
#include "benchmarks/RayRawTraversalBench.h"
#include "benchmarks/RayIntersectBench.h"
#include "benchmarks/RayDivergenceBench.h"
#include "benchmarks/RayAnyHitBench.h"
#include "benchmarks/RayProceduralBench.h"
#include "benchmarks/RaySchedulingBench.h"
#undef private

namespace {

std::unique_ptr<IComputeContext> createVulkanContext() {
  if (!ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
    return nullptr;
  }
  auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
  ctx->pickDevice(0);
  return ctx;
}

void clearPixelFillImageToZero(PixelFillRateBench &bench, uint32_t configIdx) {
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = bench.commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(bench.device, &allocInfo, &cmd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate command buffer for zero clearing!");
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &beginInfo);

  VkRenderingAttachmentInfoKHR colorAttachment{};
  colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
  colorAttachment.imageView = bench.configs[configIdx].imageView;
  colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 0.0f}};

  VkRenderingInfoKHR renderingInfo{};
  renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
  renderingInfo.renderArea.offset = {0, 0};
  renderingInfo.renderArea.extent = {bench.width, bench.height};
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments = &colorAttachment;

  bench.pfnCmdBeginRendering(cmd, &renderingInfo);
  bench.pfnCmdEndRendering(cmd);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;

  vkQueueSubmit(bench.queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(bench.queue);

  vkFreeCommandBuffers(bench.device, bench.commandPool, 1, &cmd);
}

} // namespace

// ============================================================================
// 1. PixelFillRate Adversarial Negative Validation
// ============================================================================

TEST_CASE(M3_Challenger2, PixelFill_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  PixelFillRateBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] PixelFillRate not supported" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);

  // 1. Positive validation for all configs
  for (uint32_t i = 0; i < bench.GetNumConfigs(); ++i) {
    bench.Run(i);
    ctx->waitIdle();
    ASSERT_TRUE(bench.ValidateResults(i));
  }

  // 2. Negative validation: Clear RGBA8 image to all zeroes
  clearPixelFillImageToZero(bench, 0);
  ASSERT_FALSE(bench.ValidateResults(0));

  // 3. Negative validation: Clear RGBA16F image to all zeroes
  clearPixelFillImageToZero(bench, 1);
  ASSERT_FALSE(bench.ValidateResults(1));

  // 4. Negative validation: Out-of-bounds config index
  ASSERT_FALSE(bench.ValidateResults(99));

  bench.Teardown();
}

// ============================================================================
// 2. RayRawTraversal Adversarial Negative Validation
// ============================================================================

TEST_CASE(M3_Challenger2, RayRawTraversal_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  RayRawTraversalBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] RayRawTraversal not supported" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);

  // Config 0 (Coherent Triangle)
  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Config 1 (Deep Multi-Layer Box Stress)
  bench.Run(1);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(1));

  // Negative validation: zero out resultBuffer
  std::vector<uint32_t> zeroHits(1024, 0);
  ctx->writeBuffer(bench.resultBuffer, 0, zeroHits.size() * sizeof(uint32_t), zeroHits.data());
  ctx->waitIdle();

  ASSERT_FALSE(bench.ValidateResults(0));
  ASSERT_FALSE(bench.ValidateResults(1));

  bench.Teardown();
}

// ============================================================================
// 3. RayIntersect Adversarial Negative Validation
// ============================================================================

TEST_CASE(M3_Challenger2, RayIntersect_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  RayIntersectBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] RayIntersect not supported" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Negative validation: zero out hits
  uint32_t zeroHit = 0;
  ctx->writeBuffer(bench.resultBuffer, 0, sizeof(uint32_t), &zeroHit);
  ctx->waitIdle();

  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

// ============================================================================
// 4. RayDivergence Adversarial Negative Validation
// ============================================================================

TEST_CASE(M3_Challenger2, RayDivergence_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  RayDivergenceBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] RayDivergence not supported" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Negative validation: zero out hits
  uint32_t zeroHit = 0;
  ctx->writeBuffer(bench.resultBuffer, 0, sizeof(uint32_t), &zeroHit);
  ctx->waitIdle();

  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

// ============================================================================
// 5. RayAnyHit Adversarial Negative Validation
// ============================================================================

TEST_CASE(M3_Challenger2, RayAnyHit_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  RayAnyHitBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] RayAnyHit not supported" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  bench.Run(1);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(1));

  // Negative validation: zero out hits
  uint32_t zeroHit = 0;
  ctx->writeBuffer(bench.resultBuffer, 0, sizeof(uint32_t), &zeroHit);
  ctx->waitIdle();

  ASSERT_FALSE(bench.ValidateResults(0));
  ASSERT_FALSE(bench.ValidateResults(1));

  bench.Teardown();
}

// ============================================================================
// 6. RayProcedural Adversarial Negative Validation
// ============================================================================

TEST_CASE(M3_Challenger2, RayProcedural_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  RayProceduralBench bench;
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] RayProcedural not supported" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Negative validation: zero out hits
  uint32_t zeroHit = 0;
  ctx->writeBuffer(bench.resultBuffer, 0, sizeof(uint32_t), &zeroHit);
  ctx->waitIdle();

  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

// ============================================================================
// 7. RayScheduling Adversarial Negative Validation
// ============================================================================

TEST_CASE(M3_Challenger2, RayScheduling_PositiveAndNegativeValidation) {
  auto ctx = createVulkanContext();
  ASSERT_TRUE(ctx != nullptr);

  std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());

  RaySchedulingBench bench(RaySchedulingBench::SceneType::Showroom);
  DeviceInfo info = ctx->getCurrentDeviceInfo();
  if (!bench.IsSupported(info, ctx.get())) {
    std::cout << "[SKIP] RayScheduling not supported" << std::endl;
    return;
  }

  bench.Setup(*ctx, kdir);

  bench.Run(0);
  ctx->waitIdle();
  ASSERT_TRUE(bench.ValidateResults(0));

  // Negative validation A: Zero rays, zero hits
  uint32_t statsZero[4] = {0, 0, 0, 0};
  ctx->writeBuffer(bench.resultBuffer, 0, sizeof(statsZero), statsZero);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  // Negative validation B: Rays > 0, hits == 0
  uint32_t statsRaysOnly[4] = {100000, 0, 0, 0};
  ctx->writeBuffer(bench.resultBuffer, 0, sizeof(statsRaysOnly), statsRaysOnly);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  // Negative validation C: Rays == 0, hits > 0
  uint32_t statsHitsOnly[4] = {0, 100000, 0, 0};
  ctx->writeBuffer(bench.resultBuffer, 0, sizeof(statsHitsOnly), statsHitsOnly);
  ctx->waitIdle();
  ASSERT_FALSE(bench.ValidateResults(0));

  bench.Teardown();
}

// ============================================================================
// 8. ResultFormatter & JSON Disclosures Verification
// ============================================================================

TEST_CASE(M3_Challenger2, RayRawTraversal_Disclosures_JSON_And_CLI) {
  ResultData r{};
  r.backendName = "Vulkan";
  r.deviceName = "Test GPU";
  r.benchmarkName = "RayRawTraversal (Raw BVH - Deep Multi-Layer Box Stress)";
  r.component = "Ray Tracing";
  r.subcategory = "Hardware BVH Traversal";
  r.metric = "MRays/s";
  r.operations = 192000000;
  r.time_ms = 1280.0;
  r.configIndex = 1;

  std::string jsonStr = resultsToJson({r});

  // Assert presence of required JSON disclosure fields
  ASSERT_TRUE(jsonStr.find("\"traversal_model\": \"synthetic_64_layer_aabb\"") != std::string::npos);
  ASSERT_TRUE(jsonStr.find("\"model_box_tests_per_ray\": 64") != std::string::npos);
  ASSERT_TRUE(jsonStr.find("\"synthetic_disclosure\": \"Measures synthetic 64-layer AABB traversal") != std::string::npos);

  // Assert Config Caveat contains (Synthetic 64-AABB Traversal Model)
  RayRawTraversalBench bench;
  bench.RecordRunResult(1, 1, 1000.0);
  std::string caveat = bench.GetConfigCaveat(1);
  ASSERT_TRUE(caveat.find("Synthetic 64-AABB Traversal Model") != std::string::npos);
}
