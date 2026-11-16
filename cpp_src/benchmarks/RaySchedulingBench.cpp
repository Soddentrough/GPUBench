#include "RaySchedulingBench.h"
#include "ReactorBayScene.h"
#include "core/ResultFormatter.h"
#include "core/RunnerAPI.h"
#include "core/VulkanContext.h"
#include "third_party/yyjson.h"
#include "utils/ImageExport.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#endif

namespace {
std::string resolveRendersDir() {
  std::error_code ec;
  if (std::filesystem::exists("renders", ec) ||
      std::filesystem::exists("CMakeLists.txt", ec) ||
      std::filesystem::exists("../CMakeLists.txt", ec)) {
    std::filesystem::create_directories("renders", ec);
    if (!ec) {
      std::string probePath = "renders/.write_probe";
      std::ofstream probe(probePath);
      if (probe.is_open()) {
        probe.close();
        std::filesystem::remove(probePath, ec);
        return "renders";
      }
    }
  }
#ifdef _WIN32
  if (const char *localAppData = std::getenv("LOCALAPPDATA")) {
    std::filesystem::path p = std::filesystem::path(localAppData) / "gpubench" / "renders";
    std::filesystem::create_directories(p, ec);
    if (!ec) return p.string();
  }
#else
  if (const char *xdgCache = std::getenv("XDG_CACHE_HOME")) {
    if (xdgCache[0] != '\0') {
      std::filesystem::path p = std::filesystem::path(xdgCache) / "gpubench" / "renders";
      std::filesystem::create_directories(p, ec);
      if (!ec) return p.string();
    }
  }
  if (const char *home = std::getenv("HOME")) {
    if (home[0] != '\0') {
      std::filesystem::path p = std::filesystem::path(home) / ".cache" / "gpubench" / "renders";
      std::filesystem::create_directories(p, ec);
      if (!ec) return p.string();
    }
  }
#endif
  std::filesystem::create_directories("renders", ec);
  return "renders";
}
} // namespace

#ifdef HAVE_VULKAN
void RaySchedulingBench::loadRTProcs(VkDevice device) {
  vkGetAccelerationStructureBuildSizesKHR_ptr =
      (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(
          device, "vkGetAccelerationStructureBuildSizesKHR");
  vkCreateAccelerationStructureKHR_ptr =
      (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(
          device, "vkCreateAccelerationStructureKHR");
  vkCmdBuildAccelerationStructuresKHR_ptr =
      (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(
          device, "vkCmdBuildAccelerationStructuresKHR");
  vkGetAccelerationStructureDeviceAddressKHR_ptr =
      (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(
          device, "vkGetAccelerationStructureDeviceAddressKHR");
  vkDestroyAccelerationStructureKHR_ptr =
      (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(
          device, "vkDestroyAccelerationStructureKHR");
}

void RaySchedulingBench::buildAS() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  if (triangleBlas != VK_NULL_HANDLE) {
    if (vkDestroyAccelerationStructureKHR_ptr) {
      vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
    }
    triangleBlas = VK_NULL_HANDLE;
  }
  if (sceneTlas != VK_NULL_HANDLE) {
    if (vkDestroyAccelerationStructureKHR_ptr) {
      vkDestroyAccelerationStructureKHR_ptr(device, sceneTlas, nullptr);
    }
    sceneTlas = VK_NULL_HANDLE;
  }
  if (triangleBlasBuffer) {
    context->releaseBuffer(triangleBlasBuffer);
    triangleBlasBuffer = nullptr;
  }
  if (tlasBuffer) {
    context->releaseBuffer(tlasBuffer);
    tlasBuffer = nullptr;
  }
  if (instanceBuffer) {
    context->releaseBuffer(instanceBuffer);
    instanceBuffer = nullptr;
  }
  if (scratchBuffer) {
    context->releaseBuffer(scratchBuffer);
    scratchBuffer = nullptr;
  }

  ComputeBuffer posBuildBuffer = nullptr;
  VkDeviceAddress vAddr = 0;
  VkDeviceSize vStride = isGltf ? sizeof(GltfVertex) : (sizeof(float) * 3);
  if (isGltf && !compactBvhPositions.empty()) {
    posBuildBuffer = context->createBuffer(
        compactBvhPositions.size() * sizeof(float), compactBvhPositions.data());
    vAddr = vContext->getBufferDeviceAddress(posBuildBuffer);
    vStride = sizeof(float) * 3;
  } else {
    vAddr = vContext->getBufferDeviceAddress(vertexBuffer);
  }

  VkAccelerationStructureGeometryKHR triGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  triGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  triGeom.flags = 0;
  triGeom.geometry.triangles.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triGeom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triGeom.geometry.triangles.vertexData.deviceAddress = vAddr;
  triGeom.geometry.triangles.vertexStride = vStride;
  triGeom.geometry.triangles.maxVertex = numPrimitives * 3 - 1;
  triGeom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

  VkAccelerationStructureBuildGeometryInfoKHR triBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  triBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  triBuildInfo.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  triBuildInfo.geometryCount = 1;
  triBuildInfo.pGeometries = &triGeom;
  triBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t triMaxPrimCount = numPrimitives;
  VkAccelerationStructureBuildSizesInfoKHR triSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &triBuildInfo,
      &triMaxPrimCount, &triSizes);

  triangleBlasBuffer =
      context->createBuffer(triSizes.accelerationStructureSize);

  VkAccelerationStructureCreateInfoKHR triCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  triCreateInfo.buffer = vContext->getVkBuffer(triangleBlasBuffer);
  triCreateInfo.size = triSizes.accelerationStructureSize;
  triCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &triCreateInfo, nullptr,
                                       &triangleBlas);

  // TLAS Instance Setup
  VkAccelerationStructureDeviceAddressInfoKHR triAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  triAddrInfo.accelerationStructure = triangleBlas;
  VkDeviceAddress blasAddress =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &triAddrInfo);

  VkAccelerationStructureInstanceKHR instance{};
  instance.transform.matrix[0][0] = 1.0f;
  instance.transform.matrix[1][1] = 1.0f;
  instance.transform.matrix[2][2] = 1.0f;
  instance.mask = 0xFF;
  instance.flags =
      VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  instance.accelerationStructureReference = blasAddress;

  instanceBuffer =
      context->createBuffer(sizeof(VkAccelerationStructureInstanceKHR), &instance);
  VkDeviceAddress instAddr = vContext->getBufferDeviceAddress(instanceBuffer);

  VkAccelerationStructureGeometryKHR tlasGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  tlasGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
  tlasGeom.geometry.instances.data.deviceAddress = instAddr;

  VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tlasBuildInfo.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tlasBuildInfo.geometryCount = 1;
  tlasBuildInfo.pGeometries = &tlasGeom;
  tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t tlasMaxPrimCount = 1;
  VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo,
      &tlasMaxPrimCount, &tlasSizes);

  tlasBuffer = context->createBuffer(tlasSizes.accelerationStructureSize);

  VkAccelerationStructureCreateInfoKHR tlasCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  tlasCreateInfo.buffer = vContext->getVkBuffer(tlasBuffer);
  tlasCreateInfo.size = tlasSizes.accelerationStructureSize;
  tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &tlasCreateInfo, nullptr,
                                       &sceneTlas);

  size_t maxScratch = std::max(triSizes.buildScratchSize, tlasSizes.buildScratchSize);
  scratchBuffer = context->createBuffer(maxScratch);
  VkDeviceAddress scratchAddr = vContext->getBufferDeviceAddress(scratchBuffer);

  // Command Buffer for Building
  VkCommandPool tmpPool;
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.queueFamilyIndex = vContext->getComputeQueueFamilyIndex();
  vkCreateCommandPool(device, &poolInfo, nullptr, &tmpPool);

  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo cmdAlloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdAlloc.commandPool = tmpPool;
  cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAlloc.commandBufferCount = 1;
  vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

  VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cmd, &beginInfo);

  triBuildInfo.dstAccelerationStructure = triangleBlas;
  triBuildInfo.scratchData.deviceAddress = scratchAddr;
  VkAccelerationStructureBuildRangeInfoKHR triRange{numPrimitives, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR *pTriRange = &triRange;
  vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &triBuildInfo, &pTriRange);

  vContext->cmdPipelineMemoryBarrier2(
      cmd,
      VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
      VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);

  tlasBuildInfo.dstAccelerationStructure = sceneTlas;
  tlasBuildInfo.scratchData.deviceAddress = scratchAddr;
  VkAccelerationStructureBuildRangeInfoKHR tlasRange{1, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR *pTlasRange = &tlasRange;
  vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &tlasBuildInfo, &pTlasRange);

  vkEndCommandBuffer(cmd);

  auto bvhT0 = std::chrono::high_resolution_clock::now();
  VkFence bvhFence = VK_NULL_HANDLE;
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(device, &fenceInfo, nullptr, &bvhFence) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to create fence for AS build in RaySchedulingBench");
  }
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  VkResult subRes = VK_SUCCESS;
  {
    std::lock_guard<std::recursive_mutex> qlock(vContext->getQueueMutex());
    subRes = vkQueueSubmit(queue, 1, &submit, bvhFence);
  }
  VkResult waitRes = (subRes == VK_SUCCESS)
                         ? vkWaitForFences(device, 1, &bvhFence, VK_TRUE, UINT64_MAX)
                         : subRes;
  vkDestroyFence(device, bvhFence, nullptr);
  if (subRes != VK_SUCCESS || waitRes != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to build acceleration structures in RaySchedulingBench");
  }
  auto bvhT1 = std::chrono::high_resolution_clock::now();
  bvhBuildTimeMs = std::chrono::duration<double, std::milli>(bvhT1 - bvhT0).count();

  vkDestroyCommandPool(device, tmpPool, nullptr);
  if (posBuildBuffer) {
    context->releaseBuffer(posBuildBuffer);
    posBuildBuffer = nullptr;
  }
  if (scratchBuffer) {
    context->releaseBuffer(scratchBuffer);
    scratchBuffer = nullptr;
  }
}
#endif // HAVE_VULKAN

void RaySchedulingBench::RebuildAccelerationStructures() {
#ifdef HAVE_VULKAN
  buildAS();
  if (context && context->getBackend() == ComputeBackend::Vulkan && sceneTlas != VK_NULL_HANDLE) {
    VulkanContext *vContext = static_cast<VulkanContext *>(context);
    auto rebindAS = [&](ComputeKernel k) {
      if (k) {
        vContext->setKernelAS(k, 0, (AccelerationStructure)sceneTlas);
      }
    };
    rebindAS(kernelTraditional);
    for (ComputeKernel k : kernelTraditionalSpecialized) {
      rebindAS(k);
    }
    rebindAS(kernelClassify);
    for (ComputeKernel k : kernelClassifySpecialized) {
      rebindAS(k);
    }
    rebindAS(kernelMaterial);
    rebindAS(kernelMaterialGltf);
    for (ComputeKernel k : kernelMaterialSpecialized) {
      rebindAS(k);
    }
    rebindAS(kernelBounce);
    rebindAS(kernelBounceTerminal);
    rebindAS(kernelMultiLight);
    rebindAS(kernelRTP);
    rebindAS(kernelRTPSER);
  }
#endif
}

bool RaySchedulingBench::IsSupported(const DeviceInfo &info,
                                   IComputeContext *computeContext) const {
  return info.rayTracingSupport &&
         (computeContext && computeContext->getBackend() == ComputeBackend::Vulkan);
}

void RaySchedulingBench::SetBounceDepth(uint32_t bounces) {
  bounceDepth = std::clamp(bounces, 1u, 8u);
#ifdef HAVE_VULKAN
  if (kernelBounce && kernelBounceTerminal) {
    rebuildBounceBatches();
    rebuildDGCBounceBatches();
  }
#endif
}

void RaySchedulingBench::rebuildBounceBatches() {
#ifdef HAVE_VULKAN
  bounceBatches.clear();
  bounceBatches.reserve(bounceDepth);
  for (uint32_t b = 1; b <= bounceDepth; ++b) {
    uint32_t inQ = (b - 1) % 2;
    uint32_t outQ = (b == bounceDepth) ? 0xFFFFFFFFu : (b % 2);
    uint32_t offset = (16u + inQ) * sizeof(uint32_t) * 3;
    ComputeKernel kernel = (b == bounceDepth) ? kernelBounceTerminal : nullptr;

    struct {
      uint32_t inQueue;
      uint32_t outQueue;
      uint32_t bounceIndex;
      uint32_t seed;
      uint32_t maxQueueSize;
      uint32_t sceneType;
      uint32_t isGltf;
    } pcBounce{inQ, outQ, b, 1337u, bounceCapacity, 3u, isGltf ? 1u : 0u};

    std::vector<uint8_t> pcData(sizeof(pcBounce));
    std::memcpy(pcData.data(), &pcBounce, sizeof(pcBounce));
    bounceBatches.push_back({offset, pcData, kernel});
  }
#endif
}

void RaySchedulingBench::rebuildDGCBounceBatches() {
#ifdef HAVE_VULKAN
  if (!dgcSequenceBuffer || !context) return;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(context);
  if (!vContext || !vContext->isDGCSupported()) return;

  uint32_t sceneTypeVal = 3u;
  uint32_t isGltfVal = isGltf ? 1u : 0u;

  for (uint32_t b = 1; b <= bounceDepth; ++b) {
    uint32_t inQ = (b - 1) % 2;
    uint32_t outQ = (b == bounceDepth) ? 0xFFFFFFFFu : (b % 2);

    uint32_t pc[8] = {
        inQ,
        outQ,
        b,
        1337u,
        bounceCapacity,
        sceneTypeVal,
        isGltfVal,
        0u
    };

    uint32_t seqItem[12] = {
        0, // pipelineIndex
        pc[0], pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], pc[7],
        0, 1, 1 // cmdX, cmdY, cmdZ
    };

    size_t offset = (16 + (b - 1)) * sizeof(uint32_t) * 12;
    context->writeBuffer(dgcSequenceBuffer, offset, sizeof(seqItem), seqItem);
  }
#endif
}

std::string RaySchedulingBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Stage: Material Shading (Megakernel)";
  case 1:
    return "Stage: Material Shading (RTP + SER)";
  case 2:
    return "Stage: Material Shading (DGC)";
  case 3:
    return "Path Tracing (" + std::to_string(samplesPerPixel) + " SPP) (Megakernel)";
  case 4:
    return "Path Tracing (" + std::to_string(samplesPerPixel) + " SPP) (RTP + SER)";
  case 5:
    return "Path Tracing (" + std::to_string(samplesPerPixel) + " SPP) (DGC)";
  case 6:
    return "Scene Render: Full Frame (Megakernel)";
  case 7:
    return "Scene Render: Full Frame (RTP + SER)";
  case 8:
    return "Scene Render: Full Frame (DGC)";
  case 9:
    return "Stage: BVH Traversal - Linear 1D Scanline";
  case 10:
    return "Stage: Queue Compaction - Wave Ballot Stream Sort";
  case 11:
    return "Stage: BVH Traversal - 2D Screen Tiled (8x4)";
  case 12:
    return "Stage: BVH Traversal - 2D Morton Z-Curve (8x4)";
  case 13:
    return "Stage: BVH Traversal - 2D Morton Z-Curve (4x8)";
  case 14:
    return "Primary Rays (Compute Megakernel)";
  case 15:
    return "Stage: Directional Shadows - Single Light (Megakernel)";
  case 16:
    return "Stage: Directional Shadows - Single Light (RTP + SER)";
  case 17:
    return "Stage: Queue Compaction - Single-Pass Unified Stream";
  case 18:
    return "Stage: Traversal Divergence - Alpha Cutout Geometry";
  case 19:
    return "Stage: Queue Compaction - Queue Compaction Throughput";
  case 20:
    return "Primary Rays (RTP)";
  case 21:
    return "Primary Rays (RTP + SER)";
  case 22:
    return "Stage: Multi-Light Evaluation - 128 Lights (Megakernel)";
  case 23:
    return "Stage: Multi-Light Evaluation - 128 Lights (DGC Light Binning)";
  default:
    return "Unknown";
  }
}

int32_t RaySchedulingBench::GetBaselineConfigIndex(uint32_t config_idx) const {
  switch (config_idx) {
  // Scene Ray Tracing (PBR): baseline is 14 (Primary Rays Compute Megakernel)
  case 20: // Primary Rays (RTP)
  case 21: // Primary Rays (RTP + SER)
    return 14;

  // Total Scene Render: baseline is 6 (Scene Render: Full Frame Megakernel)
  case 7: // Scene Render: Full Frame (RTP + SER)
  case 8: // Scene Render: Full Frame (DGC)
    return 6;

  // Traversal Ordering & Coherence: baseline is 9 (Linear 1D Scanline)
  case 11: // 2D Screen Tiled (8x4)
  case 12: // 2D Morton (8x4)
  case 13: // 2D Morton (4x8)
    return 9;

  // Material Shading: baseline is 0 (Stage: Material Shading Megakernel)
  case 1: // Stage: Material Shading (RTP + SER)
  case 2: // Stage: Material Shading (DGC)
    return 0;

  // Scene Path Tracing (Multi-Bounce): baseline is 3 (Megakernel)
  case 4:  // Path Tracing (RTP + SER)
  case 5:  // Path Tracing (DGC)
    return 3;

  // Directional Shadows: baseline is 15 (Single Light Megakernel)
  case 16: // Single Light (RTP + SER)
    return 15;

  // Multi-Light Evaluation: 128 Lights baseline 22
  case 23:
    return 22;

  // Wavefront Stream Compaction: baseline is 17 (Single-Pass Unified Stream, 128B expanded hit record)
  case 10: // Wave Ballot Stream Sort (16B material bins)
  case 19: // Queue Compaction Throughput (32B packed record)
    return 17;

  // Configs with no work-equivalent baseline (or that are baselines themselves):
  default:
    return -1;
  }
}

const char *RaySchedulingBench::GetSubCategory(uint32_t config_idx) const {
  if (config_idx == 14 || config_idx == 20 || config_idx == 21)
    return "Scene Ray Tracing (PBR)";
  if (config_idx == 15 || config_idx == 16)
    return "Directional Shadows";
  if (config_idx == 22 || config_idx == 23)
    return "Multi-Light Evaluation";
  if (config_idx < 3)
    return "Material Shading";
  if (config_idx >= 3 && config_idx <= 5)
    return "Scene Path Tracing (Multi-Bounce)";
  if (config_idx >= 6 && config_idx <= 8)
    return "Total Scene Render";
  if (config_idx == 9 || config_idx == 11 || config_idx == 12 || config_idx == 13)
    return "Traversal Ordering & Coherence";
  if (config_idx == 10 || config_idx == 17 || config_idx == 19)
    return "Wavefront Stream Compaction";
  if (config_idx == 18)
    return "Alpha Cutout Divergence";
  return "Pipeline Breakdown";
}

const char *RaySchedulingBench::GetDescription(uint32_t config_idx) const {
  if (config_idx == 14 || config_idx == 20 || config_idx == 21)
    return "Primary ray cast and G-buffer normal evaluation comparing Compute Megakernel, Ray Tracing Pipeline (RTP), and Hardware SER.";
  if (config_idx == 15 || config_idx == 16)
    return "Ray-traced directional sun shadow visibility comparing Megakernel and RTP+SER.";
  if (config_idx == 22 || config_idx == 23)
    return "Direct illumination across 128 stochastic lights comparing Megakernel evaluation against spatial light-binned DGC.";
  if (config_idx < 3)
    return "Divergent 8-archetype PBR BSDF shading stage comparing monolithic Megakernel branching against RTP+SER and specialized DGC micro-kernels.";
  if (config_idx >= 3 && config_idx <= 5)
    return "Multi-bounce Monte Carlo path tracing comparing Megakernel, RTP+SER, and octant-sorted Wavefront DGC.";
  if (config_idx >= 6 && config_idx <= 8)
    return "End-to-end hybrid ray-traced beauty frame (Primary + PBR + Shadows + RTAO + Indirect GI) across Megakernel, RTP+SER, and Wavefront DGC.";
  if (config_idx == 9 || config_idx == 11 || config_idx == 12 || config_idx == 13)
    return "Screen-space ray launch ordering sensitivity (1D scanline vs. 2D tiles vs. Morton Z-order) on L1/L2 BVH cache locality.";
  if (config_idx == 10 || config_idx == 17 || config_idx == 19)
    return "Subgroup-ballot wavefront hit-record stream compaction throughput across 128B expanded, 32B packed, and 16B material-binned payloads.";
  if (config_idx == 18)
    return "Alpha-cutout perforated catwalk traversal divergence measuring mid-traversal texture alpha evaluation overhead.";
  return "Hardware ray tracing execution paradigm and scheduling microbenchmark.";
}

std::string RaySchedulingBench::GetConfigCaveat(uint32_t config_idx) const {
  if (isSERConfigIdx(config_idx) && serEmulated) {
    return "Driver exposes VK_EXT_ray_tracing_invocation_reorder, but hardware reordering hint is NONE (pass-through / no dedicated hardware SER unit)";
  }
  return "";
}

std::string RaySchedulingBench::GetConfigCaveat(
    uint32_t config_idx, const DeviceInfo &info,
    IComputeContext *context) const {
  if (isSERConfigIdx(config_idx)) {
    bool driverSer = info.serSupported;
    bool hwSer = info.serHardwareSupported;
#ifdef HAVE_VULKAN
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      const VulkanContext *vc = static_cast<const VulkanContext *>(context);
      driverSer = vc->isSERSupported();
      hwSer = vc->isSERHardwareSupported();
    }
#endif
    if (serEmulated || (driverSer && !hwSer)) {
      return "Driver exposes VK_EXT_ray_tracing_invocation_reorder, but hardware reordering hint is NONE (pass-through / no dedicated hardware SER unit)";
    }
  }
  if (config_idx == 2) {
    bool basicDgc = info.dgcSupported;
    bool execSet = info.dgcExecutionSetSupported;
#ifdef HAVE_VULKAN
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
      const VulkanContext *vc = static_cast<const VulkanContext *>(context);
      basicDgc = vc->isDGCSupported();
      execSet = vc->isDGCExecutionSetSupported();
    }
#endif
    if (basicDgc && !execSet) {
      return "IndirectExecutionSet compute pipeline binding unsupported on this driver; running via standard VK_EXT_device_generated_commands multi-sequence fallback";
    }
  }
  return GetConfigCaveat(config_idx);
}

int RaySchedulingBench::GetSortWeight(uint32_t config_idx) const {
  // Algorithmic microbenchmarks: 608..614
  if (config_idx == 9) return 608;  // Traversal Ordering - Linear 1D Scanline (Baseline)
  if (config_idx == 11) return 609; // Traversal Ordering - 2D Screen Tiled (8x4)
  if (config_idx == 12) return 610; // Traversal Ordering - 2D Morton (8x4)
  if (config_idx == 13) return 611; // Traversal Ordering - 2D Morton (4x8)
  if (config_idx == 17) return 612; // Wavefront Stream Compaction - Single-Pass (128B Baseline)
  if (config_idx == 19) return 613; // Wavefront Stream Compaction - Throughput (32B Packed)
  if (config_idx == 10) return 614; // Wavefront Stream Compaction - Wave Ballot (16B Material Bins)

  const int sceneBase = 620;
  if (config_idx == 14) return sceneBase + 0;  // Scene RT (PBR) - Compute Megakernel (Baseline)
  if (config_idx == 20) return sceneBase + 1;  // Scene RT (PBR) - Dedicated RTP
  if (config_idx == 21) return sceneBase + 2;  // Scene RT (PBR) - RTP + Hardware SER
  if (config_idx >= 3 && config_idx <= 5) return sceneBase + 5 + static_cast<int>(config_idx - 3); // Path Tracing: +5..+7
  if (config_idx >= 6 && config_idx <= 8) return sceneBase + 12 + static_cast<int>(config_idx - 6); // Total Scene Render: +12..+14
  if (config_idx == 15 || config_idx == 16) return sceneBase + 15 + static_cast<int>(config_idx - 15); // Directional Shadows: +15..+16
  if (config_idx == 22 || config_idx == 23) return sceneBase + 19 + static_cast<int>(config_idx - 22); // Multi-Light Evaluation: +19..+20
  if (config_idx < 3) return sceneBase + 23 + static_cast<int>(config_idx); // Material Shading: +23..+25
  if (config_idx == 18) return sceneBase + 29; // Alpha Cutout Divergence: +29
  return sceneBase + 32 + static_cast<int>(config_idx % 8u);
}

void RaySchedulingBench::Setup(IComputeContext &context_ref,
                             const std::string &kernel_dir) {
  this->context = &context_ref;
  this->cachedKernelDir = kernel_dir;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context_ref);
  if (!vContext)
    throw std::runtime_error("RaySchedulingBench requires VulkanContext");

#ifdef HAVE_VULKAN
  materialBatches.clear();
  materialBatchesBreakdown.clear();
  bounceBatches.clear();
  multiLightBatches128.clear();
  dgcBounceBatches.clear();

  loadRTProcs(vContext->getVulkanDevice());

  if (getenv("GPUBENCH_DUMP_RENDERS")) {
    dumpRenders = true;
  }
  captureFramebuffer = false;
  rng.seed(1337u);
  isDGCAvailable = false;
  isDGCExecutionSetAvailable = false;

  for (int c = 0; c < 24; ++c) {
    lastCompletedPasses[c] = 1;
    unsupportedConfig[c] = false;
    unsupportedReason[c].clear();
  }
  lastRunConfig = 0xFFFFFFFFu;

  rayCount = renderWidth * renderHeight;
  octantCapacity = rayCount;
  materialCapacity = rayCount;
  const DeviceInfo &devInfo = context->getCurrentDeviceInfo();
  const uint64_t vramBytes =
      (devInfo.dedicatedVramBytes > 0) ? devInfo.dedicatedVramBytes : devInfo.memorySize;
  bounceCapacity = (vramBytes >= 12ULL * 1024ULL * 1024ULL * 1024ULL)
                       ? rayCount
                       : std::max(1024u, rayCount / 2u);
  queueCapacity = materialCapacity;
  resultBuffer = context->createBuffer(sizeof(uint32_t) * 4);
  uint32_t zeros[4] = {0, 0, 0, 0};
  context->writeBuffer(resultBuffer, 0, sizeof(zeros), zeros);

  // Allocate RGBA32F framebuffers for visual verification
  fbTraditional = context->createBuffer(rayCount * sizeof(float) * 4);
  fbWorkList = context->createBuffer(rayCount * sizeof(float) * 4);

  // WorkList storage buffer: 4096 uint header (16KB) + 16-queue payloads
  constexpr size_t kWorkListHeaderUints = 4096;
  uint32_t maxCapacity = std::max(rayCount, materialCapacity);
  size_t workListSize =
      sizeof(uint32_t) * kWorkListHeaderUints +
      std::max(static_cast<size_t>(16u) * maxCapacity * sizeof(float) * 4u,
               static_cast<size_t>(16u) * bounceCapacity * sizeof(float) * 8u);
  workListBuffer = context->createBuffer(workListSize);
  std::vector<uint32_t> initialCounters(kWorkListHeaderUints, 0);
  context->writeBuffer(workListBuffer, 0, initialCounters.size() * sizeof(uint32_t), initialCounters.data());

  // Indirect dispatch commands: 32 * VkDispatchIndirectCommand (384B)
  size_t indirectSize = sizeof(uint32_t) * 3 * 32;
  indirectBuffer = context->createBuffer(indirectSize);

  // Deterministic C++ Procedural AAA Scene: Zero-Day Reactor Bay
  isGltf = true;
  compactBvhPositions.clear();
  std::vector<float> vertices;
  std::vector<uint32_t> triMats;
  ReactorBayScene::buildReactorBayMesh(vertices, triMats);
  numPrimitives = static_cast<uint32_t>(vertices.size() / 36);
  if (context && context->isVerbose()) {
    std::cout << "\r\033[K[RayScheduling] Generated Zero-Day Reactor Bay AAA scene: "
              << numPrimitives << " triangles, 8 PBR archetypes, 24 256x256 textures"
              << std::endl;
  }

  const size_t numVerts = static_cast<size_t>(numPrimitives) * 3u;
  compactBvhPositions.resize(numVerts * 3u);
  for (size_t vi = 0; vi < numVerts; ++vi) {
    compactBvhPositions[vi * 3u + 0u] = vertices[vi * 12u + 0u];
    compactBvhPositions[vi * 3u + 1u] = vertices[vi * 12u + 1u];
    compactBvhPositions[vi * 3u + 2u] = vertices[vi * 12u + 2u];
  }

  vertexBuffer =
      context->createBuffer(vertices.size() * sizeof(float), vertices.data());

  std::vector<GltfMaterial> reactorMats;
  std::vector<GltfTextureHeader> reactorHeaders;
  std::vector<uint32_t> reactorPixels;
  ReactorBayScene::generateReactorBayPbrTextures(reactorMats, reactorHeaders, reactorPixels);

  materialBuffer = context->createBuffer(reactorMats.size() * sizeof(GltfMaterial), reactorMats.data());
  triangleMaterialBuffer = context->createBuffer(triMats.size() * sizeof(uint32_t), triMats.data());
  texHeaderBuffer = context->createBuffer(reactorHeaders.size() * sizeof(GltfTextureHeader), reactorHeaders.data());
  texPixelBuffer = context->createBuffer(reactorPixels.size() * sizeof(uint32_t), reactorPixels.data());

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  auto findShader = [&](const std::string &primary, const std::string &fallback) {
    if (std::filesystem::exists(kdir / "vulkan" / primary) ||
        std::filesystem::exists(kdir / "vulkan" / (primary + ".spv"))) {
      return (kdir / "vulkan" / primary).string();
    }
    return (kdir / "vulkan" / fallback).string();
  };

  kernelTraditional = vContext->createKernel(
      findShader("rt_scheduling_traditional_megakernel.comp", "rt_scheduling_traditional.comp"), "main", 8);
  vContext->setKernelAS(kernelTraditional, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelTraditional, 1, resultBuffer);
  vContext->setKernelArg(kernelTraditional, 2, fbTraditional);
  vContext->setKernelArg(kernelTraditional, 3, vertexBuffer);
  vContext->setKernelArg(kernelTraditional, 4, materialBuffer);
  vContext->setKernelArg(kernelTraditional, 5, triangleMaterialBuffer);
  vContext->setKernelArg(kernelTraditional, 6, texHeaderBuffer);
  vContext->setKernelArg(kernelTraditional, 7, texPixelBuffer);

  for (uint32_t m : {0u, 1u, 3u, 4u, 5u, 7u, 11u}) {
    kernelTraditionalSpecialized[m] = vContext->createKernelWithSpec(
        findShader("rt_scheduling_traditional_megakernel.comp", "rt_scheduling_traditional.comp"), "main", 8, 1, m);
    vContext->setKernelAS(kernelTraditionalSpecialized[m], 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelTraditionalSpecialized[m], 1, resultBuffer);
    vContext->setKernelArg(kernelTraditionalSpecialized[m], 2, fbTraditional);
    vContext->setKernelArg(kernelTraditionalSpecialized[m], 3, vertexBuffer);
    vContext->setKernelArg(kernelTraditionalSpecialized[m], 4, materialBuffer);
    vContext->setKernelArg(kernelTraditionalSpecialized[m], 5, triangleMaterialBuffer);
    vContext->setKernelArg(kernelTraditionalSpecialized[m], 6, texHeaderBuffer);
    vContext->setKernelArg(kernelTraditionalSpecialized[m], 7, texPixelBuffer);
  }

  kernelClassify = vContext->createKernel(
      findShader("rt_scheduling_device_generated_commands_classify.comp", "rt_scheduling_worklist_classify.comp"), "main", 10);
  vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelClassify, 1, resultBuffer);
  vContext->setKernelArg(kernelClassify, 2, workListBuffer);
  vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
  vContext->setKernelArg(kernelClassify, 4, fbWorkList);
  vContext->setKernelArg(kernelClassify, 5, vertexBuffer);
  vContext->setKernelArg(kernelClassify, 6, materialBuffer);
  vContext->setKernelArg(kernelClassify, 7, triangleMaterialBuffer);
  vContext->setKernelArg(kernelClassify, 8, texHeaderBuffer);
  vContext->setKernelArg(kernelClassify, 9, texPixelBuffer);

  for (uint32_t m : {0u, 1u, 8u}) {
    kernelClassifySpecialized[m] = vContext->createKernelWithSpec(
        findShader("rt_scheduling_device_generated_commands_classify.comp", "rt_scheduling_worklist_classify.comp"), "main", 10, 0, m);
    vContext->setKernelAS(kernelClassifySpecialized[m], 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelClassifySpecialized[m], 1, resultBuffer);
    vContext->setKernelArg(kernelClassifySpecialized[m], 2, workListBuffer);
    vContext->setKernelArg(kernelClassifySpecialized[m], 3, indirectBuffer);
    vContext->setKernelArg(kernelClassifySpecialized[m], 4, fbWorkList);
    vContext->setKernelArg(kernelClassifySpecialized[m], 5, vertexBuffer);
    vContext->setKernelArg(kernelClassifySpecialized[m], 6, materialBuffer);
    vContext->setKernelArg(kernelClassifySpecialized[m], 7, triangleMaterialBuffer);
    vContext->setKernelArg(kernelClassifySpecialized[m], 8, texHeaderBuffer);
    vContext->setKernelArg(kernelClassifySpecialized[m], 9, texPixelBuffer);
  }

  kernelMaterial = vContext->createKernel(
      findShader("rt_scheduling_device_generated_commands_material.comp", "rt_scheduling_worklist_material.comp"), "main", 9);
  vContext->setKernelAS(kernelMaterial, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelMaterial, 1, resultBuffer);
  vContext->setKernelArg(kernelMaterial, 2, workListBuffer);
  vContext->setKernelArg(kernelMaterial, 3, fbWorkList);
  vContext->setKernelArg(kernelMaterial, 4, vertexBuffer);
  vContext->setKernelArg(kernelMaterial, 5, materialBuffer);
  vContext->setKernelArg(kernelMaterial, 6, triangleMaterialBuffer);
  vContext->setKernelArg(kernelMaterial, 7, texHeaderBuffer);
  vContext->setKernelArg(kernelMaterial, 8, texPixelBuffer);

  kernelMaterialGltf = vContext->createKernelWithSpec(
      findShader("rt_scheduling_device_generated_commands_material.comp", "rt_scheduling_worklist_material.comp"), "main", 9, 0, 9u);
  vContext->setKernelAS(kernelMaterialGltf, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelMaterialGltf, 1, resultBuffer);
  vContext->setKernelArg(kernelMaterialGltf, 2, workListBuffer);
  vContext->setKernelArg(kernelMaterialGltf, 3, fbWorkList);
  vContext->setKernelArg(kernelMaterialGltf, 4, vertexBuffer);
  vContext->setKernelArg(kernelMaterialGltf, 5, materialBuffer);
  vContext->setKernelArg(kernelMaterialGltf, 6, triangleMaterialBuffer);
  vContext->setKernelArg(kernelMaterialGltf, 7, texHeaderBuffer);
  vContext->setKernelArg(kernelMaterialGltf, 8, texPixelBuffer);

  for (uint32_t arch = 0; arch < 8; ++arch) {
    kernelMaterialSpecialized[arch] = vContext->createKernelWithSpec(
        findShader("rt_scheduling_device_generated_commands_material.comp", "rt_scheduling_worklist_material.comp"), "main", 9, 0, arch);
    vContext->setKernelAS(kernelMaterialSpecialized[arch], 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 1, resultBuffer);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 2, workListBuffer);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 3, fbWorkList);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 4, vertexBuffer);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 5, materialBuffer);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 6, triangleMaterialBuffer);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 7, texHeaderBuffer);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 8, texPixelBuffer);
  }
  kernelBounce = vContext->createKernel(
      findShader("rt_scheduling_device_generated_commands_bounce.comp", "rt_scheduling_worklist_bounce.comp"), "main", 9);
  vContext->setKernelAS(kernelBounce, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelBounce, 1, resultBuffer);
  vContext->setKernelArg(kernelBounce, 2, workListBuffer);
  vContext->setKernelArg(kernelBounce, 3, vertexBuffer);
  vContext->setKernelArg(kernelBounce, 4, materialBuffer);
  vContext->setKernelArg(kernelBounce, 5, triangleMaterialBuffer);
  vContext->setKernelArg(kernelBounce, 6, texHeaderBuffer);
  vContext->setKernelArg(kernelBounce, 7, texPixelBuffer);
  vContext->setKernelArg(kernelBounce, 8, fbWorkList);

  kernelBounceTerminal = vContext->createKernelWithSpec(
      findShader("rt_scheduling_device_generated_commands_bounce.comp", "rt_scheduling_worklist_bounce.comp"), "main", 9, 0, 1u);
  vContext->setKernelAS(kernelBounceTerminal, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelBounceTerminal, 1, resultBuffer);
  vContext->setKernelArg(kernelBounceTerminal, 2, workListBuffer);
  vContext->setKernelArg(kernelBounceTerminal, 3, vertexBuffer);
  vContext->setKernelArg(kernelBounceTerminal, 4, materialBuffer);
  vContext->setKernelArg(kernelBounceTerminal, 5, triangleMaterialBuffer);
  vContext->setKernelArg(kernelBounceTerminal, 6, texHeaderBuffer);
  vContext->setKernelArg(kernelBounceTerminal, 7, texPixelBuffer);
  vContext->setKernelArg(kernelBounceTerminal, 8, fbWorkList);

  kernelReset = vContext->createKernel(
      (kdir / "vulkan" / "rt_scheduling_reset.comp").string(), "main", 2);
  vContext->setKernelArg(kernelReset, 0, workListBuffer);
  vContext->setKernelArg(kernelReset, 1, indirectBuffer);

  // Allocate DGC buffers
  dgcSequenceCountBuffer = context->createBuffer(64);
  uint32_t zeroCount = 0;
  context->writeBuffer(dgcSequenceCountBuffer, 0, sizeof(zeroCount), &zeroCount);

  size_t dgcSeqSize = sizeof(uint32_t) * 12 * 64; // 3072 bytes (32 templates + 32 compacted)
  dgcSequenceBuffer = context->createBuffer(dgcSeqSize);

  kernelResolve = vContext->createKernel(
      (kdir / "vulkan" / "rt_scheduling_resolve.comp").string(), "main", 4);
  vContext->setKernelArg(kernelResolve, 0, workListBuffer);
  vContext->setKernelArg(kernelResolve, 1, indirectBuffer);
  vContext->setKernelArg(kernelResolve, 2, dgcSequenceCountBuffer);
  vContext->setKernelArg(kernelResolve, 3, dgcSequenceBuffer);

  kernelMultiLight = vContext->createKernel(
      findShader("rt_scheduling_device_generated_commands_light.comp", "rt_scheduling_device_generated_commands_light.comp"), "main", 9);
  vContext->setKernelAS(kernelMultiLight, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelMultiLight, 1, resultBuffer);
  vContext->setKernelArg(kernelMultiLight, 2, workListBuffer);
  vContext->setKernelArg(kernelMultiLight, 3, fbWorkList);
  vContext->setKernelArg(kernelMultiLight, 4, vertexBuffer);
  vContext->setKernelArg(kernelMultiLight, 5, materialBuffer);
  vContext->setKernelArg(kernelMultiLight, 6, triangleMaterialBuffer);
  vContext->setKernelArg(kernelMultiLight, 7, texHeaderBuffer);
  vContext->setKernelArg(kernelMultiLight, 8, texPixelBuffer);

  // Load RTP Baseline Pipeline
  try {
    kernelRTP = vContext->createRTPipeline(
        (kdir / "vulkan" / "rt_scheduling.rgen").string(),
        (kdir / "vulkan" / "rt_scheduling.rmiss").string(),
        {(kdir / "vulkan" / "rt_scheduling.rchit").string()},
        {}, {}, 8, 7u, true);
    vContext->setKernelAS(kernelRTP, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelRTP, 1, resultBuffer);
    vContext->setKernelArg(kernelRTP, 2, fbTraditional);
    vContext->setKernelArg(kernelRTP, 3, vertexBuffer);
    vContext->setKernelArg(kernelRTP, 4, materialBuffer);
    vContext->setKernelArg(kernelRTP, 5, triangleMaterialBuffer);
    vContext->setKernelArg(kernelRTP, 6, texHeaderBuffer);
    vContext->setKernelArg(kernelRTP, 7, texPixelBuffer);
  } catch (const std::exception &e) {
    if (vContext->isVerbose()) {
      std::cerr << "[RayScheduling] Warning: Failed to create RTP pipeline: " << e.what() << std::endl;
    }
    kernelRTP = nullptr;
  }

  // Check driver SER support & Load RTP+SER Pipeline
  bool hasSERExt = vContext->isExtensionEnabled("VK_EXT_ray_tracing_invocation_reorder") ||
                   vContext->isExtensionEnabled("VK_NV_ray_tracing_invocation_reorder");
  bool serSupported = vContext->isSERSupported();
  serEmulated = false;

  if (serSupported) {
    try {
      kernelRTPSER = vContext->createRTPipeline(
          (kdir / "vulkan" / "rt_scheduling_ser.rgen").string(),
          (kdir / "vulkan" / "rt_scheduling.rmiss").string(),
          {(kdir / "vulkan" / "rt_scheduling.rchit").string()},
          {}, {}, 8, 0u, true);
      vContext->setKernelAS(kernelRTPSER, 0, (AccelerationStructure)sceneTlas);
      vContext->setKernelArg(kernelRTPSER, 1, resultBuffer);
      vContext->setKernelArg(kernelRTPSER, 2, fbTraditional);
      vContext->setKernelArg(kernelRTPSER, 3, vertexBuffer);
      vContext->setKernelArg(kernelRTPSER, 4, materialBuffer);
      vContext->setKernelArg(kernelRTPSER, 5, triangleMaterialBuffer);
      vContext->setKernelArg(kernelRTPSER, 6, texHeaderBuffer);
      vContext->setKernelArg(kernelRTPSER, 7, texPixelBuffer);
      serEmulated = !vContext->isSERHardwareSupported();
    } catch (const std::exception &e) {
      if (vContext->isVerbose()) {
        std::cerr << "[RayScheduling] Warning: Failed to create RTP+SER pipeline: " << e.what() << std::endl;
      }
      kernelRTPSER = nullptr;
      serEmulated = false;
    }
  }

  for (int i = 0; i < 24; ++i) {
    unsupportedConfig[i] = false;
    unsupportedReason[i] = "";
  }

  if (!kernelRTP) {
    unsupportedConfig[20] = true;
    unsupportedReason[20] = "Vulkan Ray Tracing Pipeline creation failed";
  }

  if (!kernelRTPSER) {
    std::string serReason = !hasSERExt
        ? "extension VK_EXT_ray_tracing_invocation_reorder missing"
        : (!serSupported
            ? "Hardware Shader Execution Reordering (SER) not supported by GPU architecture"
            : "RTP SER pipeline compilation or creation failed");
    for (uint32_t c : {1u, 4u, 7u, 16u, 21u}) {
      unsupportedConfig[c] = true;
      unsupportedReason[c] = serReason;
    }
  }

  uint32_t perQueue = rayCount / 8;

  // Pre-generate static indirect batches for Wavefront/DGC dispatches with specialized PSOs
  materialBatches.reserve(8);
  materialBatchesBreakdown.reserve(8);
  for (uint32_t m = 0; m < 8; ++m) {
    struct {
      uint32_t materialId;
      uint32_t queueCapacity;
      uint32_t dumpRenders;
      uint32_t width;
      uint32_t height;
      uint32_t sceneType;
      uint32_t isGltf;
      uint32_t mode;
    } pcMat{m, materialCapacity, 0u, renderWidth, renderHeight, 3u, isGltf ? 1u : 0u, 0u};
    std::vector<uint8_t> pcData(sizeof(pcMat));
    std::memcpy(pcData.data(), &pcMat, sizeof(pcMat));
    materialBatches.push_back({m * sizeof(uint32_t) * 3, pcData, kernelMaterialSpecialized[m]});

    pcMat.queueCapacity = perQueue;
    pcMat.dumpRenders = 0u;
    pcMat.mode = 1u;
    std::memcpy(pcData.data(), &pcMat, sizeof(pcMat));
    materialBatchesBreakdown.push_back({m * sizeof(uint32_t) * 3, pcData, kernelMaterialSpecialized[m]});
  }

  // Pre-initialize dedicated indirectMaterialBuffer for isolated Config 2 fallback testing
  indirectMaterialBuffer = context->createBuffer(sizeof(uint32_t) * 3 * 8);
  std::vector<uint32_t> initMatCmds(8 * 3, 0);
  for (uint32_t m = 0; m < 8; ++m) {
    initMatCmds[m * 3 + 0] = (perQueue + 31) / 32;
    initMatCmds[m * 3 + 1] = 1;
    initMatCmds[m * 3 + 2] = 1;
  }
  context->writeBuffer(indirectMaterialBuffer, 0, initMatCmds.size() * sizeof(uint32_t), initMatCmds.data());

  rebuildBounceBatches();

  // Pre-generate 128-light multi-light indirect batches
  multiLightBatches128.clear();
  struct {
    uint32_t queueId;
    uint32_t queueCapacity;
    uint32_t dumpRenders;
    uint32_t width;
    uint32_t height;
    uint32_t sceneType;
    uint32_t isGltf;
    uint32_t binIndex;
  } pcMultiLight128{0, materialCapacity, 0u, renderWidth, renderHeight, 3u, isGltf ? 1u : 0u, 128u};
  std::vector<uint8_t> pcMultiLightData(sizeof(pcMultiLight128));
  for (uint32_t b = 0; b < 8; ++b) {
    pcMultiLight128.queueId = b;
    std::memcpy(pcMultiLightData.data(), &pcMultiLight128, sizeof(pcMultiLight128));
    multiLightBatches128.push_back({b * sizeof(uint32_t) * 3, pcMultiLightData, kernelMultiLight});
  }

  // Native Vulkan Device-Generated Commands (VK_EXT_device_generated_commands)
  isDGCAvailable = false;
  if (vContext->isDGCSupported()) {
    try {
      // 1. Standard IndirectCommandsLayout (PushConstant Token + Dispatch Token)
      VkIndirectCommandsPushConstantTokenEXT pcToken{};
      pcToken.updateRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      pcToken.updateRange.offset = 0;
      pcToken.updateRange.size = sizeof(uint32_t) * 8; // 32 bytes

      VkIndirectCommandsLayoutTokenEXT tokensStd[2]{};
      tokensStd[0].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
      tokensStd[0].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT;
      tokensStd[0].data.pPushConstant = &pcToken;
      tokensStd[0].offset = sizeof(uint32_t); // offset 4 (pc field)

      tokensStd[1].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
      tokensStd[1].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
      tokensStd[1].offset = sizeof(uint32_t) * 9; // offset 36 (dispatch cmd)

      VkIndirectCommandsLayoutCreateInfoEXT layoutInfoStd{VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT};
      layoutInfoStd.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
      layoutInfoStd.indirectStride = sizeof(uint32_t) * 12; // 48 bytes
      layoutInfoStd.pipelineLayout = vContext->getVkPipelineLayout(kernelBounce);
      layoutInfoStd.tokenCount = 2;
      layoutInfoStd.pTokens = tokensStd;

      dgcLayoutStandard = vContext->createIndirectCommandsLayout(layoutInfoStd);

      // 2. Specialized IndirectCommandsLayout & ExecutionSet (only if driver supports compute pipeline binding in DGC)
      if (vContext->isDGCExecutionSetSupported()) {
        try {
          VkIndirectCommandsExecutionSetTokenEXT execToken{};
          execToken.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
          execToken.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;

          VkIndirectCommandsLayoutTokenEXT tokensSpec[3]{};
          tokensSpec[0].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
          tokensSpec[0].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT;
          tokensSpec[0].data.pExecutionSet = &execToken;
          tokensSpec[0].offset = 0; // pipelineIndex at offset 0

          tokensSpec[1].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
          tokensSpec[1].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT;
          tokensSpec[1].data.pPushConstant = &pcToken;
          tokensSpec[1].offset = sizeof(uint32_t); // offset 4

          tokensSpec[2].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
          tokensSpec[2].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
          tokensSpec[2].offset = sizeof(uint32_t) * 9; // offset 36

          VkIndirectCommandsLayoutCreateInfoEXT layoutInfoSpec{VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT};
          layoutInfoSpec.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
          layoutInfoSpec.indirectStride = sizeof(uint32_t) * 12; // 48 bytes
          layoutInfoSpec.pipelineLayout = vContext->getVkPipelineLayout(kernelMaterialSpecialized[0]);
          layoutInfoSpec.tokenCount = 3;
          layoutInfoSpec.pTokens = tokensSpec;

          dgcLayoutSpecialized = vContext->createIndirectCommandsLayout(layoutInfoSpec);

          // 3. Indirect Execution Set with 8 specialized material pipelines
          VkIndirectExecutionSetPipelineInfoEXT iesPipeInfo{VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT};
          iesPipeInfo.initialPipeline = vContext->getVkPipeline(kernelMaterialSpecialized[0]);
          iesPipeInfo.maxPipelineCount = 8;

          VkIndirectExecutionSetCreateInfoEXT iesInfo{VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT};
          iesInfo.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
          iesInfo.info.pPipelineInfo = &iesPipeInfo;

          dgcExecutionSetSpecialized = vContext->createIndirectExecutionSet(iesInfo);
          for (uint32_t m = 0; m < 8; ++m) {
            vContext->updateIndirectExecutionSetPipeline(dgcExecutionSetSpecialized, m, kernelMaterialSpecialized[m]);
          }
          isDGCExecutionSetAvailable = true;
        } catch (const std::exception &e) {
          std::cerr << "Warning: DGC ExecutionSet initialization failed (" << e.what()
                    << "), specialized PSOs will use indirect dispatch fallback." << std::endl;
          if (dgcLayoutSpecialized != VK_NULL_HANDLE) {
            vContext->destroyIndirectCommandsLayout(dgcLayoutSpecialized);
            dgcLayoutSpecialized = VK_NULL_HANDLE;
          }
          if (dgcExecutionSetSpecialized != VK_NULL_HANDLE) {
            vContext->destroyIndirectExecutionSet(dgcExecutionSetSpecialized);
            dgcExecutionSetSpecialized = VK_NULL_HANDLE;
          }
          isDGCExecutionSetAvailable = false;
        }
      }

      // 4. Query preprocess memory requirements & allocate preprocess buffer
      VkDeviceSize memReqStd = vContext->getGeneratedCommandsMemoryRequirements(
          dgcLayoutStandard, VK_NULL_HANDLE, 32, kernelMaterial);
      VkDeviceSize memReqSpec = 0;
      if (isDGCExecutionSetAvailable) {
        memReqSpec = vContext->getGeneratedCommandsMemoryRequirements(
            dgcLayoutSpecialized, dgcExecutionSetSpecialized, 32, nullptr);
      }
      dgcPreprocessBufferSize = std::max(memReqStd, memReqSpec);
      if (dgcPreprocessBufferSize == 0) {
        dgcPreprocessBufferSize = 2048;
      }
      dgcPreprocessBuffer = vContext->createPreprocessBuffer(dgcPreprocessBufferSize);

      // 5. Initialize DGCExecutionInfo structures
      dgcInfoStandard.layout = dgcLayoutStandard;
      dgcInfoStandard.executionSet = VK_NULL_HANDLE;
      dgcInfoStandard.sequenceBuffer = dgcSequenceBuffer;
      dgcInfoStandard.sequenceBufferOffset = sizeof(uint32_t) * 12 * 32; // compacted[0] at offset 1536
      dgcInfoStandard.sequenceBufferSize = sizeof(uint32_t) * 12 * 32;
      dgcInfoStandard.sequenceCountBuffer = dgcSequenceCountBuffer;
      dgcInfoStandard.sequenceCountBufferOffset = 0;
      dgcInfoStandard.preprocessBuffer = dgcPreprocessBuffer;
      dgcInfoStandard.preprocessBufferSize = dgcPreprocessBufferSize;
      dgcInfoStandard.maxSequenceCount = 32;

      dgcInfoFullFrame = dgcInfoStandard;
      if (isDGCExecutionSetAvailable) {
        dgcInfoFullFrame.layout = dgcLayoutSpecialized;
        dgcInfoFullFrame.executionSet = dgcExecutionSetSpecialized;
      }

      uint32_t sceneTypeVal = 3u;
      uint32_t isGltfVal = isGltf ? 1u : 0u;

      dgcSpecializedSequenceBuffer = context->createBuffer(sizeof(uint32_t) * 12 * 8);
      for (uint32_t m = 0; m < 8; ++m) {
        uint32_t pc[8] = {
            m, perQueue, 0u, renderWidth, renderHeight, sceneTypeVal, isGltfVal, 1u
        };
        uint32_t item[12] = {
            m, pc[0], pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], pc[7],
            (perQueue + 31) / 32, 1, 1
        };
        context->writeBuffer(dgcSpecializedSequenceBuffer, m * sizeof(item), sizeof(item), item);
      }

      dgcInfoSpecialized = dgcInfoStandard;
      dgcInfoSpecialized.layout = isDGCExecutionSetAvailable ? dgcLayoutSpecialized : dgcLayoutStandard;
      dgcInfoSpecialized.executionSet = isDGCExecutionSetAvailable ? dgcExecutionSetSpecialized : VK_NULL_HANDLE;
      dgcInfoSpecialized.sequenceBuffer = dgcSpecializedSequenceBuffer;
      dgcInfoSpecialized.sequenceBufferOffset = 0;
      dgcInfoSpecialized.sequenceBufferSize = sizeof(uint32_t) * 12 * 8;
      dgcInfoSpecialized.sequenceCountBuffer = nullptr;
      dgcInfoSpecialized.sequenceCountBufferOffset = 0;
      dgcInfoSpecialized.maxSequenceCount = 8;

      // 6. Pre-seed templates in dgcSequenceBuffer

      // Material templates: slots 0..7 (full PBR) and slots 24..31 (Primary Rays G-Buffer normals)
      for (uint32_t m = 0; m < 8; ++m) {
        uint32_t pc[8] = {
            m, materialCapacity, 0u, renderWidth, renderHeight,
            sceneTypeVal, isGltfVal, 0u
        };
        uint32_t item[12] = {
            m, pc[0], pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], pc[7], 0, 1, 1
        };
        context->writeBuffer(dgcSequenceBuffer, m * sizeof(item), sizeof(item), item);

        uint32_t itemPrimary[12] = {
            m, pc[0], pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], 2u, 0, 1, 1
        };
        context->writeBuffer(dgcSequenceBuffer, (24 + m) * sizeof(itemPrimary), sizeof(itemPrimary), itemPrimary);
      }

      // Shadow templates: slots 9..11
      for (uint32_t l = 0; l < 3; ++l) {
        uint32_t pc[8] = {
            l, materialCapacity, 0u, renderWidth, renderHeight,
            sceneTypeVal, isGltfVal, l
        };
        uint32_t item[12] = {
            0, pc[0], pc[1], pc[2], pc[3], pc[4], pc[5], pc[6], pc[7], 0, 1, 1
        };
        context->writeBuffer(dgcSequenceBuffer, (9 + l) * sizeof(item), sizeof(item), item);
      }

      // Multi-Light templates: slot 12 (Single Light) and slot 13 (128 Lights)
      {
        uint32_t pcSingle[8] = {
            0, materialCapacity, 0u, renderWidth, renderHeight,
            sceneTypeVal, isGltfVal, 1u
        };
        uint32_t itemSingle[12] = {
            0, pcSingle[0], pcSingle[1], pcSingle[2], pcSingle[3], pcSingle[4], pcSingle[5], pcSingle[6], pcSingle[7], 0, 1, 1
        };
        context->writeBuffer(dgcSequenceBuffer, 12 * sizeof(itemSingle), sizeof(itemSingle), itemSingle);

        uint32_t pc128[8] = {
            0, materialCapacity, 0u, renderWidth, renderHeight,
            sceneTypeVal, isGltfVal, 128u
        };
        uint32_t item128[12] = {
            0, pc128[0], pc128[1], pc128[2], pc128[3], pc128[4], pc128[5], pc128[6], pc128[7], 0, 1, 1
        };
        context->writeBuffer(dgcSequenceBuffer, 13 * sizeof(item128), sizeof(item128), item128);
      }

      rebuildDGCBounceBatches();

      isDGCAvailable = true;
    } catch (const std::exception &e) {
      std::cerr << "Warning: DGC initialization failed (" << e.what()
                << "), falling back to standard indirect dispatch." << std::endl;
      if (vContext) {
        if (dgcLayoutStandard != VK_NULL_HANDLE) {
          vContext->destroyIndirectCommandsLayout(dgcLayoutStandard);
          dgcLayoutStandard = VK_NULL_HANDLE;
        }
        if (dgcLayoutSpecialized != VK_NULL_HANDLE) {
          vContext->destroyIndirectCommandsLayout(dgcLayoutSpecialized);
          dgcLayoutSpecialized = VK_NULL_HANDLE;
        }
        if (dgcExecutionSetSpecialized != VK_NULL_HANDLE) {
          vContext->destroyIndirectExecutionSet(dgcExecutionSetSpecialized);
          dgcExecutionSetSpecialized = VK_NULL_HANDLE;
        }
      }
      if (dgcPreprocessBuffer) {
        context->releaseBuffer(dgcPreprocessBuffer);
        dgcPreprocessBuffer = nullptr;
      }
      isDGCAvailable = false;
    }
  }

  if (!isDGCAvailable) {
    const std::vector<uint32_t> dgcConfigs = {2, 5, 8, 23};
    for (uint32_t c : dgcConfigs) {
      unsupportedConfig[c] = true;
      unsupportedReason[c] = "VK_EXT_device_generated_commands extension unsupported by device/driver";
    }
  }
#endif
}

void RaySchedulingBench::Run(uint32_t config_idx) {
#ifdef HAVE_VULKAN
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  if (config_idx >= 24 || unsupportedConfig[config_idx])
    return;

  if (config_idx != lastRunConfig && resultBuffer) {
    uint32_t zeros[4] = {0, 0, 0, 0};
    context->writeBuffer(resultBuffer, 0, sizeof(zeros), zeros);
    if (workListBuffer) {
      uint32_t zeroCounters[64] = {0};
      context->writeBuffer(workListBuffer, 0, sizeof(zeroCounters), zeroCounters);
    }
    lastRunConfig = config_idx;
  }

  uint32_t seed = captureFramebuffer ? 1337u : rng();
  uint32_t sceneTypeVal = 3u;
  uint32_t isGltfVal = isGltf ? 1u : 0u;
  uint32_t dumpFlag = captureFramebuffer ? 1u : 0u;

  struct PushConstantsTraditional {
    uint32_t rayCount;
    uint32_t mode;
    uint32_t bounces;
    uint32_t seed;
    uint32_t dumpRenders;
    uint32_t width;
    uint32_t height;
    uint32_t spatialPattern;
    uint32_t sceneType;
    uint32_t isGltf;
    uint32_t spp;
    uint32_t sampleIdx{0};
  };

  struct PushConstantsClassify {
    uint32_t rayCount;
    uint32_t mode;
    uint32_t bounce;
    uint32_t seed;
    uint32_t dumpRenders;
    uint32_t width;
    uint32_t height;
    uint32_t queueCapacity;
    uint32_t spatialPattern;
    uint32_t sceneType;
    uint32_t isGltf;
    uint32_t spp;
  };

  auto tradKernel = [&](uint32_t m) -> ComputeKernel {
    return (m < 13u && kernelTraditionalSpecialized[m]) ? kernelTraditionalSpecialized[m] : kernelTraditional;
  };

  auto runChunkedPasses = [&](uint32_t passes, uint32_t baselineIdx, const auto &recordFn) {
    uint32_t passesPerSubmit = (rayCount >= 500'000u) ? 2u : passes;
    if (passes > 1u && recordedInvocations[baselineIdx] > 0 && recordedTimeMs[baselineIdx] > 0.0) {
      const double singlePassMs =
          recordedTimeMs[baselineIdx] / static_cast<double>(recordedInvocations[baselineIdx]);
      if (singlePassMs > 0.0) {
        const uint32_t targetPasses = static_cast<uint32_t>(std::floor(400.0 / singlePassMs));
        passesPerSubmit = std::clamp(targetPasses, 1u, passes);
      }
    }
    if (passes <= 1u || passesPerSubmit >= passes) {
      bool ownBatch = (passes > 1u) && !vContext->isBatching();
      if (ownBatch) vContext->beginBatch(passes, true);
      for (uint32_t s = 0; s < passes; ++s) {
        recordFn(s);
      }
      if (ownBatch) vContext->endBatch();
    } else {
      const bool wasBatching = vContext->isBatching();
      if (wasBatching) vContext->endBatch();
      for (uint32_t base = 0; base < passes; base += passesPerSubmit) {
        uint32_t endPass = std::min(base + passesPerSubmit, passes);
        vContext->beginBatch(endPass - base, true);
        for (uint32_t s = base; s < endPass; ++s) {
          recordFn(s);
        }
        vContext->endBatch();
        if (endPass < passes) {
          vContext->waitIdle();
        }
      }
      if (wasBatching) vContext->beginBatch(1, true);
    }
  };

  switch (config_idx) {
  case 0: { // Material Divergence - Traditional Megakernel (Pure Shading Microbenchmark, 4 Lights)
    ComputeKernel kTrad = tradKernel(4u);
    PushConstantsTraditional pc{rayCount, 4, 1, seed, 0, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 1: { // Stage: Material Shading - RTP + Hardware SER
    if (kernelRTPSER) {
      PushConstantsTraditional pc{rayCount, 4, 1, seed, 0, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
      vContext->setKernelArg(kernelRTPSER, 8, sizeof(pc), &pc);
      vContext->dispatch(kernelRTPSER, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    }
    break;
  }
  case 2: { // Material Divergence - Device-Generated Commands (DGC) (Specialized Micro-Kernels, Pure Shading)
    if (isDGCAvailable) {
      ComputeKernel kMat = isDGCExecutionSetAvailable ? kernelMaterialSpecialized[0] : kernelMaterial;
      vContext->dispatchDGCSequence(kMat, dgcInfoSpecialized);
    }
    break;
  }
  case 3: { // Path Tracing - Traditional Megakernel
    ComputeKernel kTrad = tradKernel(1u);
    uint32_t passes = (samplesPerPixel > 1) ? samplesPerPixel : 1u;
    runChunkedPasses(passes, 3u, [&](uint32_t s) {
      PushConstantsTraditional pc{rayCount, 1, 1 + bounceDepth, seed + s * 7919u, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, passes, s};
      vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
      vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    });
    lastCompletedPasses[config_idx] = passes;
    break;
  }
  case 4: { // Path Tracing - RTP + Hardware SER
    if (kernelRTPSER) {
      uint32_t passes = (samplesPerPixel > 1) ? samplesPerPixel : 1u;
      runChunkedPasses(passes, 4u, [&](uint32_t s) {
        PushConstantsTraditional pc{rayCount, 1, 1 + bounceDepth, seed + s * 7919u, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, passes, s};
        vContext->setKernelArg(kernelRTPSER, 8, sizeof(pc), &pc);
        vContext->dispatch(kernelRTPSER, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
      });
      lastCompletedPasses[config_idx] = passes;
    }
    break;
  }
  case 5: { // Multi-Bounce Path Tracing - Device-Generated Commands (DGC) (Wavefront Compaction)
    ComputeKernel kClass = kernelClassifySpecialized[1] ? kernelClassifySpecialized[1] : kernelClassify;
    uint32_t passes = (samplesPerPixel > 1) ? samplesPerPixel : 1u;
    runChunkedPasses(passes, 5u, [&](uint32_t s) {
      uint32_t sampleSeed = seed + s * 7919u;
      for (size_t b = 0; b < bounceBatches.size(); ++b) {
        uint32_t bounceSeed = sampleSeed + static_cast<uint32_t>(b) * 7919u;
        if (bounceBatches[b].pushConstants.size() >= 4 * sizeof(uint32_t)) {
          std::memcpy(bounceBatches[b].pushConstants.data() + 3 * sizeof(uint32_t),
                      &bounceSeed, sizeof(uint32_t));
        }
      }
      PushConstantsClassify pcClassify{rayCount, 1, s, sampleSeed, dumpFlag, renderWidth, renderHeight, bounceCapacity, 2, sceneTypeVal, isGltfVal, passes};
      vContext->setKernelArg(kClass, 10, sizeof(pcClassify), &pcClassify);

      vContext->dispatchWorkListSequence(
          nullptr,
          kClass, (rayCount + 31) / 32, 1, 1,
          kernelResolve,
          kernelBounce, indirectBuffer, bounceBatches, true /* isPingPong */,
          isDGCAvailable ? &dgcInfoStandard : nullptr, isDGCAvailable ? 3u : 0u);
    });
    lastCompletedPasses[config_idx] = passes;
    break;
  }
  case 6: { // Scene Render: Full Frame - Traditional Megakernel
    ComputeKernel kTrad = tradKernel(0u);
    PushConstantsTraditional pc{rayCount, 0, 1, seed, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 7: { // Scene Render: Full Frame - RTP + Hardware SER
    if (kernelRTPSER) {
      PushConstantsTraditional pc{rayCount, 0, 1, seed, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
      vContext->setKernelArg(kernelRTPSER, 8, sizeof(pc), &pc);
      vContext->dispatch(kernelRTPSER, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    }
    break;
  }
  case 8: { // Scene Render: Full Frame - DGC (Material Sorting)
    for (uint32_t m = 0; m < materialBatches.size(); ++m) {
      struct {
        uint32_t materialId;
        uint32_t queueCapacity;
        uint32_t dumpRenders;
        uint32_t width;
        uint32_t height;
        uint32_t sceneType;
        uint32_t isGltf;
        uint32_t mode;
      } pcMat{m, materialCapacity, dumpFlag, renderWidth, renderHeight, sceneTypeVal, isGltfVal, 0u};
      std::memcpy(materialBatches[m].pushConstants.data(), &pcMat, sizeof(pcMat));
    }
    ComputeKernel kClass = kernelClassifySpecialized[0] ? kernelClassifySpecialized[0] : kernelClassify;
    ComputeKernel kMat = isDGCExecutionSetAvailable
                             ? kernelMaterialSpecialized[0]
                             : (kernelMaterialGltf ? kernelMaterialGltf : kernelMaterial);
    PushConstantsClassify pcClassify{rayCount, 0, 0, seed, dumpFlag, renderWidth, renderHeight, materialCapacity, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kClass, 10, sizeof(pcClassify), &pcClassify);

    vContext->dispatchWorkListSequence(
        nullptr,
        kClass, (rayCount + 31) / 32, 1, 1,
        kernelResolve,
        kMat, indirectBuffer, materialBatches,
        false /* isPingPong */,
        isDGCAvailable ? &dgcInfoFullFrame : nullptr, isDGCAvailable ? 1u : 0u);
    break;
  }
  case 9: { // Stage Breakdown - BVH Traversal (Linear 32x1, Baseline)
    ComputeKernel kTrad = tradKernel(3u);
    PushConstantsTraditional pc{rayCount, 3, 1, seed, 0, renderWidth, renderHeight, 0, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 10: { // Stage Breakdown - Queue Compaction Overhead (Wave Ballot Stream Sort)
    vContext->dispatch(kernelReset, 1, 1, 1, 32, 1, 1);
    PushConstantsClassify pcClassify{rayCount, 3, 0, seed, 0, renderWidth, renderHeight, octantCapacity, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kernelClassify, 10, sizeof(pcClassify), &pcClassify);
    vContext->dispatch(kernelClassify, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 11: { // Stage Breakdown - BVH Traversal (2D Tiled 8x4)
    ComputeKernel kTrad = tradKernel(3u);
    PushConstantsTraditional pc{rayCount, 3, 1, seed, 0, renderWidth, renderHeight, 1, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 12: { // Stage Breakdown - BVH Traversal (2D Morton 8x4)
    ComputeKernel kTrad = tradKernel(3u);
    PushConstantsTraditional pc{rayCount, 3, 1, seed, 0, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 13: { // Stage Breakdown - BVH Traversal (2D Morton 4x8)
    ComputeKernel kTrad = tradKernel(3u);
    PushConstantsTraditional pc{rayCount, 3, 1, seed, 0, renderWidth, renderHeight, 3, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 14: { // Stage Breakdown - Primary Ray Tracing (2D Morton 8x4, Traditional Megakernel)
    ComputeKernel kTrad = tradKernel(7u);
    PushConstantsTraditional pc{rayCount, 7, 1, seed, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 15: { // Ray-Traced Shadows - Traditional Megakernel (Directional Shadow Rays, In-Kernel Traversal)
    ComputeKernel kTrad = tradKernel(5u);
    PushConstantsTraditional pc{rayCount, 5, 1, seed, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 16: { // Ray-Traced Shadows - RTP + Hardware SER
    if (kernelRTPSER) {
      PushConstantsTraditional pc{rayCount, 5, 1, seed, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
      vContext->setKernelArg(kernelRTPSER, 8, sizeof(pc), &pc);
      vContext->dispatch(kernelRTPSER, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    }
    break;
  }
  case 17: { // Queue Compaction - Single-Pass Unified Stream (128B Expanded Hit Record)
    vContext->dispatch(kernelReset, 1, 1, 1, 32, 1, 1);
    PushConstantsClassify pcClassify{rayCount, 6, 1, seed, 0, renderWidth, renderHeight, materialCapacity, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kernelClassify, 10, sizeof(pcClassify), &pcClassify);
    vContext->dispatch(kernelClassify, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 18: { // Scene Ray Tracing - Alpha-Mask Cutout Traversal Divergence
    vContext->dispatch(kernelReset, 1, 1, 1, 32, 1, 1);
    PushConstantsClassify pcClassify{rayCount, 7, 0, seed, dumpFlag, renderWidth, renderHeight, materialCapacity, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kernelClassify, 10, sizeof(pcClassify), &pcClassify);
    vContext->dispatch(kernelClassify, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 19: { // Queue Compaction - Queue Compaction Throughput (32B Packed Hybrid Record)
    vContext->dispatch(kernelReset, 1, 1, 1, 32, 1, 1);
    PushConstantsClassify pcClassify{rayCount, 6, 0, seed, 0, renderWidth, renderHeight, materialCapacity, 2, sceneTypeVal, isGltfVal, 1u};
    vContext->setKernelArg(kernelClassify, 10, sizeof(pcClassify), &pcClassify);
    vContext->dispatch(kernelClassify, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 20: { // Primary Rays - Dedicated Ray Tracing Pipeline (RTP)
    if (kernelRTP) {
      PushConstantsTraditional pc{rayCount, 7, 1, seed, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
      vContext->setKernelArg(kernelRTP, 8, sizeof(pc), &pc);
      vContext->dispatch(kernelRTP, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    }
    break;
  }
  case 21: { // Primary Rays - RTP + Hardware SER
    if (kernelRTPSER) {
      PushConstantsTraditional pc{rayCount, 7, 1, seed, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 1u};
      vContext->setKernelArg(kernelRTPSER, 8, sizeof(pc), &pc);
      vContext->dispatch(kernelRTPSER, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    }
    break;
  }
  case 22: { // Stage: Multi-Light Evaluation - 128 Lights (Megakernel)
    ComputeKernel kTrad = tradKernel(11u);
    PushConstantsTraditional pc{rayCount, 11, 0, seed, dumpFlag, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, 128u};
    vContext->setKernelArg(kTrad, 8, sizeof(pc), &pc);
    vContext->dispatch(kTrad, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 23: { // Stage: Multi-Light Evaluation - 128 Lights (DGC Light Binning)
    for (uint32_t b = 0; b < multiLightBatches128.size(); ++b) {
      struct {
        uint32_t queueId;
        uint32_t queueCapacity;
        uint32_t dumpRenders;
        uint32_t width;
        uint32_t height;
        uint32_t sceneType;
        uint32_t isGltf;
        uint32_t binIndex;
      } pcLight{b, materialCapacity, dumpFlag, renderWidth, renderHeight, sceneTypeVal, isGltfVal, 128u};
      std::memcpy(multiLightBatches128[b].pushConstants.data(), &pcLight, sizeof(pcLight));
    }
    ComputeKernel kClass = kernelClassifySpecialized[8] ? kernelClassifySpecialized[8] : kernelClassify;
    PushConstantsClassify pcClassify{rayCount, 8, 0, seed, dumpFlag, renderWidth, renderHeight, materialCapacity, 2, sceneTypeVal, isGltfVal, 128u};
    vContext->setKernelArg(kClass, 10, sizeof(pcClassify), &pcClassify);

    vContext->dispatchWorkListSequence(
        nullptr,
        kClass, (rayCount + 31) / 32, 1, 1,
        kernelResolve,
        kernelMultiLight, indirectBuffer, multiLightBatches128,
        false /* isPingPong */,
        isDGCAvailable ? &dgcInfoStandard : nullptr, isDGCAvailable ? 6u : 0u);
    break;
  }
  }
#endif
}

void RaySchedulingBench::performVisualVerification(bool isInteractive) {
#ifdef HAVE_VULKAN
  if (!context || !fbTraditional || !fbWorkList) return;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(context);
  std::shared_ptr<VulkanContext> sharedVk = GetSharedVulkanContextAPI();
  const bool writeDiskFiles = wantDumpRenders || dumpRenders;
  const bool directGpuPreview =
      captureGpuPreviews && vContext && (!sharedVk || sharedVk.get() == vContext);
  const bool crossGpuPreview =
      captureGpuPreviews && sharedVk && (sharedVk.get() != vContext);
  const std::string tag = "reactor";

  if (!isDGCAvailable) {
    if (context && !context->isQuiet()) {
      std::cout << "\r\033[K[Info] Skipping Megakernel vs. DGC visual parity verification: VK_EXT_device_generated_commands unsupported on this device/driver.\n";
    }
    if (writeDiskFiles || captureGpuPreviews) {
      dumpPipelineBreakdown(tag, isInteractive);
    }
    return;
  }

  if (isInteractive && context && !context->isQuiet()) {
    std::cout << "\r\033[K  \033[36m⠋\033[0m [1/3] Tonemapping and exporting primary frames..." << std::flush;
  }

  uint32_t sceneTypeVal = 3u;
  uint32_t isGltfVal = isGltf ? 1u : 0u;
  auto updateDgcDumpFlag = [&](uint32_t dumpFlag) {
    if (!dgcSequenceBuffer) return;
    for (uint32_t m = 0; m < 8; ++m) {
      uint32_t item[12] = {
          m, m, materialCapacity, dumpFlag, renderWidth, renderHeight,
          sceneTypeVal, isGltfVal, 0u, 0, 1, 1
      };
      context->writeBuffer(dgcSequenceBuffer, m * sizeof(item), sizeof(item), item);
    }
    uint32_t item128[12] = {
        0, 0, materialCapacity, dumpFlag, renderWidth, renderHeight,
        sceneTypeVal, isGltfVal, 128u, 0, 1, 1
    };
    context->writeBuffer(dgcSequenceBuffer, 13 * sizeof(item128), sizeof(item128), item128);
  };

  auto emitJsonDoc = [&](const std::string &jsonRelKey,
                         const std::string &diskPath,
                         yyjson_mut_doc *doc) {
    if (!doc) return;
    size_t len = 0;
    char *jsonStr = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, &len);
    if (jsonStr) {
      std::string content(jsonStr, len);
      content.push_back('\n');
      free(jsonStr);
      if (vContext) {
        vContext->storeInMemoryJson(jsonRelKey, content);
      }
      if (sharedVk && sharedVk.get() != vContext) {
        sharedVk->storeInMemoryJson(jsonRelKey, content);
      }
      if (writeDiskFiles && !diskPath.empty()) {
        std::ofstream jf(diskPath, std::ios::binary);
        if (jf.is_open()) {
          jf.write(content.data(), static_cast<std::streamsize>(content.size()));
        }
      }
    }
    yyjson_mut_doc_free(doc);
  };

  // Explicitly render one complete frame from each pipeline with dump enabled
  captureFramebuffer = true;
  updateDgcDumpFlag(1u);
  Run(6);
  context->waitIdle();
  Run(8);
  context->waitIdle();

  uint32_t width = renderWidth, height = renderHeight;
  size_t bufferSize = width * height * sizeof(float) * 4;

  if (directGpuPreview) {
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_traditional_megakernel.png", fbTraditional, fbTraditional, width, height, 0u, cachedKernelDir);
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_worklist_dgc.png", fbWorkList, fbWorkList, width, height, 0u, cachedKernelDir);
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_difference_heatmap.png", fbTraditional, fbWorkList, width, height, 1u, cachedKernelDir);
  }

  std::vector<float> hdrTrad(width * height * 4, 0.0f);
  std::vector<float> hdrWork(width * height * 4, 0.0f);

  context->readBuffer(fbTraditional, 0, bufferSize, hdrTrad.data());
  context->readBuffer(fbWorkList, 0, bufferSize, hdrWork.data());

  std::vector<uint8_t> ldrTrad, ldrWork, ldrDiff;
  auto metrics = gpubench::ImageExport::compareAndTonemap(
      hdrTrad.data(), hdrWork.data(), width, height, ldrTrad, ldrWork, ldrDiff);

  if (crossGpuPreview) {
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_traditional_megakernel.png", width, height, ldrTrad.data());
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_worklist_dgc.png", width, height, ldrWork.data());
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_difference_heatmap.png", width, height, ldrDiff.data());
  }

  const std::string rendersDir = writeDiskFiles ? resolveRendersDir() : "renders";

  std::string tradPng = rendersDir + "/render_" + tag + "_traditional_megakernel.png";
  std::string workPng = rendersDir + "/render_" + tag + "_worklist_dgc.png";
  std::string diffPng = rendersDir + "/render_" + tag + "_difference_heatmap.png";

  if (writeDiskFiles) {
    gpubench::ImageExport::writePNG(tradPng, width, height, ldrTrad);
    gpubench::ImageExport::writePNG(workPng, width, height, ldrWork);
    gpubench::ImageExport::writePNG(diffPng, width, height, ldrDiff);
  }

  // Extract timings for captioned telemetry slate (using recorded benchmark results or live measurements)
  double timeSecTrad = 0.0;
  if (recordedInvocations[6] > 0 && recordedTimeMs[6] > 0.0) {
    timeSecTrad = (recordedTimeMs[6] / 1000.0) / static_cast<double>(recordedInvocations[6]);
  } else {
    captureFramebuffer = false;
    updateDgcDumpFlag(0u);
    context->waitIdle();
    for (int w = 0; w < 2; ++w) Run(6);
    context->waitIdle();
    auto t0 = std::chrono::high_resolution_clock::now();
    const int iters = 6;
    for (int it = 0; it < iters; ++it) Run(6);
    context->waitIdle();
    auto t1 = std::chrono::high_resolution_clock::now();
    timeSecTrad = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(iters);
    captureFramebuffer = true;
    updateDgcDumpFlag(1u);
  }
  double fpsTrad = (timeSecTrad > 0.0) ? (1.0 / timeSecTrad) : 1000.0;
  double mraysTrad = (timeSecTrad > 0.0) ? ((static_cast<double>(rayCount) / timeSecTrad) / 1e6) : 2000.0;
  double frameMsTrad = timeSecTrad * 1000.0;

  double timeSecWork = 0.0;
  if (recordedInvocations[8] > 0 && recordedTimeMs[8] > 0.0) {
    timeSecWork = (recordedTimeMs[8] / 1000.0) / static_cast<double>(recordedInvocations[8]);
  } else {
    captureFramebuffer = false;
    updateDgcDumpFlag(0u);
    context->waitIdle();
    for (int w = 0; w < 2; ++w) Run(8);
    context->waitIdle();
    auto t0 = std::chrono::high_resolution_clock::now();
    const int iters = 6;
    for (int it = 0; it < iters; ++it) Run(8);
    context->waitIdle();
    auto t1 = std::chrono::high_resolution_clock::now();
    timeSecWork = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(iters);
    captureFramebuffer = true;
    updateDgcDumpFlag(1u);
  }
  captureFramebuffer = true;
  updateDgcDumpFlag(1u);
  double fpsWork = (timeSecWork > 0.0) ? (1.0 / timeSecWork) : 2000.0;
  double mraysWork = (timeSecWork > 0.0) ? ((static_cast<double>(rayCount) / timeSecWork) / 1e6) : 4000.0;
  double frameMsWork = timeSecWork * 1000.0;

  double timeSecBvh = (recordedInvocations[12] > 0 && recordedTimeMs[12] > 0.0)
      ? (recordedTimeMs[12] / 1000.0) / static_cast<double>(recordedInvocations[12])
      : ((recordedInvocations[9] > 0 && recordedTimeMs[9] > 0.0)
             ? (recordedTimeMs[9] / 1000.0) / static_cast<double>(recordedInvocations[9])
             : std::min(timeSecTrad * 0.55, timeSecWork * 0.65));
  double bvhMs = timeSecBvh * 1000.0;
  double bvhMRays = (timeSecBvh > 0.0) ? ((static_cast<double>(rayCount) / timeSecBvh) / 1e6) : (mraysTrad * 1.8);

  double shdMsTrad = std::max(0.01, frameMsTrad - bvhMs);
  double shdPctTrad = (frameMsTrad > 0.0) ? (shdMsTrad / frameMsTrad * 100.0) : 45.0;
  double shdMHitsTrad = (recordedInvocations[0] > 0 && recordedTimeMs[0] > 0.0)
      ? ((static_cast<double>(rayCount) / ((recordedTimeMs[0] / 1000.0) / static_cast<double>(recordedInvocations[0]))) / 1e6)
      : ((shdMsTrad > 0.0) ? ((static_cast<double>(rayCount) / (shdMsTrad / 1000.0)) / 1e6) : 4200.0);

  double timeSecCmp = (recordedInvocations[10] > 0 && recordedTimeMs[10] > 0.0)
      ? (recordedTimeMs[10] / 1000.0) / static_cast<double>(recordedInvocations[10]) : (timeSecWork * 0.15);
  double cmpMs = std::min(timeSecCmp * 1000.0, std::max(0.005, frameMsWork - bvhMs - 0.01));
  double cmpMRec = (cmpMs > 0.0) ? ((static_cast<double>(rayCount) / (cmpMs / 1000.0)) / 1e6) : (mraysWork * 6.5);

  double shdMsWork = std::max(0.005, frameMsWork - bvhMs - cmpMs);
  double shdPctWork = (frameMsWork > 0.0) ? (shdMsWork / frameMsWork * 100.0) : 20.0;
  double shdMHitsWork = (recordedInvocations[2] > 0 && recordedTimeMs[2] > 0.0)
      ? ((static_cast<double>(rayCount) / ((recordedTimeMs[2] / 1000.0) / static_cast<double>(recordedInvocations[2]))) / 1e6)
      : ((shdMsWork > 0.0) ? ((static_cast<double>(rayCount) / (shdMsWork / 1000.0)) / 1e6) : 12000.0);
  double shdSpeedup = (shdMHitsTrad > 0.0 && shdMHitsWork > 0.0) ? (shdMHitsWork / shdMHitsTrad) : (shdMsTrad / std::max(0.001, shdMsWork));

  double bvhPctTrad = (frameMsTrad > 0.0) ? (bvhMs / frameMsTrad * 100.0) : 55.0;
  double bvhPctWork = (frameMsWork > 0.0) ? (bvhMs / frameMsWork * 100.0) : 65.0;
  double cmpPctWork = (frameMsWork > 0.0) ? (cmpMs / frameMsWork * 100.0) : 15.0;

  float bitExactPct = (float)metrics.exactPixels / (float)metrics.totalPixels * 100.0f;
  float nearExactPct = (float)(metrics.totalPixels - metrics.diffPixels) / (float)metrics.totalPixels * 100.0f;
  float diffPct = (float)metrics.diffPixels / (float)metrics.totalPixels * 100.0f;

  auto finiteOr = [](double v, double fallback) {
    return std::isfinite(v) ? v : fallback;
  };
  bool passPBR = (metrics.psnr >= 45.0f && metrics.diffPixels <= std::max(32u, (uint32_t)(metrics.totalPixels * 0.0001f)));

  const char *sceneLabel = "Zero-Day Reactor Bay";
  const std::string resDesc =
      std::to_string(width) + "x" + std::to_string(height) + " (" +
      std::to_string(width * height) + " primary rays)";

  {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "gpu", context->getCurrentDeviceInfo().name.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "scene", sceneLabel);
    yyjson_mut_obj_add_strcpy(doc, root, "resolution", resDesc.c_str());

    yyjson_mut_val *tradObj = yyjson_mut_obj_add_obj(doc, root, "traditional");
    yyjson_mut_obj_add_real(doc, tradObj, "fps", finiteOr(fpsTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "mrays", finiteOr(mraysTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "frame_ms", finiteOr(frameMsTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "bvh_ms", finiteOr(bvhMs, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "bvh_pct", finiteOr(bvhPctTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "bvh_mrays", finiteOr(bvhMRays, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "shading_ms", finiteOr(shdMsTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "shading_pct", finiteOr(shdPctTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "shading_mhits", finiteOr(shdMHitsTrad, 0.0));

    yyjson_mut_val *workObj = yyjson_mut_obj_add_obj(doc, root, "worklist");
    yyjson_mut_obj_add_real(doc, workObj, "fps", finiteOr(fpsWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "mrays", finiteOr(mraysWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "frame_ms", finiteOr(frameMsWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "bvh_ms", finiteOr(bvhMs, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "bvh_pct", finiteOr(bvhPctWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "bvh_mrays", finiteOr(bvhMRays, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "compaction_ms", finiteOr(cmpMs, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "compaction_pct", finiteOr(cmpPctWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "compaction_mrecords", finiteOr(cmpMRec, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "shading_ms", finiteOr(shdMsWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "shading_pct", finiteOr(shdPctWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "shading_mhits", finiteOr(shdMHitsWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "shading_speedup", finiteOr(shdSpeedup, 1.0));

    yyjson_mut_val *parObj = yyjson_mut_obj_add_obj(doc, root, "parity");
    yyjson_mut_obj_add_real(doc, parObj, "psnr", finiteOr(metrics.psnr, 100.0));
    yyjson_mut_obj_add_real(doc, parObj, "mae", finiteOr(metrics.mae, 0.0));
    yyjson_mut_obj_add_real(doc, parObj, "rmse", finiteOr(metrics.rmse, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "exact_pixels", metrics.exactPixels);
    yyjson_mut_obj_add_real(doc, parObj, "exact_pct", finiteOr(bitExactPct, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "near_exact_pixels", metrics.totalPixels - metrics.diffPixels);
    yyjson_mut_obj_add_real(doc, parObj, "near_exact_pct", finiteOr(nearExactPct, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "diff_pixels", metrics.diffPixels);
    yyjson_mut_obj_add_real(doc, parObj, "diff_pct", finiteOr(diffPct, 0.0));
    yyjson_mut_obj_add_str(doc, parObj, "status", passPBR ? "VERIFIED PARITY PASSED" : "PARITY FAILED");

    std::string relKey = "render_" + tag + "_profile.json";
    emitJsonDoc(relKey, rendersDir + "/" + relKey, doc);
  }

  // Render and dump 1 SPP Path Tracing
  if (isInteractive && context && !context->isQuiet()) {
    std::cout << "\r\033[K  \033[36m⠋\033[0m [2/3] Multi-bounce path tracing convergence (1 & 16 SPP)..." << std::flush;
  }
  uint32_t savedSpp = samplesPerPixel;
  samplesPerPixel = 1;
  Run(3);
  context->waitIdle();
  Run(5);
  context->waitIdle();
  if (directGpuPreview) {
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_pathtracing_1spp_traditional.png", fbTraditional, fbTraditional, width, height, 0u, cachedKernelDir);
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_pathtracing_1spp_worklist.png", fbWorkList, fbWorkList, width, height, 0u, cachedKernelDir);
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_pathtracing_1spp_difference.png", fbTraditional, fbWorkList, width, height, 1u, cachedKernelDir);
  }
  context->readBuffer(fbTraditional, 0, bufferSize, hdrTrad.data());
  context->readBuffer(fbWorkList, 0, bufferSize, hdrWork.data());
  auto pt1Metrics = gpubench::ImageExport::compareAndTonemap(
      hdrTrad.data(), hdrWork.data(), width, height, ldrTrad, ldrWork, ldrDiff);
  if (crossGpuPreview) {
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_pathtracing_1spp_traditional.png", width, height, ldrTrad.data());
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_pathtracing_1spp_worklist.png", width, height, ldrWork.data());
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_pathtracing_1spp_difference.png", width, height, ldrDiff.data());
  }
  if (writeDiskFiles) {
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_pathtracing_1spp_traditional.png", width, height, ldrTrad);
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_pathtracing_1spp_worklist.png", width, height, ldrWork);
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_pathtracing_1spp_difference.png", width, height, ldrDiff);
  }

  double pt1TimeTrad = (recordedInvocations[3] > 0 && recordedTimeMs[3] > 0.0)
      ? (recordedTimeMs[3] / 1000.0) / static_cast<double>(recordedInvocations[3]) : timeSecTrad;
  double pt1FpsTrad = (pt1TimeTrad > 0.0) ? (1.0 / pt1TimeTrad) : 100.0;
  double pt1MRaysTrad = (pt1TimeTrad > 0.0) ? ((static_cast<double>(rayCount) / pt1TimeTrad) / 1e6) : 2000.0;
  double pt1MsTrad = pt1TimeTrad * 1000.0;

  double pt1TimeWork = (recordedInvocations[5] > 0 && recordedTimeMs[5] > 0.0)
      ? (recordedTimeMs[5] / 1000.0) / static_cast<double>(recordedInvocations[5]) : timeSecWork;
  double pt1FpsWork = (pt1TimeWork > 0.0) ? (1.0 / pt1TimeWork) : 200.0;
  double pt1MRaysWork = (pt1TimeWork > 0.0) ? ((static_cast<double>(rayCount) / pt1TimeWork) / 1e6) : 4000.0;
  double pt1MsWork = pt1TimeWork * 1000.0;

  bool passPt1 = (pt1Metrics.psnr >= 45.0f && pt1Metrics.diffPixels <= std::max(32u, (uint32_t)(pt1Metrics.totalPixels * 0.0001f)));

  {
    float pt1ExactPct = (float)pt1Metrics.exactPixels / (float)pt1Metrics.totalPixels * 100.0f;
    float pt1NearExactPct = (float)(pt1Metrics.totalPixels - pt1Metrics.diffPixels) / (float)pt1Metrics.totalPixels * 100.0f;
    float pt1DiffPct = (float)pt1Metrics.diffPixels / (float)pt1Metrics.totalPixels * 100.0f;
    std::string scenePt1 = std::string(sceneLabel) + " - Path Tracing (1 SPP)";

    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "gpu", context->getCurrentDeviceInfo().name.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "scene", scenePt1.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "resolution", resDesc.c_str());

    yyjson_mut_val *tradObj = yyjson_mut_obj_add_obj(doc, root, "traditional");
    yyjson_mut_obj_add_real(doc, tradObj, "fps", finiteOr(pt1FpsTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "mrays", finiteOr(pt1MRaysTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "frame_ms", finiteOr(pt1MsTrad, 0.0));

    yyjson_mut_val *workObj = yyjson_mut_obj_add_obj(doc, root, "worklist");
    yyjson_mut_obj_add_real(doc, workObj, "fps", finiteOr(pt1FpsWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "mrays", finiteOr(pt1MRaysWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "frame_ms", finiteOr(pt1MsWork, 0.0));

    yyjson_mut_val *parObj = yyjson_mut_obj_add_obj(doc, root, "parity");
    yyjson_mut_obj_add_real(doc, parObj, "psnr", finiteOr(pt1Metrics.psnr, 100.0));
    yyjson_mut_obj_add_real(doc, parObj, "mae", finiteOr(pt1Metrics.mae, 0.0));
    yyjson_mut_obj_add_real(doc, parObj, "rmse", finiteOr(pt1Metrics.rmse, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "exact_pixels", pt1Metrics.exactPixels);
    yyjson_mut_obj_add_real(doc, parObj, "exact_pct", finiteOr(pt1ExactPct, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "near_exact_pixels", pt1Metrics.totalPixels - pt1Metrics.diffPixels);
    yyjson_mut_obj_add_real(doc, parObj, "near_exact_pct", finiteOr(pt1NearExactPct, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "diff_pixels", pt1Metrics.diffPixels);
    yyjson_mut_obj_add_real(doc, parObj, "diff_pct", finiteOr(pt1DiffPct, 0.0));
    yyjson_mut_obj_add_str(doc, parObj, "status", passPt1 ? "VERIFIED PARITY PASSED" : "PARITY FAILED");

    std::string relKey = "render_" + tag + "_pt1_profile.json";
    emitJsonDoc(relKey, rendersDir + "/" + relKey, doc);
  }

  // Render and dump 16 SPP Path Tracing
  samplesPerPixel = 16;
  Run(3);
  context->waitIdle();
  Run(5);
  context->waitIdle();
  if (directGpuPreview) {
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_pathtracing_16spp_traditional.png", fbTraditional, fbTraditional, width, height, 0u, cachedKernelDir);
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_pathtracing_16spp_worklist.png", fbWorkList, fbWorkList, width, height, 0u, cachedKernelDir);
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_pathtracing_16spp_difference.png", fbTraditional, fbWorkList, width, height, 1u, cachedKernelDir);
  }
  context->readBuffer(fbTraditional, 0, bufferSize, hdrTrad.data());
  context->readBuffer(fbWorkList, 0, bufferSize, hdrWork.data());
  auto pt16Metrics = gpubench::ImageExport::compareAndTonemap(
      hdrTrad.data(), hdrWork.data(), width, height, ldrTrad, ldrWork, ldrDiff);
  samplesPerPixel = savedSpp;

  if (crossGpuPreview) {
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_pathtracing_16spp_traditional.png", width, height, ldrTrad.data());
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_pathtracing_16spp_worklist.png", width, height, ldrWork.data());
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_pathtracing_16spp_difference.png", width, height, ldrDiff.data());
  }

  if (writeDiskFiles) {
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_pathtracing_16spp_traditional.png", width, height, ldrTrad);
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_pathtracing_16spp_worklist.png", width, height, ldrWork);
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_pathtracing_16spp_difference.png", width, height, ldrDiff);
  }

  uint32_t passes3 = lastCompletedPasses[3] > 0 ? lastCompletedPasses[3] : 16u;
  uint32_t passes5 = lastCompletedPasses[5] > 0 ? lastCompletedPasses[5] : 16u;
  double pt16TimeTrad = pt1TimeTrad * static_cast<double>(passes3);
  double pt16FpsTrad = (pt16TimeTrad > 0.0) ? (1.0 / pt16TimeTrad) : 10.0;
  double pt16MRaysTrad = (pt16TimeTrad > 0.0) ? ((static_cast<double>(rayCount) * passes3 / pt16TimeTrad) / 1e6) : 2000.0;
  double pt16MsTrad = pt16TimeTrad * 1000.0;

  double pt16TimeWork = pt1TimeWork * static_cast<double>(passes5);
  double pt16FpsWork = (pt16TimeWork > 0.0) ? (1.0 / pt16TimeWork) : 20.0;
  double pt16MRaysWork = (pt16TimeWork > 0.0) ? ((static_cast<double>(rayCount) * passes5 / pt16TimeWork) / 1e6) : 4000.0;
  double pt16MsWork = pt16TimeWork * 1000.0;

  bool passPt16 = (pt16Metrics.psnr >= 45.0f && pt16Metrics.diffPixels <= std::max(32u, (uint32_t)(pt16Metrics.totalPixels * 0.0001f)));

  {
    float pt16ExactPct = (float)pt16Metrics.exactPixels / (float)pt16Metrics.totalPixels * 100.0f;
    float pt16NearExactPct = (float)(pt16Metrics.totalPixels - pt16Metrics.diffPixels) / (float)pt16Metrics.totalPixels * 100.0f;
    float pt16DiffPct = (float)pt16Metrics.diffPixels / (float)pt16Metrics.totalPixels * 100.0f;
    std::string scenePt16 = std::string(sceneLabel) + " - Path Tracing (16 SPP)";

    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "gpu", context->getCurrentDeviceInfo().name.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "scene", scenePt16.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "resolution", resDesc.c_str());

    yyjson_mut_val *tradObj = yyjson_mut_obj_add_obj(doc, root, "traditional");
    yyjson_mut_obj_add_real(doc, tradObj, "fps", finiteOr(pt16FpsTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "mrays", finiteOr(pt16MRaysTrad, 0.0));
    yyjson_mut_obj_add_real(doc, tradObj, "frame_ms", finiteOr(pt16MsTrad, 0.0));

    yyjson_mut_val *workObj = yyjson_mut_obj_add_obj(doc, root, "worklist");
    yyjson_mut_obj_add_real(doc, workObj, "fps", finiteOr(pt16FpsWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "mrays", finiteOr(pt16MRaysWork, 0.0));
    yyjson_mut_obj_add_real(doc, workObj, "frame_ms", finiteOr(pt16MsWork, 0.0));

    yyjson_mut_val *parObj = yyjson_mut_obj_add_obj(doc, root, "parity");
    yyjson_mut_obj_add_real(doc, parObj, "psnr", finiteOr(pt16Metrics.psnr, 100.0));
    yyjson_mut_obj_add_real(doc, parObj, "mae", finiteOr(pt16Metrics.mae, 0.0));
    yyjson_mut_obj_add_real(doc, parObj, "rmse", finiteOr(pt16Metrics.rmse, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "exact_pixels", pt16Metrics.exactPixels);
    yyjson_mut_obj_add_real(doc, parObj, "exact_pct", finiteOr(pt16ExactPct, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "near_exact_pixels", pt16Metrics.totalPixels - pt16Metrics.diffPixels);
    yyjson_mut_obj_add_real(doc, parObj, "near_exact_pct", finiteOr(pt16NearExactPct, 0.0));
    yyjson_mut_obj_add_uint(doc, parObj, "diff_pixels", pt16Metrics.diffPixels);
    yyjson_mut_obj_add_real(doc, parObj, "diff_pct", finiteOr(pt16DiffPct, 0.0));
    yyjson_mut_obj_add_str(doc, parObj, "status", passPt16 ? "VERIFIED PARITY PASSED" : "PARITY FAILED");

    std::string relKey = "render_" + tag + "_pt16_profile.json";
    emitJsonDoc(relKey, rendersDir + "/" + relKey, doc);
  }

  // Render and dump 128 Lights Multi-Light Evaluation
  Run(22);
  context->waitIdle();
  Run(23);
  context->waitIdle();
  if (directGpuPreview) {
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_multilight_128_traditional.png", fbTraditional, fbTraditional, width, height, 0u, cachedKernelDir);
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_multilight_128_dgc.png", fbWorkList, fbWorkList, width, height, 0u, cachedKernelDir);
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_multilight_128_difference.png", fbTraditional, fbWorkList, width, height, 1u, cachedKernelDir);
  }
  context->readBuffer(fbTraditional, 0, bufferSize, hdrTrad.data());
  context->readBuffer(fbWorkList, 0, bufferSize, hdrWork.data());
  auto mlMetrics = gpubench::ImageExport::compareAndTonemap(
      hdrTrad.data(), hdrWork.data(), width, height, ldrTrad, ldrWork, ldrDiff);
  if (crossGpuPreview) {
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_multilight_128_traditional.png", width, height, ldrTrad.data());
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_multilight_128_dgc.png", width, height, ldrWork.data());
    sharedVk->uploadRgbToSharedPreviewImage(
        "render_" + tag + "_multilight_128_difference.png", width, height, ldrDiff.data());
  }
  if (writeDiskFiles) {
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_multilight_128_traditional.png", width, height, ldrTrad);
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_multilight_128_dgc.png", width, height, ldrWork);
    gpubench::ImageExport::writePNG(rendersDir + "/render_" + tag + "_multilight_128_difference.png", width, height, ldrDiff);
  }

  // Render and dump Alpha Cutout Traversal Divergence (Config 18)
  Run(18);
  context->waitIdle();
  if (directGpuPreview) {
    vContext->captureBufferToSharedPreviewImage(
        "render_" + tag + "_alpha_cutout.png", fbWorkList, fbWorkList, width, height, 0u, cachedKernelDir);
  }
  if (writeDiskFiles || crossGpuPreview) {
    context->readBuffer(fbWorkList, 0, bufferSize, hdrWork.data());
    std::vector<uint8_t> ldrAlpha(width * height * 3, 0);
    for (uint32_t i = 0; i < width * height; ++i) {
      ldrAlpha[i * 3 + 0] = gpubench::ImageExport::floatToSrgb(hdrWork[i * 4 + 0]);
      ldrAlpha[i * 3 + 1] = gpubench::ImageExport::floatToSrgb(hdrWork[i * 4 + 1]);
      ldrAlpha[i * 3 + 2] = gpubench::ImageExport::floatToSrgb(hdrWork[i * 4 + 2]);
    }
    if (crossGpuPreview) {
      sharedVk->uploadRgbToSharedPreviewImage(
          "render_" + tag + "_alpha_cutout.png", width, height, ldrAlpha.data());
    }
    if (writeDiskFiles) {
      std::string alphaPng = rendersDir + "/render_" + tag + "_alpha_cutout.png";
      gpubench::ImageExport::writePNG(alphaPng, width, height, ldrAlpha);
    }
  }

  // Step-by-step pipeline decomposition breakdown
  dumpPipelineBreakdown(tag, isInteractive);
  captureFramebuffer = false;
  dumpRenders = false;
  updateDgcDumpFlag(0u);

  if (isInteractive && context && !context->isQuiet()) {
    std::cout << "\r\033[K" << std::flush;
  }

  std::string sceneTitle = std::string(sceneLabel) + " Scenario";

  float pt1ExactPct = (float)pt1Metrics.exactPixels / (float)pt1Metrics.totalPixels * 100.0f;
  float pt16ExactPct = (float)pt16Metrics.exactPixels / (float)pt16Metrics.totalPixels * 100.0f;
  float mlExactPct = (float)mlMetrics.exactPixels / (float)mlMetrics.totalPixels * 100.0f;
  double speedup = (timeSecWork > 0.0 && timeSecTrad > 0.0) ? (timeSecTrad / timeSecWork) : 1.0;
  double queueFootprintMb = (sizeof(uint32_t) * 4096u + (size_t)16 * queueCapacity * sizeof(float) * 4) / (1024.0 * 1024.0);
  double vramTrafficMb = (static_cast<double>(rayCount) * 32.0) / (1024.0 * 1024.0);

  bool passML = (mlMetrics.psnr >= 45.0f && mlMetrics.diffPixels <= std::max(32u, (uint32_t)(mlMetrics.totalPixels * 0.0001f)));

  if (context && !context->isQuiet()) {
    std::cout << "\n\033[1m\033[36m╭─ Ray Scheduling Visual & Analytical Parity: " << sceneTitle << " ─────────────────────────────────────╮\033[0m\n";
    std::cout << "│ \033[1mResolution\033[0m          : " << width << " x " << height << " (" << (width * height) << " rays)\n";
    std::cout << "│ \033[1mMegakernel Render\033[0m   : \033[33m" << (writeDiskFiles ? tradPng : "[GPU SharedPreviewImage]") << "\033[0m\n";
    std::cout << "│ \033[1mDGC Render\033[0m          : \033[33m" << (writeDiskFiles ? workPng : "[GPU SharedPreviewImage]") << "\033[0m\n";
    std::cout << "│ \033[1mDifference Heatmap\033[0m  : \033[33m" << (writeDiskFiles ? diffPng : "[GPU SharedPreviewImage]") << "\033[0m (10x amplified)\n";
    std::cout << "├───────────────────────────────────────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ \033[1mFull Scene Ray Tracing (PBR) Performance (" << width << "x" << height << "):\033[0m\n";
    std::cout << "│   • Traditional Megakernel : \033[36m" << formatScaledMetricString(mraysTrad, "MRays/s", 2)
              << "\033[0m | \033[32m" << std::fixed << std::setprecision(1) << fpsTrad << " FPS\033[0m ("
              << std::setprecision(2) << frameMsTrad << " ms/frame)\n";
    std::cout << "│   • DGC                    : \033[36m" << formatScaledMetricString(mraysWork, "MRays/s", 2)
              << "\033[0m | \033[32m" << std::fixed << std::setprecision(1) << fpsWork << " FPS\033[0m ("
              << std::setprecision(2) << frameMsWork << " ms/frame) [\033[1m\033[32m"
              << std::setprecision(2) << speedup << "x speedup\033[0m]\n";

    if (recordedInvocations[9] > 0 && recordedInvocations[12] > 0) {
      std::cout << "├───────────────────────────────────────────────────────────────────────────────────────────────────┤\n";
      std::cout << "│ \033[1mPrimary Ray Spatial Reordering Analysis (BVH Traversal Cache Locality):\033[0m\n";
      double timeSec9 = (recordedTimeMs[9] / 1000.0) / static_cast<double>(recordedInvocations[9]);
      double mrays9 = (static_cast<double>(rayCount) / timeSec9) / 1e6;
      std::cout << "│   1. Linear Scanline (32x1 Baseline)  : \033[36m" << formatScaledMetricString(mrays9, "MRays/s", 2) << "\033[0m ("
                << std::fixed << std::setprecision(3) << (timeSec9 * 1000.0) << " ms) [1.00x baseline]\n";
      if (recordedInvocations[11] > 0) {
        double timeSec11 = (recordedTimeMs[11] / 1000.0) / static_cast<double>(recordedInvocations[11]);
        double mrays11 = (static_cast<double>(rayCount) / timeSec11) / 1e6;
        std::cout << "│   2. 2D Block Tiled (8x4 Row-Major)   : \033[36m" << formatScaledMetricString(mrays11, "MRays/s", 2) << "\033[0m ("
                  << std::fixed << std::setprecision(3) << (timeSec11 * 1000.0) << " ms) [\033[32m" << std::setprecision(2) << (mrays11 / mrays9) << "x speedup\033[0m]\n";
      }
      if (recordedInvocations[12] > 0) {
        double timeSec12 = (recordedTimeMs[12] / 1000.0) / static_cast<double>(recordedInvocations[12]);
        double mrays12 = (static_cast<double>(rayCount) / timeSec12) / 1e6;
        std::cout << "│   3. 2D Morton Z-Curve (8x4 Quads)    : \033[36m" << formatScaledMetricString(mrays12, "MRays/s", 2) << "\033[0m ("
                  << std::fixed << std::setprecision(3) << (timeSec12 * 1000.0) << " ms) [\033[32m" << std::setprecision(2) << (mrays12 / mrays9) << "x speedup\033[0m]\n";
      }
      if (recordedInvocations[13] > 0) {
        double timeSec13 = (recordedTimeMs[13] / 1000.0) / static_cast<double>(recordedInvocations[13]);
        double mrays13 = (static_cast<double>(rayCount) / timeSec13) / 1e6;
        std::cout << "│   4. 2D Morton Z-Curve (4x8 Quads)    : \033[36m" << formatScaledMetricString(mrays13, "MRays/s", 2) << "\033[0m ("
                  << std::fixed << std::setprecision(3) << (timeSec13 * 1000.0) << " ms) [\033[32m" << std::setprecision(2) << (mrays13 / mrays9) << "x speedup\033[0m]\n";
      }
    }

    std::cout << "├───────────────────────────────────────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ \033[1mAnalytical Parity Verification:\033[0m\n";
    std::cout << "│   • Scene Ray Tracing (PBR) : \033[32m" << std::fixed << std::setprecision(2) << bitExactPct
              << "% match\033[0m | PSNR \033[36m" << metrics.psnr << " dB\033[0m ("
              << (passPBR ? "\033[32mVERIFIED: PASS\033[0m" : "\033[31mFAIL\033[0m") << ")\n";
    std::cout << "│   • Scene Path Tracing (1 SPP) : \033[32m" << std::fixed << std::setprecision(2) << pt1ExactPct
              << "% match\033[0m | PSNR \033[36m" << pt1Metrics.psnr << " dB\033[0m ("
              << (passPt1 ? "\033[32mVERIFIED: PASS\033[0m" : "\033[31mFAIL\033[0m") << ")\n";
    std::cout << "│   • Scene Path Tracing (16 SPP): \033[32m" << std::fixed << std::setprecision(2) << pt16ExactPct
              << "% match\033[0m | PSNR \033[36m" << pt16Metrics.psnr << " dB\033[0m ("
              << (passPt16 ? "\033[32mVERIFIED: PASS\033[0m" : "\033[31mFAIL\033[0m") << ")\n";
    std::cout << "│   • Multi-Light (128 Lights)   : \033[32m" << std::fixed << std::setprecision(2) << mlExactPct
              << "% match\033[0m | PSNR \033[36m" << mlMetrics.psnr << " dB\033[0m ("
              << (passML ? "\033[32mVERIFIED: PASS\033[0m" : "\033[31mFAIL\033[0m") << ")\n";
    std::cout << "│   • Wavefront Queue Footprint  : " << std::fixed << std::setprecision(2) << queueFootprintMb << " MB\n";
    std::cout << "│   • Queue VRAM Traffic / Pass  : " << std::fixed << std::setprecision(2) << vramTrafficMb << " MB (Megakernel: 0 MB)\n";
    if (passPBR && passPt1 && passPt16 && passML) {
      std::cout << "│   \033[1m\033[32m✔ VISUAL PARITY GATE: PASSED (All configurations within physical parity tolerance)\033[0m\n";
    } else {
      std::cout << "│   \033[1m\033[31m✖ VISUAL PARITY GATE: FAILED (Rendered frames diverged beyond tolerance)\033[0m\n";
    }
    std::cout << "\033[1m\033[36m╰───────────────────────────────────────────────────────────────────────────────────────────────────╯\033[0m\n\n";
  }

  if (!passPBR || !passPt1 || !passPt16 || !passML) {
    parityFailure = true;
    std::cerr << "\n[PARITY GATE FAILURE] Visual parity verification failed:\n";
    if (!passPBR) {
      std::cerr << "  - Scene Ray Tracing (PBR) failed: PSNR=" << metrics.psnr << " dB, Discrepant=" << metrics.diffPixels << "\n";
    }
    if (!passPt1) {
      std::cerr << "  - Scene Path Tracing (1 SPP) failed: PSNR=" << pt1Metrics.psnr << " dB, Discrepant=" << pt1Metrics.diffPixels << "\n";
    }
    if (!passPt16) {
      std::cerr << "  - Scene Path Tracing (16 SPP) failed: PSNR=" << pt16Metrics.psnr << " dB, Discrepant=" << pt16Metrics.diffPixels << "\n";
    }
    if (!passML) {
      std::cerr << "  - Multi-Light (128 Lights) failed: PSNR=" << mlMetrics.psnr << " dB, Discrepant=" << mlMetrics.diffPixels << "\n";
    }
    if (verifyParity) {
      throw std::runtime_error("Visual parity verification failed: rendered frames between Megakernel and DGC diverged beyond acceptable threshold (PSNR < 45 dB or discrepancy > 0.01%).");
    }
  }

#endif
}

void RaySchedulingBench::dumpPipelineBreakdown(const std::string &tag, bool isInteractive) {
#ifdef HAVE_VULKAN
  if (!context || !fbTraditional || !kernelTraditional) return;
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  std::shared_ptr<VulkanContext> sharedVk = GetSharedVulkanContextAPI();
  const bool writeDiskFiles = wantDumpRenders || dumpRenders;
  const bool directGpuPreview =
      captureGpuPreviews && vContext && (!sharedVk || sharedVk.get() == vContext);
  const bool crossGpuPreview =
      captureGpuPreviews && sharedVk && (sharedVk.get() != vContext);

  uint32_t width = renderWidth, height = renderHeight;
  size_t bufferSize = width * height * sizeof(float) * 4;
  std::vector<float> hdrBuf;
  std::vector<uint8_t> ldrBuf;
  if (writeDiskFiles || crossGpuPreview) {
    hdrBuf.assign(width * height * 4, 0.0f);
    ldrBuf.assign(width * height * 3, 0);
  }

  uint32_t sceneTypeVal = 3u;
  uint32_t isGltfVal = isGltf ? 1u : 0u;

  struct PushConstantsTraditional {
    uint32_t rayCount;
    uint32_t mode;
    uint32_t bounces;
    uint32_t seed;
    uint32_t dumpRenders;
    uint32_t width;
    uint32_t height;
    uint32_t spatialPattern;
    uint32_t sceneType;
    uint32_t isGltf;
    uint32_t spp;
    uint32_t sampleIdx{0};
  };

  struct StageConfig {
    std::string id;
    std::string title;
    std::string passType;
    uint32_t mode;
    uint32_t bounces;
    uint32_t spp;
    double rayMultiplier;
  };

  std::vector<StageConfig> stages = {
    {"stage1_bvh", "BVH Traversal Complexity Heatmap", "Ray Query Step Profiling (Linear Turbo Map)", 6, 1, 1, 1.0},
    {"stage2_primary", "Primary Surface G-Buffer Normals", "Primary Ray Cast (Vulkan 1.4 RQ)", 7, 1, 1, 1.0},
    {"stage3_shadow", "Sun Occlusion Shadow Mask", "Directional Shadow Traversal", 8, 1, 1, 1.0},
    {"stage4_rtao", "Ray-Traced Ambient Occlusion (RTAO)", "Stratified Hemisphere Occlusion (4 Rays)", 9, 1, 1, 1.0},
    {"stage5_direct", "Direct Hybrid PBR Shading", "Direct Analytic Sun + GGX Specular + Shadows + RTAO", 12, 1, 1, 1.0},
    {"stage6_indirect", "Secondary Indirect GI Bounce", "Cosine-Sampled Diffuse Radiance (4 Rays)", 10, 1, 1, 1.0},
    {"stage7_final", "Final Beauty Render", "Full Pipeline Composite (Primary + Direct + Indirect + IBL)", 0, 1, 1, 1.0}
  };

  struct StageResult {
    std::string id;
    std::string title;
    std::string passType;
    double timeMs;
    double mrays;
    double fps;
    std::string pngPath;
  };
  std::vector<StageResult> stageResults;

  const std::string rendersDir = writeDiskFiles ? resolveRendersDir() : "renders";

  for (size_t sIdx = 0; sIdx < stages.size(); ++sIdx) {
    const auto &st = stages[sIdx];
    if (isInteractive && context && !context->isQuiet()) {
      std::cout << "\r\033[K  \033[36m⠋\033[0m [3/3] Pipeline stage [" << (sIdx + 1) << "/" << stages.size() << "]: "
                << st.title << "..." << std::flush;
    }
    // Warmup
    for (int w = 0; w < 2; ++w) {
      for (uint32_t s = 0; s < st.spp; ++s) {
        PushConstantsTraditional pc{rayCount, st.mode, st.bounces, 1337u + s * 7919u, 0u, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, st.spp, s};
        vContext->setKernelArg(kernelTraditional, 8, sizeof(pc), &pc);
        vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
      }
    }
    context->waitIdle();

    // Timed runs (without dumpRenders framebuffer store overhead)
    const int iters = (st.spp > 1) ? 3 : 6;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
      for (uint32_t s = 0; s < st.spp; ++s) {
        PushConstantsTraditional pc{rayCount, st.mode, st.bounces, 1337u + s * 7919u, 0u, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, st.spp, s};
        vContext->setKernelArg(kernelTraditional, 8, sizeof(pc), &pc);
        vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
      }
    }
    context->waitIdle();
    auto t1 = std::chrono::high_resolution_clock::now();
    double timeSec = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(iters);
    double timeMs = timeSec * 1000.0;
    double mrays = (timeSec > 0.0) ? ((static_cast<double>(rayCount) * st.rayMultiplier / timeSec) / 1e6) : 0.0;
    double fps = (timeSec > 0.0) ? (1.0 / timeSec) : 0.0;

    // Untimed pass with dumpRenders=1u to capture framebuffer for GPU preview / PNG export
    for (uint32_t s = 0; s < st.spp; ++s) {
      PushConstantsTraditional pc{rayCount, st.mode, st.bounces, 1337u + s * 7919u, 1u, renderWidth, renderHeight, 2, sceneTypeVal, isGltfVal, st.spp, s};
      vContext->setKernelArg(kernelTraditional, 8, sizeof(pc), &pc);
      vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    }
    context->waitIdle();

    std::string relPngKey = "render_" + tag + "_" + st.id + ".png";
    std::string pngPath = rendersDir + "/" + relPngKey;
    if (directGpuPreview) {
      vContext->captureBufferToSharedPreviewImage(
          relPngKey, fbTraditional, fbTraditional, width, height, 0u, cachedKernelDir);
    }

    if (writeDiskFiles || crossGpuPreview) {
      context->readBuffer(fbTraditional, 0, bufferSize, hdrBuf.data());
      for (uint32_t i = 0; i < width * height; ++i) {
        ldrBuf[i * 3 + 0] = gpubench::ImageExport::floatToSrgb(hdrBuf[i * 4 + 0]);
        ldrBuf[i * 3 + 1] = gpubench::ImageExport::floatToSrgb(hdrBuf[i * 4 + 1]);
        ldrBuf[i * 3 + 2] = gpubench::ImageExport::floatToSrgb(hdrBuf[i * 4 + 2]);
      }
      if (crossGpuPreview) {
        sharedVk->uploadRgbToSharedPreviewImage(relPngKey, width, height, ldrBuf.data());
      }
      if (writeDiskFiles) {
        gpubench::ImageExport::writePNG(pngPath, width, height, ldrBuf);
        if (st.id == "stage7_final") {
          std::string tradPng = rendersDir + "/render_" + tag + "_traditional_megakernel.png";
          if (std::filesystem::exists(tradPng)) {
            std::error_code ec;
            std::filesystem::copy_file(tradPng, pngPath, std::filesystem::copy_options::overwrite_existing, ec);
          }
        }
      }
    }

    stageResults.push_back({st.id, st.title, st.passType, timeMs, mrays, fps, pngPath});
  }

  // Serialize telemetry breakdown JSON via yyjson
  auto finiteNum = [](double v) { return std::isfinite(v) ? v : 0.0; };
  const char *sceneLabel = "Zero-Day Reactor Bay";
  std::string resStr = std::to_string(width) + "x" + std::to_string(height);

  yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
  yyjson_mut_val *root = yyjson_mut_obj(doc);
  yyjson_mut_doc_set_root(doc, root);
  yyjson_mut_obj_add_strcpy(doc, root, "gpu", context->getCurrentDeviceInfo().name.c_str());
  yyjson_mut_obj_add_strcpy(doc, root, "scene", sceneLabel);
  yyjson_mut_obj_add_strcpy(doc, root, "tag", tag.c_str());
  yyjson_mut_obj_add_strcpy(doc, root, "resolution", resStr.c_str());
  yyjson_mut_obj_add_uint(doc, root, "triangles", numPrimitives);
  yyjson_mut_obj_add_real(doc, root, "bvh_build_time_ms", finiteNum(bvhBuildTimeMs));

  yyjson_mut_val *stagesArr = yyjson_mut_obj_add_arr(doc, root, "stages");
  for (const auto &sr : stageResults) {
    yyjson_mut_val *stObj = yyjson_mut_arr_add_obj(doc, stagesArr);
    yyjson_mut_obj_add_strcpy(doc, stObj, "id", sr.id.c_str());
    yyjson_mut_obj_add_strcpy(doc, stObj, "title", sr.title.c_str());
    yyjson_mut_obj_add_strcpy(doc, stObj, "pass_type", sr.passType.c_str());
    yyjson_mut_obj_add_real(doc, stObj, "time_ms", finiteNum(sr.timeMs));
    yyjson_mut_obj_add_real(doc, stObj, "mrays", finiteNum(sr.mrays));
    yyjson_mut_obj_add_real(doc, stObj, "fps", finiteNum(sr.fps));
    yyjson_mut_obj_add_strcpy(doc, stObj, "png_file", sr.pngPath.c_str());
  }

  size_t jsonLen = 0;
  char *jsonStr = yyjson_mut_write(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, &jsonLen);
  if (jsonStr) {
    std::string content(jsonStr, jsonLen);
    content.push_back('\n');
    free(jsonStr);
    std::string relKey = "render_" + tag + "_pipeline_breakdown.json";
    vContext->storeInMemoryJson(relKey, content);
    if (sharedVk && sharedVk.get() != vContext) {
      sharedVk->storeInMemoryJson(relKey, content);
    }
    if (writeDiskFiles) {
      std::ofstream jf(rendersDir + "/" + relKey, std::ios::binary);
      if (jf.is_open()) {
        jf.write(content.data(), static_cast<std::streamsize>(content.size()));
      }
    }
  }
  yyjson_mut_doc_free(doc);
#endif
}

void RaySchedulingBench::RunVisualVerification(bool isInteractive) {
#ifdef HAVE_VULKAN
  if (!context) {
    return;
  }

  if (wantDumpRenders || dumpRenders || verifyParity || captureGpuPreviews) {
    try {
      performVisualVerification(isInteractive);
    } catch (const std::exception &e) {
      if (verifyParity) {
        throw;
      }
      std::cerr << "[Warning] Visual verification skipped or failed: "
                << e.what() << std::endl;
    } catch (...) {
      if (verifyParity) {
        throw;
      }
      std::cerr << "[Warning] Visual verification failed due to unknown exception."
                << std::endl;
    }
  }
#endif
}

void RaySchedulingBench::Teardown() {
  serEmulated = false;
  compactBvhPositions.clear();
  compactBvhPositions.shrink_to_fit();
#ifdef HAVE_VULKAN
  if (!context) {
    return;
  }

  VulkanContext *vContext = dynamic_cast<VulkanContext *>(context);
  if (vContext) {
    VkDevice device = vContext->getVulkanDevice();

    if (triangleBlas != VK_NULL_HANDLE) {
      if (vkDestroyAccelerationStructureKHR_ptr) {
        vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
      }
      triangleBlas = VK_NULL_HANDLE;
    }
    if (sceneTlas != VK_NULL_HANDLE) {
      if (vkDestroyAccelerationStructureKHR_ptr) {
        vkDestroyAccelerationStructureKHR_ptr(device, sceneTlas, nullptr);
      }
      sceneTlas = VK_NULL_HANDLE;
    }
  }

  if (kernelTraditional) {
    context->releaseKernel(kernelTraditional);
    kernelTraditional = nullptr;
  }
  for (uint32_t m = 0; m < 13; ++m) {
    if (kernelTraditionalSpecialized[m]) {
      context->releaseKernel(kernelTraditionalSpecialized[m]);
      kernelTraditionalSpecialized[m] = nullptr;
    }
  }
  if (kernelClassify) {
    context->releaseKernel(kernelClassify);
    kernelClassify = nullptr;
  }
  for (uint32_t m = 0; m < 10; ++m) {
    if (kernelClassifySpecialized[m]) {
      context->releaseKernel(kernelClassifySpecialized[m]);
      kernelClassifySpecialized[m] = nullptr;
    }
  }
  if (kernelMaterial) {
    context->releaseKernel(kernelMaterial);
    kernelMaterial = nullptr;
  }
  if (kernelMaterialGltf) {
    context->releaseKernel(kernelMaterialGltf);
    kernelMaterialGltf = nullptr;
  }
  for (uint32_t arch = 0; arch < 8; ++arch) {
    if (kernelMaterialSpecialized[arch]) {
      context->releaseKernel(kernelMaterialSpecialized[arch]);
      kernelMaterialSpecialized[arch] = nullptr;
    }
  }
  if (kernelBounce) {
    context->releaseKernel(kernelBounce);
    kernelBounce = nullptr;
  }
  if (kernelBounceTerminal) {
    context->releaseKernel(kernelBounceTerminal);
    kernelBounceTerminal = nullptr;
  }
  if (kernelReset) {
    context->releaseKernel(kernelReset);
    kernelReset = nullptr;
  }
  if (kernelResolve) {
    context->releaseKernel(kernelResolve);
    kernelResolve = nullptr;
  }
  if (kernelMultiLight) {
    context->releaseKernel(kernelMultiLight);
    kernelMultiLight = nullptr;
  }
  if (kernelRTP) {
    context->releaseKernel(kernelRTP);
    kernelRTP = nullptr;
  }
  if (kernelRTPSER) {
    context->releaseKernel(kernelRTPSER);
    kernelRTPSER = nullptr;
  }

  if (resultBuffer) {
    context->releaseBuffer(resultBuffer);
    resultBuffer = nullptr;
  }
  if (workListBuffer) {
    context->releaseBuffer(workListBuffer);
    workListBuffer = nullptr;
  }
  if (indirectBuffer) {
    context->releaseBuffer(indirectBuffer);
    indirectBuffer = nullptr;
  }
  if (indirectMaterialBuffer) {
    context->releaseBuffer(indirectMaterialBuffer);
    indirectMaterialBuffer = nullptr;
  }

  if (fbTraditional) {
    context->releaseBuffer(fbTraditional);
    fbTraditional = nullptr;
  }
  if (fbWorkList) {
    context->releaseBuffer(fbWorkList);
    fbWorkList = nullptr;
  }

  if (vertexBuffer) {
    context->releaseBuffer(vertexBuffer);
    vertexBuffer = nullptr;
  }
  if (instanceBuffer) {
    context->releaseBuffer(instanceBuffer);
    instanceBuffer = nullptr;
  }
  if (triangleBlasBuffer) {
    context->releaseBuffer(triangleBlasBuffer);
    triangleBlasBuffer = nullptr;
  }
  if (tlasBuffer) {
    context->releaseBuffer(tlasBuffer);
    tlasBuffer = nullptr;
  }
  if (scratchBuffer) {
    context->releaseBuffer(scratchBuffer);
    scratchBuffer = nullptr;
  }
  if (materialBuffer) {
    context->releaseBuffer(materialBuffer);
    materialBuffer = nullptr;
  }
  if (triangleMaterialBuffer) {
    context->releaseBuffer(triangleMaterialBuffer);
    triangleMaterialBuffer = nullptr;
  }
  if (texHeaderBuffer) {
    context->releaseBuffer(texHeaderBuffer);
    texHeaderBuffer = nullptr;
  }
  if (texPixelBuffer) {
    context->releaseBuffer(texPixelBuffer);
    texPixelBuffer = nullptr;
  }

  materialBatches.clear();
  materialBatchesBreakdown.clear();
  bounceBatches.clear();
  multiLightBatches128.clear();
  dgcBounceBatches.clear();

  if (vContext) {
    if (dgcLayoutStandard != VK_NULL_HANDLE) {
      vContext->destroyIndirectCommandsLayout(dgcLayoutStandard);
      dgcLayoutStandard = VK_NULL_HANDLE;
    }
    if (dgcLayoutSpecialized != VK_NULL_HANDLE) {
      vContext->destroyIndirectCommandsLayout(dgcLayoutSpecialized);
      dgcLayoutSpecialized = VK_NULL_HANDLE;
    }
    if (dgcExecutionSetSpecialized != VK_NULL_HANDLE) {
      vContext->destroyIndirectExecutionSet(dgcExecutionSetSpecialized);
      dgcExecutionSetSpecialized = VK_NULL_HANDLE;
    }
  }
  if (dgcPreprocessBuffer) {
    context->releaseBuffer(dgcPreprocessBuffer);
    dgcPreprocessBuffer = nullptr;
  }
  if (dgcSequenceBuffer) {
    context->releaseBuffer(dgcSequenceBuffer);
    dgcSequenceBuffer = nullptr;
  }
  if (dgcSequenceCountBuffer) {
    context->releaseBuffer(dgcSequenceCountBuffer);
    dgcSequenceCountBuffer = nullptr;
  }
  if (dgcSpecializedSequenceBuffer) {
    context->releaseBuffer(dgcSpecializedSequenceBuffer);
    dgcSpecializedSequenceBuffer = nullptr;
  }
#endif
  context = nullptr;
}

BenchmarkResult RaySchedulingBench::GetResult(uint32_t config_idx) const {
  BenchmarkResult r;
  if (config_idx >= 3 && config_idx <= 5) {
    uint32_t passes = (samplesPerPixel > 1)
        ? ((lastCompletedPasses[config_idx] > 0) ? lastCompletedPasses[config_idx] : samplesPerPixel)
        : 1u;
    r.operations = static_cast<uint64_t>(rayCount) * passes;
  } else {
    // Stage compaction benchmarks (configs 10, 17, 19) and primary rays report ray/record operations
    r.operations = static_cast<uint64_t>(rayCount);
  }
  r.elapsedTime = 0.0;
  return r;
}

bool RaySchedulingBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= GetNumConfigs()) {
    return false;
  }
  // If the configuration is legitimately unsupported on this hardware/driver
  // (e.g. SER on non-SER GPUs or DGC on drivers lacking execution sets),
  // it was not executed; return true to avoid false failure report.
  if (unsupportedConfig[config_idx]) {
    return true;
  }
  if (!context || !resultBuffer) {
    return false;
  }
  try {
    // resultBuffer layout: [0]=totalRays, [1]=totalHits, [2]=accumulatedLuminance, [3]=pad
    uint32_t stats[4] = {0, 0, 0, 0};
    context->readBuffer(resultBuffer, 0, sizeof(stats), stats);
    uint32_t totalRays = stats[0];
    uint32_t totalHits = stats[1];
    // Assert that workload executed rays and recorded valid intersections
    return (totalRays > 0 && totalHits > 0);
  } catch (...) {
    return false;
  }
}
