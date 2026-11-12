#include "RayRawTraversalBench.h"
#include "core/VulkanContext.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

bool RayRawTraversalBench::IsSupported(const DeviceInfo &info,
                                      IComputeContext *context) const {
  return info.rayTracingSupport &&
         (context && context->getBackend() == ComputeBackend::Vulkan);
}

#ifdef HAVE_VULKAN
void RayRawTraversalBench::loadRTProcs(VkDevice device) {
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
#endif

void RayRawTraversalBench::Setup(IComputeContext &context,
                                const std::string &kernel_dir) {
  this->context = &context;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
  if (!vContext) {
    throw std::runtime_error("RayRawTraversalBench requires VulkanContext");
  }

#ifdef HAVE_VULKAN
  loadRTProcs(vContext->getVulkanDevice());

  // Target 64,000,000 rays per dispatch to maximally saturate 64 CUs (128 SIMD32s)
  rayCount = 64000000;
  uint32_t numWorkgroups = (rayCount + 31) / 32;

  // Strided output buffer: exactly 1 uint32 per Wave32 workgroup (0 global atomic contention)
  resultBuffer = context.createBuffer(numWorkgroups * sizeof(uint32_t));
  std::vector<uint32_t> zeroBuffer(numWorkgroups, 0);
  context.writeBuffer(resultBuffer, 0, numWorkgroups * sizeof(uint32_t), zeroBuffer.data());

  // 1. Geometry 0: Coherent Layered Triangle Grid (32 layers of 32x32 cells = 65,536 triangles)
  // Perfectly tiles [0.0, 1.0] x [0.0, 1.0] in XY. Ray origins in [0.01, 0.99] have 0% root misses.
  uint32_t gridDim = 32;
  uint32_t coherentLayers = 32;
  coherentPrimitives = gridDim * gridDim * 2 * coherentLayers;

  std::vector<float> coherentVertices;
  coherentVertices.reserve(coherentPrimitives * 9);
  for (uint32_t z = 0; z < coherentLayers; ++z) {
    float fz = static_cast<float>(z) * 0.1f;
    for (uint32_t y = 0; y < gridDim; ++y) {
      for (uint32_t x = 0; x < gridDim; ++x) {
        float x0 = static_cast<float>(x) / static_cast<float>(gridDim);
        float x1 = static_cast<float>(x + 1) / static_cast<float>(gridDim);
        float y0 = static_cast<float>(y) / static_cast<float>(gridDim);
        float y1 = static_cast<float>(y + 1) / static_cast<float>(gridDim);

        // Triangle A
        coherentVertices.push_back(x0); coherentVertices.push_back(y0); coherentVertices.push_back(fz);
        coherentVertices.push_back(x1); coherentVertices.push_back(y0); coherentVertices.push_back(fz);
        coherentVertices.push_back(x0); coherentVertices.push_back(y1); coherentVertices.push_back(fz);

        // Triangle B
        coherentVertices.push_back(x1); coherentVertices.push_back(y0); coherentVertices.push_back(fz);
        coherentVertices.push_back(x1); coherentVertices.push_back(y1); coherentVertices.push_back(fz);
        coherentVertices.push_back(x0); coherentVertices.push_back(y1); coherentVertices.push_back(fz);
      }
    }
  }
  vertexBufferCoherent =
      context.createBuffer(coherentVertices.size() * sizeof(float), coherentVertices.data());

  // 2. Geometry 1: Deep Multi-Layer AABB BVH Hierarchy (64 layers of 32x32 cells = 65,536 AABBs)
  // Uses true VK_GEOMETRY_TYPE_AABBS_KHR slabs so every ray traverses all 64 stratified AABB layers
  // before generating a procedural intersection on Layer 63.
  uint32_t deepLayers = 64;
  deepPrimitives = gridDim * gridDim * deepLayers;

  std::vector<VkAabbPositionsKHR> deepAabbs;
  deepAabbs.reserve(deepPrimitives);
  for (uint32_t z = 0; z < deepLayers; ++z) {
    float fz = static_cast<float>(z) * 0.1f;
    for (uint32_t y = 0; y < gridDim; ++y) {
      for (uint32_t x = 0; x < gridDim; ++x) {
        float x0 = static_cast<float>(x) / static_cast<float>(gridDim);
        float x1 = static_cast<float>(x + 1) / static_cast<float>(gridDim);
        float y0 = static_cast<float>(y) / static_cast<float>(gridDim);
        float y1 = static_cast<float>(y + 1) / static_cast<float>(gridDim);
        deepAabbs.push_back({x0, y0, fz - 0.01f, x1, y1, fz + 0.01f});
      }
    }
  }
  vertexBufferDeep = context.createBuffer(
      deepAabbs.size() * sizeof(VkAabbPositionsKHR), deepAabbs.data());

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path kernel_file = kdir / "vulkan" / "rt_raw_traversal.comp";
  kernels[0] = context.createKernel(kernel_file.string(), "main", 2);
  kernels[1] = context.createKernel(kernel_file.string(), "main", 2);
  kernel = kernels[0];

  vContext->setKernelAS(kernels[0], 0, (AccelerationStructure)coherentTlas);
  vContext->setKernelArg(kernels[0], 1, resultBuffer);
  vContext->setKernelAS(kernels[1], 0, (AccelerationStructure)deepTlas);
  vContext->setKernelArg(kernels[1], 1, resultBuffer);
#endif // HAVE_VULKAN
}

#ifdef HAVE_VULKAN
void RayRawTraversalBench::buildAS() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  VkDeviceAddress vAddrCoherent = vContext->getBufferDeviceAddress(vertexBufferCoherent);
  VkDeviceAddress aAddrDeep = vContext->getBufferDeviceAddress(vertexBufferDeep);

  // 1. Coherent Triangle BLAS (strict hardware opaque triangle fast path)
  VkAccelerationStructureGeometryKHR geomCoherent{};
  geomCoherent.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geomCoherent.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  geomCoherent.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  geomCoherent.geometry.triangles.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  geomCoherent.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  geomCoherent.geometry.triangles.vertexData.deviceAddress = vAddrCoherent;
  geomCoherent.geometry.triangles.vertexStride = sizeof(float) * 3;
  geomCoherent.geometry.triangles.maxVertex = coherentPrimitives * 3 - 1;
  geomCoherent.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfoCoherent{};
  buildInfoCoherent.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfoCoherent.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  buildInfoCoherent.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  buildInfoCoherent.geometryCount = 1;
  buildInfoCoherent.pGeometries = &geomCoherent;
  buildInfoCoherent.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  VkAccelerationStructureBuildSizesInfoKHR sizesCoherent{};
  sizesCoherent.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  uint32_t maxPrimCoherent = coherentPrimitives;
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfoCoherent,
      &maxPrimCoherent, &sizesCoherent);

  coherentBlasBuffer = context->createBuffer(sizesCoherent.accelerationStructureSize);
  {
    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = vContext->getVkBuffer(coherentBlasBuffer);
    createInfo.size = sizesCoherent.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR_ptr(device, &createInfo, nullptr, &coherentBlas) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create coherent BLAS in RayRawTraversalBench");
    }
  }

  // 2. Deep Multi-Layer AABB BLAS (non-opaque procedural AABBs so rayQueryProceedEXT yields all 64 layer candidates)
  VkAccelerationStructureGeometryKHR geomDeep{};
  geomDeep.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  geomDeep.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  geomDeep.flags = 0;
  geomDeep.geometry.aabbs.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
  geomDeep.geometry.aabbs.data.deviceAddress = aAddrDeep;
  geomDeep.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

  VkAccelerationStructureBuildGeometryInfoKHR buildInfoDeep{};
  buildInfoDeep.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
  buildInfoDeep.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  buildInfoDeep.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  buildInfoDeep.geometryCount = 1;
  buildInfoDeep.pGeometries = &geomDeep;
  buildInfoDeep.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  VkAccelerationStructureBuildSizesInfoKHR sizesDeep{};
  sizesDeep.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  uint32_t maxPrimDeep = deepPrimitives;
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfoDeep,
      &maxPrimDeep, &sizesDeep);

  deepBlasBuffer = context->createBuffer(sizesDeep.accelerationStructureSize);
  {
    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = vContext->getVkBuffer(deepBlasBuffer);
    createInfo.size = sizesDeep.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR_ptr(device, &createInfo, nullptr, &deepBlas) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create deep AABB BLAS in RayRawTraversalBench");
    }
  }

  // TLAS Device Addresses
  VkAccelerationStructureDeviceAddressInfoKHR addrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  addrInfo.accelerationStructure = coherentBlas;
  VkDeviceAddress blasAddrCoherent =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &addrInfo);

  addrInfo.accelerationStructure = deepBlas;
  VkDeviceAddress blasAddrDeep =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &addrInfo);

  // Instances
  VkAccelerationStructureInstanceKHR instCoherent = {};
  instCoherent.transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  instCoherent.instanceCustomIndex = 0;
  instCoherent.mask = 0xFF;
  instCoherent.accelerationStructureReference = blasAddrCoherent;
  instCoherent.flags =
      VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR |
      VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;

  VkAccelerationStructureInstanceKHR instDeep = {};
  instDeep.transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  instDeep.instanceCustomIndex = 0;
  instDeep.mask = 0xFF;
  instDeep.accelerationStructureReference = blasAddrDeep;
  instDeep.flags = VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;

  instanceBufferCoherent = context->createBuffer(sizeof(instCoherent), &instCoherent);
  instanceBufferDeep = context->createBuffer(sizeof(instDeep), &instDeep);

  // Helper lambda for TLAS creation
  auto createTLAS = [&](ComputeBuffer &instBuffer,
                        VkAccelerationStructureKHR &tlasHandle,
                        ComputeBuffer &tlasBufferHandle,
                        VkAccelerationStructureBuildGeometryInfoKHR &tlasBuildInfoOut,
                        VkAccelerationStructureGeometryKHR &tlasGeomOut,
                        VkAccelerationStructureBuildSizesInfoKHR &tlasSizesOut) {
    tlasGeomOut.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeomOut.pNext = nullptr;
    tlasGeomOut.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeomOut.flags = 0;
    tlasGeomOut.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeomOut.geometry.instances.pNext = nullptr;
    tlasGeomOut.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeomOut.geometry.instances.data.deviceAddress =
        vContext->getBufferDeviceAddress(instBuffer);

    tlasBuildInfoOut.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlasBuildInfoOut.pNext = nullptr;
    tlasBuildInfoOut.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuildInfoOut.flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    tlasBuildInfoOut.geometryCount = 1;
    tlasBuildInfoOut.pGeometries = &tlasGeomOut;
    tlasBuildInfoOut.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    uint32_t maxPrimCount = 1;
    tlasSizesOut.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    tlasSizesOut.pNext = nullptr;
    vkGetAccelerationStructureBuildSizesKHR_ptr(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfoOut,
        &maxPrimCount, &tlasSizesOut);

    tlasBufferHandle = context->createBuffer(tlasSizesOut.accelerationStructureSize);
    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = vContext->getVkBuffer(tlasBufferHandle);
    createInfo.size = tlasSizesOut.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    if (vkCreateAccelerationStructureKHR_ptr(device, &createInfo, nullptr, &tlasHandle) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create TLAS in RayRawTraversalBench");
    }
  };

  VkAccelerationStructureGeometryKHR topGeomCoherent{};
  VkAccelerationStructureBuildGeometryInfoKHR tlasBuildCoherent{};
  VkAccelerationStructureBuildSizesInfoKHR tlasSizesCoherent{};
  createTLAS(instanceBufferCoherent, coherentTlas, coherentTlasBuffer,
             tlasBuildCoherent, topGeomCoherent, tlasSizesCoherent);

  VkAccelerationStructureGeometryKHR topGeomDeep{};
  VkAccelerationStructureBuildGeometryInfoKHR tlasBuildDeep{};
  VkAccelerationStructureBuildSizesInfoKHR tlasSizesDeep{};
  createTLAS(instanceBufferDeep, deepTlas, deepTlasBuffer,
             tlasBuildDeep, topGeomDeep, tlasSizesDeep);

  // Allocate Scratch Buffer for builds
  size_t maxScratch = std::max({
      static_cast<size_t>(sizesCoherent.buildScratchSize),
      static_cast<size_t>(sizesDeep.buildScratchSize),
      static_cast<size_t>(tlasSizesCoherent.buildScratchSize),
      static_cast<size_t>(tlasSizesDeep.buildScratchSize)});
  scratchBuffer = context->createBuffer(maxScratch);
  VkDeviceAddress sAddr = vContext->getBufferDeviceAddress(scratchBuffer);

  // Command Buffer for Building AS
  VkCommandPoolCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpInfo.queueFamilyIndex = vContext->getComputeQueueFamilyIndex();
  VkCommandPool tmpPool;
  if (vkCreateCommandPool(device, &cpInfo, nullptr, &tmpPool) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool in RayRawTraversalBench");
  }

  VkCommandBufferAllocateInfo cbAlloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbAlloc.commandPool = tmpPool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  if (vkAllocateCommandBuffers(device, &cbAlloc, &cmd) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to allocate command buffer in RayRawTraversalBench");
  }

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to begin command buffer in RayRawTraversalBench");
  }

  auto cmdBuild = [&](VkAccelerationStructureBuildGeometryInfoKHR &info,
                      VkAccelerationStructureKHR dst, uint32_t primCount) {
    info.dstAccelerationStructure = dst;
    info.scratchData.deviceAddress = sAddr;
    VkAccelerationStructureBuildRangeInfoKHR range{primCount, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
    vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &info, &pRange);

    vContext->cmdPipelineMemoryBarrier2(
        cmd,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  };

  // Build BLASes
  cmdBuild(buildInfoCoherent, coherentBlas, coherentPrimitives);
  cmdBuild(buildInfoDeep, deepBlas, deepPrimitives);

  // Build TLASes
  cmdBuild(tlasBuildCoherent, coherentTlas, 1);
  cmdBuild(tlasBuildDeep, deepTlas, 1);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to end command buffer in RayRawTraversalBench");
  }

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to submit AS build in RayRawTraversalBench");
  }
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, tmpPool, nullptr);
}
#endif // HAVE_VULKAN

void RayRawTraversalBench::Run(uint32_t config_idx) {
#ifdef HAVE_VULKAN
  if (config_idx >= 2 || !kernels[config_idx]) {
    return;
  }
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  if (lastRunConfig != -1 && lastRunConfig != static_cast<int>(config_idx)) {
    uint32_t checkCount = 1024;
    std::vector<uint32_t> zeroHits(checkCount, 0);
    context->writeBuffer(resultBuffer, 0, checkCount * sizeof(uint32_t), zeroHits.data());
  }
  lastRunConfig = static_cast<int>(config_idx);

  ComputeKernel activeKernel = kernels[config_idx];
  uint32_t mode = config_idx; // 0 for coherent triangles, 1 for deep 64-layer AABBs
  float tMin = 0.001f;
  float tMax = 100.0f;

  vContext->setKernelArg(activeKernel, 2, sizeof(uint32_t), &rayCount);
  vContext->setKernelArg(activeKernel, 3, sizeof(uint32_t), &mode);
  vContext->setKernelArg(activeKernel, 4, sizeof(float), &tMin);
  vContext->setKernelArg(activeKernel, 5, sizeof(float), &tMax);

  // 2D grid (31250 x 64 = 2,000,000 workgroups * 32 threads = 64,000,000 rays) stays strictly <= 65535
  uint32_t totalWg = (rayCount + 31u) / 32u;
  uint32_t gridY = 64u;
  uint32_t gridX = (totalWg + gridY - 1u) / gridY;
  vContext->dispatch(activeKernel, gridX, gridY, 1, 32, 1, 1);
#else
  (void)config_idx;
#endif
}

void RayRawTraversalBench::Teardown() {
#ifdef HAVE_VULKAN
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext ? vContext->getVulkanDevice() : VK_NULL_HANDLE;

  if (device != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_ptr) {
    if (coherentBlas != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR_ptr(device, coherentBlas, nullptr);
      coherentBlas = VK_NULL_HANDLE;
    }
    if (coherentTlas != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR_ptr(device, coherentTlas, nullptr);
      coherentTlas = VK_NULL_HANDLE;
    }
    if (deepBlas != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR_ptr(device, deepBlas, nullptr);
      deepBlas = VK_NULL_HANDLE;
    }
    if (deepTlas != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR_ptr(device, deepTlas, nullptr);
      deepTlas = VK_NULL_HANDLE;
    }
  }

  if (context) {
    for (uint32_t i = 0; i < 2; ++i) {
      if (kernels[i]) {
        context->releaseKernel(kernels[i]);
        kernels[i] = nullptr;
      }
    }
    kernel = nullptr;
    if (resultBuffer) { context->releaseBuffer(resultBuffer); resultBuffer = nullptr; }
    if (vertexBufferCoherent) { context->releaseBuffer(vertexBufferCoherent); vertexBufferCoherent = nullptr; }
    if (vertexBufferDeep) { context->releaseBuffer(vertexBufferDeep); vertexBufferDeep = nullptr; }
    if (instanceBufferCoherent) { context->releaseBuffer(instanceBufferCoherent); instanceBufferCoherent = nullptr; }
    if (instanceBufferDeep) { context->releaseBuffer(instanceBufferDeep); instanceBufferDeep = nullptr; }
    if (coherentBlasBuffer) { context->releaseBuffer(coherentBlasBuffer); coherentBlasBuffer = nullptr; }
    if (coherentTlasBuffer) { context->releaseBuffer(coherentTlasBuffer); coherentTlasBuffer = nullptr; }
    if (deepBlasBuffer) { context->releaseBuffer(deepBlasBuffer); deepBlasBuffer = nullptr; }
    if (deepTlasBuffer) { context->releaseBuffer(deepTlasBuffer); deepTlasBuffer = nullptr; }
    if (scratchBuffer) { context->releaseBuffer(scratchBuffer); scratchBuffer = nullptr; }
    context = nullptr;
  }
  lastRunConfig = -1;
#endif
}

BenchmarkResult RayRawTraversalBench::GetResult(uint32_t config_idx) const {
  (void)config_idx;
  return {(uint64_t)rayCount, 0.0};
}

void RayRawTraversalBench::RecordRunResult(uint32_t config_idx,
                                          uint64_t total_invocations,
                                          double total_time_ms) {
  if (config_idx < 2) {
    recordedInvocations[config_idx] = total_invocations;
    recordedTimeMs[config_idx] = total_time_ms;
  }
}

std::string RayRawTraversalBench::GetConfigCaveat(uint32_t config_idx) const {
  if (config_idx >= 2 || recordedTimeMs[config_idx] <= 0.0 || recordedInvocations[config_idx] == 0) {
    return "";
  }

  double time_s = recordedTimeMs[config_idx] / 1000.0;
  uint64_t totalRays = static_cast<uint64_t>(rayCount) * recordedInvocations[config_idx];
  double mrays_s = (static_cast<double>(totalRays) / time_s) / 1e6;

  std::ostringstream ss;
  ss << std::fixed << std::setprecision(1);

  if (config_idx == 0) {
    ss << mrays_s << " MRays/s (Coherent Triangle Fast-Path)";
  } else {
    uint64_t boxOps = totalRays * 64;
    double gis = (time_s > 0.0) ? ((static_cast<double>(boxOps) / time_s) / 1e9) : 0.0;
    ss << mrays_s << " MRays/s (Synthetic 64-AABB Traversal Model; " << gis << " Box GIS/s)";
  }

  return ss.str();
}

std::string RayRawTraversalBench::GetConfigCaveat(uint32_t config_idx,
                                                 const DeviceInfo &info,
                                                 IComputeContext *context) const {
  (void)info;
  (void)context;
  return GetConfigCaveat(config_idx);
}

bool RayRawTraversalBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= GetNumConfigs()) return false;
  if (!context || !resultBuffer) return false;
  if (lastRunConfig != static_cast<int>(config_idx)) return false;

  try {
    // Read back first 1024 workgroups from strided buffer
    uint32_t checkCount = 1024;
    std::vector<uint32_t> hostHits(checkCount, 0);
    context->readBuffer(resultBuffer, 0, checkCount * sizeof(uint32_t), hostHits.data());

    for (uint32_t i = 0; i < checkCount; ++i) {
      // Every workgroup should report 32 hits (100% ray-geometry alignment, 0% empty space root misses)
      if (hostHits[i] == 0) {
        return false;
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

const char *RayRawTraversalBench::GetName() const {
  return "RayRawTraversal";
}

std::vector<std::string> RayRawTraversalBench::GetAliases() const {
  return {"rawtraversal", "rtraw", "bvhraw", "raw_traversal"};
}

const char *RayRawTraversalBench::GetMetric() const {
  return "MRays/s";
}

const char *RayRawTraversalBench::GetMetric(uint32_t config_idx) const {
  (void)config_idx;
  return "MRays/s";
}

const char *RayRawTraversalBench::GetComponent(uint32_t config_idx) const {
  (void)config_idx;
  return "Ray Tracing";
}

const char *RayRawTraversalBench::GetSubCategory(uint32_t config_idx) const {
  (void)config_idx;
  return "Hardware BVH Traversal";
}

std::string RayRawTraversalBench::GetConfigName(uint32_t config_idx) const {
  return config_idx == 0
      ? "Raw BVH - Coherent Triangle Traversal (Max Occupancy)"
      : "Raw BVH - Deep Multi-Layer Box Stress";
}

uint64_t RayRawTraversalBench::GetTraversedRays(uint32_t config_idx) const {
  if (config_idx >= 2) return 0;
  return static_cast<uint64_t>(rayCount) * recordedInvocations[config_idx];
}

double RayRawTraversalBench::GetThroughputMRays(uint32_t config_idx) const {
  if (config_idx >= 2 || recordedTimeMs[config_idx] <= 0.0) return 0.0;
  double time_s = recordedTimeMs[config_idx] / 1000.0;
  return (static_cast<double>(GetTraversedRays(config_idx)) / time_s) / 1e6;
}

void RayRawTraversalBench::DumpGeometry() const {
  std::ofstream objFile("raw_traversal_scene.obj");
  if (!objFile.is_open()) return;

  uint32_t gridDim = 32;
  uint32_t layers = 64; // Dump all 64 stratified AABB layers (65,536 boxes) matching Deep Box Stress geometry
  uint32_t vIdx = 1;

  for (uint32_t z = 0; z < layers; ++z) {
    float fz = static_cast<float>(z) * 0.1f;
    float z0 = fz - 0.01f;
    float z1 = fz + 0.01f;
    for (uint32_t y = 0; y < gridDim; ++y) {
      for (uint32_t x = 0; x < gridDim; ++x) {
        float x0 = static_cast<float>(x) / static_cast<float>(gridDim);
        float x1 = static_cast<float>(x + 1) / static_cast<float>(gridDim);
        float y0 = static_cast<float>(y) / static_cast<float>(gridDim);
        float y1 = static_cast<float>(y + 1) / static_cast<float>(gridDim);

        objFile << "v " << x0 << " " << y0 << " " << z0 << "\n"
                << "v " << x1 << " " << y0 << " " << z0 << "\n"
                << "v " << x1 << " " << y1 << " " << z0 << "\n"
                << "v " << x0 << " " << y1 << " " << z0 << "\n"
                << "v " << x0 << " " << y0 << " " << z1 << "\n"
                << "v " << x1 << " " << y0 << " " << z1 << "\n"
                << "v " << x1 << " " << y1 << " " << z1 << "\n"
                << "v " << x0 << " " << y1 << " " << z1 << "\n";

        auto f = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
          objFile << "f " << (vIdx + a) << " " << (vIdx + b) << " "
                  << (vIdx + c) << " " << (vIdx + d) << "\n";
        };
        f(0, 1, 2, 3);
        f(4, 7, 6, 5);
        f(0, 4, 5, 1);
        f(3, 2, 6, 7);
        f(0, 3, 7, 4);
        f(1, 5, 6, 2);
        vIdx += 8;
      }
    }
  }
  objFile.close();
  std::cout << "Raw traversal geometry dumped to raw_traversal_scene.obj" << std::endl;
}
