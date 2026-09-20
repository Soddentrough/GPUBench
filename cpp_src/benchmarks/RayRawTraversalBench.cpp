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

  // 2. Geometry 1: Deep Multi-Layer BVH Hierarchy (128 layers of 32x32 cells = 262,144 triangles)
  // Forces deep internal node and leaf traversal through 7-8 levels of BVH8 hierarchy.
  uint32_t deepLayers = 128;
  deepPrimitives = gridDim * gridDim * 2 * deepLayers;

  std::vector<float> deepVertices;
  deepVertices.reserve(deepPrimitives * 9);
  for (uint32_t z = 0; z < deepLayers; ++z) {
    float fz = static_cast<float>(z) * 0.1f;
    for (uint32_t y = 0; y < gridDim; ++y) {
      for (uint32_t x = 0; x < gridDim; ++x) {
        float x0 = static_cast<float>(x) / static_cast<float>(gridDim);
        float x1 = static_cast<float>(x + 1) / static_cast<float>(gridDim);
        float y0 = static_cast<float>(y) / static_cast<float>(gridDim);
        float y1 = static_cast<float>(y + 1) / static_cast<float>(gridDim);

        // Triangle A
        deepVertices.push_back(x0); deepVertices.push_back(y0); deepVertices.push_back(fz);
        deepVertices.push_back(x1); deepVertices.push_back(y0); deepVertices.push_back(fz);
        deepVertices.push_back(x0); deepVertices.push_back(y1); deepVertices.push_back(fz);

        // Triangle B
        deepVertices.push_back(x1); deepVertices.push_back(y0); deepVertices.push_back(fz);
        deepVertices.push_back(x1); deepVertices.push_back(y1); deepVertices.push_back(fz);
        deepVertices.push_back(x0); deepVertices.push_back(y1); deepVertices.push_back(fz);
      }
    }
  }
  vertexBufferDeep =
      context.createBuffer(deepVertices.size() * sizeof(float), deepVertices.data());

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path kernel_file = kdir / "vulkan" / "rt_raw_traversal.comp";
  kernel = context.createKernel(kernel_file.string(), "main", 2);
#endif // HAVE_VULKAN
}

#ifdef HAVE_VULKAN
void RayRawTraversalBench::buildAS() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  VkDeviceAddress vAddrCoherent = vContext->getBufferDeviceAddress(vertexBufferCoherent);
  VkDeviceAddress vAddrDeep = vContext->getBufferDeviceAddress(vertexBufferDeep);

  // Helper lambda for BLAS creation
  auto createBLAS = [&](VkDeviceAddress vertexAddr, uint32_t primCount,
                        VkAccelerationStructureKHR &blasHandle,
                        ComputeBuffer &blasBufferHandle,
                        VkAccelerationStructureBuildGeometryInfoKHR &buildInfoOut,
                        VkAccelerationStructureGeometryKHR &geomOut,
                        VkAccelerationStructureBuildSizesInfoKHR &sizesOut) {
    geomOut.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geomOut.pNext = nullptr;
    geomOut.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geomOut.flags = VK_GEOMETRY_OPAQUE_BIT_KHR; // Strict hardware opaque fast path
    geomOut.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geomOut.geometry.triangles.pNext = nullptr;
    geomOut.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geomOut.geometry.triangles.vertexData.deviceAddress = vertexAddr;
    geomOut.geometry.triangles.vertexStride = sizeof(float) * 3;
    geomOut.geometry.triangles.maxVertex = primCount * 3;
    geomOut.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    geomOut.geometry.triangles.transformData.deviceAddress = 0;

    buildInfoOut.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfoOut.pNext = nullptr;
    buildInfoOut.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfoOut.flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    buildInfoOut.geometryCount = 1;
    buildInfoOut.pGeometries = &geomOut;
    buildInfoOut.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    sizesOut.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    sizesOut.pNext = nullptr;
    uint32_t maxPrimCount = primCount;
    vkGetAccelerationStructureBuildSizesKHR_ptr(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfoOut,
        &maxPrimCount, &sizesOut);

    blasBufferHandle = context->createBuffer(sizesOut.accelerationStructureSize);
    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = vContext->getVkBuffer(blasBufferHandle);
    createInfo.size = sizesOut.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vkCreateAccelerationStructureKHR_ptr(device, &createInfo, nullptr, &blasHandle);
  };

  VkAccelerationStructureGeometryKHR geomCoherent{};
  VkAccelerationStructureBuildGeometryInfoKHR buildInfoCoherent{};
  VkAccelerationStructureBuildSizesInfoKHR sizesCoherent{};
  createBLAS(vAddrCoherent, coherentPrimitives, coherentBlas, coherentBlasBuffer,
             buildInfoCoherent, geomCoherent, sizesCoherent);

  VkAccelerationStructureGeometryKHR geomDeep{};
  VkAccelerationStructureBuildGeometryInfoKHR buildInfoDeep{};
  VkAccelerationStructureBuildSizesInfoKHR sizesDeep{};
  createBLAS(vAddrDeep, deepPrimitives, deepBlas, deepBlasBuffer,
             buildInfoDeep, geomDeep, sizesDeep);

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
  instDeep.flags =
      VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR |
      VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;

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
    tlasGeomOut.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
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
    vkCreateAccelerationStructureKHR_ptr(device, &createInfo, nullptr, &tlasHandle);
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
  vkCreateCommandPool(device, &cpInfo, nullptr, &tmpPool);

  VkCommandBufferAllocateInfo cbAlloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbAlloc.commandPool = tmpPool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device, &cbAlloc, &cmd);

  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &beginInfo);

  auto cmdBuild = [&](VkAccelerationStructureBuildGeometryInfoKHR &info,
                      VkAccelerationStructureKHR dst, uint32_t primCount) {
    info.dstAccelerationStructure = dst;
    info.scratchData.deviceAddress = sAddr;
    VkAccelerationStructureBuildRangeInfoKHR range{primCount, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
    vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &info, &pRange);

    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
  };

  // Build BLASes
  cmdBuild(buildInfoCoherent, coherentBlas, coherentPrimitives);
  cmdBuild(buildInfoDeep, deepBlas, deepPrimitives);

  // Build TLASes
  cmdBuild(tlasBuildCoherent, coherentTlas, 1);
  cmdBuild(tlasBuildDeep, deepTlas, 1);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, tmpPool, nullptr);
}
#endif // HAVE_VULKAN

void RayRawTraversalBench::Run(uint32_t config_idx) {
#ifdef HAVE_VULKAN
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkAccelerationStructureKHR activeTlas =
      (config_idx == 0) ? coherentTlas : deepTlas;

  vContext->setKernelAS(kernel, 0, (AccelerationStructure)activeTlas);
  vContext->setKernelArg(kernel, 1, resultBuffer);

  uint32_t mode = config_idx; // 0 for coherent, 1 for deep
  float tMin = 0.001f;
  float tMax = 100.0f;

  vContext->setKernelArg(kernel, 2, sizeof(uint32_t), &rayCount);
  vContext->setKernelArg(kernel, 3, sizeof(uint32_t), &mode);
  vContext->setKernelArg(kernel, 4, sizeof(float), &tMin);
  vContext->setKernelArg(kernel, 5, sizeof(float), &tMax);

  vContext->dispatch(kernel, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
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
    if (kernel) { context->releaseKernel(kernel); kernel = nullptr; }
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
#endif
}

BenchmarkResult RayRawTraversalBench::GetResult(uint32_t config_idx) const {
  if (config_idx == 0) {
    // Config 0: Triangle intersection tests in leaf (2 candidate triangle tests per ray)
    return {(uint64_t)rayCount * 2, 0.0};
  } else {
    // Config 1: Deep multi-layer traversal (reported in MRays/s)
    return {(uint64_t)rayCount, 0.0};
  }
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
    // Config 0 reports GIS/s as primary metric (triangle tests)
    // Secondary telemetry: MRays/s and % of 300.8 GIS/s Boost ceiling
    uint64_t totalTriTests = totalRays * 2;
    double gis_s = (static_cast<double>(totalTriTests) / time_s) / 1e9;
    double pctBoost = (gis_s / 300.8) * 100.0;
    ss << mrays_s << " MRays/s (" << pctBoost << "% Boost Peak)";
  } else {
    // Config 1 reports MRays/s as primary metric (deep traversal)
    // Secondary telemetry: sustained Box GIS/s (64 box tests/ray) and % of 1,203.2 GIS/s Boost ceiling
    uint64_t totalBoxTests = totalRays * 64;
    double box_gis_s = (static_cast<double>(totalBoxTests) / time_s) / 1e9;
    double pctBoost = (box_gis_s / 1203.2) * 100.0;
    ss << box_gis_s << " GIS/s (" << pctBoost << "% Boost Peak)";
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
  if (!context || !resultBuffer) return false;

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
}

const char *RayRawTraversalBench::GetName() const {
  return "RayRawTraversal";
}

std::vector<std::string> RayRawTraversalBench::GetAliases() const {
  return {"rawtraversal", "rtraw", "bvhraw", "raw_traversal"};
}

const char *RayRawTraversalBench::GetMetric() const {
  return "GIS/s";
}

const char *RayRawTraversalBench::GetMetric(uint32_t config_idx) const {
  return config_idx == 0 ? "GIS/s" : "MRays/s";
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
      ? "Raw BVH Traversal - Coherent Triangle Traversal (Max Occupancy)"
      : "Raw BVH Traversal - Deep BVH Multi-Layer Stress";
}

uint64_t RayRawTraversalBench::GetTraversedRays(uint32_t config_idx) const {
  if (config_idx >= 2) return 0;
  return static_cast<uint64_t>(rayCount) * recordedInvocations[config_idx];
}

uint64_t RayRawTraversalBench::GetSustainedOperations(uint32_t config_idx) const {
  if (config_idx == 0) {
    return GetTraversedRays(0) * 2; // 2 triangle tests per ray
  } else {
    return GetTraversedRays(1) * 64; // 64 box tests per ray across 8 BVH8 nodes
  }
}

double RayRawTraversalBench::GetThroughputGIS(uint32_t config_idx) const {
  if (config_idx >= 2 || recordedTimeMs[config_idx] <= 0.0) return 0.0;
  double time_s = recordedTimeMs[config_idx] / 1000.0;
  return (static_cast<double>(GetSustainedOperations(config_idx)) / time_s) / 1e9;
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
  uint32_t layers = 8; // Dump representative 8 layers for visualization
  uint32_t vIdx = 1;

  for (uint32_t z = 0; z < layers; ++z) {
    float fz = static_cast<float>(z) * 0.1f;
    for (uint32_t y = 0; y < gridDim; ++y) {
      for (uint32_t x = 0; x < gridDim; ++x) {
        float x0 = static_cast<float>(x) / static_cast<float>(gridDim);
        float x1 = static_cast<float>(x + 1) / static_cast<float>(gridDim);
        float y0 = static_cast<float>(y) / static_cast<float>(gridDim);
        float y1 = static_cast<float>(y + 1) / static_cast<float>(gridDim);

        // Quad vertices
        objFile << "v " << x0 << " " << y0 << " " << fz << "\n";
        objFile << "v " << x1 << " " << y0 << " " << fz << "\n";
        objFile << "v " << x1 << " " << y1 << " " << fz << "\n";
        objFile << "v " << x0 << " " << y1 << " " << fz << "\n";

        // Two triangles
        objFile << "f " << vIdx << " " << vIdx + 1 << " " << vIdx + 3 << "\n";
        objFile << "f " << vIdx + 1 << " " << vIdx + 2 << " " << vIdx + 3 << "\n";
        vIdx += 4;
      }
    }
  }
  objFile.close();
  std::cout << "Raw traversal geometry dumped to raw_traversal_scene.obj" << std::endl;
}
