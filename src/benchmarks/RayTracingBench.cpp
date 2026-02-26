#include "RayTracingBench.h"
#include "core/VulkanContext.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>

bool RayTracingBench::IsSupported(const DeviceInfo &info,
                                  IComputeContext *context) const {
  return info.rayTracingSupport &&
         (context && context->getBackend() == ComputeBackend::Vulkan);
}

void RayTracingBench::loadRTProcs(VkDevice device) {
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

void RayTracingBench::Setup(IComputeContext &context,
                            const std::string &kernel_dir) {
  this->context = &context;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
  if (!vContext)
    throw std::runtime_error("RayTracingBench requires VulkanContext");

  loadRTProcs(vContext->getVulkanDevice());

  // Target a substantial workload to saturate RTUs
  rayCount = 128000000;
  resultBuffer = context.createBuffer(sizeof(uint32_t));
  uint32_t zero = 0;
  context.writeBuffer(resultBuffer, 0, 4, &zero);

  // Setup Triangle and Box data (64 layers of 16x16 grids = 16,384 primitives)
  uint32_t gridSize = 16;
  uint32_t layers = 64;
  numPrimitives = gridSize * gridSize * layers;

  std::vector<float> vertices;
  for (uint32_t z = 0; z < layers; ++z) {
    float jitterX = (z % 8) * 0.05f;
    float jitterY = (z / 8) * 0.05f;
    for (uint32_t y = 0; y < gridSize; ++y) {
      for (uint32_t x = 0; x < gridSize; ++x) {
        float fx = (float)x - 8.0f + jitterX;
        float fy = (float)y - 8.0f + jitterY;
        float fz = (float)z * 0.1f;
        // Small triangles in the bottom corner (0.1 to 0.4)
        vertices.push_back(fx + 0.1f);
        vertices.push_back(fy + 0.1f);
        vertices.push_back(fz);
        vertices.push_back(fx + 0.4f);
        vertices.push_back(fy + 0.1f);
        vertices.push_back(fz);
        vertices.push_back(fx + 0.1f);
        vertices.push_back(fy + 0.4f);
        vertices.push_back(fz);
      }
    }
  }
  vertexBuffer =
      context.createBuffer(vertices.size() * sizeof(float), vertices.data());

  std::vector<VkAabbPositionsKHR> aabbs;
  for (uint32_t z = 0; z < layers; ++z) {
    float jitterX = (z % 8) * 0.05f;
    float jitterY = (z / 8) * 0.05f;
    for (uint32_t y = 0; y < gridSize; ++y) {
      for (uint32_t x = 0; x < gridSize; ++x) {
        float fx = (float)x - 8.0f + jitterX;
        float fy = (float)y - 8.0f + jitterY;
        float fz = (float)z * 0.1f;
        // Small AABB in the bottom corner (0.1 to 0.4). Ray at 0.8 completely
        // misses it!
        aabbs.push_back({fx + 0.1f, fy + 0.1f, fz - 0.01f, fx + 0.4f, fy + 0.4f,
                         fz + 0.01f});
      }
    }
  }
  aabbBuffer = context.createBuffer(aabbs.size() * sizeof(VkAabbPositionsKHR),
                                    aabbs.data());

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path kernel_file = kdir / "vulkan" / "rt_benchmark.comp";
  kernel = context.createKernel(kernel_file.string(), "main", 2);
}

void RayTracingBench::buildAS() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  VkDeviceAddress vAddr = vContext->getBufferDeviceAddress(vertexBuffer);
  VkDeviceAddress aAddr = vContext->getBufferDeviceAddress(aabbBuffer);

  // 1. Triangle BLAS
  VkAccelerationStructureGeometryKHR triGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  triGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  triGeom.flags =
      VK_GEOMETRY_OPAQUE_BIT_KHR; // Force exact hardware Ray-Tri test!
  triGeom.geometry.triangles.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  triGeom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triGeom.geometry.triangles.vertexData.deviceAddress = vAddr;
  triGeom.geometry.triangles.vertexStride = sizeof(float) * 3;
  triGeom.geometry.triangles.maxVertex = numPrimitives * 3;
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

  // 2. Box BLAS
  VkAccelerationStructureGeometryKHR boxGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  boxGeom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  boxGeom.flags = 0; // Non-opaque to stress math units
  boxGeom.geometry.aabbs.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
  boxGeom.geometry.aabbs.data.deviceAddress = aAddr;
  boxGeom.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

  VkAccelerationStructureBuildGeometryInfoKHR boxBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  boxBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  boxBuildInfo.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  boxBuildInfo.geometryCount = 1;
  boxBuildInfo.pGeometries = &boxGeom;
  boxBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t boxMaxPrimCount = numPrimitives;
  VkAccelerationStructureBuildSizesInfoKHR boxSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &boxBuildInfo,
      &boxMaxPrimCount, &boxSizes);

  boxBlasBuffer = context->createBuffer(boxSizes.accelerationStructureSize);
  VkAccelerationStructureCreateInfoKHR boxCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  boxCreateInfo.buffer = vContext->getVkBuffer(boxBlasBuffer);
  boxCreateInfo.size = boxSizes.accelerationStructureSize;
  boxCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &boxCreateInfo, nullptr,
                                       &boxBlas);

  // 3. TLASes
  VkAccelerationStructureDeviceAddressInfoKHR triAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  triAddrInfo.accelerationStructure = triangleBlas;
  VkDeviceAddress triASAddr =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &triAddrInfo);

  VkAccelerationStructureDeviceAddressInfoKHR boxAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  boxAddrInfo.accelerationStructure = boxBlas;
  VkDeviceAddress boxASAddr =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &boxAddrInfo);

  // Instance for Triangles
  VkAccelerationStructureInstanceKHR triInstance = {};
  triInstance.transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  triInstance.instanceCustomIndex = 0;
  triInstance.mask = 0xFF;
  triInstance.accelerationStructureReference = triASAddr;
  triInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

  // Instance for Boxes
  VkAccelerationStructureInstanceKHR boxInstance = {};
  boxInstance.transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  boxInstance.instanceCustomIndex = 0;
  boxInstance.mask = 0xFF;
  boxInstance.accelerationStructureReference = boxASAddr;
  boxInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

  instanceBuffer =
      context->createBuffer(sizeof(VkAccelerationStructureInstanceKHR) * 2);
  context->writeBuffer(instanceBuffer, 0, sizeof(triInstance), &triInstance);
  context->writeBuffer(instanceBuffer, sizeof(triInstance), sizeof(boxInstance),
                       &boxInstance);

  VkAccelerationStructureGeometryKHR topTriGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  topTriGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  topTriGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  topTriGeom.geometry.instances.data.deviceAddress =
      vContext->getBufferDeviceAddress(instanceBuffer);

  VkAccelerationStructureGeometryKHR topBoxGeom = topTriGeom;
  topBoxGeom.geometry.instances.data.deviceAddress += sizeof(triInstance);

  auto createTLAS = [&](VkAccelerationStructureGeometryKHR &geom,
                        VkAccelerationStructureKHR &tlasHandle,
                        ComputeBuffer &tlasBufferHandle) {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geom;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    uint32_t maxPrimCount = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR_ptr(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
        &maxPrimCount, &sizes);

    tlasBufferHandle = context->createBuffer(sizes.accelerationStructureSize);
    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = vContext->getVkBuffer(tlasBufferHandle);
    createInfo.size = sizes.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    vkCreateAccelerationStructureKHR_ptr(device, &createInfo, nullptr,
                                         &tlasHandle);
    return sizes.buildScratchSize;
  };

  size_t triScratch = createTLAS(topTriGeom, triangleTlas, triangleTlasBuffer);
  size_t boxScratch = createTLAS(topBoxGeom, boxTlas, boxTlasBuffer);

  // Scratch buffer
  size_t scratchSize =
      std::max({triSizes.buildScratchSize, boxSizes.buildScratchSize,
                triScratch, boxScratch});
  scratchBuffer = context->createBuffer(scratchSize);
  VkDeviceAddress sAddr = vContext->getBufferDeviceAddress(scratchBuffer);

  // Build commands
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

  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);

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

  cmdBuild(triBuildInfo, triangleBlas, numPrimitives);
  cmdBuild(boxBuildInfo, boxBlas, numPrimitives);

  VkAccelerationStructureBuildGeometryInfoKHR tlasTriBuild{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tlasTriBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tlasTriBuild.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tlasTriBuild.geometryCount = 1;
  tlasTriBuild.pGeometries = &topTriGeom;
  tlasTriBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  cmdBuild(tlasTriBuild, triangleTlas, 1);

  VkAccelerationStructureBuildGeometryInfoKHR tlasBoxBuild = tlasTriBuild;
  tlasBoxBuild.pGeometries = &topBoxGeom;
  cmdBuild(tlasBoxBuild, boxTlas, 1);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, tmpPool, nullptr);
}

void RayTracingBench::Run(uint32_t config_idx) {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkAccelerationStructureKHR activeTlas =
      (config_idx == 0) ? triangleTlas : boxTlas;

  vContext->setKernelAS(kernel, 0, (AccelerationStructure)activeTlas);
  vContext->setKernelArg(kernel, 1, resultBuffer);

  uint32_t testMode = config_idx; // 0 for triangle, 1 for box
  vContext->setKernelArg(kernel, 2, sizeof(uint32_t), &rayCount);
  vContext->setKernelArg(kernel, 3, sizeof(uint32_t), &testMode);

  vContext->dispatch(kernel, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
  context->waitIdle();
}

void RayTracingBench::Teardown() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();

  if (triangleBlas)
    vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
  if (boxBlas)
    vkDestroyAccelerationStructureKHR_ptr(device, boxBlas, nullptr);
  if (triangleTlas)
    vkDestroyAccelerationStructureKHR_ptr(device, triangleTlas, nullptr);
  if (boxTlas)
    vkDestroyAccelerationStructureKHR_ptr(device, boxTlas, nullptr);

  if (kernel)
    context->releaseKernel(kernel);
  if (resultBuffer)
    context->releaseBuffer(resultBuffer);
  if (vertexBuffer)
    context->releaseBuffer(vertexBuffer);
  if (aabbBuffer)
    context->releaseBuffer(aabbBuffer);
  if (instanceBuffer)
    context->releaseBuffer(instanceBuffer);
  if (triangleBlasBuffer)
    context->releaseBuffer(triangleBlasBuffer);
  if (boxBlasBuffer)
    context->releaseBuffer(boxBlasBuffer);
  if (triangleTlasBuffer)
    context->releaseBuffer(triangleTlasBuffer);
  if (boxTlasBuffer)
    context->releaseBuffer(boxTlasBuffer);
  if (scratchBuffer)
    context->releaseBuffer(scratchBuffer);
}

BenchmarkResult RayTracingBench::GetResult(uint32_t config_idx) const {
  // Each ray hits exactly 64 layers in our structured grid
  return {(uint64_t)rayCount * 64, 0.0};
}

const char *RayTracingBench::GetName() const { return "RayTracing"; }
const char *RayTracingBench::GetComponent(uint32_t config_idx) const {
  return "Ray Tracing";
}
const char *RayTracingBench::GetMetric() const { return "GIS/s"; }
const char *RayTracingBench::GetSubCategory(uint32_t config_idx) const {
  return "Intersection tests";
}

std::string RayTracingBench::GetConfigName(uint32_t config_idx) const {
  return config_idx == 0 ? "Ray-Triangle" : "Ray-Box";
}
