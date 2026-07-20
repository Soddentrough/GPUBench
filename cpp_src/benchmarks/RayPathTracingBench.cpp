#include "RayPathTracingBench.h"
#include "core/VulkanContext.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>

bool RayPathTracingBench::IsSupported(const DeviceInfo &info,
                                     IComputeContext *context) const {
  return info.rayTracingSupport &&
         (context && context->getBackend() == ComputeBackend::Vulkan);
}

void RayPathTracingBench::loadRTProcs(VkDevice device) {
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

void RayPathTracingBench::Setup(IComputeContext &context,
                                const std::string &kernel_dir) {
  this->context = &context;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
  if (!vContext)
    throw std::runtime_error("RayPathTracingBench requires VulkanContext");

  loadRTProcs(vContext->getVulkanDevice());

  // Setup ray workload (4M paths)
  rayCount = 4000000;
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

  aabbBuffer = nullptr;

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path kernel_file = kdir / "vulkan" / "rt_path_tracing.comp";
  kernel = context.createKernel(kernel_file.string(), "main", 2);
}

void RayPathTracingBench::buildAS() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  VkDeviceAddress vAddr = vContext->getBufferDeviceAddress(vertexBuffer);

  // 1. Triangle BLAS
  VkAccelerationStructureGeometryKHR triGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  triGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  triGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
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

  // 2. TLAS
  VkAccelerationStructureDeviceAddressInfoKHR triAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  triAddrInfo.accelerationStructure = triangleBlas;
  VkDeviceAddress triASAddr =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &triAddrInfo);

  // Triangle Instance
  VkAccelerationStructureInstanceKHR triInstance = {};
  triInstance.transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  triInstance.instanceCustomIndex = 0;
  triInstance.mask = 0xFF;
  triInstance.accelerationStructureReference = triASAddr;
  triInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

  instanceBuffer =
      context->createBuffer(sizeof(VkAccelerationStructureInstanceKHR));
  context->writeBuffer(instanceBuffer, 0, sizeof(triInstance), &triInstance);

  VkAccelerationStructureGeometryKHR topGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  topGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  topGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  topGeom.geometry.instances.data.deviceAddress =
      vContext->getBufferDeviceAddress(instanceBuffer);

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &topGeom;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t maxPrimCount = 1; // One instance
  VkAccelerationStructureBuildSizesInfoKHR sizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
      &maxPrimCount, &sizes);

  tlasBuffer = context->createBuffer(sizes.accelerationStructureSize);
  VkAccelerationStructureCreateInfoKHR createInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  createInfo.buffer = vContext->getVkBuffer(tlasBuffer);
  createInfo.size = sizes.accelerationStructureSize;
  createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &createInfo, nullptr,
                                       &sceneTlas);

  size_t scratchSize = std::max(triSizes.buildScratchSize, sizes.buildScratchSize);
  scratchBuffer = context->createBuffer(scratchSize);
  VkDeviceAddress sAddr = vContext->getBufferDeviceAddress(scratchBuffer);

  // Command compilation for structure builds
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

  buildInfo.dstAccelerationStructure = sceneTlas;
  buildInfo.scratchData.deviceAddress = sAddr;
  cmdBuild(buildInfo, sceneTlas, 1);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, tmpPool, nullptr);
}

void RayPathTracingBench::Run(uint32_t config_idx) {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);

  // Clear results hits buffer for this timed run iteration
  uint32_t zero = 0;
  context->writeBuffer(resultBuffer, 0, 4, &zero);

  vContext->setKernelAS(kernel, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernel, 1, resultBuffer);

  uint32_t bounces = (config_idx == 0) ? 2 : ((config_idx == 1) ? 4 : 8);
  uint32_t seed = rand();

  vContext->setKernelArg(kernel, 2, sizeof(uint32_t), &rayCount);
  vContext->setKernelArg(kernel, 3, sizeof(uint32_t), &bounces);
  vContext->setKernelArg(kernel, 4, sizeof(uint32_t), &seed);

  auto start = std::chrono::high_resolution_clock::now();
  vContext->dispatch(kernel, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
  context->waitIdle();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  results[config_idx] = diff.count();
}

void RayPathTracingBench::Teardown() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();

  if (triangleBlas)
    vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
  if (boxBlas)
    vkDestroyAccelerationStructureKHR_ptr(device, boxBlas, nullptr);
  if (sceneTlas)
    vkDestroyAccelerationStructureKHR_ptr(device, sceneTlas, nullptr);

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
  if (tlasBuffer)
    context->releaseBuffer(tlasBuffer);
  if (scratchBuffer)
    context->releaseBuffer(scratchBuffer);
}

BenchmarkResult RayPathTracingBench::GetResult(uint32_t config_idx) const {
  // Metric is primary-ray throughput: rays launched from the camera per
  // second. Do NOT multiply by bounce count — counting secondary rays
  // inverts the scaling (paths terminate early, so total time grows
  // slower than the bounce count, making throughput appear to increase).
  return {(uint64_t)rayCount, results[config_idx]};
}

const char *RayPathTracingBench::GetName() const { return "RayPathTracing"; }

const char *RayPathTracingBench::GetComponent(uint32_t config_idx) const {
  return "Ray Tracing";
}

const char *RayPathTracingBench::GetMetric(uint32_t config_idx) const {
  return "MRays/s";
}

const char *RayPathTracingBench::GetSubCategory(uint32_t config_idx) const {
  return "Path Tracing";
}

std::string RayPathTracingBench::GetConfigName(uint32_t config_idx) const {
  return config_idx == 0 ? "2 Bounces" : ((config_idx == 1) ? "4 Bounces" : "8 Bounces");
}
