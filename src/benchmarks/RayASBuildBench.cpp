#include "RayASBuildBench.h"
#include "core/VulkanContext.h"
#include <chrono>
#include <iostream>

bool RayASBuildBench::IsSupported(const DeviceInfo &info,
                                  IComputeContext *context) const {
  return info.rayTracingSupport &&
         (context && context->getBackend() == ComputeBackend::Vulkan);
}

void RayASBuildBench::loadRTProcs(VkDevice device) {
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

void RayASBuildBench::Setup(IComputeContext &context,
                            const std::string &kernel_dir) {
  this->context = &context;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
  if (!vContext)
    throw std::runtime_error("RayASBuildBench requires VulkanContext");

  VkDevice device = vContext->getVulkanDevice();
  loadRTProcs(device);

  numPrimitives = 1000000; // 1M triangles
  numInstances = 10000;    // 10K instances

  std::vector<float> vertices(numPrimitives * 9, 0.0f);
  for (uint32_t i = 0; i < vertices.size(); ++i) {
    vertices[i] = (float(rand()) / RAND_MAX);
  }
  vertexBuffer =
      context.createBuffer(vertices.size() * sizeof(float), vertices.data());

  // Setup BLAS Info
  VkDeviceAddress vAddr = vContext->getBufferDeviceAddress(vertexBuffer);
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

  VkAccelerationStructureBuildGeometryInfoKHR blasBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  blasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  blasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  blasBuildInfo.geometryCount = 1;
  blasBuildInfo.pGeometries = &triGeom;
  blasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t maxPrimCount = numPrimitives;
  blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuildInfo,
      &maxPrimCount, &blasSizes);

  blasBuffer = context.createBuffer(blasSizes.accelerationStructureSize);
  VkAccelerationStructureCreateInfoKHR blasCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  blasCreateInfo.buffer = vContext->getVkBuffer(blasBuffer);
  blasCreateInfo.size = blasSizes.accelerationStructureSize;
  blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &blasCreateInfo, nullptr, &blas);

  // Setup TLAS Info
  VkAccelerationStructureDeviceAddressInfoKHR blasAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  blasAddrInfo.accelerationStructure = blas;
  VkDeviceAddress blasAddr =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &blasAddrInfo);

  std::vector<VkAccelerationStructureInstanceKHR> instances(numInstances);
  for (uint32_t i = 0; i < numInstances; ++i) {
    instances[i].transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    instances[i].instanceCustomIndex = i;
    instances[i].mask = 0xFF;
    instances[i].accelerationStructureReference = blasAddr;
    instances[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  }
  instanceBuffer = context.createBuffer(
      instances.size() * sizeof(VkAccelerationStructureInstanceKHR),
      instances.data());

  VkAccelerationStructureGeometryKHR instGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  instGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  instGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  instGeom.geometry.instances.data.deviceAddress =
      vContext->getBufferDeviceAddress(instanceBuffer);

  VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tlasBuildInfo.geometryCount = 1;
  tlasBuildInfo.pGeometries = &instGeom;
  tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t maxInstCount = numInstances;
  tlasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo,
      &maxInstCount, &tlasSizes);

  tlasBuffer = context.createBuffer(tlasSizes.accelerationStructureSize);
  VkAccelerationStructureCreateInfoKHR tlasCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  tlasCreateInfo.buffer = vContext->getVkBuffer(tlasBuffer);
  tlasCreateInfo.size = tlasSizes.accelerationStructureSize;
  tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &tlasCreateInfo, nullptr, &tlas);

  // Scratch buffers
  scratchBuffer = context.createBuffer(std::max(blasSizes.buildScratchSize, tlasSizes.buildScratchSize));
  updateScratchBuffer = context.createBuffer(blasSizes.updateScratchSize);
  
  // Initial build for updates to work
  Run(0);
}

void RayASBuildBench::Run(uint32_t config_idx) {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

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
  begin.flags = 0;
  vkBeginCommandBuffer(cmd, &begin);

  VkDeviceAddress vAddr = vContext->getBufferDeviceAddress(vertexBuffer);
  VkDeviceAddress iAddr = vContext->getBufferDeviceAddress(instanceBuffer);

  VkAccelerationStructureGeometryKHR geom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  VkAccelerationStructureBuildRangeInfoKHR range{};

  if (config_idx == 0 || config_idx == 2) { // BLAS
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geom.geometry.triangles.vertexData.deviceAddress = vAddr;
    geom.geometry.triangles.vertexStride = sizeof(float) * 3;
    geom.geometry.triangles.maxVertex = numPrimitives * 3;
    geom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geom;
    buildInfo.dstAccelerationStructure = blas;
    
    if (config_idx == 0) {
      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.scratchData.deviceAddress = vContext->getBufferDeviceAddress(scratchBuffer);
    } else {
      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      buildInfo.srcAccelerationStructure = blas;
      buildInfo.scratchData.deviceAddress = vContext->getBufferDeviceAddress(updateScratchBuffer);
    }
    
    range.primitiveCount = numPrimitives;
  } else { // TLAS
    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.data.deviceAddress = iAddr;

    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geom;
    buildInfo.dstAccelerationStructure = tlas;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.scratchData.deviceAddress = vContext->getBufferDeviceAddress(scratchBuffer);
    range.primitiveCount = numInstances;
  }

  const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
  vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &buildInfo, &pRange);

  vkEndCommandBuffer(cmd);

  auto start = std::chrono::high_resolution_clock::now();
  
  // We launch 10 submits to measure a stable time without too much CPU overhead
  uint32_t iters = 10;
  for (uint32_t i = 0; i < iters; ++i) {
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  buildTimes[config_idx] = (diff.count() / iters) * 1000.0; // Time in milliseconds
  iterations = iters;

  vkDestroyCommandPool(device, tmpPool, nullptr);
}

void RayASBuildBench::Teardown() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();

  if (blas) vkDestroyAccelerationStructureKHR_ptr(device, blas, nullptr);
  if (tlas) vkDestroyAccelerationStructureKHR_ptr(device, tlas, nullptr);

  if (vertexBuffer) context->releaseBuffer(vertexBuffer);
  if (instanceBuffer) context->releaseBuffer(instanceBuffer);
  if (blasBuffer) context->releaseBuffer(blasBuffer);
  if (tlasBuffer) context->releaseBuffer(tlasBuffer);
  if (scratchBuffer) context->releaseBuffer(scratchBuffer);
  if (updateScratchBuffer) context->releaseBuffer(updateScratchBuffer);
}

BenchmarkResult RayASBuildBench::GetResult(uint32_t config_idx) const {
  return {1, buildTimes.at(config_idx)};
}

const char *RayASBuildBench::GetName() const { return "RayASBuild"; }
const char *RayASBuildBench::GetComponent(uint32_t config_idx) const {
  return "Ray Tracing";
}
const char *RayASBuildBench::GetMetric() const { return "ms/op"; }
const char *RayASBuildBench::GetSubCategory(uint32_t config_idx) const {
  return "AS Build Performance";
}

std::string RayASBuildBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx == 0) return "BLAS Build (1M Tris)";
  if (config_idx == 1) return "TLAS Build (10K Inst)";
  return "BLAS Update (1M Tris)";
}
