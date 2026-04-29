#include "RayMaterialDivergenceBench.h"
#include "core/VulkanContext.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

bool RayMaterialDivergenceBench::IsSupported(const DeviceInfo &info,
                                     IComputeContext *context) const {
  return info.rayTracingSupport &&
         (context && context->getBackend() == ComputeBackend::Vulkan);
}

void RayMaterialDivergenceBench::loadRTProcs(VkDevice device) {
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

void RayMaterialDivergenceBench::Setup(IComputeContext &context,
                               const std::string &kernel_dir) {
  this->context = &context;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
  if (!vContext)
    throw std::runtime_error("RayMaterialDivergenceBench requires VulkanContext");

  loadRTProcs(vContext->getVulkanDevice());

  rayCount = 4000000;
  resultBuffer = context.createBuffer(sizeof(uint32_t));
  uint32_t zero = 0;
  context.writeBuffer(resultBuffer, 0, 4, &zero);

  numPrimitives = 12; // A simple cube or low poly shape
  std::vector<float> vertices;
  vertices.reserve(numPrimitives * 9);
  
  srand(1337);
  for (uint32_t i = 0; i < numPrimitives; ++i) {
    for (int j = 0; j < 9; ++j) {
      vertices.push_back((float(rand()) / RAND_MAX) * 0.8f - 0.4f);
    }
  }

  vertexBuffer = context.createBuffer(vertices.size() * sizeof(float), vertices.data());

  std::filesystem::path kdir(kernel_dir);
  
  if (vContext) {
    std::vector<std::string> hits = {
        (kdir / "vulkan" / "raymatdiv_mat0.rchit").string(),
        (kdir / "vulkan" / "raymatdiv_mat1.rchit").string(),
        (kdir / "vulkan" / "raymatdiv_mat2.rchit").string(),
        (kdir / "vulkan" / "raymatdiv_mat3.rchit").string()
    };
    
    kernel = vContext->createRTPipeline(
        (kdir / "vulkan" / "raymatdiv.rgen").string(),
        (kdir / "vulkan" / "raymatdiv.rmiss").string(), hits, {}, {}, 2);
  }
}

void RayMaterialDivergenceBench::buildAS(uint32_t config_idx) {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  if (triangleBlas) vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
  if (triangleTlas) vkDestroyAccelerationStructureKHR_ptr(device, triangleTlas, nullptr);
  if (triangleBlasBuffer) context->releaseBuffer(triangleBlasBuffer);
  if (triangleTlasBuffer) context->releaseBuffer(triangleTlasBuffer);
  if (instanceBuffer) context->releaseBuffer(instanceBuffer);
  if (scratchBuffer) context->releaseBuffer(scratchBuffer);

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

  VkAccelerationStructureDeviceAddressInfoKHR triAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  triAddrInfo.accelerationStructure = triangleBlas;
  VkDeviceAddress triASAddr =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &triAddrInfo);

  numInstances = 40000;
  std::vector<VkAccelerationStructureInstanceKHR> instances(numInstances);
  
  uint32_t gridWidth = sqrt(numInstances);
  float spacing = 200.0f / gridWidth;

  srand(1337);
  for(uint32_t i=0; i<numInstances; ++i) {
      instances[i].transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
      
      // Arrange in a 2D grid across X and Y, with a fixed Z
      uint32_t ix = i % gridWidth;
      uint32_t iy = i / gridWidth;
      
      instances[i].transform.matrix[0][3] = (ix * spacing) - 100.0f;
      instances[i].transform.matrix[1][3] = (iy * spacing) - 100.0f;
      instances[i].transform.matrix[2][3] = 0.0f; // Fixed depth
      
      instances[i].instanceCustomIndex = i;
      instances[i].mask = 0xFF;
      
      // Coherent assigns material 0, divergent assigns random material 0-3
      uint32_t material_idx = (config_idx == 0) ? 0 : (rand() % 4);
      instances[i].instanceShaderBindingTableRecordOffset = material_idx;
      
      instances[i].accelerationStructureReference = triASAddr;
      instances[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  }

  instanceBuffer = context->createBuffer(instances.size() * sizeof(VkAccelerationStructureInstanceKHR), instances.data());

  VkAccelerationStructureGeometryKHR topTriGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  topTriGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  topTriGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  topTriGeom.geometry.instances.data.deviceAddress =
      vContext->getBufferDeviceAddress(instanceBuffer);

  VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tlasBuildInfo.geometryCount = 1;
  tlasBuildInfo.pGeometries = &topTriGeom;
  tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t tlasMaxPrimCount = numInstances;
  VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo,
      &tlasMaxPrimCount, &tlasSizes);

  triangleTlasBuffer = context->createBuffer(tlasSizes.accelerationStructureSize);
  VkAccelerationStructureCreateInfoKHR tlasCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  tlasCreateInfo.buffer = vContext->getVkBuffer(triangleTlasBuffer);
  tlasCreateInfo.size = tlasSizes.accelerationStructureSize;
  tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &tlasCreateInfo, nullptr,
                                       &triangleTlas);

  size_t scratchSize = std::max(triSizes.buildScratchSize, tlasSizes.buildScratchSize);
  scratchBuffer = context->createBuffer(scratchSize);
  VkDeviceAddress sAddr = vContext->getBufferDeviceAddress(scratchBuffer);

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
  cmdBuild(tlasBuildInfo, triangleTlas, numInstances);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, tmpPool, nullptr);
}

void RayMaterialDivergenceBench::Run(uint32_t config_idx) {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  
  buildAS(config_idx);
  
  vContext->setKernelAS(kernel, 0, (AccelerationStructure)triangleTlas);
  vContext->setKernelArg(kernel, 1, resultBuffer);
  vContext->setKernelArg(kernel, 2, sizeof(uint32_t), &rayCount);

  auto start = std::chrono::high_resolution_clock::now();
  vContext->dispatch(kernel, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
  context->waitIdle();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  rtResults[config_idx] = diff.count();
}

void RayMaterialDivergenceBench::Teardown() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();

  if (triangleBlas) vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
  if (triangleTlas) vkDestroyAccelerationStructureKHR_ptr(device, triangleTlas, nullptr);

  if (kernel) context->releaseKernel(kernel);
  if (resultBuffer) context->releaseBuffer(resultBuffer);
  if (vertexBuffer) context->releaseBuffer(vertexBuffer);
  if (instanceBuffer) context->releaseBuffer(instanceBuffer);
  if (triangleBlasBuffer) context->releaseBuffer(triangleBlasBuffer);
  if (triangleTlasBuffer) context->releaseBuffer(triangleTlasBuffer);
  if (scratchBuffer) context->releaseBuffer(scratchBuffer);
}

BenchmarkResult RayMaterialDivergenceBench::GetResult(uint32_t config_idx) const {
  return {(uint64_t)rayCount, rtResults.at(config_idx)};
}

const char *RayMaterialDivergenceBench::GetName() const { return "RayMaterialDivergence"; }
const char *RayMaterialDivergenceBench::GetComponent(uint32_t config_idx) const {
  return "Ray Tracing";
}
const char *RayMaterialDivergenceBench::GetMetric() const { return "GRays/s"; }
const char *RayMaterialDivergenceBench::GetSubCategory(uint32_t config_idx) const {
  return "Material Divergence";
}

std::string RayMaterialDivergenceBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx == 0) return "Coherent Material (1 Shader)";
  return "Divergent Material (4 Shaders)";
}
