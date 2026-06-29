#include "RayProceduralBench.h"
#include "core/VulkanContext.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>

bool RayProceduralBench::IsSupported(const DeviceInfo &info,
                                     IComputeContext *context) const {
  return info.rayTracingSupport &&
         (context && context->getBackend() == ComputeBackend::Vulkan);
}

void RayProceduralBench::loadRTProcs(VkDevice device) {
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

void RayProceduralBench::Setup(IComputeContext &context,
                               const std::string &kernel_dir) {
  this->context = &context;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
  if (!vContext)
    throw std::runtime_error("RayProceduralBench requires VulkanContext");

  loadRTProcs(vContext->getVulkanDevice());

  rayCount = 4000000;
  resultBuffer = context.createBuffer(sizeof(uint32_t));
  uint32_t zero = 0;
  context.writeBuffer(resultBuffer, 0, 4, &zero);

  numPrimitives = 200000; 
  std::vector<VkAabbPositionsKHR> aabbs;
  aabbs.reserve(numPrimitives);
  std::vector<float> spheres;
  spheres.reserve(numPrimitives * 4);

  srand(1337);
  for (uint32_t i = 0; i < numPrimitives; ++i) {
    float x = (float(rand()) / RAND_MAX) * 200.0f - 100.0f;
    float y = (float(rand()) / RAND_MAX) * 200.0f - 100.0f;
    float z = (float(rand()) / RAND_MAX) * 200.0f - 100.0f;
    float r = 2.0f; 

    aabbs.push_back({x - r, y - r, z - r, x + r, y + r, z + r});
    spheres.push_back(x);
    spheres.push_back(y);
    spheres.push_back(z);
    spheres.push_back(r);
  }

  aabbBuffer =
      context.createBuffer(aabbs.size() * sizeof(VkAabbPositionsKHR), aabbs.data());
  sphereBuffer =
      context.createBuffer(spheres.size() * sizeof(float), spheres.data());

  buildAS();


  std::filesystem::path kdir(kernel_dir);
  
  if (vContext) {
    std::vector<std::string> hit = {(kdir / "vulkan" / "rayprocedural.rchit").string()};
    std::vector<std::string> rint = {(kdir / "vulkan" / "rayprocedural.rint").string()};
    
    kernel = vContext->createRTPipeline(
        (kdir / "vulkan" / "rayprocedural.rgen").string(),
        (kdir / "vulkan" / "rayprocedural.rmiss").string(), hit, {}, rint, 3);

  }
}

void RayProceduralBench::buildAS() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  VkDeviceAddress aAddr = vContext->getBufferDeviceAddress(aabbBuffer);

  VkAccelerationStructureGeometryKHR aabbGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  aabbGeom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  aabbGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR; 
  aabbGeom.geometry.aabbs.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
  aabbGeom.geometry.aabbs.data.deviceAddress = aAddr;
  aabbGeom.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);

  VkAccelerationStructureBuildGeometryInfoKHR blasBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  blasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  blasBuildInfo.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  blasBuildInfo.geometryCount = 1;
  blasBuildInfo.pGeometries = &aabbGeom;
  blasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t blasMaxPrimCount = numPrimitives;
  VkAccelerationStructureBuildSizesInfoKHR blasSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuildInfo,
      &blasMaxPrimCount, &blasSizes);

  aabbBlasBuffer =
      context->createBuffer(blasSizes.accelerationStructureSize);
  VkAccelerationStructureCreateInfoKHR blasCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  blasCreateInfo.buffer = vContext->getVkBuffer(aabbBlasBuffer);
  blasCreateInfo.size = blasSizes.accelerationStructureSize;
  blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &blasCreateInfo, nullptr,
                                       &aabbBlas);

  VkAccelerationStructureDeviceAddressInfoKHR blasAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  blasAddrInfo.accelerationStructure = aabbBlas;
  VkDeviceAddress blasASAddr =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &blasAddrInfo);

  VkAccelerationStructureInstanceKHR instance = {};
  instance.transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  instance.instanceCustomIndex = 0;
  instance.mask = 0xFF;
  instance.accelerationStructureReference = blasASAddr;
  instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

  instanceBuffer =
      context->createBuffer(sizeof(VkAccelerationStructureInstanceKHR));
  context->writeBuffer(instanceBuffer, 0, sizeof(instance), &instance);

  VkAccelerationStructureGeometryKHR topGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  topGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  topGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  topGeom.geometry.instances.data.deviceAddress =
      vContext->getBufferDeviceAddress(instanceBuffer);

  VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tlasBuildInfo.geometryCount = 1;
  tlasBuildInfo.pGeometries = &topGeom;
  tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t tlasMaxPrimCount = 1;
  VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo,
      &tlasMaxPrimCount, &tlasSizes);

  aabbTlasBuffer = context->createBuffer(tlasSizes.accelerationStructureSize);
  VkAccelerationStructureCreateInfoKHR tlasCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  tlasCreateInfo.buffer = vContext->getVkBuffer(aabbTlasBuffer);
  tlasCreateInfo.size = tlasSizes.accelerationStructureSize;
  tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &tlasCreateInfo, nullptr,
                                       &aabbTlas);

  size_t scratchSize = std::max(blasSizes.buildScratchSize, tlasSizes.buildScratchSize);
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

  cmdBuild(blasBuildInfo, aabbBlas, numPrimitives);
  cmdBuild(tlasBuildInfo, aabbTlas, 1);



  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);



  vkDestroyCommandPool(device, tmpPool, nullptr);
}

void RayProceduralBench::Run(uint32_t config_idx) {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  
  vContext->setKernelAS(kernel, 0, (AccelerationStructure)aabbTlas);
  vContext->setKernelArg(kernel, 1, resultBuffer);
  vContext->setKernelArg(kernel, 2, sphereBuffer);

  vContext->setKernelArg(kernel, 3, sizeof(uint32_t), &rayCount);

  auto start = std::chrono::high_resolution_clock::now();
  vContext->dispatch(kernel, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
  context->waitIdle();
  auto end = std::chrono::high_resolution_clock::now();


  std::chrono::duration<double> diff = end - start;
  rtResults[config_idx] = diff.count();
}

void RayProceduralBench::Teardown() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();

  if (aabbBlas) vkDestroyAccelerationStructureKHR_ptr(device, aabbBlas, nullptr);
  if (aabbTlas) vkDestroyAccelerationStructureKHR_ptr(device, aabbTlas, nullptr);

  if (kernel) context->releaseKernel(kernel);
  if (resultBuffer) context->releaseBuffer(resultBuffer);
  if (aabbBuffer) context->releaseBuffer(aabbBuffer);
  if (sphereBuffer) context->releaseBuffer(sphereBuffer);
  if (instanceBuffer) context->releaseBuffer(instanceBuffer);
  if (aabbBlasBuffer) context->releaseBuffer(aabbBlasBuffer);
  if (aabbTlasBuffer) context->releaseBuffer(aabbTlasBuffer);
  if (scratchBuffer) context->releaseBuffer(scratchBuffer);
}

BenchmarkResult RayProceduralBench::GetResult(uint32_t config_idx) const {
  return {(uint64_t)rayCount, rtResults.at(config_idx)};
}

const char *RayProceduralBench::GetName() const { return "RayProcedural"; }
const char *RayProceduralBench::GetComponent(uint32_t config_idx) const {
  return "Ray Tracing";
}
const char *RayProceduralBench::GetMetric() const { return "GRays/s"; }
const char *RayProceduralBench::GetSubCategory(uint32_t config_idx) const {
  return "Procedural Intersection";
}

std::string RayProceduralBench::GetConfigName(uint32_t config_idx) const {
  return "Spheres (AABB)";
}
