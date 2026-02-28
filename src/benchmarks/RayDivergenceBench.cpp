#include "RayDivergenceBench.h"
#include "core/VulkanContext.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>

bool RayDivergenceBench::IsSupported(const DeviceInfo &info,
                                     IComputeContext *context) const {
  return info.rayTracingSupport &&
         (context && context->getBackend() == ComputeBackend::Vulkan);
}

void RayDivergenceBench::loadRTProcs(VkDevice device) {
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

void RayDivergenceBench::Setup(IComputeContext &context,
                               const std::string &kernel_dir) {
  this->context = &context;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
  if (!vContext)
    throw std::runtime_error("RayDivergenceBench requires VulkanContext");

  loadRTProcs(vContext->getVulkanDevice());

  // Target a substantial workload to saturate RTUs
  rayCount = 4000000;
  resultBuffer = context.createBuffer(sizeof(uint32_t));
  uint32_t zero = 0;
  context.writeBuffer(resultBuffer, 0, 4, &zero);

  // Setup a high-resolution flat floor plane (Z=0) and ceiling plane (Z=-20)
  uint32_t gridSize = 256;
  uint32_t primitivesPerPlane = gridSize * gridSize * 2;
  numPrimitives = primitivesPerPlane * 2; // Floor + Ceiling

  std::vector<float> vertices;
  vertices.reserve(numPrimitives * 9);

  auto addPlane = [&](float z) {
    float scale = 200.0f / gridSize;
    for (uint32_t y = 0; y < gridSize; ++y) {
      for (uint32_t x = 0; x < gridSize; ++x) {
        float fx0 = (float)x * scale - 100.0f;
        float fy0 = (float)y * scale - 100.0f;
        float fx1 = (float)(x + 1) * scale - 100.0f;
        float fy1 = (float)(y + 1) * scale - 100.0f;

        // Triangle 1
        vertices.push_back(fx0);
        vertices.push_back(fy0);
        vertices.push_back(z);
        vertices.push_back(fx1);
        vertices.push_back(fy0);
        vertices.push_back(z);
        vertices.push_back(fx0);
        vertices.push_back(fy1);
        vertices.push_back(z);
        // Triangle 2
        vertices.push_back(fx1);
        vertices.push_back(fy0);
        vertices.push_back(z);
        vertices.push_back(fx1);
        vertices.push_back(fy1);
        vertices.push_back(z);
        vertices.push_back(fx0);
        vertices.push_back(fy1);
        vertices.push_back(z);
      }
    }
  };

  addPlane(0.0f);   // Floor
  addPlane(-20.0f); // Ceiling

  // Export the generated geometry to an OBJ file for visualization in Blender
  std::ofstream objFile("raydiv_scene.obj");
  for (size_t i = 0; i < vertices.size(); i += 3) {
    objFile << "v " << vertices[i] << " " << vertices[i + 1] << " "
            << vertices[i + 2] << "\n";
  }
  for (size_t i = 1; i <= vertices.size() / 3; i += 3) {
    objFile << "f " << i << " " << i + 1 << " " << i + 2 << "\n";
  }
  objFile.close();

  vertexBuffer =
      context.createBuffer(vertices.size() * sizeof(float), vertices.data());

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  std::vector<std::string> hit_shaders = {
      (kdir / "vulkan" / "raydiv_pipeline_a.rchit").string(),
      (kdir / "vulkan" / "raydiv_pipeline_b.rchit").string(),
      (kdir / "vulkan" / "raydiv_pipeline_c.rchit").string(),
      (kdir / "vulkan" / "raydiv_pipeline_d.rchit").string()};

  try {
    VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
    if (vContext) {
      kernel = vContext->createRTPipeline(
          (kdir / "vulkan" / "raydiv_pipeline.rgen").string(),
          (kdir / "vulkan" / "raydiv_pipeline.rmiss").string(), hit_shaders,
          2); // 2 buffer descriptors (TLAS + Result Buffer)
    } else {
      std::cerr
          << "RayDivergenceBench currently only supports Vulkan RT backend."
          << std::endl;
      kernel = nullptr;
    }
  } catch (const std::exception &e) {
    std::cerr << "RT Pipeline creation failed: " << e.what() << std::endl;
    kernel = nullptr;
  }
}

void RayDivergenceBench::buildAS() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  VkDeviceAddress vAddr = vContext->getBufferDeviceAddress(vertexBuffer);

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

  VkAccelerationStructureDeviceAddressInfoKHR triAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  triAddrInfo.accelerationStructure = triangleBlas;
  VkDeviceAddress triASAddr =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &triAddrInfo);

  // Instance for Triangles
  VkAccelerationStructureInstanceKHR triInstance = {};
  triInstance.transform = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  triInstance.instanceCustomIndex = 0;
  triInstance.mask = 0xFF;
  triInstance.accelerationStructureReference = triASAddr;
  triInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

  instanceBuffer =
      context->createBuffer(sizeof(VkAccelerationStructureInstanceKHR));
  context->writeBuffer(instanceBuffer, 0, sizeof(triInstance), &triInstance);

  VkAccelerationStructureGeometryKHR topTriGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  topTriGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  topTriGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  topTriGeom.geometry.instances.data.deviceAddress =
      vContext->getBufferDeviceAddress(instanceBuffer);

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

  // Scratch buffer
  size_t scratchSize = std::max(triSizes.buildScratchSize, triScratch);
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

  VkAccelerationStructureBuildGeometryInfoKHR tlasTriBuild{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tlasTriBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tlasTriBuild.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tlasTriBuild.geometryCount = 1;
  tlasTriBuild.pGeometries = &topTriGeom;
  tlasTriBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  cmdBuild(tlasTriBuild, triangleTlas, 1);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, tmpPool, nullptr);
}

void RayDivergenceBench::Run(uint32_t config_idx) {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkAccelerationStructureKHR activeTlas = triangleTlas;

  vContext->setKernelAS(kernel, 0, (AccelerationStructure)activeTlas);
  vContext->setKernelArg(kernel, 1, resultBuffer);

  // config_idx 0 = 100% coherence. config_idx 4 = 0% coherence.
  float coherenceFactor = 1.0f - (float(config_idx) * 0.25f);
  uint32_t seed = config_idx * 1337;

  // Push Constants: rayCount, coherenceFactor, seed
  vContext->setKernelArg(kernel, 2, sizeof(uint32_t), &rayCount);
  vContext->setKernelArg(kernel, 3, sizeof(float), &coherenceFactor);
  vContext->setKernelArg(kernel, 4, sizeof(uint32_t), &seed);

  auto start = std::chrono::high_resolution_clock::now();
  vContext->dispatch(kernel, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
  context->waitIdle();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = end - start;
  rtResults[config_idx] = diff.count();
}

void RayDivergenceBench::Teardown() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();

  if (triangleBlas)
    vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
  if (triangleTlas)
    vkDestroyAccelerationStructureKHR_ptr(device, triangleTlas, nullptr);

  if (kernel)
    context->releaseKernel(kernel);
  if (resultBuffer)
    context->releaseBuffer(resultBuffer);
  if (vertexBuffer)
    context->releaseBuffer(vertexBuffer);
  if (instanceBuffer)
    context->releaseBuffer(instanceBuffer);
  if (triangleBlasBuffer)
    context->releaseBuffer(triangleBlasBuffer);
  if (triangleTlasBuffer)
    context->releaseBuffer(triangleTlasBuffer);
  if (scratchBuffer)
    context->releaseBuffer(scratchBuffer);
}

BenchmarkResult RayDivergenceBench::GetResult(uint32_t config_idx) const {
  return {(uint64_t)rayCount, rtResults[config_idx]};
}

const char *RayDivergenceBench::GetName() const { return "RayDivergence"; }
const char *RayDivergenceBench::GetComponent(uint32_t config_idx) const {
  return "Ray Tracing";
}
const char *RayDivergenceBench::GetMetric() const { return "GRays/s"; }
const char *RayDivergenceBench::GetSubCategory(uint32_t config_idx) const {
  return "Material Divergence";
}

std::string RayDivergenceBench::GetConfigName(uint32_t config_idx) const {
  int coherencePercentage = 100 - (config_idx * 25);
  std::string label = std::to_string(coherencePercentage) + "% Coherence";

  if (coherencePercentage == 100)
    label += " (Perfect Mirror)";
  else if (coherencePercentage == 50)
    label += " (Half Diffuse)";
  else if (coherencePercentage == 0)
    label += " (Perfectly Diffuse)";

  return label;
}
