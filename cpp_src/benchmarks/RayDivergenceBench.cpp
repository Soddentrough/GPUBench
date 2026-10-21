#include "RayDivergenceBench.h"
#include "core/VulkanContext.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

bool RayDivergenceBench::IsSupported(const DeviceInfo &info,
                                     IComputeContext *context) const {
  if (!info.rayTracingSupport) return false;
  if (!context) return true;
  if (context->getBackend() != ComputeBackend::Vulkan) return false;
  const VulkanContext *vc = dynamic_cast<const VulkanContext *>(context);
  if (vc && !vc->isExtensionEnabled(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
    return false;
  }
  return true;
}

std::string RayDivergenceBench::GetSupportNote() const {
  return "Ray tracing requires Vulkan ray tracing pipeline extensions";
}

std::string RayDivergenceBench::GetSupportNote(const DeviceInfo &info,
                                               IComputeContext *context) const {
  if (context && context->getBackend() == ComputeBackend::OpenCL) {
    return "No support for ray tracing pipelines in OpenCL API";
  }
  if (context && context->getBackend() == ComputeBackend::ROCm) {
    return "No support for ray tracing pipelines in ROCm/HIP API";
  }
  if (!info.rayTracingSupport) {
    return "VK_KHR_acceleration_structure or VK_KHR_ray_query missing";
  }
  if (context) {
    const VulkanContext *vc = dynamic_cast<const VulkanContext *>(context);
    if (vc && !vc->isExtensionEnabled(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
      return "VK_KHR_ray_tracing_pipeline missing";
    }
  }
  return "Ray tracing requires Vulkan ray tracing pipeline extensions";
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

  // Setup enclosed chamber geometry with 3D relief floor, ceiling, and boundary walls
  std::vector<float> vertices;
  generateGeometry(vertices);
  numPrimitives = static_cast<uint32_t>(vertices.size() / 9);

  vertexBuffer =
      context.createBuffer(vertices.size() * sizeof(float), vertices.data());

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  std::vector<std::string> hit_shaders = {
      (kdir / "vulkan" / "raydiv_pipeline_a.rchit").string()};

  try {
    VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context);
    if (vContext) {
      kernel = vContext->createRTPipeline(
          (kdir / "vulkan" / "raydiv_pipeline.rgen").string(),
          (kdir / "vulkan" / "raydiv_pipeline.rmiss").string(), hit_shaders,
          {}, // empty rahit_paths
          {}, // empty rint_paths
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
  if (vkCreateAccelerationStructureKHR_ptr(device, &triCreateInfo, nullptr,
                                           &triangleBlas) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create triangle BLAS in RayDivergenceBench");
  }

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
    if (vkCreateAccelerationStructureKHR_ptr(device, &createInfo, nullptr,
                                             &tlasHandle) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create TLAS in RayDivergenceBench");
    }
    return sizes.buildScratchSize;
  };

  size_t triScratch = createTLAS(topTriGeom, triangleTlas, triangleTlasBuffer);

  // Scratch buffer
  size_t scratchSize = std::max(static_cast<size_t>(triSizes.buildScratchSize), triScratch);
  scratchBuffer = context->createBuffer(scratchSize);
  VkDeviceAddress sAddr = vContext->getBufferDeviceAddress(scratchBuffer);

  // Build commands
  VkCommandPoolCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpInfo.queueFamilyIndex = vContext->getComputeQueueFamilyIndex();
  VkCommandPool tmpPool;
  if (vkCreateCommandPool(device, &cpInfo, nullptr, &tmpPool) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool in RayDivergenceBench");
  }

  VkCommandBufferAllocateInfo cbAlloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbAlloc.commandPool = tmpPool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  if (vkAllocateCommandBuffers(device, &cbAlloc, &cmd) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to allocate command buffer in RayDivergenceBench");
  }

  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to begin command buffer in RayDivergenceBench");
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

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to end command buffer in RayDivergenceBench");
  }

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  if (vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
    vkDestroyCommandPool(device, tmpPool, nullptr);
    throw std::runtime_error("Failed to submit AS build in RayDivergenceBench");
  }
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, tmpPool, nullptr);
}

void RayDivergenceBench::Run(uint32_t config_idx) {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  if (lastRunConfig != static_cast<int>(config_idx)) {
    uint32_t zero = 0;
    context->writeBuffer(resultBuffer, 0, sizeof(uint32_t), &zero);
    lastRunConfig = static_cast<int>(config_idx);
  }
  VkAccelerationStructureKHR activeTlas = triangleTlas;

  vContext->setKernelAS(kernel, 0, (AccelerationStructure)activeTlas);
  vContext->setKernelArg(kernel, 1, resultBuffer);

  // config_idx 0 = 0% coherence (100% diffuse baseline). config_idx 4 = 100% coherence (0° beam).
  float coherenceFactor = float(config_idx) * 0.25f;
  uint32_t seed = config_idx * 1337;

  // Push Constants: rayCount, coherenceFactor, seed
  vContext->setKernelArg(kernel, 2, sizeof(uint32_t), &rayCount);
  vContext->setKernelArg(kernel, 3, sizeof(float), &coherenceFactor);
  vContext->setKernelArg(kernel, 4, sizeof(uint32_t), &seed);

  vContext->dispatch(kernel, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
}

void RayDivergenceBench::Teardown() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext ? vContext->getVulkanDevice() : VK_NULL_HANDLE;

  if (device != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_ptr) {
    if (triangleBlas != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
      triangleBlas = VK_NULL_HANDLE;
    }
    if (triangleTlas != VK_NULL_HANDLE) {
      vkDestroyAccelerationStructureKHR_ptr(device, triangleTlas, nullptr);
      triangleTlas = VK_NULL_HANDLE;
    }
  }

  if (context) {
    if (kernel) { context->releaseKernel(kernel); kernel = nullptr; }
    if (resultBuffer) { context->releaseBuffer(resultBuffer); resultBuffer = nullptr; }
    if (vertexBuffer) { context->releaseBuffer(vertexBuffer); vertexBuffer = nullptr; }
    if (instanceBuffer) { context->releaseBuffer(instanceBuffer); instanceBuffer = nullptr; }
    if (triangleBlasBuffer) { context->releaseBuffer(triangleBlasBuffer); triangleBlasBuffer = nullptr; }
    if (triangleTlasBuffer) { context->releaseBuffer(triangleTlasBuffer); triangleTlasBuffer = nullptr; }
    if (scratchBuffer) { context->releaseBuffer(scratchBuffer); scratchBuffer = nullptr; }
    context = nullptr;
  }
  lastRunConfig = -1;
}

BenchmarkResult RayDivergenceBench::GetResult(uint32_t config_idx) const {
  return {(uint64_t)rayCount, 0.0};
}

const char *RayDivergenceBench::GetName() const { return "RayDivergence"; }
const char *RayDivergenceBench::GetComponent(uint32_t config_idx) const {
  return "Ray Tracing";
}
const char *RayDivergenceBench::GetMetric() const { return "MRays/s"; }
const char *RayDivergenceBench::GetSubCategory(uint32_t config_idx) const {
  return "Ray Directional Coherence";
}

std::string RayDivergenceBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Bounce Divergence: Rough Diffuse (r=1.0, 90° Spread)";
  case 1:
    return "Bounce Divergence: Matte Satin (r=0.75, 67.5° Spread)";
  case 2:
    return "Bounce Divergence: Brushed Metal (r=0.5, 45° Spread)";
  case 3:
    return "Bounce Divergence: Glossy Coat (r=0.25, 22.5° Spread)";
  case 4:
    return "Bounce Divergence: Mirror Reflection (r=0.0, 0° Coherent)";
  default:
    return "Bounce Divergence: Level " + std::to_string(config_idx);
  }
}

void RayDivergenceBench::generateGeometry(std::vector<float> &vertices) const {
  // Uniformly tessellated 6-face enclosing chamber [-100, 100]^3 (160x160 quads = 51,200 tris per face, 307,200 tris total).
  // Every direction in the hemisphere encounters identical triangle density and BVH tree depth,
  // avoiding grazing-angle multi-leaf overlap traps or early ray misses.
  const uint32_t gridSize = 160;
  const uint32_t trisPerFace = gridSize * gridSize * 2;
  vertices.reserve(trisPerFace * 6 * 9);

  const float halfExtent = 100.0f;
  const float step = (2.0f * halfExtent) / static_cast<float>(gridSize);

  auto pushTri = [&](float x0, float y0, float z0,
                     float x1, float y1, float z1,
                     float x2, float y2, float z2) {
    vertices.push_back(x0); vertices.push_back(y0); vertices.push_back(z0);
    vertices.push_back(x1); vertices.push_back(y1); vertices.push_back(z1);
    vertices.push_back(x2); vertices.push_back(y2); vertices.push_back(z2);
  };

  for (uint32_t iv = 0; iv < gridSize; ++iv) {
    float v0 = -halfExtent + static_cast<float>(iv) * step;
    float v1 = v0 + step;
    for (uint32_t iu = 0; iu < gridSize; ++iu) {
      float u0 = -halfExtent + static_cast<float>(iu) * step;
      float u1 = u0 + step;

      // 1. Top Dome / Ceiling (+Z = +100)
      pushTri(u0, v0,  halfExtent, u1, v0,  halfExtent, u0, v1,  halfExtent);
      pushTri(u1, v0,  halfExtent, u1, v1,  halfExtent, u0, v1,  halfExtent);

      // 2. Floor (-Z = -100)
      pushTri(u0, v0, -halfExtent, u0, v1, -halfExtent, u1, v0, -halfExtent);
      pushTri(u1, v0, -halfExtent, u0, v1, -halfExtent, u1, v1, -halfExtent);

      // 3. East Wall (+X = +100)
      pushTri( halfExtent, u0, v0,  halfExtent, u1, v0,  halfExtent, u0, v1);
      pushTri( halfExtent, u1, v0,  halfExtent, u1, v1,  halfExtent, u0, v1);

      // 4. West Wall (-X = -100)
      pushTri(-halfExtent, u0, v0, -halfExtent, u0, v1, -halfExtent, u1, v0);
      pushTri(-halfExtent, u1, v0, -halfExtent, u0, v1, -halfExtent, u1, v1);

      // 5. North Wall (+Y = +100)
      pushTri(u0,  halfExtent, v0, u0,  halfExtent, v1, u1,  halfExtent, v0);
      pushTri(u1,  halfExtent, v0, u0,  halfExtent, v1, u1,  halfExtent, v1);

      // 6. South Wall (-Y = -100)
      pushTri(u0, -halfExtent, v0, u1, -halfExtent, v0, u0, -halfExtent, v1);
      pushTri(u1, -halfExtent, v0, u1, -halfExtent, v1, u0, -halfExtent, v1);
    }
  }
}

void RayDivergenceBench::DumpGeometry() const {
  std::vector<float> vertices;
  generateGeometry(vertices);

  std::ofstream objFile("raydiv_scene.obj");
  for (size_t i = 0; i < vertices.size(); i += 3) {
    objFile << "v " << vertices[i] << " " << vertices[i + 1] << " "
            << vertices[i + 2] << "\n";
  }
  for (size_t i = 1; i <= vertices.size() / 3; i += 3) {
    objFile << "f " << i << " " << i + 1 << " " << i + 2 << "\n";
  }
  objFile.close();
  std::cout << "Geometry dumped to raydiv_scene.obj" << std::endl;
}

bool RayDivergenceBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= GetNumConfigs()) {
    return false;
  }
  if (!context || !resultBuffer) {
    return false;
  }
  if (lastRunConfig != static_cast<int>(config_idx)) {
    return false;
  }
  try {
    uint32_t hits = 0;
    context->readBuffer(resultBuffer, 0, sizeof(uint32_t), &hits);
    // Chamber enclosure guarantees 100% of rays intersect geometry;
    // hits must be non-zero after any valid execution.
    return hits > 0;
  } catch (...) {
    return false;
  }
}
