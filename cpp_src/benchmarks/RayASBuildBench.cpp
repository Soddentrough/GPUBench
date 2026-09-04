#include "RayASBuildBench.h"
#include "core/VulkanContext.h"
#include <chrono>
#include <cmath>
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

void RayASBuildBench::Setup(IComputeContext &context_ref,
                            const std::string &kernel_dir) {
  this->context = &context_ref;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context_ref);
  if (!vContext)
    throw std::runtime_error("RayASBuildBench requires VulkanContext");

  VkDevice device = vContext->getVulkanDevice();
  loadRTProcs(device);

  // 1. Allocate primary vertex pool (10M triangles max = 360 MB)
  uint32_t maxPrimitives = 10000000;
  std::vector<float> vertices(maxPrimitives * 9, 0.0f);
  for (uint32_t i = 0; i < vertices.size(); ++i) {
    vertices[i] = (float(rand()) / RAND_MAX);
  }
  vertexBuffer =
      context_ref.createBuffer(vertices.size() * sizeof(float), vertices.data());

  VkDeviceAddress vAddr = vContext->getBufferDeviceAddress(vertexBuffer);

  // 2. Setup 3 standalone BLAS tiers (1M, 5M, 10M triangles)
  std::vector<uint32_t> primCounts = {1000000, 5000000, 10000000};
  blases.resize(primCounts.size());
  size_t maxBuildScratch = 0;
  size_t maxUpdateScratch = 0;

  for (size_t b = 0; b < primCounts.size(); ++b) {
    blases[b].numPrimitives = primCounts[b];
    VkAccelerationStructureGeometryKHR triGeom{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    triGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    triGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    triGeom.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triGeom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triGeom.geometry.triangles.vertexData.deviceAddress = vAddr;
    triGeom.geometry.triangles.vertexStride = sizeof(float) * 3;
    triGeom.geometry.triangles.maxVertex = primCounts[b] * 3 - 1;
    triGeom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR blasBuildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    blasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    blasBuildInfo.geometryCount = 1;
    blasBuildInfo.pGeometries = &triGeom;
    blasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    uint32_t maxPrimCount = primCounts[b];
    blases[b].sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR_ptr(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasBuildInfo,
        &maxPrimCount, &blases[b].sizes);

    blases[b].buffer = context_ref.createBuffer(blases[b].sizes.accelerationStructureSize);
    VkAccelerationStructureCreateInfoKHR blasCreateInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    blasCreateInfo.buffer = vContext->getVkBuffer(blases[b].buffer);
    blasCreateInfo.size = blases[b].sizes.accelerationStructureSize;
    blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vkCreateAccelerationStructureKHR_ptr(device, &blasCreateInfo, nullptr, &blases[b].handle);

    maxBuildScratch = std::max(maxBuildScratch, (size_t)blases[b].sizes.buildScratchSize);
    maxUpdateScratch = std::max(maxUpdateScratch, (size_t)blases[b].sizes.updateScratchSize);
  }

  // 3. Setup Multi-BLAS Library (5,000 distinct geometries for branched TLAS)
  uint32_t numBlasLib = 5000;
  blasLibBuffers.resize(numBlasLib);
  blasLibHandles.resize(numBlasLib);
  blasLibAddrs.resize(numBlasLib);

  std::vector<uint32_t> libPrims(numBlasLib);
  std::vector<VkAccelerationStructureBuildSizesInfoKHR> libSizes(numBlasLib);

  for (uint32_t b = 0; b < numBlasLib; ++b) {
    libPrims[b] = 50 + (b % 15) * 100;
    VkAccelerationStructureGeometryKHR geom{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geom.geometry.triangles.vertexData.deviceAddress =
        vAddr + (b * 1000 * 9 * sizeof(float));
    geom.geometry.triangles.vertexStride = sizeof(float) * 3;
    geom.geometry.triangles.maxVertex = libPrims[b] * 3 - 1;
    geom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR bInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    bInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bInfo.geometryCount = 1;
    bInfo.pGeometries = &geom;

    libSizes[b].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR_ptr(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bInfo,
        &libPrims[b], &libSizes[b]);

    blasLibBuffers[b] = context_ref.createBuffer(libSizes[b].accelerationStructureSize);
    VkAccelerationStructureCreateInfoKHR cInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    cInfo.buffer = vContext->getVkBuffer(blasLibBuffers[b]);
    cInfo.offset = 0;
    cInfo.size = libSizes[b].accelerationStructureSize;
    cInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vkCreateAccelerationStructureKHR_ptr(device, &cInfo, nullptr, &blasLibHandles[b]);

    VkAccelerationStructureDeviceAddressInfoKHR dInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    dInfo.accelerationStructure = blasLibHandles[b];
    blasLibAddrs[b] = vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &dInfo);

    maxBuildScratch = std::max(maxBuildScratch, (size_t)libSizes[b].buildScratchSize);
  }

  // Batch build the 5,000 BLASes in a temporary command buffer
  {
    VkQueue queue = vContext->getComputeQueue();
    VkCommandPoolCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
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

    ComputeBuffer libScratch = context_ref.createBuffer(maxBuildScratch);
    VkDeviceAddress libScratchAddr = vContext->getBufferDeviceAddress(libScratch);

    VkCommandBufferBeginInfo bBegin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bBegin);

    for (uint32_t b = 0; b < numBlasLib; ++b) {
      VkAccelerationStructureGeometryKHR geom{
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
      geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
      geom.geometry.triangles.sType =
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
      geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
      geom.geometry.triangles.vertexData.deviceAddress =
          vAddr + (b * 1000 * 9 * sizeof(float));
      geom.geometry.triangles.vertexStride = sizeof(float) * 3;
      geom.geometry.triangles.maxVertex = libPrims[b] * 3 - 1;
      geom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

      VkAccelerationStructureBuildGeometryInfoKHR bInfo{
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
      bInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      bInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
      bInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      bInfo.geometryCount = 1;
      bInfo.pGeometries = &geom;
      bInfo.dstAccelerationStructure = blasLibHandles[b];
      bInfo.scratchData.deviceAddress = libScratchAddr;

      VkAccelerationStructureBuildRangeInfoKHR range{};
      range.primitiveCount = libPrims[b];
      const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;

      vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &bInfo, &pRange);

      if (b + 1 < numBlasLib) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                           VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
      }
    }
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkDestroyCommandPool(device, tmpPool, nullptr);
    context_ref.releaseBuffer(libScratch);
  }

  // 4. Setup 3 Real-World Branched Scene TLASes
  tlases.resize(3);

  // Scene 0: Indoor Corridor (20,000 instances, 5,000 unique meshes, ~20M virtual triangles)
  // Architecture: Room/hallway clustering with high geometric uniqueness (~1:4 ratio)
  {
    tlases[0].name = "TLAS: Indoor Corridor";
    tlases[0].numInstances = 20000;
    std::vector<VkAccelerationStructureInstanceKHR> inst0(tlases[0].numInstances);
    for (uint32_t i = 0; i < tlases[0].numInstances; ++i) {
      uint32_t bIdx = i % numBlasLib;
      uint32_t room = i / 200; // 100 rooms of 200 instances
      float rx = (float)(room % 10) * 40.0f;
      float ry = (float)(room / 10) * 40.0f;
      float dx = ((float)(i % 14) - 7.0f) * 2.0f;
      float dy = ((float)((i / 14) % 14) - 7.0f) * 2.0f;
      float dz = (float)(i % 5) * 1.5f;

      inst0[i].transform = {
          1.0f, 0.0f, 0.0f, rx + dx,
          0.0f, 1.0f, 0.0f, ry + dy,
          0.0f, 0.0f, 1.0f, dz
      };
      inst0[i].instanceCustomIndex = i;
      inst0[i].mask = 0xFF;
      inst0[i].instanceShaderBindingTableRecordOffset = 0;
      inst0[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      inst0[i].accelerationStructureReference = blasLibAddrs[bIdx];
    }
    tlases[0].instanceBuffer = context_ref.createBuffer(
        inst0.size() * sizeof(VkAccelerationStructureInstanceKHR), inst0.data());
  }

  // Scene 1: Dense Jungle / Forest (50,000 instances, 500 unique meshes, ~50M virtual triangles)
  // Architecture: Dense overlapping foliage on undulating terrain (~1:100 ratio, heavy AABB overlap)
  {
    tlases[1].name = "TLAS: Dense Jungle";
    tlases[1].numInstances = 50000;
    std::vector<VkAccelerationStructureInstanceKHR> inst1(tlases[1].numInstances);
    for (uint32_t i = 0; i < tlases[1].numInstances; ++i) {
      uint32_t bIdx = i % 500; // 500 complex foliage meshes
      float x = ((float)(i % 250) - 125.0f) * 1.5f;
      float y = ((float)(i / 250) - 100.0f) * 1.5f;
      float z = sinf(x * 0.05f) * 5.0f + cosf(y * 0.05f) * 5.0f + (float)(i % 7) * 2.0f;

      inst1[i].transform = {
          1.0f, 0.0f, 0.0f, x,
          0.0f, 1.0f, 0.0f, y,
          0.0f, 0.0f, 1.0f, z
      };
      inst1[i].instanceCustomIndex = i;
      inst1[i].mask = 0xFF;
      inst1[i].instanceShaderBindingTableRecordOffset = 0;
      inst1[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      inst1[i].accelerationStructureReference = blasLibAddrs[bIdx];
    }
    tlases[1].instanceBuffer = context_ref.createBuffer(
        inst1.size() * sizeof(VkAccelerationStructureInstanceKHR), inst1.data());
  }

  // Scene 2: Massive Open World (200,000 instances, 5,000 unique meshes, ~500M virtual triangles)
  // Architecture: Multi-scale geographic distribution across broad world sectors (~1:40 ratio)
  {
    tlases[2].name = "TLAS: Massive Open World";
    tlases[2].numInstances = 200000;
    std::vector<VkAccelerationStructureInstanceKHR> inst2(tlases[2].numInstances);
    for (uint32_t i = 0; i < tlases[2].numInstances; ++i) {
      uint32_t bIdx = i % numBlasLib;
      uint32_t sector = i / 10000; // 20 sectors
      float sx = ((float)(sector % 5) - 2.0f) * 500.0f;
      float sy = ((float)(sector / 5) - 2.0f) * 500.0f;
      float cx = ((float)(i % 100) - 50.0f) * 4.0f;
      float cy = ((float)((i / 100) % 100) - 50.0f) * 4.0f;
      float cz = (float)(i % 10) * 3.0f;

      inst2[i].transform = {
          1.0f, 0.0f, 0.0f, sx + cx,
          0.0f, 1.0f, 0.0f, sy + cy,
          0.0f, 0.0f, 1.0f, cz
      };
      inst2[i].instanceCustomIndex = i;
      inst2[i].mask = 0xFF;
      inst2[i].instanceShaderBindingTableRecordOffset = 0;
      inst2[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      inst2[i].accelerationStructureReference = blasLibAddrs[bIdx];
    }
    tlases[2].instanceBuffer = context_ref.createBuffer(
        inst2.size() * sizeof(VkAccelerationStructureInstanceKHR), inst2.data());
  }

  // Allocate TLAS buffers and handles
  for (size_t t = 0; t < tlases.size(); ++t) {
    VkAccelerationStructureGeometryKHR instGeom{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    instGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    instGeom.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instGeom.geometry.instances.data.deviceAddress =
        vContext->getBufferDeviceAddress(tlases[t].instanceBuffer);

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlasBuildInfo.geometryCount = 1;
    tlasBuildInfo.pGeometries = &instGeom;
    tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

    uint32_t maxInstCount = tlases[t].numInstances;
    tlases[t].sizes.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR_ptr(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo,
        &maxInstCount, &tlases[t].sizes);

    tlases[t].buffer = context_ref.createBuffer(tlases[t].sizes.accelerationStructureSize);
    VkAccelerationStructureCreateInfoKHR tlasCreateInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    tlasCreateInfo.buffer = vContext->getVkBuffer(tlases[t].buffer);
    tlasCreateInfo.size = tlases[t].sizes.accelerationStructureSize;
    tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    vkCreateAccelerationStructureKHR_ptr(device, &tlasCreateInfo, nullptr, &tlases[t].handle);

    maxBuildScratch = std::max(maxBuildScratch, (size_t)tlases[t].sizes.buildScratchSize);
  }

  scratchBuffer = context_ref.createBuffer(maxBuildScratch);
  updateScratchBuffer = context_ref.createBuffer(maxUpdateScratch);

  // Initial builds for standalone BLASes so refit updates work cleanly
  Run(0);
  Run(2);
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

  VkAccelerationStructureGeometryKHR geom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  VkAccelerationStructureBuildRangeInfoKHR range{};

  uint32_t iters = 10;

  if (config_idx < 5) { // BLAS: 0=1M Build, 1=1M Update, 2=5M Build, 3=5M Update, 4=10M Build
    size_t bIdx = 0;
    bool isUpdate = false;
    if (config_idx == 0) { bIdx = 0; isUpdate = false; iters = 10; }
    else if (config_idx == 1) { bIdx = 0; isUpdate = true; iters = 10; }
    else if (config_idx == 2) { bIdx = 1; isUpdate = false; iters = 5; }
    else if (config_idx == 3) { bIdx = 1; isUpdate = true; iters = 5; }
    else if (config_idx == 4) { bIdx = 2; isUpdate = false; iters = 5; }

    uint32_t prims = blases[bIdx].numPrimitives;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geom.geometry.triangles.vertexData.deviceAddress = vAddr;
    geom.geometry.triangles.vertexStride = sizeof(float) * 3;
    geom.geometry.triangles.maxVertex = prims * 3 - 1;
    geom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geom;
    buildInfo.dstAccelerationStructure = blases[bIdx].handle;

    if (!isUpdate) {
      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfo.scratchData.deviceAddress = vContext->getBufferDeviceAddress(scratchBuffer);
    } else {
      buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
      buildInfo.srcAccelerationStructure = blases[bIdx].handle;
      buildInfo.scratchData.deviceAddress = vContext->getBufferDeviceAddress(updateScratchBuffer);
    }
    range.primitiveCount = prims;
  } else { // TLAS: 5=Indoor Corridor (20K), 6=Dense Jungle (50K), 7=Massive Open World (200K)
    size_t tIdx = config_idx - 5;
    uint32_t instCount = tlases[tIdx].numInstances;
    iters = (config_idx == 7) ? 5 : 10;

    VkDeviceAddress iAddr = vContext->getBufferDeviceAddress(tlases[tIdx].instanceBuffer);

    geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geom.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geom.geometry.instances.data.deviceAddress = iAddr;

    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geom;
    buildInfo.dstAccelerationStructure = tlases[tIdx].handle;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.scratchData.deviceAddress = vContext->getBufferDeviceAddress(scratchBuffer);
    range.primitiveCount = instCount;
  }

  const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
  vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &buildInfo, &pRange);

  vkEndCommandBuffer(cmd);

  auto start = std::chrono::high_resolution_clock::now();

  for (uint32_t i = 0; i < iters; ++i) {
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VkResult res = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
      vkDestroyCommandPool(device, tmpPool, nullptr);
      throw std::runtime_error("vkQueueSubmit failed in RayASBuild: " + std::to_string(res));
    }
    res = vkQueueWaitIdle(queue);
    if (res != VK_SUCCESS) {
      vkDestroyCommandPool(device, tmpPool, nullptr);
      throw std::runtime_error("vkQueueWaitIdle failed in RayASBuild: " + std::to_string(res));
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  buildTimes[config_idx] = (diff.count() / iters) * 1000.0;
  iterations = iters;

  vkDestroyCommandPool(device, tmpPool, nullptr);
}

void RayASBuildBench::Teardown() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext ? vContext->getVulkanDevice() : VK_NULL_HANDLE;

  if (device != VK_NULL_HANDLE && vkDestroyAccelerationStructureKHR_ptr) {
    for (auto &b : blases) {
      if (b.handle != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR_ptr(device, b.handle, nullptr);
        b.handle = VK_NULL_HANDLE;
      }
    }
    for (auto h : blasLibHandles) {
      if (h != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR_ptr(device, h, nullptr);
      }
    }
    blasLibHandles.clear();

    for (auto &t : tlases) {
      if (t.handle != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR_ptr(device, t.handle, nullptr);
        t.handle = VK_NULL_HANDLE;
      }
    }
  }

  if (context) {
    for (auto &b : blases) {
      if (b.buffer) { context->releaseBuffer(b.buffer); b.buffer = nullptr; }
    }
    blases.clear();

    for (auto b : blasLibBuffers) {
      if (b) { context->releaseBuffer(b); }
    }
    blasLibBuffers.clear();
    blasLibAddrs.clear();

    for (auto &t : tlases) {
      if (t.buffer) { context->releaseBuffer(t.buffer); t.buffer = nullptr; }
      if (t.instanceBuffer) { context->releaseBuffer(t.instanceBuffer); t.instanceBuffer = nullptr; }
    }
    tlases.clear();

    if (vertexBuffer) { context->releaseBuffer(vertexBuffer); vertexBuffer = nullptr; }
    if (scratchBuffer) { context->releaseBuffer(scratchBuffer); scratchBuffer = nullptr; }
    if (updateScratchBuffer) { context->releaseBuffer(updateScratchBuffer); updateScratchBuffer = nullptr; }
    context = nullptr;
  }
}

BenchmarkResult RayASBuildBench::GetResult(uint32_t config_idx) const {
  uint64_t ops = 0;
  switch (config_idx) {
  case 0: ops = 1000000; break;
  case 1: ops = 1000000; break;
  case 2: ops = 5000000; break;
  case 3: ops = 5000000; break;
  case 4: ops = 10000000; break;
  case 5: ops = 20000; break;
  case 6: ops = 50000; break;
  case 7: ops = 200000; break;
  default: ops = 1; break;
  }
  return {ops, buildTimes.at(config_idx)};
}

const char *RayASBuildBench::GetName() const { return "RayASBuild"; }
const char *RayASBuildBench::GetComponent(uint32_t config_idx) const {
  return "Ray Tracing";
}
const char *RayASBuildBench::GetMetric(uint32_t config_idx) const {
  if (config_idx >= 5) return "MInst/s";
  return "MTris/s";
}
const char *RayASBuildBench::GetSubCategory(uint32_t config_idx) const {
  switch (config_idx) {
  case 0: return "BLAS Build (1M)";
  case 1: return "BLAS Update (1M)";
  case 2: return "BLAS Build (5M)";
  case 3: return "BLAS Update (5M)";
  case 4: return "BLAS Build (10M)";
  case 5: return "TLAS: Indoor Corridor";
  case 6: return "TLAS: Dense Jungle";
  case 7: return "TLAS: Open World";
  default: return "AS Build";
  }
}

std::string RayASBuildBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0: return "BLAS Build (1M Tris)";
  case 1: return "BLAS Update (1M Tris)";
  case 2: return "BLAS Build (5M Tris)";
  case 3: return "BLAS Update (5M Tris)";
  case 4: return "BLAS Build (10M Tris)";
  case 5: return "TLAS: Indoor Corridor (20K Inst, 5K Meshes)";
  case 6: return "TLAS: Dense Jungle (50K Inst, 500 Meshes)";
  case 7: return "TLAS: Massive Open World (200K Inst, 5K Meshes)";
  default: return "Unknown";
  }
}

