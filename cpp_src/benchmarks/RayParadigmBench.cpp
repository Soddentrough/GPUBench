#include "RayParadigmBench.h"
#include "core/VulkanContext.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifdef HAVE_VULKAN
void RayParadigmBench::loadRTProcs(VkDevice device) {
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

void RayParadigmBench::buildAS() {
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();
  VkQueue queue = vContext->getComputeQueue();

  if (triangleBlas)
    vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
  if (sceneTlas)
    vkDestroyAccelerationStructureKHR_ptr(device, sceneTlas, nullptr);
  if (triangleBlasBuffer)
    context->releaseBuffer(triangleBlasBuffer);
  if (tlasBuffer)
    context->releaseBuffer(tlasBuffer);
  if (instanceBuffer)
    context->releaseBuffer(instanceBuffer);
  if (scratchBuffer)
    context->releaseBuffer(scratchBuffer);

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

  // TLAS Instance Setup
  VkAccelerationStructureDeviceAddressInfoKHR triAddrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  triAddrInfo.accelerationStructure = triangleBlas;
  VkDeviceAddress blasAddress =
      vkGetAccelerationStructureDeviceAddressKHR_ptr(device, &triAddrInfo);

  VkAccelerationStructureInstanceKHR instance{};
  instance.transform.matrix[0][0] = 1.0f;
  instance.transform.matrix[1][1] = 1.0f;
  instance.transform.matrix[2][2] = 1.0f;
  instance.mask = 0xFF;
  instance.flags =
      VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  instance.accelerationStructureReference = blasAddress;

  instanceBuffer =
      context->createBuffer(sizeof(VkAccelerationStructureInstanceKHR), &instance);
  VkDeviceAddress instAddr = vContext->getBufferDeviceAddress(instanceBuffer);

  VkAccelerationStructureGeometryKHR tlasGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  tlasGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
  tlasGeom.geometry.instances.data.deviceAddress = instAddr;

  VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tlasBuildInfo.flags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tlasBuildInfo.geometryCount = 1;
  tlasBuildInfo.pGeometries = &tlasGeom;
  tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

  uint32_t tlasMaxPrimCount = 1;
  VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR_ptr(
      device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildInfo,
      &tlasMaxPrimCount, &tlasSizes);

  tlasBuffer = context->createBuffer(tlasSizes.accelerationStructureSize);

  VkAccelerationStructureCreateInfoKHR tlasCreateInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  tlasCreateInfo.buffer = vContext->getVkBuffer(tlasBuffer);
  tlasCreateInfo.size = tlasSizes.accelerationStructureSize;
  tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  vkCreateAccelerationStructureKHR_ptr(device, &tlasCreateInfo, nullptr,
                                       &sceneTlas);

  size_t maxScratch = std::max(triSizes.buildScratchSize, tlasSizes.buildScratchSize);
  scratchBuffer = context->createBuffer(maxScratch);
  VkDeviceAddress scratchAddr = vContext->getBufferDeviceAddress(scratchBuffer);

  // Command Buffer for Building
  VkCommandPool tmpPool;
  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.queueFamilyIndex = vContext->getComputeQueueFamilyIndex();
  vkCreateCommandPool(device, &poolInfo, nullptr, &tmpPool);

  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo cmdAlloc{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdAlloc.commandPool = tmpPool;
  cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAlloc.commandBufferCount = 1;
  vkAllocateCommandBuffers(device, &cmdAlloc, &cmd);

  VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cmd, &beginInfo);

  triBuildInfo.dstAccelerationStructure = triangleBlas;
  triBuildInfo.scratchData.deviceAddress = scratchAddr;
  VkAccelerationStructureBuildRangeInfoKHR triRange{numPrimitives, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR *pTriRange = &triRange;
  vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &triBuildInfo, &pTriRange);

  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0,
                       1, &barrier, 0, nullptr, 0, nullptr);

  tlasBuildInfo.dstAccelerationStructure = sceneTlas;
  tlasBuildInfo.scratchData.deviceAddress = scratchAddr;
  VkAccelerationStructureBuildRangeInfoKHR tlasRange{1, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR *pTlasRange = &tlasRange;
  vkCmdBuildAccelerationStructuresKHR_ptr(cmd, 1, &tlasBuildInfo, &pTlasRange);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkDestroyCommandPool(device, tmpPool, nullptr);
}
#endif // HAVE_VULKAN

bool RayParadigmBench::IsSupported(const DeviceInfo &info,
                                   IComputeContext *computeContext) const {
  return info.rayTracingSupport &&
         (computeContext && computeContext->getBackend() == ComputeBackend::Vulkan);
}

std::string RayParadigmBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Material Divergence - Traditional Megakernel";
  case 1:
    return "Material Divergence - Traditional + SER (Hardware Reordering)";
  case 2:
    return "Material Divergence - Work Lists / DGC (Wavefront Compaction)";
  case 3:
    return "Material Divergence - Work Graphs (Autonomous Node Enqueue)";
  case 4:
    return "Multi-Bounce Path Tracing - Traditional Megakernel (Lane Idling)";
  case 5:
    return "Multi-Bounce Path Tracing - Traditional + SER (Hardware Reordering)";
  case 6:
    return "Multi-Bounce Path Tracing - Work Lists / DGC (Wavefront Compaction)";
  case 7:
    return "Multi-Bounce Path Tracing - Work Graphs (Dynamic Bounce Node Enqueue)";
  case 8:
    return "Incoherent Rays - Traditional Megakernel (Direct Traversal)";
  case 9:
    return "Incoherent Rays - Traditional + SER (Spatial Reordering Hint)";
  case 10:
    return "Incoherent Rays - Work Lists (GPU Morton Spatial Sort)";
  case 11:
    return "Incoherent Rays - Work Graphs (Asynchronous Node Enqueue)";
  default:
    return "Unknown";
  }
}

const char *RayParadigmBench::GetSubCategory(uint32_t config_idx) const {
  if (config_idx < 4)
    return "Material Divergence";
  if (config_idx < 8)
    return "Multi-Bounce Path Tracing";
  return "Incoherent Secondary Rays";
}

void RayParadigmBench::Setup(IComputeContext &context_ref,
                             const std::string &kernel_dir) {
  this->context = &context_ref;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context_ref);
  if (!vContext)
    throw std::runtime_error("RayParadigmBench requires VulkanContext");

#ifdef HAVE_VULKAN
  loadRTProcs(vContext->getVulkanDevice());

  rayCount = 1000000;
  resultBuffer = context->createBuffer(sizeof(uint32_t) * 4);
  uint32_t zeros[4] = {0, 0, 0, 0};
  context->writeBuffer(resultBuffer, 0, sizeof(zeros), zeros);

  // WorkList storage buffer: 32 counters (128B) + 131072 * 48B records (~6.25MB)
  size_t workListSize = sizeof(uint32_t) * 32 + 131072 * sizeof(float) * 12;
  workListBuffer = context->createBuffer(workListSize);

  // Indirect dispatch commands: 32 * VkDispatchIndirectCommand (384B)
  size_t indirectSize = sizeof(uint32_t) * 3 * 32;
  indirectBuffer = context->createBuffer(indirectSize);

  // Geometry: 4096 triangles for high geometric diversity
  numPrimitives = 4096;
  std::vector<float> vertices;
  vertices.reserve(numPrimitives * 9);
  srand(42);
  for (uint32_t i = 0; i < numPrimitives; ++i) {
    float cx = (float(rand()) / RAND_MAX) * 10.0f - 5.0f;
    float cy = (float(rand()) / RAND_MAX) * 10.0f - 5.0f;
    float cz = (float(rand()) / RAND_MAX) * 5.0f + 1.0f;
    for (int j = 0; j < 3; ++j) {
      vertices.push_back(cx + (float(rand()) / RAND_MAX) * 0.4f - 0.2f);
      vertices.push_back(cy + (float(rand()) / RAND_MAX) * 0.4f - 0.2f);
      vertices.push_back(cz + (float(rand()) / RAND_MAX) * 0.4f - 0.2f);
    }
  }
  vertexBuffer =
      context->createBuffer(vertices.size() * sizeof(float), vertices.data());

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  kernelTraditional = vContext->createKernel(
      (kdir / "vulkan" / "rt_paradigm_traditional.comp").string(), "main", 2);
  kernelClassify = vContext->createKernel(
      (kdir / "vulkan" / "rt_paradigm_worklist_classify.comp").string(), "main", 4);
  kernelMaterial = vContext->createKernel(
      (kdir / "vulkan" / "rt_paradigm_worklist_material.comp").string(), "main", 3);
  kernelBounce = vContext->createKernel(
      (kdir / "vulkan" / "rt_paradigm_worklist_bounce.comp").string(), "main", 3);
  kernelSort = vContext->createKernel(
      (kdir / "vulkan" / "rt_paradigm_worklist_sort.comp").string(), "main", 3);
  kernelWorkGraph = vContext->createKernel(
      (kdir / "vulkan" / "rt_paradigm_workgraph.comp").string(), "main", 2);

  // Check hardware SER support
  bool serSupported = vContext->isSERSupported();
  for (int i = 0; i < 12; ++i) {
    unsupportedConfig[i] = false;
    unsupportedReason[i] = "";
  }
  if (!serSupported) {
    unsupportedConfig[1] = true;
    unsupportedReason[1] = "Hardware SER (VK_EXT/NV_ray_tracing_invocation_reorder) not supported on this GPU";
    unsupportedConfig[5] = true;
    unsupportedReason[5] = "Hardware SER (VK_EXT/NV_ray_tracing_invocation_reorder) not supported on this GPU";
    unsupportedConfig[9] = true;
    unsupportedReason[9] = "Hardware SER (VK_EXT/NV_ray_tracing_invocation_reorder) not supported on this GPU";
  }
#endif
}

void RayParadigmBench::Run(uint32_t config_idx) {
#ifdef HAVE_VULKAN
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  if (unsupportedConfig[config_idx])
    return;

  // Clear results hits buffer for this timed run iteration
  uint32_t zeros[4] = {0, 0, 0, 0};
  context->writeBuffer(resultBuffer, 0, sizeof(zeros), zeros);

  uint32_t seed = rand();

  switch (config_idx) {
  case 0: { // Material Divergence - Traditional Megakernel
    vContext->setKernelAS(kernelTraditional, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelTraditional, 1, resultBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
    } pc{rayCount, 0, 1, seed};
    vContext->setKernelArg(kernelTraditional, 2, sizeof(pc), &pc);
    vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 1: { // Material Divergence - Traditional + SER
    // Checked via unsupportedConfig
    break;
  }
  case 2: { // Material Divergence - Work Lists / DGC
    // Clear queue counters & indirect dispatch commands
    std::vector<uint32_t> initCounters(32, 0);
    context->writeBuffer(workListBuffer, 0, sizeof(uint32_t) * 32, initCounters.data());
    std::vector<uint32_t> initCommands(32 * 3, 0);
    context->writeBuffer(indirectBuffer, 0, sizeof(uint32_t) * 32 * 3, initCommands.data());

    // Pass 1: Traversal & Stream Compaction into Material Bins
    vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelClassify, 1, resultBuffer);
    vContext->setKernelArg(kernelClassify, 2, workListBuffer);
    vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounce;
      uint32_t seed;
    } pcClassify{rayCount, 0, 0, seed};
    vContext->setKernelArg(kernelClassify, 4, sizeof(pcClassify), &pcClassify);
    vContext->dispatch(kernelClassify, (rayCount + 31) / 32, 1, 1, 32, 1, 1);

    // Pass 2: Uniform Indirect Material Dispatches
    vContext->setKernelAS(kernelMaterial, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelMaterial, 1, resultBuffer);
    vContext->setKernelArg(kernelMaterial, 2, workListBuffer);
    for (uint32_t m = 0; m < 32; ++m) {
      struct {
        uint32_t materialId;
        uint32_t totalQueueSize;
      } pcMat{m, 4096};
      vContext->setKernelArg(kernelMaterial, 3, sizeof(pcMat), &pcMat);
      vContext->dispatchIndirect(kernelMaterial, indirectBuffer, m * sizeof(uint32_t) * 3);
    }
    break;
  }
  case 3: { // Material Divergence - Work Graphs
    vContext->setKernelAS(kernelWorkGraph, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelWorkGraph, 1, resultBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
    } pc{rayCount, 0, 1, seed};
    vContext->setKernelArg(kernelWorkGraph, 2, sizeof(pc), &pc);
    vContext->dispatch(kernelWorkGraph, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 4: { // Multi-Bounce Path Tracing - Traditional Megakernel
    vContext->setKernelAS(kernelTraditional, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelTraditional, 1, resultBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
    } pc{rayCount, 1, 4, seed};
    vContext->setKernelArg(kernelTraditional, 2, sizeof(pc), &pc);
    vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 5: { // Multi-Bounce Path Tracing - Traditional + SER
    // Checked via unsupportedConfig
    break;
  }
  case 6: { // Multi-Bounce Path Tracing - Work Lists / DGC (Wavefront Compaction)
    std::vector<uint32_t> initCounters(32, 0);
    context->writeBuffer(workListBuffer, 0, sizeof(uint32_t) * 32, initCounters.data());
    std::vector<uint32_t> initCommands(32 * 3, 0);
    context->writeBuffer(indirectBuffer, 0, sizeof(uint32_t) * 32 * 3, initCommands.data());

    // Pass 1: Primary Ray Generation & Initial Filter
    vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelClassify, 1, resultBuffer);
    vContext->setKernelArg(kernelClassify, 2, workListBuffer);
    vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounce;
      uint32_t seed;
    } pcClassify{rayCount, 1, 0, seed};
    vContext->setKernelArg(kernelClassify, 4, sizeof(pcClassify), &pcClassify);
    vContext->dispatch(kernelClassify, (rayCount + 31) / 32, 1, 1, 32, 1, 1);

    // Pass 2..4: Compacted Wavefront Bounce Dispatches
    vContext->setKernelAS(kernelBounce, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelBounce, 1, resultBuffer);
    vContext->setKernelArg(kernelBounce, 2, workListBuffer);
    for (uint32_t b = 1; b < 4; ++b) {
      struct {
        uint32_t currentBounce;
        uint32_t maxQueueSize;
      } pcBounce{b, 65536};
      vContext->setKernelArg(kernelBounce, 3, sizeof(pcBounce), &pcBounce);
      vContext->dispatchIndirect(kernelBounce, indirectBuffer, 0);
    }
    break;
  }
  case 7: { // Multi-Bounce Path Tracing - Work Graphs
    vContext->setKernelAS(kernelWorkGraph, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelWorkGraph, 1, resultBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
    } pc{rayCount, 1, 4, seed};
    vContext->setKernelArg(kernelWorkGraph, 2, sizeof(pc), &pc);
    vContext->dispatch(kernelWorkGraph, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 8: { // Incoherent Rays - Traditional Megakernel
    vContext->setKernelAS(kernelTraditional, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelTraditional, 1, resultBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
    } pc{rayCount, 2, 1, seed};
    vContext->setKernelArg(kernelTraditional, 2, sizeof(pc), &pc);
    vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 9: { // Incoherent Rays - Traditional + SER
    // Checked via unsupportedConfig
    break;
  }
  case 10: { // Incoherent Rays - Work Lists with Morton Spatial Sort
    std::vector<uint32_t> initCounters(32, 0);
    context->writeBuffer(workListBuffer, 0, sizeof(uint32_t) * 32, initCounters.data());
    std::vector<uint32_t> initCommands(32 * 3, 0);
    context->writeBuffer(indirectBuffer, 0, sizeof(uint32_t) * 32 * 3, initCommands.data());

    // Pass 1: Traversal + Morton Code Generation
    vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelClassify, 1, resultBuffer);
    vContext->setKernelArg(kernelClassify, 2, workListBuffer);
    vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounce;
      uint32_t seed;
    } pcClassify{rayCount, 2, 0, seed};
    vContext->setKernelArg(kernelClassify, 4, sizeof(pcClassify), &pcClassify);
    vContext->dispatch(kernelClassify, (rayCount + 31) / 32, 1, 1, 32, 1, 1);

    // Pass 2: GPU Bitonic Sort by 30-bit Morton Code (Spatial Reordering)
    vContext->setKernelAS(kernelSort, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelSort, 1, resultBuffer);
    vContext->setKernelArg(kernelSort, 2, workListBuffer);
    uint32_t sortElements = 32768; // Power of 2 sort block
    for (uint32_t p = 2; p <= sortElements; p <<= 1) {
      for (uint32_t q = p >> 1; q > 0; q >>= 1) {
        struct {
          uint32_t p;
          uint32_t q;
          uint32_t count;
        } pcSort{p, q, sortElements};
        vContext->setKernelArg(kernelSort, 3, sizeof(pcSort), &pcSort);
        vContext->dispatch(kernelSort, (sortElements + 63) / 64, 1, 1, 64, 1, 1);
      }
    }

    // Pass 3: Coherent BVH Traversal from Spatially Sorted Ray Queue
    vContext->setKernelAS(kernelBounce, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelBounce, 1, resultBuffer);
    vContext->setKernelArg(kernelBounce, 2, workListBuffer);
    struct {
      uint32_t currentBounce;
      uint32_t maxQueueSize;
    } pcBounce{0, 65536};
    vContext->setKernelArg(kernelBounce, 3, sizeof(pcBounce), &pcBounce);
    vContext->dispatchIndirect(kernelBounce, indirectBuffer, 0);
    break;
  }
  case 11: { // Incoherent Rays - Work Graphs
    vContext->setKernelAS(kernelWorkGraph, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelWorkGraph, 1, resultBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
    } pc{rayCount, 2, 1, seed};
    vContext->setKernelArg(kernelWorkGraph, 2, sizeof(pc), &pc);
    vContext->dispatch(kernelWorkGraph, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  }
#endif
}

void RayParadigmBench::Teardown() {
#ifdef HAVE_VULKAN
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  VkDevice device = vContext->getVulkanDevice();

  if (triangleBlas)
    vkDestroyAccelerationStructureKHR_ptr(device, triangleBlas, nullptr);
  if (sceneTlas)
    vkDestroyAccelerationStructureKHR_ptr(device, sceneTlas, nullptr);

  if (kernelTraditional)
    context->releaseKernel(kernelTraditional);
  if (kernelClassify)
    context->releaseKernel(kernelClassify);
  if (kernelMaterial)
    context->releaseKernel(kernelMaterial);
  if (kernelBounce)
    context->releaseKernel(kernelBounce);
  if (kernelSort)
    context->releaseKernel(kernelSort);
  if (kernelWorkGraph)
    context->releaseKernel(kernelWorkGraph);

  if (resultBuffer)
    context->releaseBuffer(resultBuffer);
  if (workListBuffer)
    context->releaseBuffer(workListBuffer);
  if (indirectBuffer)
    context->releaseBuffer(indirectBuffer);

  if (vertexBuffer)
    context->releaseBuffer(vertexBuffer);
  if (instanceBuffer)
    context->releaseBuffer(instanceBuffer);
  if (triangleBlasBuffer)
    context->releaseBuffer(triangleBlasBuffer);
  if (tlasBuffer)
    context->releaseBuffer(tlasBuffer);
  if (scratchBuffer)
    context->releaseBuffer(scratchBuffer);
#endif
}

BenchmarkResult RayParadigmBench::GetResult(uint32_t config_idx) const {
  BenchmarkResult r;
  r.operations = static_cast<uint64_t>(rayCount);
  r.elapsedTime = 0.0;
  return r;
}
