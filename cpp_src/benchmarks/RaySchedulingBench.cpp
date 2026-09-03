#include "RaySchedulingBench.h"
#include "core/VulkanContext.h"
#include "utils/ImageExport.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifdef HAVE_VULKAN
void RaySchedulingBench::loadRTProcs(VkDevice device) {
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

void RaySchedulingBench::buildAS() {
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

bool RaySchedulingBench::IsSupported(const DeviceInfo &info,
                                   IComputeContext *computeContext) const {
  return info.rayTracingSupport &&
         (computeContext && computeContext->getBackend() == ComputeBackend::Vulkan);
}

std::string RaySchedulingBench::GetConfigName(uint32_t config_idx) const {
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
    return "Multi-Bounce Path Tracing (4 Bounces, Russian Roulette) - Traditional Megakernel (Lane Idling)";
  case 5:
    return "Multi-Bounce Path Tracing (4 Bounces, Russian Roulette) - Traditional + SER (Hardware Reordering)";
  case 6:
    return "Multi-Bounce Path Tracing (4 Bounces, Russian Roulette) - Work Lists / DGC (Wavefront Compaction)";
  case 7:
    return "Multi-Bounce Path Tracing (4 Bounces, Russian Roulette) - Work Graphs (Dynamic Bounce Node Enqueue)";
  case 8:
    return "Incoherent Rays - Traditional Megakernel (Direct Traversal)";
  case 9:
    return "Incoherent Rays - Traditional + SER (Spatial Reordering Hint)";
  case 10:
    return "Incoherent Rays - Work Lists / DGC (Directional Octant Binning)";
  case 11:
    return "Incoherent Rays - Work Graphs (Autonomous Directional Node Enqueue)";
  case 12:
    return "Stage Breakdown - Pure BVH Traversal (98K Triangles, No Shading)";
  case 13:
    return "Stage Breakdown - Pure Material Shading (Traditional Megakernel, 4 Lights)";
  case 14:
    return "Stage Breakdown - Pure Material Shading (Work Lists Specialized, 4 Lights)";
  case 15:
    return "Stage Breakdown - Stream Compaction & Memory Spilling Overhead";
  default:
    return "Unknown";
  }
}

const char *RaySchedulingBench::GetSubCategory(uint32_t config_idx) const {
  if (config_idx < 4)
    return "Material Divergence";
  if (config_idx < 8)
    return "Multi-Bounce Path Tracing";
  if (config_idx < 12)
    return "Incoherent Secondary Rays";
  return "Stage Breakdown Analysis";
}

void RaySchedulingBench::Setup(IComputeContext &context_ref,
                             const std::string &kernel_dir) {
  this->context = &context_ref;
  VulkanContext *vContext = dynamic_cast<VulkanContext *>(&context_ref);
  if (!vContext)
    throw std::runtime_error("RaySchedulingBench requires VulkanContext");

#ifdef HAVE_VULKAN
  loadRTProcs(vContext->getVulkanDevice());

  if (getenv("GPUBENCH_DUMP_RENDERS")) {
    dumpRenders = true;
  }

  rayCount = 1048576; // 1024 x 1024 square ray grid
  resultBuffer = context->createBuffer(sizeof(uint32_t) * 4);
  uint32_t zeros[4] = {0, 0, 0, 0};
  context->writeBuffer(resultBuffer, 0, sizeof(zeros), zeros);

  // Allocate 1024x1024 RGBA32F framebuffers for visual verification
  fbTraditional = context->createBuffer(rayCount * sizeof(float) * 4);
  fbWorkList = context->createBuffer(rayCount * sizeof(float) * 4);

  // WorkList storage buffer: 32 counters (128B) + 32 * 65536 * 48B records (~100MB)
  size_t workListSize = sizeof(uint32_t) * 32 + 32 * 65536 * sizeof(float) * 12;
  workListBuffer = context->createBuffer(workListSize);

  // Indirect dispatch commands: 32 * VkDispatchIndirectCommand (384B)
  size_t indirectSize = sizeof(uint32_t) * 3 * 32;
  indirectBuffer = context->createBuffer(indirectSize);

  // High-Density Continuous Showroom Geometry (98,304 triangles)
  // Part 1: Continuous Parametric Trefoil Torus Knot (65,536 triangles)
  // Part 2: Concave Showroom Cavity Dish (32,768 triangles)
  std::vector<float> vertices;
  const uint32_t knot_u = 256;
  const uint32_t knot_v = 128;
  const float p_knot = 2.0f, q_knot = 3.0f;
  const float R_knot = 2.0f, r0_knot = 0.8f, r_tube = 0.45f;
  const float pi2 = 6.283185307179586f;

  std::vector<std::vector<std::array<float, 3>>> knot_grid(knot_u, std::vector<std::array<float, 3>>(knot_v));
  for (uint32_t i = 0; i < knot_u; ++i) {
    float u = (float(i) / float(knot_u)) * pi2;
    float r = R_knot + r0_knot * std::cos(q_knot * u);
    float cx = r * std::cos(p_knot * u);
    float cy = r * std::sin(p_knot * u);
    float cz = -r0_knot * std::sin(q_knot * u);

    // Tangent approximation
    float u_next = u + 0.001f;
    float r_n = R_knot + r0_knot * std::cos(q_knot * u_next);
    float tx = r_n * std::cos(p_knot * u_next) - cx;
    float ty = r_n * std::sin(p_knot * u_next) - cy;
    float tz = -r0_knot * std::sin(q_knot * u_next) - cz;
    float tlen = std::sqrt(tx * tx + ty * ty + tz * tz);
    tx /= tlen; ty /= tlen; tz /= tlen;

    // Normal & binormal frame
    float nx = -ty, ny = tx, nz = 0.0f;
    float nlen = std::sqrt(nx * nx + ny * ny);
    if (nlen < 0.001f) { nx = 1.0f; ny = 0.0f; nz = 0.0f; } else { nx /= nlen; ny /= nlen; }
    float bx = ty * nz - tz * ny;
    float by = tz * nx - tx * nz;
    float bz = tx * ny - ty * nx;

    // 3D Euler rotation matching Blender showroom scene (25°, 20°, -15°) + location (0, 0, 0.45)
    const float R[3][3] = {
      { 0.907673f,  0.243210f,  0.342020f},
      {-0.094951f,  0.912837f, -0.397131f},
      {-0.408795f,  0.327990f,  0.851651f}
    };

    for (uint32_t j = 0; j < knot_v; ++j) {
      float v = (float(j) / float(knot_v)) * pi2;
      float cv = std::cos(v), sv = std::sin(v);
      float px = cx + r_tube * (cv * nx + sv * bx);
      float py = cy + r_tube * (cv * ny + sv * by);
      float pz = cz + r_tube * (cv * nz + sv * bz);
      float rx = R[0][0] * px + R[0][1] * py + R[0][2] * pz;
      float ry = R[1][0] * px + R[1][1] * py + R[1][2] * pz;
      float rz = R[2][0] * px + R[2][1] * py + R[2][2] * pz + 0.45f;
      knot_grid[i][j] = {rx, ry, rz};
    }
  }

  // Triangulate Knot (256 * 128 * 2 = 65,536 triangles)
  for (uint32_t i = 0; i < knot_u; ++i) {
    uint32_t i_next = (i + 1) % knot_u;
    for (uint32_t j = 0; j < knot_v; ++j) {
      uint32_t j_next = (j + 1) % knot_v;
      const auto &p00 = knot_grid[i][j];
      const auto &p10 = knot_grid[i_next][j];
      const auto &p11 = knot_grid[i_next][j_next];
      const auto &p01 = knot_grid[i][j_next];

      // Tri 1
      vertices.insert(vertices.end(), {p00[0], p00[1], p00[2], p10[0], p10[1], p10[2], p11[0], p11[1], p11[2]});
      // Tri 2
      vertices.insert(vertices.end(), {p00[0], p00[1], p00[2], p11[0], p11[1], p11[2], p01[0], p01[1], p01[2]});
    }
  }

  // Part 2: Concave Showroom Cavity Dish matching Blender README scene (128 * 128 * 2 = 32,768 triangles)
  const uint32_t dish_n = 128;
  for (uint32_t i = 0; i < dish_n; ++i) {
    float u0 = (float(i) / float(dish_n) - 0.5f) * 16.0f;
    float u1 = (float(i + 1) / float(dish_n) - 0.5f) * 16.0f;
    for (uint32_t j = 0; j < dish_n; ++j) {
      float v0 = (float(j) / float(dish_n) - 0.5f) * 16.0f;
      float v1 = (float(j + 1) / float(dish_n) - 0.5f) * 16.0f;

      float z00 = -1.5f + 0.035f * (u0 * u0 + v0 * v0);
      float z10 = -1.5f + 0.035f * (u1 * u1 + v0 * v0);
      float z11 = -1.5f + 0.035f * (u1 * u1 + v1 * v1);
      float z01 = -1.5f + 0.035f * (u0 * u0 + v1 * v1);

      vertices.insert(vertices.end(), {u0, v0, z00, u1, v0, z10, u1, v1, z11});
      vertices.insert(vertices.end(), {u0, v0, z00, u1, v1, z11, u0, v1, z01});
    }
  }

  numPrimitives = static_cast<uint32_t>(vertices.size() / 9);
  vertexBuffer =
      context->createBuffer(vertices.size() * sizeof(float), vertices.data());

  buildAS();

  std::filesystem::path kdir(kernel_dir);
  kernelTraditional = vContext->createKernel(
      (kdir / "vulkan" / "rt_scheduling_traditional.comp").string(), "main", 3);
  vContext->setKernelAS(kernelTraditional, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelTraditional, 1, resultBuffer);
  vContext->setKernelArg(kernelTraditional, 2, fbTraditional);

  kernelClassify = vContext->createKernel(
      (kdir / "vulkan" / "rt_scheduling_worklist_classify.comp").string(), "main", 5);
  vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelClassify, 1, resultBuffer);
  vContext->setKernelArg(kernelClassify, 2, workListBuffer);
  vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
  vContext->setKernelArg(kernelClassify, 4, fbWorkList);

  kernelMaterial = vContext->createKernel(
      (kdir / "vulkan" / "rt_scheduling_worklist_material.comp").string(), "main", 4);
  vContext->setKernelAS(kernelMaterial, 0, (AccelerationStructure)sceneTlas);
  vContext->setKernelArg(kernelMaterial, 1, resultBuffer);
  vContext->setKernelArg(kernelMaterial, 2, workListBuffer);
  vContext->setKernelArg(kernelMaterial, 3, fbWorkList);

  for (uint32_t arch = 0; arch < 6; ++arch) {
    kernelMaterialSpecialized[arch] = vContext->createKernelWithSpec(
        (kdir / "vulkan" / "rt_scheduling_worklist_material.comp").string(), "main", 4, 0, arch);
    vContext->setKernelAS(kernelMaterialSpecialized[arch], 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 1, resultBuffer);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 2, workListBuffer);
    vContext->setKernelArg(kernelMaterialSpecialized[arch], 3, fbWorkList);
  }
  kernelBounce = vContext->createKernel(
      (kdir / "vulkan" / "rt_scheduling_worklist_bounce.comp").string(), "main", 3);
  kernelWorkGraph = vContext->createKernel(
      (kdir / "vulkan" / "rt_scheduling_workgraph.comp").string(), "main", 2);

  // Check hardware SER support
  bool serSupported = vContext->isSERSupported();
  for (int i = 0; i < 16; ++i) {
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

  // Check Work Graphs support (VK_AMDX_shader_enqueue)
  bool workGraphsSupported = vContext->isWorkGraphsSupported();
  if (!workGraphsSupported) {
    unsupportedConfig[3] = true;
    unsupportedReason[3] = "Work Graphs (VK_AMDX_shader_enqueue) not exposed by driver";
    unsupportedConfig[7] = true;
    unsupportedReason[7] = "Work Graphs (VK_AMDX_shader_enqueue) not exposed by driver";
    unsupportedConfig[11] = true;
    unsupportedReason[11] = "Work Graphs (VK_AMDX_shader_enqueue) not exposed by driver";
  }

  // Pre-generate static indirect batches for Work Lists dispatches with specialized PSOs
  materialBatches.reserve(32);
  materialBatchesBreakdown.reserve(32);
  for (uint32_t m = 0; m < 32; ++m) {
    struct {
      uint32_t materialId;
      uint32_t totalQueueSize;
      uint32_t dumpRenders;
    } pcMat{m, 65536, dumpRenders ? 1u : 0u};
    std::vector<uint8_t> pcData(sizeof(pcMat));
    std::memcpy(pcData.data(), &pcMat, sizeof(pcMat));
    uint32_t arch = (m < 20) ? (m / 4) : 5;
    materialBatches.push_back({m * sizeof(uint32_t) * 3, pcData, kernelMaterialSpecialized[arch]});

    pcMat.dumpRenders = 0u;
    std::memcpy(pcData.data(), &pcMat, sizeof(pcMat));
    materialBatchesBreakdown.push_back({m * sizeof(uint32_t) * 3, pcData, kernelMaterialSpecialized[arch]});
  }

  // Pre-initialize indirectBuffer commands and workList counters for isolated stage testing
  std::vector<uint32_t> initCmds(32 * 3);
  std::vector<uint32_t> initCounters(32);
  uint32_t perQueue = rayCount / 32;
  for (uint32_t m = 0; m < 32; ++m) {
    initCmds[m * 3 + 0] = (perQueue + 63) / 64;
    initCmds[m * 3 + 1] = 1;
    initCmds[m * 3 + 2] = 1;
    initCounters[m] = perQueue;
  }
  context->writeBuffer(indirectBuffer, 0, initCmds.size() * sizeof(uint32_t), initCmds.data());
  context->writeBuffer(workListBuffer, 0, initCounters.size() * sizeof(uint32_t), initCounters.data());

  bounceBatches.reserve(3);
  for (uint32_t b = 1; b < 4; ++b) {
    struct {
      uint32_t queueIndex;
      uint32_t maxQueueSize;
    } pcBounce{0, 65536};
    std::vector<uint8_t> pcData(sizeof(pcBounce));
    std::memcpy(pcData.data(), &pcBounce, sizeof(pcBounce));
    bounceBatches.push_back({0, pcData});
  }

  octantBatches.reserve(8);
  for (uint32_t oct = 0; oct < 8; ++oct) {
    struct {
      uint32_t queueIndex;
      uint32_t maxQueueSize;
    } pcBounce{oct, 131072};
    std::vector<uint8_t> pcData(sizeof(pcBounce));
    std::memcpy(pcData.data(), &pcBounce, sizeof(pcBounce));
    octantBatches.push_back({oct * sizeof(uint32_t) * 3, pcData});
  }
#endif
}

void RaySchedulingBench::Run(uint32_t config_idx) {
#ifdef HAVE_VULKAN
  VulkanContext *vContext = static_cast<VulkanContext *>(context);
  if (unsupportedConfig[config_idx])
    return;

  uint32_t seed = dumpRenders ? 1337u : rand();

  switch (config_idx) {
  case 0: { // Material Divergence - Traditional Megakernel
    vContext->setKernelAS(kernelTraditional, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelTraditional, 1, resultBuffer);
    vContext->setKernelArg(kernelTraditional, 2, fbTraditional);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
      uint32_t dumpRenders;
    } pc{rayCount, 0, 1, seed, dumpRenders ? 1u : 0u};
    vContext->setKernelArg(kernelTraditional, 3, sizeof(pc), &pc);
    vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 1: { // Material Divergence - Traditional + SER
    // Checked via unsupportedConfig
    break;
  }
  case 2: { // Material Divergence - Work Lists / DGC
    // Pass 1 & 2: Traversal, stream compaction into material bins, and uniform indirect dispatches
    // executed seamlessly in a single GPU command stream with hardware pipeline barriers.
    vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelClassify, 1, resultBuffer);
    vContext->setKernelArg(kernelClassify, 2, workListBuffer);
    vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
    vContext->setKernelArg(kernelClassify, 4, fbWorkList);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounce;
      uint32_t seed;
      uint32_t dumpRenders;
    } pcClassify{rayCount, 0, 0, seed, dumpRenders ? 1u : 0u};
    vContext->setKernelArg(kernelClassify, 5, sizeof(pcClassify), &pcClassify);

    vContext->setKernelAS(kernelMaterial, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelMaterial, 1, resultBuffer);
    vContext->setKernelArg(kernelMaterial, 2, workListBuffer);
    vContext->setKernelArg(kernelMaterial, 3, fbWorkList);

    vContext->dispatchWorkListSequence(
        workListBuffer, sizeof(uint32_t) * 32,
        indirectBuffer, sizeof(uint32_t) * 32 * 3,
        kernelClassify, (rayCount + 31) / 32, 1, 1,
        kernelMaterial, indirectBuffer, materialBatches);
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
    vContext->setKernelArg(kernelTraditional, 2, fbTraditional);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
      uint32_t dumpRenders;
    } pc{rayCount, 1, 4, seed, 0};
    vContext->setKernelArg(kernelTraditional, 3, sizeof(pc), &pc);
    vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 5: { // Multi-Bounce Path Tracing - Traditional + SER
    // Checked via unsupportedConfig
    break;
  }
  case 6: { // Multi-Bounce Path Tracing - Work Lists / DGC (Wavefront Compaction)
    vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelClassify, 1, resultBuffer);
    vContext->setKernelArg(kernelClassify, 2, workListBuffer);
    vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
    vContext->setKernelArg(kernelClassify, 4, fbWorkList);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounce;
      uint32_t seed;
      uint32_t dumpRenders;
    } pcClassify{rayCount, 1, 0, seed, 0};
    vContext->setKernelArg(kernelClassify, 5, sizeof(pcClassify), &pcClassify);

    vContext->setKernelAS(kernelBounce, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelBounce, 1, resultBuffer);
    vContext->setKernelArg(kernelBounce, 2, workListBuffer);

    vContext->dispatchWorkListSequence(
        workListBuffer, sizeof(uint32_t) * 32,
        indirectBuffer, sizeof(uint32_t) * 32 * 3,
        kernelClassify, (rayCount + 31) / 32, 1, 1,
        kernelBounce, indirectBuffer, bounceBatches);
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
    vContext->setKernelArg(kernelTraditional, 2, fbTraditional);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
      uint32_t dumpRenders;
    } pc{rayCount, 2, 1, seed, 0};
    vContext->setKernelArg(kernelTraditional, 3, sizeof(pc), &pc);
    vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 9: { // Incoherent Rays - Traditional + SER
    // Checked via unsupportedConfig
    break;
  }
  case 10: { // Incoherent Rays - Work Lists / DGC (Directional Octant Binning)
    // Pass 1 & 2: Traversal, directional octant binning, and coherent secondary ray dispatches
    // executed in a single command stream on the GPU.
    vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelClassify, 1, resultBuffer);
    vContext->setKernelArg(kernelClassify, 2, workListBuffer);
    vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
    vContext->setKernelArg(kernelClassify, 4, fbWorkList);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounce;
      uint32_t seed;
      uint32_t dumpRenders;
    } pcClassify{rayCount, 2, 0, seed, 0};
    vContext->setKernelArg(kernelClassify, 5, sizeof(pcClassify), &pcClassify);

    vContext->setKernelAS(kernelBounce, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelBounce, 1, resultBuffer);
    vContext->setKernelArg(kernelBounce, 2, workListBuffer);

    vContext->dispatchWorkListSequence(
        workListBuffer, sizeof(uint32_t) * 32,
        indirectBuffer, sizeof(uint32_t) * 32 * 3,
        kernelClassify, (rayCount + 63) / 64, 1, 1,
        kernelBounce, indirectBuffer, octantBatches);
    break;
  }
  case 11: { // Incoherent Rays - Work Graphs (Autonomous Directional Node Enqueue)
    vContext->setKernelAS(kernelWorkGraph, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelWorkGraph, 1, resultBuffer);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
    } pc{rayCount, 2, 1, seed};
    vContext->setKernelArg(kernelWorkGraph, 2, sizeof(pc), &pc);
    vContext->dispatch(kernelWorkGraph, (rayCount + 63) / 64, 1, 1, 64, 1, 1);
    break;
  }
  case 12: { // Stage Breakdown - Pure BVH Traversal (98K Triangles, No Shading)
    vContext->setKernelAS(kernelTraditional, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelTraditional, 1, resultBuffer);
    vContext->setKernelArg(kernelTraditional, 2, fbTraditional);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
      uint32_t dumpRenders;
    } pc{rayCount, 3, 1, seed, 0};
    vContext->setKernelArg(kernelTraditional, 3, sizeof(pc), &pc);
    vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 13: { // Stage Breakdown - Pure Material Shading (Traditional Megakernel, 4 Lights)
    vContext->setKernelAS(kernelTraditional, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelTraditional, 1, resultBuffer);
    vContext->setKernelArg(kernelTraditional, 2, fbTraditional);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounces;
      uint32_t seed;
      uint32_t dumpRenders;
    } pc{rayCount, 4, 1, seed, 0};
    vContext->setKernelArg(kernelTraditional, 3, sizeof(pc), &pc);
    vContext->dispatch(kernelTraditional, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  case 14: { // Stage Breakdown - Pure Material Shading (Work Lists Specialized, 4 Lights)
    vContext->dispatchIndirectSequence(kernelMaterial, indirectBuffer, materialBatchesBreakdown);
    break;
  }
  case 15: { // Stage Breakdown - Stream Compaction & Memory Spilling Overhead
    vContext->setKernelAS(kernelClassify, 0, (AccelerationStructure)sceneTlas);
    vContext->setKernelArg(kernelClassify, 1, resultBuffer);
    vContext->setKernelArg(kernelClassify, 2, workListBuffer);
    vContext->setKernelArg(kernelClassify, 3, indirectBuffer);
    vContext->setKernelArg(kernelClassify, 4, fbWorkList);
    struct {
      uint32_t rayCount;
      uint32_t mode;
      uint32_t bounce;
      uint32_t seed;
      uint32_t dumpRenders;
    } pcClassify{rayCount, 3, 0, seed, 0};
    vContext->setKernelArg(kernelClassify, 5, sizeof(pcClassify), &pcClassify);
    vContext->dispatch(kernelClassify, (rayCount + 31) / 32, 1, 1, 32, 1, 1);
    break;
  }
  }
#endif
}

void RaySchedulingBench::performVisualVerification() {
#ifdef HAVE_VULKAN
  if (!context || !fbTraditional || !fbWorkList) return;

  uint32_t width = 1024, height = 1024;
  size_t bufferSize = width * height * sizeof(float) * 4;

  std::vector<float> hdrTrad(width * height * 4, 0.0f);
  std::vector<float> hdrWork(width * height * 4, 0.0f);

  context->readBuffer(fbTraditional, 0, bufferSize, hdrTrad.data());
  context->readBuffer(fbWorkList, 0, bufferSize, hdrWork.data());

  std::vector<uint8_t> ldrTrad, ldrWork, ldrDiff;
  auto metrics = gpubench::ImageExport::compareAndTonemap(
      hdrTrad.data(), hdrWork.data(), width, height, ldrTrad, ldrWork, ldrDiff);

  std::filesystem::create_directories("renders");
  std::string tradPpm = "renders/render_traditional_megakernel.ppm";
  std::string workPpm = "renders/render_worklist_dgc.ppm";
  std::string diffPpm = "renders/render_difference_heatmap.ppm";

  std::string tradPng = "renders/render_traditional_megakernel.png";
  std::string workPng = "renders/render_worklist_dgc.png";
  std::string diffPng = "renders/render_difference_heatmap.png";

  gpubench::ImageExport::writePPM(tradPpm, width, height, ldrTrad);
  gpubench::ImageExport::writePPM(workPpm, width, height, ldrWork);
  gpubench::ImageExport::writePPM(diffPpm, width, height, ldrDiff);

  gpubench::ImageExport::convertPPMtoPNG(tradPpm, tradPng);
  gpubench::ImageExport::convertPPMtoPNG(workPpm, workPng);
  gpubench::ImageExport::convertPPMtoPNG(diffPpm, diffPng);

  std::cout << std::endl;
  std::cout << "================================================================================" << std::endl;
  std::cout << "            RAY SCHEDULING VISUAL & ANALYTICAL PARITY VERIFICATION              " << std::endl;
  std::cout << "================================================================================" << std::endl;
  std::cout << "  Resolution          : " << width << " x " << height << " (" << (width * height) << " rays)" << std::endl;
  std::cout << "  Megakernel Render   : " << tradPng << std::endl;
  std::cout << "  Work Lists Render   : " << workPng << std::endl;
  std::cout << "  Difference Heatmap  : " << diffPng << " (10x amplified)" << std::endl;
  std::cout << "--------------------------------------------------------------------------------" << std::endl;
  std::cout << "  Max Color Delta     : " << std::fixed << std::setprecision(6) << metrics.maxDelta
            << " (" << static_cast<int>(metrics.maxDelta * 255.0f + 0.5f) << " / 255)" << std::endl;
  std::cout << "  Mean Abs Error (MAE): " << std::fixed << std::setprecision(6) << metrics.mae << std::endl;
  std::cout << "  RMSE                : " << std::fixed << std::setprecision(6) << metrics.rmse << std::endl;
  std::cout << "  PSNR                : " << std::fixed << std::setprecision(2) << metrics.psnr << " dB" << std::endl;
  float bitExactPct = (float)metrics.exactPixels / (float)metrics.totalPixels * 100.0f;
  float nearExactPct = (float)(metrics.totalPixels - metrics.diffPixels) / (float)metrics.totalPixels * 100.0f;
  std::cout << "  Bit-Exact Match     : " << metrics.exactPixels << " / " << metrics.totalPixels
            << " (" << std::fixed << std::setprecision(2) << bitExactPct << "%)" << std::endl;
  std::cout << "  Near-Exact (<=1 LSB): " << (metrics.totalPixels - metrics.diffPixels) << " / " << metrics.totalPixels
            << " (" << std::fixed << std::setprecision(3) << nearExactPct << "%)" << std::endl;
  std::cout << "  Discrepant (> 1 LSB): " << metrics.diffPixels << " / " << metrics.totalPixels
            << " (" << (metrics.diffPixels <= 32 ? "VERIFIED: PARITY PASSED" : "DEVIATION DETECTED") << ")" << std::endl;
  std::cout << "================================================================================" << std::endl;
  std::cout << std::endl;
#endif
}

void RaySchedulingBench::Teardown() {
#ifdef HAVE_VULKAN
  if (dumpRenders) {
    performVisualVerification();
  }

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
  for (uint32_t arch = 0; arch < 5; ++arch) {
    if (kernelMaterialSpecialized[arch])
      context->releaseKernel(kernelMaterialSpecialized[arch]);
  }
  if (kernelBounce)
    context->releaseKernel(kernelBounce);
  if (kernelWorkGraph)
    context->releaseKernel(kernelWorkGraph);

  if (resultBuffer)
    context->releaseBuffer(resultBuffer);
  if (workListBuffer)
    context->releaseBuffer(workListBuffer);
  if (indirectBuffer)
    context->releaseBuffer(indirectBuffer);

  if (fbTraditional) {
    context->releaseBuffer(fbTraditional);
    fbTraditional = nullptr;
  }
  if (fbWorkList) {
    context->releaseBuffer(fbWorkList);
    fbWorkList = nullptr;
  }

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

BenchmarkResult RaySchedulingBench::GetResult(uint32_t config_idx) const {
  BenchmarkResult r;
  r.operations = static_cast<uint64_t>(rayCount);
  r.elapsedTime = 0.0;
  return r;
}
