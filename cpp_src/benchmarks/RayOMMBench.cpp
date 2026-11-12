#include "RayOMMBench.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {

bool evalLeafOpaqueHost(uint32_t primId, float u, float v) {
  uint32_t layer = primId >> 7u;
  float shiftX = static_cast<float>((layer * 3u) & 7u) * 0.025f - 0.08f;
  float shiftY = static_cast<float>((layer * 5u) & 7u) * 0.025f - 0.08f;
  float dx = u - (0.32f + shiftX);
  float dy = v - (0.32f + shiftY);
  return (dx * dx + dy * dy) <= 0.055f;
}

// Bird-curve / hierarchical barycentric sample for subdivision level 3 (64 micro-triangles)
void sampleMicroTriangleVertices(uint32_t microIdx, float &u0, float &v0,
                                 float &u1, float &v1, float &u2, float &v2) {
  // Subdivision level 3 -> 8 segments per edge. Approximate micro-triangle i in [0..63]
  // by row/col on the 8-row triangular lattice: row r has (2*(7-r)+1) micro-triangles.
  uint32_t rem = microIdx & 63u;
  uint32_t row = 0u;
  while (row < 8u) {
    uint32_t rowCount = 2u * (7u - row) + 1u;
    if (rem < rowCount) break;
    rem -= rowCount;
    row++;
  }
  uint32_t col = rem >> 1u;
  bool upsideDown = (rem & 1u) != 0u;
  constexpr float inv8 = 0.125f;
  if (!upsideDown) {
    u0 = static_cast<float>(col) * inv8;
    v0 = static_cast<float>(row) * inv8;
    u1 = static_cast<float>(col + 1u) * inv8;
    v1 = static_cast<float>(row) * inv8;
    u2 = static_cast<float>(col) * inv8;
    v2 = static_cast<float>(row + 1u) * inv8;
  } else {
    u0 = static_cast<float>(col + 1u) * inv8;
    v0 = static_cast<float>(row) * inv8;
    u1 = static_cast<float>(col + 1u) * inv8;
    v1 = static_cast<float>(row + 1u) * inv8;
    u2 = static_cast<float>(col) * inv8;
    v2 = static_cast<float>(row + 1u) * inv8;
  }
}

} // namespace

std::string RayOMMBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Software AnyHit Alpha-Cutout (Baseline)";
  case 1:
    return "Hardware Opacity Micromap (2-State OMM)";
  case 2:
    return "Hardware Opacity Micromap (4-State OMM)";
  default:
    return "";
  }
}

const char *RayOMMBench::GetDescription(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Traces primary rays through 16 stacked alpha-cutout foliage layers invoking software AnyHit shaders on every hit candidate.";
  case 1:
    return "Evaluates alpha-cutout micro-triangles in fixed-function BVH hardware via VK_EXT_opacity_micromap (2-State OMM, zero AnyHit invocations).";
  case 2:
    return "Uses 4-State hardware Opacity Micromaps to resolve interior/exterior micro-triangles in BVH hardware and invoke AnyHit only along leaf silhouettes.";
  default:
    return "Alpha-cutout foliage ray traversal benchmark comparing AnyHit vs VK_EXT_opacity_micromap.";
  }
}

bool RayOMMBench::IsSupported(const DeviceInfo &info,
                              IComputeContext *context) const {
  if (!info.rayTracingSupport) return false;
  if (!context) return true;
  if (context->getBackend() != ComputeBackend::Vulkan) return false;
  const auto *vc = dynamic_cast<const VulkanContext *>(context);
  return vc && vc->isExtensionEnabled(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
}

bool RayOMMBench::IsConfigSupported(uint32_t config_idx,
                                     const DeviceInfo &info,
                                     IComputeContext *context) const {
  if (!IsSupported(info, context)) return false;
  if (config_idx == 0) return true;
  const auto *vc = dynamic_cast<const VulkanContext *>(context);
  if (config_idx == 1 || config_idx == 2) {
    return vc ? vc->isOpacityMicromapSupported() : info.opacityMicromapSupport;
  }
  return false;
}

IBenchmark::SupportLimitation
RayOMMBench::GetSupportLimitation(const DeviceInfo &info,
                                  IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if (!info.rayTracingSupport) {
    return SupportLimitation::kHardware;
  }
  return SupportLimitation::kApi;
}

IBenchmark::SupportLimitation RayOMMBench::GetConfigSupportLimitation(
    uint32_t config_idx, const DeviceInfo &info,
    IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if (!info.rayTracingSupport) {
    return SupportLimitation::kHardware;
  }
  if (config_idx == 1 || config_idx == 2) {
    return SupportLimitation::kHardware;
  }
  return SupportLimitation::kNone;
}

std::string RayOMMBench::GetSupportNote(const DeviceInfo &info,
                                        IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4 ray tracing pipeline";
  }
  if (!info.rayTracingSupport) {
    return "GPU lacks hardware ray tracing support";
  }
  return "VK_KHR_ray_tracing_pipeline not enabled";
}

std::string RayOMMBench::GetConfigSupportNote(uint32_t config_idx,
                                              const DeviceInfo &info,
                                              IComputeContext *context) const {
  if (!IsSupported(info, context)) {
    return GetSupportNote(info, context);
  }
  if (config_idx == 1 || config_idx == 2) {
    return "Requires VK_EXT_opacity_micromap (hardware Opacity Micromap traversal)";
  }
  return "";
}

void RayOMMBench::Setup(IComputeContext &context,
                        const std::string &kernel_dir) {
  m_vk = dynamic_cast<VulkanContext *>(&context);
  if (!m_vk) {
    throw std::runtime_error("RayOMMBench requires VulkanContext");
  }
  m_kernelDir = kernel_dir;
  VkDevice dev = m_vk->getDevice();

  m_pfnGetASBuildSizes =
      reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
          vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR"));
  m_pfnCreateAS = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
      vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR"));
  m_pfnDestroyAS = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
      vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR"));
  m_pfnCmdBuildAS = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
      vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR"));
  m_pfnGetASAddress =
      reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
          vkGetDeviceProcAddr(dev,
                              "vkGetAccelerationStructureDeviceAddressKHR"));

  if (m_vk->isOpacityMicromapSupported()) {
    m_pfnCreateMicromap = reinterpret_cast<PFN_vkCreateMicromapEXT>(
        vkGetDeviceProcAddr(dev, "vkCreateMicromapEXT"));
    m_pfnDestroyMicromap = reinterpret_cast<PFN_vkDestroyMicromapEXT>(
        vkGetDeviceProcAddr(dev, "vkDestroyMicromapEXT"));
    m_pfnGetMicromapBuildSizes =
        reinterpret_cast<PFN_vkGetMicromapBuildSizesEXT>(
            vkGetDeviceProcAddr(dev, "vkGetMicromapBuildSizesEXT"));
    m_pfnCmdBuildMicromaps = reinterpret_cast<PFN_vkCmdBuildMicromapsEXT>(
        vkGetDeviceProcAddr(dev, "vkCmdBuildMicromapsEXT"));
  }

  // Generate 16 stacked planes of 64x64 foliage quads spanning [-1.1, 1.1] x [-1.1, 1.1]
  std::vector<float> verts;
  verts.reserve(static_cast<size_t>(kNumPrimitives) * 9u);
  const float cellSize = 2.2f / static_cast<float>(kGridDim);
  for (uint32_t layer = 0; layer < kNumLayers; ++layer) {
    float z = static_cast<float>(layer) * 0.35f;
    float jitter = static_cast<float>(layer & 3u) * (cellSize * 0.22f);
    for (uint32_t y = 0; y < kGridDim; ++y) {
      for (uint32_t x = 0; x < kGridDim; ++x) {
        float x0 = -1.1f + static_cast<float>(x) * cellSize + jitter;
        float y0 = -1.1f + static_cast<float>(y) * cellSize + jitter;
        float x1 = x0 + cellSize;
        float y1 = y0 + cellSize;
        // Tri 0
        verts.insert(verts.end(), {x0, y0, z, x1, y0, z, x0, y1, z});
        // Tri 1
        verts.insert(verts.end(), {x1, y0, z, x1, y1, z, x0, y1, z});
      }
    }
  }
  m_vertexBuf =
      m_vk->createBuffer(verts.size() * sizeof(float), verts.data());
}

void RayOMMBench::buildFoliageAS(uint32_t config_idx) {
  VkDevice dev = m_vk->getDevice();
  VmaAllocator vma = m_vk->getVmaAllocator();

  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolCi.queueFamilyIndex = m_vk->getComputeQueueFamilyIndex();
  poolCi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkCreateCommandPool(dev, &poolCi, nullptr, &pool);

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo cbAi{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbAi.commandPool = pool;
  cbAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAi.commandBufferCount = 1;
  vkAllocateCommandBuffers(dev, &cbAi, &cmd);

  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fCi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(dev, &fCi, nullptr, &fence);

  auto submitAndWait = [&]() {
    vkEndCommandBuffer(cmd);
    vkResetFences(dev, 1, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    {
      std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
      vkQueueSubmit(m_vk->getComputeQueue(), 1, &si, fence);
    }
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
  };

  VkMicromapUsageEXT ommUsage{};
  VkAccelerationStructureTrianglesOpacityMicromapEXT triOmm{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT};

  if ((config_idx == 1 || config_idx == 2) && m_pfnCreateMicromap &&
      m_pfnGetMicromapBuildSizes && m_pfnCmdBuildMicromaps) {
    const uint16_t ommFormat =
        (config_idx == 1)
            ? static_cast<uint16_t>(VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT)
            : static_cast<uint16_t>(VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT);
    const uint32_t bytesPerTri = (config_idx == 1) ? 8u : 16u;

    ommUsage.count = kNumPrimitives;
    ommUsage.subdivisionLevel = 3u;
    ommUsage.format = ommFormat;

    std::vector<uint8_t> bitData(static_cast<size_t>(kNumPrimitives) *
                                     bytesPerTri,
                                 0u);
    std::vector<VkMicromapTriangleEXT> triDescs(kNumPrimitives);

    for (uint32_t p = 0; p < kNumPrimitives; ++p) {
      triDescs[p].dataOffset = p * bytesPerTri;
      triDescs[p].subdivisionLevel = 3u;
      triDescs[p].format = ommFormat;

      uint8_t *dstBytes = bitData.data() + static_cast<size_t>(p) * bytesPerTri;
      for (uint32_t m = 0; m < 64u; ++m) {
        float u0, v0, u1, v1, u2, v2;
        sampleMicroTriangleVertices(m, u0, v0, u1, v1, u2, v2);
        float uc = (u0 + u1 + u2) * (1.0f / 3.0f);
        float vc = (v0 + v1 + v2) * (1.0f / 3.0f);
        if (config_idx == 1) {
          if (evalLeafOpaqueHost(p, uc, vc)) {
            dstBytes[m >> 3u] |= static_cast<uint8_t>(1u << (m & 7u));
          }
        } else {
          bool s0 = evalLeafOpaqueHost(p, u0, v0);
          bool s1 = evalLeafOpaqueHost(p, u1, v1);
          bool s2 = evalLeafOpaqueHost(p, u2, v2);
          bool sc = evalLeafOpaqueHost(p, uc, vc);
          uint8_t state2bit = 0u; // 0 = TRANSPARENT
          if (s0 && s1 && s2 && sc) {
            state2bit = 1u; // 1 = OPAQUE
          } else if (!s0 && !s1 && !s2 && !sc) {
            state2bit = 0u; // 0 = TRANSPARENT
          } else {
            state2bit = 3u; // 3 = UNKNOWN_OPAQUE -> invokes AnyHit on border
          }
          uint32_t byteIdx = m >> 2u;
          uint32_t shift = (m & 3u) * 2u;
          dstBytes[byteIdx] |= static_cast<uint8_t>(state2bit << shift);
        }
      }
    }

    VkMicromapBuildInfoEXT mmBuildInfo{
        VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
    mmBuildInfo.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    mmBuildInfo.flags = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
    mmBuildInfo.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
    mmBuildInfo.usageCountsCount = 1;
    mmBuildInfo.pUsageCounts = &ommUsage;

    VkMicromapBuildSizesInfoEXT mmSizes{
        VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
    m_pfnGetMicromapBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                               &mmBuildInfo, &mmSizes);

    // Allocate micromap backing storage with VK_BUFFER_USAGE_2_MICROMAP_STORAGE_BIT_EXT
    VkBufferUsageFlags2CreateInfoKHR mmStoreUsage2{
        VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR};
    mmStoreUsage2.usage = VK_BUFFER_USAGE_2_MICROMAP_STORAGE_BIT_EXT |
                          VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR;
    VkBufferCreateInfo mmStoreCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    mmStoreCi.pNext = &mmStoreUsage2;
    mmStoreCi.size = std::max<VkDeviceSize>(mmSizes.micromapSize, 4096);
    mmStoreCi.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    mmStoreCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo devAllocCi{};
    devAllocCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateBuffer(vma, &mmStoreCi, &devAllocCi, &m_ommStorageBuf,
                        &m_ommStorageAlloc, nullptr) != VK_SUCCESS) {
      throw std::runtime_error("RayOMMBench: failed to allocate OMM storage");
    }

    VkMicromapCreateInfoEXT mmCi{VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
    mmCi.buffer = m_ommStorageBuf;
    mmCi.offset = 0;
    mmCi.size = mmSizes.micromapSize;
    mmCi.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    if (m_pfnCreateMicromap(dev, &mmCi, nullptr, &m_micromap) != VK_SUCCESS) {
      throw std::runtime_error("RayOMMBench: vkCreateMicromapEXT failed");
    }

    // Allocate temporary build inputs with MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT
    auto makeBuildInputBuf = [&](const void *srcData, size_t byteLen,
                                 VkBuffer &outBuf, VmaAllocation &outAlloc) {
      VkBufferUsageFlags2CreateInfoKHR u2{
          VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR};
      u2.usage = VK_BUFFER_USAGE_2_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT |
                 VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR;
      VkBufferCreateInfo bCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bCi.pNext = &u2;
      bCi.size = byteLen;
      bCi.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
      bCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      VmaAllocationCreateInfo hostCi{};
      hostCi.usage = VMA_MEMORY_USAGE_AUTO;
      hostCi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
      VmaAllocationInfo aInfo{};
      if (vmaCreateBuffer(vma, &bCi, &hostCi, &outBuf, &outAlloc, &aInfo) !=
          VK_SUCCESS) {
        throw std::runtime_error("RayOMMBench: failed to allocate OMM input");
      }
      std::memcpy(aInfo.pMappedData, srcData, byteLen);
      vmaFlushAllocation(vma, outAlloc, 0, byteLen);
      VkBufferDeviceAddressInfo bda{
          VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
      bda.buffer = outBuf;
      return vkGetBufferDeviceAddress(dev, &bda);
    };

    VkBuffer dataBuf = VK_NULL_HANDLE, triArrBuf = VK_NULL_HANDLE;
    VmaAllocation dataAlloc = VK_NULL_HANDLE, triArrAlloc = VK_NULL_HANDLE;
    VkDeviceAddress dataAddr =
        makeBuildInputBuf(bitData.data(), bitData.size(), dataBuf, dataAlloc);
    VkDeviceAddress triArrAddr = makeBuildInputBuf(
        triDescs.data(), triDescs.size() * sizeof(VkMicromapTriangleEXT),
        triArrBuf, triArrAlloc);
    ComputeBuffer mmScratch = m_vk->createBuffer(
        static_cast<size_t>(std::max<VkDeviceSize>(mmSizes.buildScratchSize, 4096)));

    mmBuildInfo.dstMicromap = m_micromap;
    mmBuildInfo.data.deviceAddress = dataAddr;
    mmBuildInfo.scratchData.deviceAddress =
        m_vk->getBufferDeviceAddress(mmScratch);
    mmBuildInfo.triangleArray.deviceAddress = triArrAddr;
    mmBuildInfo.triangleArrayStride = sizeof(VkMicromapTriangleEXT);

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    m_pfnCmdBuildMicromaps(cmd, 1, &mmBuildInfo);
    submitAndWait();

    m_vk->releaseBuffer(mmScratch);
    vmaDestroyBuffer(vma, dataBuf, dataAlloc);
    vmaDestroyBuffer(vma, triArrBuf, triArrAlloc);

    triOmm.indexType = VK_INDEX_TYPE_NONE_KHR;
    triOmm.baseTriangle = 0;
    triOmm.usageCountsCount = 1;
    triOmm.pUsageCounts = &ommUsage;
    triOmm.micromap = m_micromap;
  }

  // Build Foliage Triangle BLAS
  VkAccelerationStructureGeometryKHR triGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  triGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  triGeom.flags = 0; // Non-opaque so AnyHit / OMM controls transparency
  triGeom.geometry.triangles.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  if (m_micromap != VK_NULL_HANDLE) {
    triGeom.geometry.triangles.pNext = &triOmm;
  }
  triGeom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  triGeom.geometry.triangles.vertexData.deviceAddress =
      m_vk->getBufferDeviceAddress(m_vertexBuf);
  triGeom.geometry.triangles.vertexStride = sizeof(float) * 3u;
  triGeom.geometry.triangles.maxVertex = kNumPrimitives * 3u - 1u;
  triGeom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

  VkAccelerationStructureBuildGeometryInfoKHR blasBuild{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  blasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  blasBuild.geometryCount = 1;
  blasBuild.pGeometries = &triGeom;

  uint32_t maxPrimCount = kNumPrimitives;
  VkAccelerationStructureBuildSizesInfoKHR blasSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  m_pfnGetASBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                       &blasBuild, &maxPrimCount, &blasSizes);

  m_blasBuf =
      m_vk->createBuffer(static_cast<size_t>(blasSizes.accelerationStructureSize));
  VkAccelerationStructureCreateInfoKHR blasCi{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  blasCi.buffer = m_vk->getVkBuffer(m_blasBuf);
  blasCi.size = blasSizes.accelerationStructureSize;
  blasCi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  m_pfnCreateAS(dev, &blasCi, nullptr, &m_blas);

  ComputeBuffer blasScratch = m_vk->createBuffer(
      static_cast<size_t>(std::max<VkDeviceSize>(blasSizes.buildScratchSize, 4096)));
  blasBuild.dstAccelerationStructure = m_blas;
  blasBuild.scratchData.deviceAddress =
      m_vk->getBufferDeviceAddress(blasScratch);

  VkAccelerationStructureBuildRangeInfoKHR blasRange{};
  blasRange.primitiveCount = kNumPrimitives;
  const VkAccelerationStructureBuildRangeInfoKHR *pBlasRange = &blasRange;

  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  m_pfnCmdBuildAS(cmd, 1, &blasBuild, &pBlasRange);
  submitAndWait();
  m_vk->releaseBuffer(blasScratch);

  // Build TLAS
  VkAccelerationStructureDeviceAddressInfoKHR addrInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
  addrInfo.accelerationStructure = m_blas;
  VkDeviceAddress blasAddr = m_pfnGetASAddress(dev, &addrInfo);

  VkAccelerationStructureInstanceKHR inst{};
  inst.transform.matrix[0][0] = 1.0f;
  inst.transform.matrix[1][1] = 1.0f;
  inst.transform.matrix[2][2] = 1.0f;
  inst.mask = 0xFFu;
  inst.accelerationStructureReference = blasAddr;
  ComputeBuffer instBuf = m_vk->createBuffer(sizeof(inst), &inst);

  VkAccelerationStructureGeometryKHR tlasGeom{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  tlasGeom.geometry.instances.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
  tlasGeom.geometry.instances.data.deviceAddress =
      m_vk->getBufferDeviceAddress(instBuf);

  VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  tlasBuild.geometryCount = 1;
  tlasBuild.pGeometries = &tlasGeom;

  uint32_t oneInst = 1u;
  VkAccelerationStructureBuildSizesInfoKHR tlasSizes{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  m_pfnGetASBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                       &tlasBuild, &oneInst, &tlasSizes);

  m_tlasBuf =
      m_vk->createBuffer(static_cast<size_t>(tlasSizes.accelerationStructureSize));
  VkAccelerationStructureCreateInfoKHR tlasCi{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
  tlasCi.buffer = m_vk->getVkBuffer(m_tlasBuf);
  tlasCi.size = tlasSizes.accelerationStructureSize;
  tlasCi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  m_pfnCreateAS(dev, &tlasCi, nullptr, &m_tlas);

  ComputeBuffer tlasScratch = m_vk->createBuffer(
      static_cast<size_t>(std::max<VkDeviceSize>(tlasSizes.buildScratchSize, 4096)));
  tlasBuild.dstAccelerationStructure = m_tlas;
  tlasBuild.scratchData.deviceAddress =
      m_vk->getBufferDeviceAddress(tlasScratch);

  VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
  tlasRange.primitiveCount = 1u;
  const VkAccelerationStructureBuildRangeInfoKHR *pTlasRange = &tlasRange;

  vkResetCommandBuffer(cmd, 0);
  vkBeginCommandBuffer(cmd, &bi);
  m_pfnCmdBuildAS(cmd, 1, &tlasBuild, &pTlasRange);
  submitAndWait();

  m_vk->releaseBuffer(tlasScratch);
  m_vk->releaseBuffer(instBuf);
  vkDestroyFence(dev, fence, nullptr);
  vkDestroyCommandPool(dev, pool, nullptr);
}

void RayOMMBench::PrepareConfig(uint32_t config_idx) {
  ReleaseConfigResources(config_idx);
  if (!m_vk) return;

  size_t pixelBytes =
      static_cast<size_t>(m_width) * static_cast<size_t>(m_height) * sizeof(float) * 4u;
  m_outBuf = m_vk->createBuffer(pixelBytes);

  buildFoliageAS(config_idx);

  std::filesystem::path vkDir = std::filesystem::path(m_kernelDir) / "vulkan";
  bool useOmmShader = (config_idx == 1 || config_idx == 2);
  std::string rgenPath =
      (vkDir / (useOmmShader ? "rayomm_omm.rgen" : "rayomm.rgen")).string();
  std::string rmissPath = (vkDir / "rayomm.rmiss").string();
  std::vector<std::string> rchitPaths = {(vkDir / "rayomm.rchit").string()};
  std::vector<std::string> rahitPaths = {(vkDir / "rayomm.rahit").string()};

  uint32_t force2State = (config_idx == 1) ? 1u : 0u;
  m_rtKernel = m_vk->createRTPipeline(rgenPath, rmissPath, rchitPaths,
                                      rahitPaths, {}, 2, force2State,
                                      useOmmShader);
  m_vk->setKernelAS(m_rtKernel, 0, reinterpret_cast<AccelerationStructure>(m_tlas));
  m_vk->setKernelArg(m_rtKernel, 1, m_outBuf);
}

void RayOMMBench::ReleaseConfigResources(uint32_t /*config_idx*/) {
  if (!m_vk) return;
  m_vk->waitIdle();
  VkDevice dev = m_vk->getDevice();

  if (m_rtKernel) {
    m_vk->releaseKernel(m_rtKernel);
    m_rtKernel = nullptr;
  }
  if (m_tlas != VK_NULL_HANDLE && m_pfnDestroyAS) {
    m_pfnDestroyAS(dev, m_tlas, nullptr);
    m_tlas = VK_NULL_HANDLE;
  }
  if (m_tlasBuf) {
    m_vk->releaseBuffer(m_tlasBuf);
    m_tlasBuf = nullptr;
  }
  if (m_blas != VK_NULL_HANDLE && m_pfnDestroyAS) {
    m_pfnDestroyAS(dev, m_blas, nullptr);
    m_blas = VK_NULL_HANDLE;
  }
  if (m_blasBuf) {
    m_vk->releaseBuffer(m_blasBuf);
    m_blasBuf = nullptr;
  }
  if (m_micromap != VK_NULL_HANDLE && m_pfnDestroyMicromap) {
    m_pfnDestroyMicromap(dev, m_micromap, nullptr);
    m_micromap = VK_NULL_HANDLE;
  }
  if (m_ommStorageBuf != VK_NULL_HANDLE &&
      m_vk->getVmaAllocator() != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_vk->getVmaAllocator(), m_ommStorageBuf,
                     m_ommStorageAlloc);
    m_ommStorageBuf = VK_NULL_HANDLE;
    m_ommStorageAlloc = VK_NULL_HANDLE;
  }
  if (m_outBuf) {
    m_vk->releaseBuffer(m_outBuf);
    m_outBuf = nullptr;
  }
}

void RayOMMBench::Run(uint32_t /*config_idx*/) {
  if (!m_vk || !m_rtKernel) return;
  m_vk->dispatch(m_rtKernel, m_width, m_height, 1, 1, 1, 1);
}

BenchmarkResult RayOMMBench::GetResult(uint32_t /*config_idx*/) const {
  return {static_cast<uint64_t>(m_width) * static_cast<uint64_t>(m_height), 0.0};
}

bool RayOMMBench::ValidateResults(uint32_t /*config_idx*/) const {
  if (!m_vk || !m_outBuf) return false;
  // Read middle scanlines where both foliage hits and sky misses exist
  uint32_t samplePixels = std::min<uint32_t>(m_width * 8u, 16384u);
  size_t byteOffset =
      static_cast<size_t>(m_height / 2u) * static_cast<size_t>(m_width) * sizeof(float) * 4u;
  std::vector<float> rgba(samplePixels * 4u, 0.0f);
  m_vk->readBuffer(m_outBuf, byteOffset, rgba.size() * sizeof(float),
                   rgba.data());

  uint32_t leafHits = 0;
  uint32_t skyMisses = 0;
  for (uint32_t p = 0; p < samplePixels; ++p) {
    float r = rgba[p * 4u + 0u];
    float g = rgba[p * 4u + 1u];
    float b = rgba[p * 4u + 2u];
    float w = rgba[p * 4u + 3u];
    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b) ||
        !std::isfinite(w)) {
      return false;
    }
    if (w > 0.5f && g > 0.05f) {
      leafHits++;
    } else if (std::fabs(w) < 1e-4f && b > 0.4f) {
      skyMisses++;
    }
  }
  return (leafHits > 32u) && (skyMisses > 32u);
}

void RayOMMBench::Teardown() {
  ReleaseConfigResources(0);
  if (m_vk && m_vertexBuf) {
    m_vk->releaseBuffer(m_vertexBuf);
    m_vertexBuf = nullptr;
  }
  m_vk = nullptr;
}
