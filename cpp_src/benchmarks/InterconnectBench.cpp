#include "InterconnectBench.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace {

#if defined(__x86_64__) || defined(_M_X64)
bool cpuSupportsAvx2ForRebar() {
#ifdef _MSC_VER
  int regs[4] = {0, 0, 0, 0};
  __cpuid(regs, 0);
  if (regs[0] < 7) return false;
  __cpuidex(regs, 1, 0);
  bool osxsave = (regs[2] & (1 << 27)) != 0;
  bool avx = (regs[2] & (1 << 28)) != 0;
  if (!osxsave || !avx) return false;
  unsigned long long xcr0 = _xgetbv(0);
  if ((xcr0 & 0x6ULL) != 0x6ULL) return false;
  __cpuidex(regs, 7, 0);
  return (regs[1] & (1 << 5)) != 0;
#else
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
  if ((ecx & (1u << 27)) == 0 || (ecx & (1u << 28)) == 0) return false;
  unsigned int xcr0_lo = 0, xcr0_hi = 0;
  __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
  if ((xcr0_lo & 0x6u) != 0x6u) return false;
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
  return (ebx & (1u << 5)) != 0;
#endif
}

#if defined(_MSC_VER)
void streamWriteRebarAvx2(uint32_t *dst, size_t wordCount, uint32_t seed) {
  size_t vecCount = wordCount / 8;
  __m256i vSeed = _mm256_set1_epi32(static_cast<int>(seed));
  __m256i vLane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  __m256i vStep = _mm256_set1_epi32(8);
  __m256i vIdx = vLane;
  for (size_t i = 0; i < vecCount; ++i) {
    __m256i val = _mm256_xor_si256(vSeed, vIdx);
    _mm256_stream_si256(reinterpret_cast<__m256i *>(dst + i * 8), val);
    vIdx = _mm256_add_epi32(vIdx, vStep);
  }
  _mm_sfence();
}
#else
__attribute__((target("avx2"))) void streamWriteRebarAvx2(uint32_t *dst,
                                                          size_t wordCount,
                                                          uint32_t seed) {
  size_t vecCount = wordCount / 8;
  __m256i vSeed = _mm256_set1_epi32(static_cast<int>(seed));
  __m256i vLane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  __m256i vStep = _mm256_set1_epi32(8);
  __m256i vIdx = vLane;
  for (size_t i = 0; i < vecCount; ++i) {
    __m256i val = _mm256_xor_si256(vSeed, vIdx);
    _mm256_stream_si256(reinterpret_cast<__m256i *>(dst + i * 8), val);
    vIdx = _mm256_add_epi32(vIdx, vStep);
  }
  _mm_sfence();
}
#endif
#endif

} // namespace

std::string InterconnectBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Pinned DMA Host-to-Device (H2D)";
  case 1:
    return "Pinned DMA Device-to-Host (D2H)";
  case 2:
    return "Bidirectional H2D + D2H DMA";
  case 3:
    return "Resizable BAR Direct CPU->VRAM Write";
  case 4:
    return "Unified Memory CPU<->GPU Coherency Latency";
  case 5:
    return "Multi-GPU Peer-to-Peer (P2P)";
  default:
    return "";
  }
}

const char *InterconnectBench::GetDescription(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Measures sustained PCIe Host-to-Device DMA read throughput from pinned system memory into local VRAM.";
  case 1:
    return "Measures sustained PCIe Device-to-Host DMA write throughput from local VRAM into cached pinned system memory.";
  case 2:
    return "Measures full-duplex PCIe interconnect bandwidth by executing concurrent H2D and D2H DMA transfers across dual engines.";
  case 3:
    return "Streams non-temporal SIMD CPU writes directly across PCIe Resizable BAR into HOST_VISIBLE | DEVICE_LOCAL VRAM.";
  case 4:
    return "Measures fine-grained coherent atomic cache-line round-trip latency between CPU and GPU on Unified Memory APUs.";
  case 5:
    return "Measures direct GPU-to-GPU Peer-to-Peer (P2P) DMA bandwidth across PCIe / xGMI / NVLink device groups.";
  default:
    return "PCIe, Resizable BAR, Unified Memory, and Multi-GPU Peer-to-Peer interconnect benchmark.";
  }
}

bool InterconnectBench::probeVulkanDeviceGroupP2P(const VulkanContext *vc,
                                                  std::string &outReason) const {
  if (!vc || vc->getInstance() == VK_NULL_HANDLE) {
    outReason = "Requires Vulkan 1.4 context";
    return false;
  }
  if (vc->getPhysicalDevices().size() < 2) {
    outReason = "Requires >= 2 GPUs for Multi-GPU Peer-to-Peer (P2P) transfer";
    return false;
  }

  uint32_t groupCount = 0;
  if (vkEnumeratePhysicalDeviceGroups(vc->getInstance(), &groupCount,
                                      nullptr) != VK_SUCCESS ||
      groupCount == 0) {
    outReason = "No Vulkan physical device groups enumerated";
    return false;
  }
  std::vector<VkPhysicalDeviceGroupProperties> groups(groupCount);
  for (auto &g : groups) {
    g.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
  }
  if (vkEnumeratePhysicalDeviceGroups(vc->getInstance(), &groupCount,
                                      groups.data()) != VK_SUCCESS) {
    outReason = "Failed to query Vulkan physical device groups";
    return false;
  }
  for (const auto &g : groups) {
    if (g.physicalDeviceCount >= 2) {
      for (uint32_t d = 0; d < g.physicalDeviceCount; ++d) {
        if (g.physicalDevices[d] == vc->getPhysicalDevice()) {
          return true;
        }
      }
    }
  }
  outReason =
      "GPUs are in independent VkDeviceGroups (driver does not expose linked Vulkan P2P device group)";
  return false;
}

bool InterconnectBench::IsSupported(const DeviceInfo & /*info*/,
                                    IComputeContext *context) const {
  if (!context) return true;
  return context->getBackend() == ComputeBackend::Vulkan;
}

bool InterconnectBench::IsConfigSupported(uint32_t config_idx,
                                          const DeviceInfo &info,
                                          IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return false;
  }
  switch (config_idx) {
  case 0:
  case 1:
  case 2:
    return true;
  case 3:
    return info.rebarSupported && !info.isApu;
  case 4:
    return info.isApu;
  case 5: {
    const auto *vc = dynamic_cast<const VulkanContext *>(context);
    if (!vc) return false;
    std::string reason;
    return probeVulkanDeviceGroupP2P(vc, reason);
  }
  default:
    return false;
  }
}

IBenchmark::SupportLimitation
InterconnectBench::GetSupportLimitation(const DeviceInfo & /*info*/,
                                        IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  return SupportLimitation::kNone;
}

IBenchmark::SupportLimitation InterconnectBench::GetConfigSupportLimitation(
    uint32_t config_idx, const DeviceInfo & /*info*/,
    IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if (config_idx == 3 || config_idx == 4) {
    return SupportLimitation::kHardware;
  }
  if (config_idx == 5) {
    const auto *vc = dynamic_cast<const VulkanContext *>(context);
    if (vc && vc->getPhysicalDevices().size() < 2) {
      return SupportLimitation::kHardware;
    }
    return SupportLimitation::kApi;
  }
  return SupportLimitation::kNone;
}

std::string InterconnectBench::GetSupportNote(const DeviceInfo & /*info*/,
                                              IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4 compute & DMA transfer queues";
  }
  return "";
}

std::string InterconnectBench::GetConfigSupportNote(
    uint32_t config_idx, const DeviceInfo &info,
    IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4 backend";
  }
  if (config_idx == 3) {
    if (info.isApu) {
      return "Unified memory APU detected (no discrete PCIe BAR aperture; see Config 4)";
    }
    if (!info.rebarSupported) {
      return "Resizable BAR " +
             gpubench::formatRebarStatus(info.rebarSupported, info.rebarSizeBytes,
                                         info.isApu, true) +
             " (requires >= 512 MB HOST_VISIBLE | DEVICE_LOCAL heap)";
    }
  }
  if (config_idx == 4 && !info.isApu) {
    return "Requires Unified Memory APU / SoC architecture (discrete PCIe GPU detected)";
  }
  if (config_idx == 5) {
    const auto *vc = dynamic_cast<const VulkanContext *>(context);
    std::string reason;
    if (!probeVulkanDeviceGroupP2P(vc, reason)) {
      return reason;
    }
  }
  return "";
}

std::string InterconnectBench::GetConfigCaveat(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context) const {
  std::string caveat;
  if (config_idx == 0 || config_idx == 1 || config_idx == 2 ||
      config_idx == 3 || config_idx == 5) {
    caveat = gpubench::formatRebarCaveat(info, context);
  }
  if (config_idx == 2) {
    const auto *vc = dynamic_cast<const VulkanContext *>(context);
    if (vc && !vc->hasDedicatedTransferQueue()) {
      caveat = gpubench::appendCaveat(
          caveat,
          "No dedicated SDMA transfer queue family; H2D and D2H recorded on shared queue");
    } else if (vc && !vc->hasSecondaryTransferQueue()) {
      caveat = gpubench::appendCaveat(
          caveat,
          "Driver exposes 1 dedicated SDMA transfer queue; H2D issued on primary queue alongside SDMA D2H");
    }
  }
  return caveat;
}

bool InterconnectBench::UsesCustomTiming(uint32_t /*config_idx*/) const {
  return true;
}

double InterconnectBench::GetLastRunDurationMs(uint32_t config_idx) const {
  if (config_idx < m_lastDurationMs.size()) {
    return m_lastDurationMs[config_idx];
  }
  return 0.0;
}

void InterconnectBench::Setup(IComputeContext &context,
                              const std::string &kernel_dir) {
  m_context = &context;
  m_vk = dynamic_cast<VulkanContext *>(&context);
  if (!m_vk) {
    throw std::runtime_error("InterconnectBench requires VulkanContext");
  }
  m_kernelDir = kernel_dir;
  m_lastDurationMs.fill(0.0);

  std::filesystem::path shaderPath =
      std::filesystem::path(kernel_dir) / "vulkan" / "interconnect_latency.comp";
  m_helperKernel =
      m_vk->createKernel(shaderPath.string(), "main", 2);

  VkDevice dev = m_vk->getDevice();

  VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolCi.queueFamilyIndex = m_vk->getComputeQueueFamilyIndex();
  poolCi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(dev, &poolCi, nullptr, &m_primaryPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("InterconnectBench: failed to create primary pool");
  }

  VkCommandBufferAllocateInfo cmdAi{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdAi.commandPool = m_primaryPool;
  cmdAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAi.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(dev, &cmdAi, &m_primaryCmd) != VK_SUCCESS) {
    throw std::runtime_error("InterconnectBench: failed to allocate primary cmd");
  }

  VkFenceCreateInfo fenceCi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(dev, &fenceCi, nullptr, &m_primaryFence);

  VkQueryPoolCreateInfo qpCi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
  qpCi.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpCi.queryCount = 4;
  vkCreateQueryPool(dev, &qpCi, nullptr, &m_tsPool);

  if (m_vk->hasDedicatedTransferQueue()) {
    VkCommandPoolCreateInfo xferPoolCi{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    xferPoolCi.queueFamilyIndex = m_vk->getTransferQueueFamilyIndex();
    xferPoolCi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(dev, &xferPoolCi, nullptr, &m_transferPool) ==
        VK_SUCCESS) {
      VkCommandBufferAllocateInfo xferCmdAi{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      xferCmdAi.commandPool = m_transferPool;
      xferCmdAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      xferCmdAi.commandBufferCount = 1;
      vkAllocateCommandBuffers(dev, &xferCmdAi, &m_transferCmd);
      vkCreateFence(dev, &fenceCi, nullptr, &m_transferFence);
    }
    if (m_vk->hasSecondaryTransferQueue() &&
        vkCreateCommandPool(dev, &xferPoolCi, nullptr,
                            &m_secondaryTransferPool) == VK_SUCCESS) {
      VkCommandBufferAllocateInfo secCmdAi{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      secCmdAi.commandPool = m_secondaryTransferPool;
      secCmdAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      secCmdAi.commandBufferCount = 1;
      vkAllocateCommandBuffers(dev, &secCmdAi, &m_secondaryTransferCmd);
      vkCreateFence(dev, &fenceCi, nullptr, &m_secondaryTransferFence);
    }
  }
}

void InterconnectBench::PrepareConfig(uint32_t config_idx) {
  ReleaseConfigResources(config_idx);
  if (!m_vk) return;

  if (config_idx == 0) {
    m_pinnedUpload = m_vk->createPinnedHostBuffer(kDmaBytes, true, &m_mappedUpload);
    m_devDst = m_vk->createBuffer(kDmaBytes);
    auto *u32 = static_cast<uint32_t *>(m_mappedUpload);
    const size_t words = kDmaBytes / sizeof(uint32_t);
    for (size_t i = 0; i < words; ++i) {
      u32[i] = 0xA55A0000u ^ static_cast<uint32_t>(i);
    }
  } else if (config_idx == 1) {
    std::vector<uint32_t> initWords(kDmaBytes / sizeof(uint32_t));
    for (size_t i = 0; i < initWords.size(); ++i) {
      initWords[i] = 0x5AA50000u ^ static_cast<uint32_t>(i);
    }
    m_devSrc = m_vk->createBuffer(kDmaBytes, initWords.data());
    m_pinnedDownload =
        m_vk->createPinnedHostBuffer(kDmaBytes, false, &m_mappedDownload);
    std::memset(m_mappedDownload, 0, kDmaBytes);
  } else if (config_idx == 2) {
    m_pinnedUpload = m_vk->createPinnedHostBuffer(kDmaBytes, true, &m_mappedUpload);
    m_devDst = m_vk->createBuffer(kDmaBytes);
    auto *u32 = static_cast<uint32_t *>(m_mappedUpload);
    const size_t words = kDmaBytes / sizeof(uint32_t);
    for (size_t i = 0; i < words; ++i) {
      u32[i] = 0x3CC30000u ^ static_cast<uint32_t>(i);
    }

    if (m_vk->hasDedicatedTransferQueue() &&
        m_vk->getTransferQueueFamilyIndex() !=
            m_vk->getComputeQueueFamilyIndex() &&
        m_vk->getVmaAllocator() != VK_NULL_HANDLE) {
      uint32_t qFamilies[2] = {m_vk->getComputeQueueFamilyIndex(),
                               m_vk->getTransferQueueFamilyIndex()};
      VkBufferCreateInfo srcCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      srcCi.size = kDmaBytes;
      srcCi.usage =
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      srcCi.sharingMode = VK_SHARING_MODE_CONCURRENT;
      srcCi.queueFamilyIndexCount = 2;
      srcCi.pQueueFamilyIndices = qFamilies;

      VmaAllocationCreateInfo devAllocCi{};
      devAllocCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
      if (vmaCreateBuffer(m_vk->getVmaAllocator(), &srcCi, &devAllocCi,
                          &m_xferDevSrcBuf, &m_xferDevSrcAlloc,
                          nullptr) != VK_SUCCESS) {
        throw std::runtime_error("InterconnectBench: failed to create concurrent D2H src");
      }

      VkBufferCreateInfo dstCi = srcCi;
      VmaAllocationCreateInfo hostAllocCi{};
      hostAllocCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
      hostAllocCi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
      VmaAllocationInfo hostInfo{};
      if (vmaCreateBuffer(m_vk->getVmaAllocator(), &dstCi, &hostAllocCi,
                          &m_xferHostDstBuf, &m_xferHostDstAlloc,
                          &hostInfo) != VK_SUCCESS) {
        throw std::runtime_error("InterconnectBench: failed to create concurrent D2H dst");
      }
      m_xferHostDstMapped = hostInfo.pMappedData;
      std::memset(m_xferHostDstMapped, 0, kDmaBytes);

      if (m_vk->hasSecondaryTransferQueue() &&
          m_secondaryTransferCmd != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo hostUploadCi{};
        hostUploadCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        hostUploadCi.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo upInfo{};
        if (vmaCreateBuffer(m_vk->getVmaAllocator(), &srcCi, &hostUploadCi,
                            &m_xferHostSrcBuf, &m_xferHostSrcAlloc,
                            &upInfo) != VK_SUCCESS) {
          throw std::runtime_error(
              "InterconnectBench: failed to create concurrent H2D src");
        }
        m_xferHostSrcMapped = upInfo.pMappedData;
        auto *upU32 = static_cast<uint32_t *>(m_xferHostSrcMapped);
        for (size_t i = 0; i < words; ++i) {
          upU32[i] = 0x3CC30000u ^ static_cast<uint32_t>(i);
        }
        if (vmaCreateBuffer(m_vk->getVmaAllocator(), &srcCi, &devAllocCi,
                            &m_xferDevDstBuf, &m_xferDevDstAlloc,
                            nullptr) != VK_SUCCESS) {
          throw std::runtime_error(
              "InterconnectBench: failed to create concurrent H2D dst");
        }
        m_pinnedDownload =
            m_vk->createPinnedHostBuffer(65536, false, &m_mappedDownload);
      }

      // Initialize m_xferDevSrcBuf via m_pinnedUpload copy on compute queue
      auto *initPtr = static_cast<uint32_t *>(m_mappedUpload);
      for (size_t i = 0; i < words; ++i) {
        initPtr[i] = 0x96690000u ^ static_cast<uint32_t>(i);
      }
      vkResetCommandBuffer(m_primaryCmd, 0);
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(m_primaryCmd, &bi);
      VkBufferCopy copyRegion{0, 0, kDmaBytes};
      vkCmdCopyBuffer(m_primaryCmd, m_vk->getVkBuffer(m_pinnedUpload),
                      m_xferDevSrcBuf, 1, &copyRegion);
      vkEndCommandBuffer(m_primaryCmd);
      vkResetFences(m_vk->getDevice(), 1, &m_primaryFence);
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &m_primaryCmd;
      {
        std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
        vkQueueSubmit(m_vk->getComputeQueue(), 1, &si, m_primaryFence);
      }
      vkWaitForFences(m_vk->getDevice(), 1, &m_primaryFence, VK_TRUE,
                      UINT64_MAX);
      // Restore H2D pattern in m_mappedUpload
      for (size_t i = 0; i < words; ++i) {
        initPtr[i] = 0x3CC30000u ^ static_cast<uint32_t>(i);
      }
    } else {
      std::vector<uint32_t> initWords(words);
      for (size_t i = 0; i < words; ++i) {
        initWords[i] = 0x96690000u ^ static_cast<uint32_t>(i);
      }
      m_devSrc = m_vk->createBuffer(kDmaBytes, initWords.data());
      m_pinnedDownload =
          m_vk->createPinnedHostBuffer(kDmaBytes, false, &m_mappedDownload);
      std::memset(m_mappedDownload, 0, kDmaBytes);
    }
  } else if (config_idx == 3) {
    m_rebarBuf = m_vk->createRebarBuffer(kRebarBytes, &m_mappedRebar);
    if (!m_rebarBuf || !m_mappedRebar) {
      throw std::runtime_error("InterconnectBench: failed to allocate ReBAR buffer");
    }
    std::memset(m_mappedRebar, 0, kRebarBytes);
    uint32_t zeros[4] = {0, 0, 0, 0};
    m_verifyOutBuf = m_vk->createBuffer(sizeof(zeros), zeros);
  } else if (config_idx == 4) {
    m_pinnedUpload = m_vk->createPinnedHostBuffer(4096, true, &m_mappedUpload);
    std::memset(m_mappedUpload, 0, 4096);
    uint32_t zeros[4] = {0, 0, 0, 0};
    m_verifyOutBuf = m_vk->createBuffer(sizeof(zeros), zeros);
  } else if (config_idx == 5) {
    // Device group P2P logical device setup
    uint32_t groupCount = 0;
    vkEnumeratePhysicalDeviceGroups(m_vk->getInstance(), &groupCount, nullptr);
    std::vector<VkPhysicalDeviceGroupProperties> groups(groupCount);
    for (auto &g : groups) {
      g.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
    }
    vkEnumeratePhysicalDeviceGroups(m_vk->getInstance(), &groupCount,
                                    groups.data());
    const VkPhysicalDeviceGroupProperties *selectedGroup = nullptr;
    for (const auto &g : groups) {
      if (g.physicalDeviceCount >= 2) {
        selectedGroup = &g;
        break;
      }
    }
    if (!selectedGroup) {
      throw std::runtime_error("No multi-GPU VkDeviceGroup available");
    }

    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(selectedGroup->physicalDevices[0],
                                             &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(selectedGroup->physicalDevices[0],
                                             &qCount, qProps.data());
    uint32_t qFam = 0;
    for (uint32_t q = 0; q < qCount; ++q) {
      if (qProps[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        qFam = q;
        break;
      }
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qCi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qCi.queueFamilyIndex = qFam;
    qCi.queueCount = 1;
    qCi.pQueuePriorities = &prio;

    VkDeviceGroupDeviceCreateInfo dgCi{
        VK_STRUCTURE_TYPE_DEVICE_GROUP_DEVICE_CREATE_INFO};
    dgCi.physicalDeviceCount = 2;
    dgCi.pPhysicalDevices = selectedGroup->physicalDevices;

    VkDeviceCreateInfo devCi{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devCi.pNext = &dgCi;
    devCi.queueCreateInfoCount = 1;
    devCi.pQueueCreateInfos = &qCi;
    if (vkCreateDevice(selectedGroup->physicalDevices[0], &devCi, nullptr,
                       &m_p2pDevice) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create multi-GPU device group");
    }
    vkGetDeviceQueue(m_p2pDevice, qFam, 0, &m_p2pQueue);

    VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolCi.queueFamilyIndex = qFam;
    poolCi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(m_p2pDevice, &poolCi, nullptr, &m_p2pPool);

    VkCommandBufferAllocateInfo cmdAi{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdAi.commandPool = m_p2pPool;
    cmdAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAi.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_p2pDevice, &cmdAi, &m_p2pCmd);

    VkFenceCreateInfo fCi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(m_p2pDevice, &fCi, nullptr, &m_p2pFence);

    VkBufferCreateInfo bCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bCi.size = kDmaBytes;
    bCi.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(m_p2pDevice, &bCi, nullptr, &m_p2pSrcBuf);
    vkCreateBuffer(m_p2pDevice, &bCi, nullptr, &m_p2pDstBuf);
    vkCreateBuffer(m_p2pDevice, &bCi, nullptr, &m_p2pStageBuf);

    VkMemoryRequirements memReq{};
    vkGetBufferMemoryRequirements(m_p2pDevice, m_p2pSrcBuf, &memReq);
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(selectedGroup->physicalDevices[0],
                                        &memProps);
    uint32_t devMemType = UINT32_MAX;
    uint32_t hostMemType = UINT32_MAX;
    for (uint32_t t = 0; t < memProps.memoryTypeCount; ++t) {
      if ((memReq.memoryTypeBits & (1u << t)) == 0) continue;
      auto flags = memProps.memoryTypes[t].propertyFlags;
      if (devMemType == UINT32_MAX &&
          (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        devMemType = t;
      }
      if (hostMemType == UINT32_MAX &&
          (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        hostMemType = t;
      }
    }
    VkPeerMemoryFeatureFlags peerFlags = 0;
    vkGetDeviceGroupPeerMemoryFeatures(m_p2pDevice,
                                       memProps.memoryTypes[devMemType].heapIndex,
                                       0, 1, &peerFlags);
    if ((peerFlags & VK_PEER_MEMORY_FEATURE_COPY_DST_BIT) == 0) {
      throw std::runtime_error(
          "Device group heap does not support VK_PEER_MEMORY_FEATURE_COPY_DST_BIT");
    }

    VkMemoryAllocateFlagsInfo mask0{
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    mask0.flags = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT;
    mask0.deviceMask = 0x1u;
    VkMemoryAllocateInfo a0{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    a0.pNext = &mask0;
    a0.allocationSize = memReq.size;
    a0.memoryTypeIndex = devMemType;
    vkAllocateMemory(m_p2pDevice, &a0, nullptr, &m_p2pSrcMem);

    VkMemoryAllocateFlagsInfo mask1 = mask0;
    mask1.deviceMask = 0x2u;
    VkMemoryAllocateInfo a1 = a0;
    a1.pNext = &mask1;
    vkAllocateMemory(m_p2pDevice, &a1, nullptr, &m_p2pDstMem);

    VkMemoryAllocateInfo aHost{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    aHost.pNext = &mask0;
    aHost.allocationSize = memReq.size;
    aHost.memoryTypeIndex = hostMemType;
    vkAllocateMemory(m_p2pDevice, &aHost, nullptr, &m_p2pStageMem);
    vkMapMemory(m_p2pDevice, m_p2pStageMem, 0, kDmaBytes, 0,
                &m_p2pStageMapped);

    VkBindBufferMemoryDeviceGroupInfo bindDg0{
        VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_DEVICE_GROUP_INFO};
    uint32_t devIndices0[2] = {0, 0};
    bindDg0.deviceIndexCount = 2;
    bindDg0.pDeviceIndices = devIndices0;

    VkBindBufferMemoryDeviceGroupInfo bindDg1 = bindDg0;
    uint32_t devIndices1[2] = {1, 1};
    bindDg1.pDeviceIndices = devIndices1;

    VkBindBufferMemoryInfo binds[3]{};
    binds[0].sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    binds[0].pNext = &bindDg0;
    binds[0].buffer = m_p2pSrcBuf;
    binds[0].memory = m_p2pSrcMem;

    binds[1].sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    binds[1].pNext = &bindDg1;
    binds[1].buffer = m_p2pDstBuf;
    binds[1].memory = m_p2pDstMem;

    binds[2].sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
    binds[2].pNext = &bindDg0;
    binds[2].buffer = m_p2pStageBuf;
    binds[2].memory = m_p2pStageMem;
    vkBindBufferMemory2(m_p2pDevice, 3, binds);

    auto *stageU32 = static_cast<uint32_t *>(m_p2pStageMapped);
    for (size_t i = 0; i < kDmaBytes / sizeof(uint32_t); ++i) {
      stageU32[i] = 0x77880000u ^ static_cast<uint32_t>(i);
    }
    vkResetCommandBuffer(m_p2pCmd, 0);
    VkDeviceGroupCommandBufferBeginInfo dgBegin{
        VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO};
    dgBegin.deviceMask = 0x1u;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.pNext = &dgBegin;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_p2pCmd, &bi);
    VkBufferCopy region{0, 0, kDmaBytes};
    vkCmdCopyBuffer(m_p2pCmd, m_p2pStageBuf, m_p2pSrcBuf, 1, &region);
    vkEndCommandBuffer(m_p2pCmd);
    vkResetFences(m_p2pDevice, 1, &m_p2pFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_p2pCmd;
    vkQueueSubmit(m_p2pQueue, 1, &si, m_p2pFence);
    vkWaitForFences(m_p2pDevice, 1, &m_p2pFence, VK_TRUE, UINT64_MAX);
  }
}

void InterconnectBench::ReleaseConfigResources(uint32_t /*config_idx*/) {
  if (!m_vk) return;
  m_vk->waitIdle();

  if (m_pinnedUpload) {
    m_vk->releaseBuffer(m_pinnedUpload);
    m_pinnedUpload = nullptr;
    m_mappedUpload = nullptr;
  }
  if (m_devDst) {
    m_vk->releaseBuffer(m_devDst);
    m_devDst = nullptr;
  }
  if (m_devSrc) {
    m_vk->releaseBuffer(m_devSrc);
    m_devSrc = nullptr;
  }
  if (m_pinnedDownload) {
    m_vk->releaseBuffer(m_pinnedDownload);
    m_pinnedDownload = nullptr;
    m_mappedDownload = nullptr;
  }
  if (m_xferHostSrcBuf != VK_NULL_HANDLE &&
      m_vk->getVmaAllocator() != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_vk->getVmaAllocator(), m_xferHostSrcBuf,
                     m_xferHostSrcAlloc);
    m_xferHostSrcBuf = VK_NULL_HANDLE;
    m_xferHostSrcAlloc = VK_NULL_HANDLE;
    m_xferHostSrcMapped = nullptr;
  }
  if (m_xferDevDstBuf != VK_NULL_HANDLE &&
      m_vk->getVmaAllocator() != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_vk->getVmaAllocator(), m_xferDevDstBuf,
                     m_xferDevDstAlloc);
    m_xferDevDstBuf = VK_NULL_HANDLE;
    m_xferDevDstAlloc = VK_NULL_HANDLE;
  }
  if (m_xferDevSrcBuf != VK_NULL_HANDLE &&
      m_vk->getVmaAllocator() != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_vk->getVmaAllocator(), m_xferDevSrcBuf,
                     m_xferDevSrcAlloc);
    m_xferDevSrcBuf = VK_NULL_HANDLE;
    m_xferDevSrcAlloc = VK_NULL_HANDLE;
  }
  if (m_xferHostDstBuf != VK_NULL_HANDLE &&
      m_vk->getVmaAllocator() != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_vk->getVmaAllocator(), m_xferHostDstBuf,
                     m_xferHostDstAlloc);
    m_xferHostDstBuf = VK_NULL_HANDLE;
    m_xferHostDstAlloc = VK_NULL_HANDLE;
    m_xferHostDstMapped = nullptr;
  }
  if (m_rebarBuf) {
    m_vk->releaseBuffer(m_rebarBuf);
    m_rebarBuf = nullptr;
    m_mappedRebar = nullptr;
  }
  if (m_verifyOutBuf) {
    m_vk->releaseBuffer(m_verifyOutBuf);
    m_verifyOutBuf = nullptr;
  }

  if (m_p2pDevice != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_p2pDevice);
    if (m_p2pStageMem != VK_NULL_HANDLE) {
      vkUnmapMemory(m_p2pDevice, m_p2pStageMem);
      m_p2pStageMapped = nullptr;
    }
    if (m_p2pSrcBuf != VK_NULL_HANDLE)
      vkDestroyBuffer(m_p2pDevice, m_p2pSrcBuf, nullptr);
    if (m_p2pDstBuf != VK_NULL_HANDLE)
      vkDestroyBuffer(m_p2pDevice, m_p2pDstBuf, nullptr);
    if (m_p2pStageBuf != VK_NULL_HANDLE)
      vkDestroyBuffer(m_p2pDevice, m_p2pStageBuf, nullptr);
    if (m_p2pSrcMem != VK_NULL_HANDLE)
      vkFreeMemory(m_p2pDevice, m_p2pSrcMem, nullptr);
    if (m_p2pDstMem != VK_NULL_HANDLE)
      vkFreeMemory(m_p2pDevice, m_p2pDstMem, nullptr);
    if (m_p2pStageMem != VK_NULL_HANDLE)
      vkFreeMemory(m_p2pDevice, m_p2pStageMem, nullptr);
    if (m_p2pFence != VK_NULL_HANDLE)
      vkDestroyFence(m_p2pDevice, m_p2pFence, nullptr);
    if (m_p2pPool != VK_NULL_HANDLE)
      vkDestroyCommandPool(m_p2pDevice, m_p2pPool, nullptr);
    vkDestroyDevice(m_p2pDevice, nullptr);
    m_p2pDevice = VK_NULL_HANDLE;
    m_p2pQueue = VK_NULL_HANDLE;
    m_p2pPool = VK_NULL_HANDLE;
    m_p2pCmd = VK_NULL_HANDLE;
    m_p2pFence = VK_NULL_HANDLE;
    m_p2pSrcBuf = VK_NULL_HANDLE;
    m_p2pDstBuf = VK_NULL_HANDLE;
    m_p2pStageBuf = VK_NULL_HANDLE;
    m_p2pSrcMem = VK_NULL_HANDLE;
    m_p2pDstMem = VK_NULL_HANDLE;
    m_p2pStageMem = VK_NULL_HANDLE;
  }
}

void InterconnectBench::Run(uint32_t config_idx) {
  if (!m_vk) return;
  VkDevice dev = m_vk->getDevice();
  double tsPeriodNs =
      static_cast<double>(m_vk->getPhysicalDeviceProperties().limits.timestampPeriod);

  if (config_idx == 0 || config_idx == 1) {
    VkBuffer src = (config_idx == 0) ? m_vk->getVkBuffer(m_pinnedUpload)
                                     : m_vk->getVkBuffer(m_devSrc);
    VkBuffer dst = (config_idx == 0) ? m_vk->getVkBuffer(m_devDst)
                                     : m_vk->getVkBuffer(m_pinnedDownload);

    vkResetCommandBuffer(m_primaryCmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_primaryCmd, &bi);
    vkCmdResetQueryPool(m_primaryCmd, m_tsPool, 0, 2);
    vkCmdWriteTimestamp(m_primaryCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        m_tsPool, 0);
    VkBufferCopy region{0, 0, kDmaBytes};
    vkCmdCopyBuffer(m_primaryCmd, src, dst, 1, &region);
    if (config_idx == 1) {
      m_vk->cmdPipelineMemoryBarrier2(
          m_primaryCmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
          VK_ACCESS_2_HOST_READ_BIT);
    }
    vkCmdWriteTimestamp(m_primaryCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        m_tsPool, 1);
    vkEndCommandBuffer(m_primaryCmd);

    vkResetFences(dev, 1, &m_primaryFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_primaryCmd;
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
      vkQueueSubmit(m_vk->getComputeQueue(), 1, &si, m_primaryFence);
    }
    vkWaitForFences(dev, 1, &m_primaryFence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();

    uint64_t ts[2] = {0, 0};
    if (tsPeriodNs > 0.0 &&
        vkGetQueryPoolResults(dev, m_tsPool, 0, 2, sizeof(ts), ts,
                              sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT |
                                  VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS &&
        ts[1] > ts[0]) {
      m_lastDurationMs[config_idx] =
          static_cast<double>(ts[1] - ts[0]) * tsPeriodNs * 1e-6;
    } else {
      m_lastDurationMs[config_idx] =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
  } else if (config_idx == 2) {
    if (m_transferCmd != VK_NULL_HANDLE &&
        m_secondaryTransferCmd != VK_NULL_HANDLE &&
        m_vk->hasSecondaryTransferQueue() &&
        m_xferHostSrcBuf != VK_NULL_HANDLE &&
        m_xferDevDstBuf != VK_NULL_HANDLE &&
        m_xferDevSrcBuf != VK_NULL_HANDLE) {
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VkBufferCopy region{0, 0, kDmaBytes};

      vkResetCommandBuffer(m_transferCmd, 0);
      vkBeginCommandBuffer(m_transferCmd, &bi);
      vkCmdCopyBuffer(m_transferCmd, m_xferHostSrcBuf, m_xferDevDstBuf, 1,
                      &region);
      vkEndCommandBuffer(m_transferCmd);

      vkResetCommandBuffer(m_secondaryTransferCmd, 0);
      vkBeginCommandBuffer(m_secondaryTransferCmd, &bi);
      vkCmdCopyBuffer(m_secondaryTransferCmd, m_xferDevSrcBuf, m_xferHostDstBuf,
                      1, &region);
      vkEndCommandBuffer(m_secondaryTransferCmd);

      VkFence fences[2] = {m_transferFence, m_secondaryTransferFence};
      vkResetFences(dev, 2, fences);

      VkSubmitInfo si0{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si0.commandBufferCount = 1;
      si0.pCommandBuffers = &m_transferCmd;
      VkSubmitInfo si1{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si1.commandBufferCount = 1;
      si1.pCommandBuffers = &m_secondaryTransferCmd;

      auto t0 = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
        vkQueueSubmit(m_vk->getTransferQueue(), 1, &si0, m_transferFence);
        vkQueueSubmit(m_vk->getSecondaryTransferQueue(), 1, &si1,
                      m_secondaryTransferFence);
      }
      vkWaitForFences(dev, 2, fences, VK_TRUE, UINT64_MAX);
      auto t1 = std::chrono::steady_clock::now();
      m_lastDurationMs[2] =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else if (m_transferCmd != VK_NULL_HANDLE &&
               m_xferDevSrcBuf != VK_NULL_HANDLE) {
      vkResetCommandBuffer(m_primaryCmd, 0);
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(m_primaryCmd, &bi);
      VkBufferCopy region{0, 0, kDmaBytes};
      vkCmdCopyBuffer(m_primaryCmd, m_vk->getVkBuffer(m_pinnedUpload),
                      m_vk->getVkBuffer(m_devDst), 1, &region);
      vkEndCommandBuffer(m_primaryCmd);

      vkResetCommandBuffer(m_transferCmd, 0);
      vkBeginCommandBuffer(m_transferCmd, &bi);
      vkCmdCopyBuffer(m_transferCmd, m_xferDevSrcBuf, m_xferHostDstBuf, 1,
                      &region);
      vkEndCommandBuffer(m_transferCmd);

      VkFence fences[2] = {m_primaryFence, m_transferFence};
      vkResetFences(dev, 2, fences);

      VkSubmitInfo si0{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si0.commandBufferCount = 1;
      si0.pCommandBuffers = &m_primaryCmd;
      VkSubmitInfo si1{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si1.commandBufferCount = 1;
      si1.pCommandBuffers = &m_transferCmd;

      auto t0 = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
        vkQueueSubmit(m_vk->getComputeQueue(), 1, &si0, m_primaryFence);
        vkQueueSubmit(m_vk->getTransferQueue(), 1, &si1, m_transferFence);
      }
      vkWaitForFences(dev, 2, fences, VK_TRUE, UINT64_MAX);
      auto t1 = std::chrono::steady_clock::now();
      m_lastDurationMs[2] =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    } else {
      vkResetCommandBuffer(m_primaryCmd, 0);
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(m_primaryCmd, &bi);
      VkBufferCopy region{0, 0, kDmaBytes};
      vkCmdCopyBuffer(m_primaryCmd, m_vk->getVkBuffer(m_pinnedUpload),
                      m_vk->getVkBuffer(m_devDst), 1, &region);
      vkCmdCopyBuffer(m_primaryCmd, m_vk->getVkBuffer(m_devSrc),
                      m_vk->getVkBuffer(m_pinnedDownload), 1, &region);
      vkEndCommandBuffer(m_primaryCmd);
      vkResetFences(dev, 1, &m_primaryFence);
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &m_primaryCmd;
      auto t0 = std::chrono::steady_clock::now();
      {
        std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
        vkQueueSubmit(m_vk->getComputeQueue(), 1, &si, m_primaryFence);
      }
      vkWaitForFences(dev, 1, &m_primaryFence, VK_TRUE, UINT64_MAX);
      auto t1 = std::chrono::steady_clock::now();
      m_lastDurationMs[2] =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
  } else if (config_idx == 3) {
    m_rebarSeed += 0x01010101u;
    const size_t wordCount = kRebarBytes / sizeof(uint32_t);
    auto *dst = static_cast<uint32_t *>(m_mappedRebar);
    auto t0 = std::chrono::steady_clock::now();
#if defined(__x86_64__) || defined(_M_X64)
    if (cpuSupportsAvx2ForRebar()) {
      streamWriteRebarAvx2(dst, wordCount, m_rebarSeed);
    } else
#endif
    {
      for (size_t i = 0; i < wordCount; ++i) {
        dst[i] = m_rebarSeed ^ static_cast<uint32_t>(i);
      }
      std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    auto t1 = std::chrono::steady_clock::now();
    m_lastDurationMs[3] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  } else if (config_idx == 4) {
    auto *mail = static_cast<uint32_t *>(m_mappedUpload);
    std::atomic_ref<uint32_t> atomicFlag(mail[0]);
    atomicFlag.store(0u, std::memory_order_release);
    uint32_t zero = 0u;
    m_vk->writeBuffer(m_verifyOutBuf, 0, sizeof(uint32_t), &zero);

    uint32_t mode = 0u;
    uint32_t count = kPingPongSteps;
    uint32_t seed = 0u;
    m_vk->setKernelArg(m_helperKernel, 0, m_pinnedUpload);
    m_vk->setKernelArg(m_helperKernel, 1, m_verifyOutBuf);
    m_vk->setKernelArg(m_helperKernel, 2, sizeof(uint32_t), &mode);
    m_vk->setKernelArg(m_helperKernel, 3, sizeof(uint32_t), &count);
    m_vk->setKernelArg(m_helperKernel, 4, sizeof(uint32_t), &seed);

    // Launch 1 workgroup asynchronously via dispatch (returns immediately before waitIdle)
    m_vk->dispatch(m_helperKernel, 1, 1, 1, 64, 1, 1);

    // Wait for GPU ready signal (state == 1)
    auto waitStart = std::chrono::steady_clock::now();
    while (atomicFlag.load(std::memory_order_acquire) != 1u) {
      if (std::chrono::steady_clock::now() - waitStart >
          std::chrono::milliseconds(500)) {
        break;
      }
    }

    auto t0 = std::chrono::steady_clock::now();
    for (uint32_t step = 0; step < kPingPongSteps; ++step) {
      uint32_t sendVal = step * 2u + 2u;
      uint32_t expectReply = sendVal + 1u;
      atomicFlag.store(sendVal, std::memory_order_release);
      while (atomicFlag.load(std::memory_order_acquire) != expectReply) {
        if (std::chrono::steady_clock::now() - t0 >
            std::chrono::milliseconds(1000)) {
          break;
        }
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    m_vk->waitIdle();
    m_lastDurationMs[4] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  } else if (config_idx == 5 && m_p2pDevice != VK_NULL_HANDLE) {
    vkResetCommandBuffer(m_p2pCmd, 0);
    VkDeviceGroupCommandBufferBeginInfo dgBegin{
        VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO};
    dgBegin.deviceMask = 0x1u;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.pNext = &dgBegin;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_p2pCmd, &bi);
    VkBufferCopy region{0, 0, kDmaBytes};
    vkCmdCopyBuffer(m_p2pCmd, m_p2pSrcBuf, m_p2pDstBuf, 1, &region);
    vkEndCommandBuffer(m_p2pCmd);

    vkResetFences(m_p2pDevice, 1, &m_p2pFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_p2pCmd;
    auto t0 = std::chrono::steady_clock::now();
    vkQueueSubmit(m_p2pQueue, 1, &si, m_p2pFence);
    vkWaitForFences(m_p2pDevice, 1, &m_p2pFence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();
    m_lastDurationMs[5] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
}

BenchmarkResult InterconnectBench::GetResult(uint32_t config_idx) const {
  uint64_t ops = kDmaBytes;
  if (config_idx == 2) {
    ops = kDmaBytes * 2ULL;
  } else if (config_idx == 3) {
    ops = kRebarBytes;
  } else if (config_idx == 4) {
    ops = static_cast<uint64_t>(kPingPongSteps) * 2ULL; // one-way coherency hops
  }
  return {ops, GetLastRunDurationMs(config_idx)};
}

bool InterconnectBench::ValidateResults(uint32_t config_idx) const {
  if (!m_vk) return false;
  constexpr size_t kSliceWords = 4096;
  const size_t totalDmaWords = kDmaBytes / sizeof(uint32_t);
  const size_t midWordOffset = totalDmaWords / 2;
  const size_t tailWordOffset = totalDmaWords - kSliceWords;

  if (config_idx == 0) {
    if (!m_devDst) return false;
    std::vector<uint32_t> slice(kSliceWords, 0);
    const size_t offsets[3] = {0, midWordOffset, tailWordOffset};
    for (size_t wordOff : offsets) {
      m_vk->readBuffer(m_devDst, wordOff * sizeof(uint32_t),
                       kSliceWords * sizeof(uint32_t), slice.data());
      for (size_t i = 0; i < kSliceWords; ++i) {
        uint32_t expected =
            0xA55A0000u ^ static_cast<uint32_t>(wordOff + i);
        if (slice[i] != expected) return false;
      }
    }
    return true;
  }
  if (config_idx == 1) {
    if (!m_mappedDownload) return false;
    const auto *dst = static_cast<const uint32_t *>(m_mappedDownload);
    const size_t offsets[3] = {0, midWordOffset, tailWordOffset};
    for (size_t wordOff : offsets) {
      for (size_t i = 0; i < kSliceWords; ++i) {
        uint32_t expected =
            0x5AA50000u ^ static_cast<uint32_t>(wordOff + i);
        if (dst[wordOff + i] != expected) return false;
      }
    }
    return true;
  }
  if (config_idx == 2) {
    std::vector<uint32_t> slice(kSliceWords, 0);
    if (m_xferDevDstBuf != VK_NULL_HANDLE && m_pinnedDownload &&
        m_mappedDownload) {
      vkResetCommandBuffer(m_primaryCmd, 0);
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(m_primaryCmd, &bi);
      VkBufferCopy region{tailWordOffset * sizeof(uint32_t), 0,
                          kSliceWords * sizeof(uint32_t)};
      vkCmdCopyBuffer(m_primaryCmd, m_xferDevDstBuf,
                      m_vk->getVkBuffer(m_pinnedDownload), 1, &region);
      vkEndCommandBuffer(m_primaryCmd);
      vkResetFences(m_vk->getDevice(), 1, &m_primaryFence);
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &m_primaryCmd;
      {
        std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
        vkQueueSubmit(m_vk->getComputeQueue(), 1, &si, m_primaryFence);
      }
      vkWaitForFences(m_vk->getDevice(), 1, &m_primaryFence, VK_TRUE,
                      UINT64_MAX);
      std::memcpy(slice.data(), m_mappedDownload,
                  kSliceWords * sizeof(uint32_t));
    } else {
      if (!m_devDst) return false;
      m_vk->readBuffer(m_devDst, tailWordOffset * sizeof(uint32_t),
                       kSliceWords * sizeof(uint32_t), slice.data());
    }
    for (size_t i = 0; i < kSliceWords; ++i) {
      if (slice[i] != (0x3CC30000u ^ static_cast<uint32_t>(tailWordOffset + i)))
        return false;
    }
    const uint32_t *hostDst = static_cast<const uint32_t *>(
        m_xferHostDstMapped ? m_xferHostDstMapped : m_mappedDownload);
    if (!hostDst) return false;
    for (size_t i = 0; i < kSliceWords; ++i) {
      if (hostDst[tailWordOffset + i] !=
          (0x96690000u ^ static_cast<uint32_t>(tailWordOffset + i)))
        return false;
    }
    return true;
  }
  if (config_idx == 3) {
    if (!m_rebarBuf || !m_verifyOutBuf || !m_helperKernel) return false;
    uint32_t zeros[4] = {0, 0, 0, 0};
    m_vk->writeBuffer(m_verifyOutBuf, 0, sizeof(zeros), zeros);
    uint32_t mode = 1u;
    uint32_t wordCount = static_cast<uint32_t>(kRebarBytes / sizeof(uint32_t));
    uint32_t seed = m_rebarSeed;
    m_vk->setKernelArg(m_helperKernel, 0, m_rebarBuf);
    m_vk->setKernelArg(m_helperKernel, 1, m_verifyOutBuf);
    m_vk->setKernelArg(m_helperKernel, 2, sizeof(uint32_t), &mode);
    m_vk->setKernelArg(m_helperKernel, 3, sizeof(uint32_t), &wordCount);
    m_vk->setKernelArg(m_helperKernel, 4, sizeof(uint32_t), &seed);
    m_vk->dispatch(m_helperKernel, (wordCount + 63u) / 64u, 1, 1, 64, 1, 1);
    m_vk->waitIdle();

    uint32_t status[4] = {1u, 0u, 0u, 0u};
    m_vk->readBuffer(m_verifyOutBuf, 0, sizeof(status), status);
    return (status[0] == 0u) && (status[1] == m_rebarSeed);
  }
  if (config_idx == 4) {
    if (!m_verifyOutBuf) return false;
    uint32_t completed = 0;
    m_vk->readBuffer(m_verifyOutBuf, 0, sizeof(uint32_t), &completed);
    return completed == kPingPongSteps;
  }
  if (config_idx == 5) {
    if (m_p2pDevice == VK_NULL_HANDLE || !m_p2pStageMapped) return false;
    std::memset(m_p2pStageMapped, 0, kSliceWords * sizeof(uint32_t));
    vkResetCommandBuffer(m_p2pCmd, 0);
    VkDeviceGroupCommandBufferBeginInfo dgBegin{
        VK_STRUCTURE_TYPE_DEVICE_GROUP_COMMAND_BUFFER_BEGIN_INFO};
    dgBegin.deviceMask = 0x1u;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.pNext = &dgBegin;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_p2pCmd, &bi);
    VkBufferCopy region{0, 0, kSliceWords * sizeof(uint32_t)};
    vkCmdCopyBuffer(m_p2pCmd, m_p2pDstBuf, m_p2pStageBuf, 1, &region);
    vkEndCommandBuffer(m_p2pCmd);
    vkResetFences(m_p2pDevice, 1, &m_p2pFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_p2pCmd;
    vkQueueSubmit(m_p2pQueue, 1, &si, m_p2pFence);
    vkWaitForFences(m_p2pDevice, 1, &m_p2pFence, VK_TRUE, UINT64_MAX);
    const auto *stageU32 = static_cast<const uint32_t *>(m_p2pStageMapped);
    for (size_t i = 0; i < kSliceWords; ++i) {
      if (stageU32[i] != (0x77880000u ^ static_cast<uint32_t>(i))) return false;
    }
    return true;
  }
  return false;
}

void InterconnectBench::Teardown() {
  ReleaseConfigResources(0);
  if (m_vk) {
    VkDevice dev = m_vk->getDevice();
    if (m_helperKernel) {
      m_vk->releaseKernel(m_helperKernel);
      m_helperKernel = nullptr;
    }
    if (m_tsPool != VK_NULL_HANDLE) {
      vkDestroyQueryPool(dev, m_tsPool, nullptr);
      m_tsPool = VK_NULL_HANDLE;
    }
    if (m_primaryFence != VK_NULL_HANDLE) {
      vkDestroyFence(dev, m_primaryFence, nullptr);
      m_primaryFence = VK_NULL_HANDLE;
    }
    if (m_primaryPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(dev, m_primaryPool, nullptr);
      m_primaryPool = VK_NULL_HANDLE;
      m_primaryCmd = VK_NULL_HANDLE;
    }
    if (m_secondaryTransferFence != VK_NULL_HANDLE) {
      vkDestroyFence(dev, m_secondaryTransferFence, nullptr);
      m_secondaryTransferFence = VK_NULL_HANDLE;
    }
    if (m_secondaryTransferPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(dev, m_secondaryTransferPool, nullptr);
      m_secondaryTransferPool = VK_NULL_HANDLE;
      m_secondaryTransferCmd = VK_NULL_HANDLE;
    }
    if (m_transferFence != VK_NULL_HANDLE) {
      vkDestroyFence(dev, m_transferFence, nullptr);
      m_transferFence = VK_NULL_HANDLE;
    }
    if (m_transferPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(dev, m_transferPool, nullptr);
      m_transferPool = VK_NULL_HANDLE;
      m_transferCmd = VK_NULL_HANDLE;
    }
  }
  m_vk = nullptr;
  m_context = nullptr;
}
