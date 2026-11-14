#pragma once

#include "IComputeContext.h"
#include "BlockSuballocator.h"
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>
#include "vk_mem_alloc.h"

class VulkanContext : public IComputeContext {
public:
  VulkanContext(bool verbose = false, bool debug = false,
                const std::vector<const char *> &extraInstanceExtensions = {});
  ~VulkanContext();

  VulkanContext(const VulkanContext &) = delete;
  VulkanContext &operator=(const VulkanContext &) = delete;

  // IComputeContext interface
  ComputeBackend getBackend() const override { return ComputeBackend::Vulkan; }
  bool isAvailable() const override { return instance != VK_NULL_HANDLE; }
  const std::vector<DeviceInfo> &getDevices() const override;
  void pickDevice(uint32_t index) override;
  DeviceInfo getCurrentDeviceInfo() const override;
  uint32_t getSelectedDeviceIndex() const override {
    return selectedDeviceIndex;
  }

  // Buffer management (VMA-backed)
  ComputeBuffer createBuffer(size_t size,
                             const void *host_ptr = nullptr) override;
  ComputeBuffer createPinnedHostBuffer(size_t size, bool forUpload,
                                       void **outMappedPtr = nullptr);
  ComputeBuffer createRebarBuffer(size_t size, void **outMappedPtr = nullptr);
  void *getBufferMappedPtr(ComputeBuffer buffer) const;
  void writeBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                   const void *host_ptr) override;
  void readBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                  void *host_ptr) const override;
  void releaseBuffer(ComputeBuffer buffer) override;
  VkDeviceAddress getBufferDeviceAddress(ComputeBuffer buffer) const;

  struct VmaBudgetSummary {
    bool available = false;
    uint64_t usageBytes = 0;
    uint64_t budgetBytes = 0;
  };
  VmaAllocator getVmaAllocator() const { return m_vmaAllocator; }
  VmaBudgetSummary queryVramBudget() const;

  static constexpr VkComponentTypeKHR kVkComponentTypeFloat6E3M2EXT =
      static_cast<VkComponentTypeKHR>(1000672001);
  static constexpr VkComponentTypeKHR kVkComponentTypeFloat4E2M1EXT =
      static_cast<VkComponentTypeKHR>(1000672002);

  struct CoopMatrixTile {
    bool supported = false;
    uint32_t M = 16;
    uint32_t N = 16;
    uint32_t K = 16;
    uint32_t subgroupSize = 32;
  };
  CoopMatrixTile selectCoopMatrixTile(VkComponentTypeKHR abType,
                                      VkComponentTypeKHR cType) const;

  // Kernel management
  ComputeKernel createKernel(const std::string &file_name,
                             const std::string &kernel_name,
                             uint32_t num_buffer_args) override;
  ComputeKernel createKernelWithSpec(const std::string &file_name,
                                     const std::string &kernel_name,
                                     uint32_t num_buffer_args,
                                     uint32_t spec_id,
                                     uint32_t spec_val);
  ComputeKernel createKernelWithSpecConstants(
      const std::string &file_name, const std::string &kernel_name,
      uint32_t num_buffer_args,
      const std::vector<uint32_t> &spec_uints) override;
  ComputeKernel createRTPipeline(const std::string &rgen_path,
                                 const std::string &rmiss_path,
                                 const std::vector<std::string> &rchit_paths,
                                 const std::vector<std::string> &rahit_paths,
                                 const std::vector<std::string> &rint_paths,
                                 uint32_t num_buffer_args,
                                 uint32_t spec_const0 = 7u,
                                 bool has_spec_const0 = false) override;
  void setKernelArg(ComputeKernel kernel, uint32_t arg_index,
                    ComputeBuffer buffer) override;
  void setKernelAS(ComputeKernel kernel, uint32_t arg_index,
                   AccelerationStructure as) override;
  void setKernelArg(ComputeKernel kernel, uint32_t arg_index, size_t arg_size,
                    const void *arg_value) override;
  void dispatch(ComputeKernel kernel, uint32_t grid_x, uint32_t grid_y,
                uint32_t grid_z, uint32_t block_x, uint32_t block_y,
                uint32_t block_z) override;
  void releaseKernel(ComputeKernel kernel) override;
  void waitIdle() override;

  // Command buffer batching (Optimization O-2 + Group 3 barrier control)
  void beginBatch(uint32_t expected_dispatches = 0,
                  bool inter_dispatch_barrier = false) override;
  void endBatch() override;
  bool isBatching() const override { return m_inBatch; }

  // Persistent disk-backed VkPipelineCache
  VkPipelineCache getPipelineCache() const { return m_pipelineCache; }
  void flushPipelineCacheToDisk() const;

  // Shared GUI/Compute presentation + multi-queue accessors
  void setPresentationSurface(VkSurfaceKHR surface) {
    m_presentationSurface = surface;
  }
  VkQueue getPresentationQueue() const {
    return presentationQueue != VK_NULL_HANDLE ? presentationQueue
                                               : graphicsQueue;
  }
  uint32_t getPresentationQueueFamilyIndex() const {
    return presentationQueueFamilyIndex;
  }
  std::recursive_mutex &getQueueMutex() const { return m_queueMutex; }

  bool hasAsyncComputeQueue() const {
    return asyncComputeQueue != VK_NULL_HANDLE;
  }
  VkQueue getAsyncComputeQueue() const { return asyncComputeQueue; }
  uint32_t getAsyncComputeQueueFamilyIndex() const {
    return asyncComputeQueueFamilyIndex;
  }

  bool hasDedicatedTransferQueue() const {
    return transferQueue != VK_NULL_HANDLE;
  }
  VkQueue getTransferQueue() const { return transferQueue; }
  bool hasSecondaryTransferQueue() const {
    return secondaryTransferQueue != VK_NULL_HANDLE;
  }
  VkQueue getSecondaryTransferQueue() const { return secondaryTransferQueue; }
  uint32_t getTransferQueueFamilyIndex() const {
    return transferQueueFamilyIndex;
  }

  // Zero-disk on-GPU shared preview images & in-memory profile JSONs
  struct SharedPreviewImage {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t generation = 0;
  };
  bool captureBufferToSharedPreviewImage(const std::string &key,
                                         ComputeBuffer hdrBufA,
                                         ComputeBuffer hdrBufB,
                                         uint32_t width, uint32_t height,
                                         uint32_t mode,
                                         const std::string &kernelDir);
  bool uploadRgbToSharedPreviewImage(const std::string &key,
                                     uint32_t width, uint32_t height,
                                     const uint8_t *rgbPixels);
  bool getSharedPreviewImage(const std::string &key,
                             SharedPreviewImage &outInfo) const;
  void clearSharedPreviewImages();
  void storeInMemoryJson(const std::string &key,
                         const std::string &jsonContent);
  bool getInMemoryJson(const std::string &key,
                       std::string &outJsonContent) const;

  void setExpectedKernelCount(uint32_t count) override;
  void notifyKernelCreated(const std::string &kernel_name) override;
  void setVerbose(bool v) override { verbose = v; }
  bool isVerbose() const override { return verbose; }
  void setQuiet(bool q) override { quiet = q; }
  bool isQuiet() const override { return quiet; }

  VkPhysicalDevice getVulkanPhysicalDevice() const override {
    return physicalDevice;
  }
  VkDevice getVulkanDevice() const override { return device; }
  void *getVulkanContext() const override { return (void *)this; }

  // Hardware GPU timestamp profiling
  void startTiming() override { startTiming(VK_NULL_HANDLE); }
  double stopTiming() override { return stopTiming(VK_NULL_HANDLE); }
  void startTiming(VkCommandBuffer cmd);
  double stopTiming(VkCommandBuffer cmd);
  bool hasGpuTiming() const override { return timingSupported; }

  // Vulkan-specific accessors
  VkInstance getInstance() const { return instance; }
  VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
  VkDevice getDevice() const { return device; }
  uint32_t getComputeQueueFamilyIndex() const {
    return computeQueueFamilyIndex;
  }
  VkQueue getComputeQueue() const { return computeQueue; }
  uint32_t getGraphicsQueueFamilyIndex() const {
    return graphicsQueueFamilyIndex;
  }
  VkQueue getGraphicsQueue() const { return graphicsQueue; }
  const VkPhysicalDeviceProperties &getPhysicalDeviceProperties() const {
    return properties;
  }
  VkBuffer getVkBuffer(ComputeBuffer buffer) const;

  bool isExtensionEnabled(const std::string &ext) const {
    return enabledExtensionsSet.find(ext) != enabledExtensionsSet.end();
  }
  bool isRTMaint1Supported() const {
    return isExtensionEnabled("VK_KHR_ray_tracing_maintenance1");
  }
  bool isDGCSupported() const {
    return dgcSupported;
  }
  bool isDGCExecutionSetSupported() const {
    return dgcExecutionSetSupported;
  }
  bool isMaintenance5Supported() const {
    return maintenance5Supported;
  }
  bool isWorkGraphsSupported() const {
    return workGraphsSupported;
  }
  bool isCooperativeVectorSupported() const {
    return cooperativeVectorSupported;
  }
  bool isOpacityMicromapSupported() const {
    return opacityMicromapSupported;
  }
  bool isStructuredSparsitySupported() const {
    return structuredSparsitySupported;
  }
  bool isSERSupported() const {
    return serSupported;
  }
  bool isPipelineExecutablePropertiesSupported() const {
    return pipelineExecutablePropertiesSupported;
  }
  bool isDynamicRenderingSupported() const {
    return dynamicRenderingSupported;
  }
  PFN_vkCmdBeginRenderingKHR getCmdBeginRendering() const {
    return vkCmdBeginRendering_ptr;
  }
  PFN_vkCmdEndRenderingKHR getCmdEndRendering() const {
    return vkCmdEndRendering_ptr;
  }
  KernelResourceUsage getKernelResourceUsage(ComputeKernel kernel) const override;
  bool isKernelBf16Emulated(ComputeKernel kernel) const;

  bool isPerformanceQuerySupported() const {
    return performanceQuerySupported;
  }
  bool hasPerformanceQuery() const override {
    return performanceQueryEnabled;
  }
  void startPerformanceQuery() override;
  HardwarePerformanceCounters stopPerformanceQuery() override;
  HardwarePerformanceCounters getLastPerformanceCounters() const override {
    return lastPerfCounters;
  }

  struct IndirectBatchEntry {
    VkDeviceSize offset;
    std::vector<uint8_t> pushConstants;
    ComputeKernel specializedKernel = nullptr;
  };

  struct DGCExecutionInfo {
    VkIndirectCommandsLayoutEXT layout = VK_NULL_HANDLE;
    VkIndirectExecutionSetEXT executionSet = VK_NULL_HANDLE;
    ComputeBuffer sequenceBuffer = nullptr;
    VkDeviceSize sequenceBufferOffset = 0;
    VkDeviceSize sequenceBufferSize = 0;
    ComputeBuffer sequenceCountBuffer = nullptr;
    VkDeviceSize sequenceCountBufferOffset = 0;
    ComputeBuffer preprocessBuffer = nullptr;
    VkDeviceSize preprocessBufferSize = 0;
    uint32_t maxSequenceCount = 0;
  };

  void dispatchIndirect(ComputeKernel kernel, ComputeBuffer indirectBuffer,
                        VkDeviceSize offset = 0);
  void dispatchIndirectSequence(ComputeKernel kernel, ComputeBuffer indirectBuffer,
                                const std::vector<IndirectBatchEntry> &entries);
  void dispatchWorkListSequence(
      ComputeKernel resetKernel,
      ComputeKernel classifyKernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
      ComputeKernel resolveKernel,
      ComputeKernel secondKernel, ComputeBuffer indirectBuffer,
      const std::vector<IndirectBatchEntry> &entries,
      bool isPingPong = false,
      const DGCExecutionInfo *dgcInfo = nullptr,
      uint32_t dgcMode = 0);
  void dispatchRayTracingIndirect(ComputeKernel kernel, ComputeBuffer indirectBuffer,
                                 VkDeviceSize offset = 0);

  // Native Vulkan Device-Generated Commands (VK_EXT_device_generated_commands)
  VkIndirectCommandsLayoutEXT createIndirectCommandsLayout(
      const VkIndirectCommandsLayoutCreateInfoEXT &createInfo);
  void destroyIndirectCommandsLayout(VkIndirectCommandsLayoutEXT layout);

  VkIndirectExecutionSetEXT createIndirectExecutionSet(
      const VkIndirectExecutionSetCreateInfoEXT &createInfo);
  void updateIndirectExecutionSetPipeline(VkIndirectExecutionSetEXT set,
                                         uint32_t index,
                                         ComputeKernel kernel);
  void destroyIndirectExecutionSet(VkIndirectExecutionSetEXT set);

  VkDeviceSize getGeneratedCommandsMemoryRequirements(
      VkIndirectCommandsLayoutEXT layout,
      VkIndirectExecutionSetEXT execSet,
      uint32_t maxSequenceCount,
      ComputeKernel fallbackKernel = nullptr);

  ComputeBuffer createPreprocessBuffer(size_t size);
  VkPipeline getVkPipeline(ComputeKernel kernel) const;
  VkPipelineLayout getVkPipelineLayout(ComputeKernel kernel) const;

  void dispatchDGCWorkListSequence(
      ComputeKernel resetKernel,
      ComputeKernel classifyKernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
      ComputeKernel resolveKernel,
      ComputeKernel secondKernel,
      const DGCExecutionInfo &dgcInfo,
      const void *resolvePc = nullptr, size_t resolvePcSize = 0);

  void dispatchDGCSequence(ComputeKernel kernel,
                           const DGCExecutionInfo &dgcInfo);

  // Headless presentation hooks for profiling / tracing tools (e.g. RRA)
  void enableHeadlessSwapchain() override;
  void presentFrame() override;
  bool isHeadlessSwapchainEnabled() const override { return headlessSwapchain != VK_NULL_HANDLE; }

public:
  const std::vector<VkPhysicalDevice> &getPhysicalDevices() const {
    return physicalDevices;
  }
  void pickPhysicalDevice(uint32_t index);

  bool isSync2Supported() const { return sync2Supported; }
  void cmdPipelineMemoryBarrier2(
      VkCommandBuffer cmd,
      VkPipelineStageFlags2 srcStageMask,
      VkAccessFlags2 srcAccessMask,
      VkPipelineStageFlags2 dstStageMask,
      VkAccessFlags2 dstAccessMask);

private:
  struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation vmaAllocation = VK_NULL_HANDLE;
    void *mappedPtr = nullptr;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
    bool isSuballocated = false;
    uint32_t blockIndex = 0;
  };

  struct VulkanKernel {
    VkShaderModule shaderModule;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    std::map<uint32_t, ComputeBuffer> arg_buffers;
    uint32_t numBufferDescriptors;
    std::vector<uint8_t> pushConstantData;
    bool isRTPipeline = false;
    VkStridedDeviceAddressRegionKHR rgenRegion{};
    VkStridedDeviceAddressRegionKHR missRegion{};
    VkStridedDeviceAddressRegionKHR hitRegion{};
    VkStridedDeviceAddressRegionKHR callRegion{};
    ComputeBuffer sbtBuffer = nullptr;
  };

  void createInstance();
  void enumeratePhysicalDevices();
  void createDevice();
  void initVmaAllocator(bool bufferDeviceAddressEnabled, bool memoryBudgetEnabled);
  void cleanupVmaAllocator();
  void initPipelineCache();
  void cleanupPipelineCache();
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) const;

  std::vector<std::string> m_extraInstanceExtensions;
  VkSurfaceKHR m_presentationSurface = VK_NULL_HANDLE;
  mutable std::recursive_mutex m_queueMutex;

  VkInstance instance = VK_NULL_HANDLE;
  std::vector<VkPhysicalDevice> physicalDevices;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties;
  VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;
  VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;

  uint32_t computeQueueFamilyIndex = 0;
  VkQueue computeQueue = VK_NULL_HANDLE;
  uint32_t graphicsQueueFamilyIndex = 0;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  uint32_t presentationQueueFamilyIndex = 0;
  VkQueue presentationQueue = VK_NULL_HANDLE;
  uint32_t asyncComputeQueueFamilyIndex = UINT32_MAX;
  VkQueue asyncComputeQueue = VK_NULL_HANDLE;
  uint32_t transferQueueFamilyIndex = UINT32_MAX;
  VkQueue transferQueue = VK_NULL_HANDLE;
  VkQueue secondaryTransferQueue = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;

  static constexpr size_t kMaxInFlight = 4;
  struct InFlightFrame {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool inUse = false;
  };
  std::array<InFlightFrame, kMaxInFlight> inFlightFrames{};
  size_t currentFrameIndex = 0;
  bool m_inBatch = false;
  bool m_batchCmdRecording = false;
  bool m_batchInterDispatchBarrier = false;
  uint32_t m_batchDispatchCount = 0;

  VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
  bool timingSupported = false;
  bool isTimingActive = false;
  double timestampPeriod = 1.0;
  uint32_t currentQueryIndex = 0;
  static constexpr uint32_t kMaxTimestamps = 4096;
  struct TimestampQueryPair {
    uint32_t startQuery = 0;
    uint32_t stopQuery = 0;
  };
  std::vector<TimestampQueryPair> m_activeTimingPairs;
  bool timingRecording = false;
  bool timingSubmitted = false;
  VkFence timingSubmittedFence = VK_NULL_HANDLE;

  void maybeBeginTiming(VkCommandBuffer cmd);
  void maybeEndTiming(VkCommandBuffer cmd);
  bool hostQueryResetSupported = false;

  PFN_vkCmdPipelineBarrier2KHR vkCmdPipelineBarrier2KHR_ptr = nullptr;
  bool sync2Supported = false;

  PFN_vkCmdBeginRenderingKHR vkCmdBeginRendering_ptr = nullptr;
  PFN_vkCmdEndRenderingKHR vkCmdEndRendering_ptr = nullptr;
  bool dynamicRenderingSupported = false;

  PFN_vkGetPipelineExecutablePropertiesKHR vkGetPipelineExecutablePropertiesKHR_ptr = nullptr;
  PFN_vkGetPipelineExecutableStatisticsKHR vkGetPipelineExecutableStatisticsKHR_ptr = nullptr;
  PFN_vkGetPipelineExecutableInternalRepresentationsKHR vkGetPipelineExecutableInternalRepresentationsKHR_ptr = nullptr;
  bool pipelineExecutablePropertiesSupported = false;

  VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
  PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT_ptr = nullptr;
  PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT_ptr = nullptr;

  std::map<ComputeBuffer, VulkanBuffer *> buffers;
  std::map<ComputeKernel, VulkanKernel *> kernels;
  VulkanKernel *getKernel(ComputeKernel handle) const;

  mutable std::mutex m_sharedPreviewMutex;
  std::map<std::string, SharedPreviewImage> m_sharedPreviewImages;
  std::map<std::string, std::string> m_inMemoryJsons;
  uint64_t m_previewGenerationCounter = 0;

  // Staging buffer pool (VMA-backed)
  static constexpr size_t kStagingBufferSize = 64ULL * 1024ULL * 1024ULL;
  mutable std::mutex stagingMutex;
  mutable VkBuffer stagingBuffer = VK_NULL_HANDLE;
  mutable VmaAllocation stagingVmaAllocation = VK_NULL_HANDLE;
  mutable VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  mutable void *stagingMappedPtr = nullptr;
  mutable VkCommandBuffer stagingCmdBuffer = VK_NULL_HANDLE;
  mutable VkFence stagingFence = VK_NULL_HANDLE;
  void initStagingBuffer();
  void cleanupStagingBuffer();

  // Retained legacy helpers
  static constexpr VkDeviceSize kDefaultBlockSize = gpubench::BlockSuballocator::kDefaultBlockSize;
  static constexpr VkDeviceSize kMaxSuballocSize = gpubench::BlockSuballocator::kMaxSuballocSize;
  mutable std::mutex suballocatorMutex;
  gpubench::BlockSuballocator m_suballocator;
  std::vector<VkDeviceMemory> m_blockMemories;
  bool allocateSuballocatedBuffer(VkDeviceSize size, VkDeviceSize alignment,
                                  uint32_t memoryTypeIndex, bool needDeviceAddress,
                                  VulkanBuffer *outBuf);
  void freeSuballocatedBuffer(const VulkanBuffer *buf);
  void cleanupMemoryBlocks();

  mutable std::vector<DeviceInfo> deviceInfos;
  uint32_t selectedDeviceIndex = 0;
  bool verbose = false;
  bool quiet = false;
  bool debug = false;

  uint32_t expectedKernelCount = 0;
  uint32_t createdKernelCount = 0;
  bool subgroupSizeControlSupported = false;

  ComputeKernel createKernelInternal(const std::string &file_name,
                                     const std::string &kernel_name,
                                     uint32_t num_buffer_args,
                                     const uint32_t *spec_ids,
                                     const uint32_t *spec_vals,
                                     uint32_t spec_count);
  std::set<std::string> enabledExtensionsSet;
  void printProgressBar(uint32_t current, uint32_t total,
                        const std::string &kernel_name);

  VkSurfaceKHR headlessSurface = VK_NULL_HANDLE;
  VkSwapchainKHR headlessSwapchain = VK_NULL_HANDLE;
  std::vector<VkImage> swapchainImages;
  bool headlessSurfaceSupported = false;
  bool swapchainSupported = false;
  bool serSupported = false;
  bool maintenance5Supported = false;
  bool dgcSupported = false;
  bool dgcExecutionSetSupported = false;
  bool workGraphsSupported = false;
  bool cooperativeVectorSupported = false;
  bool opacityMicromapSupported = false;
  bool structuredSparsitySupported = false;
  bool memoryBudgetSupported = false;
  bool amdDeviceCoherentMemorySupported = false;

  // DGC (VK_EXT_device_generated_commands) function pointers
  PFN_vkGetGeneratedCommandsMemoryRequirementsEXT vkGetGeneratedCommandsMemoryRequirementsEXT_ptr = nullptr;
  PFN_vkCmdPreprocessGeneratedCommandsEXT vkCmdPreprocessGeneratedCommandsEXT_ptr = nullptr;
  PFN_vkCmdExecuteGeneratedCommandsEXT vkCmdExecuteGeneratedCommandsEXT_ptr = nullptr;
  PFN_vkCreateIndirectCommandsLayoutEXT vkCreateIndirectCommandsLayoutEXT_ptr = nullptr;
  PFN_vkDestroyIndirectCommandsLayoutEXT vkDestroyIndirectCommandsLayoutEXT_ptr = nullptr;
  PFN_vkCreateIndirectExecutionSetEXT vkCreateIndirectExecutionSetEXT_ptr = nullptr;
  PFN_vkDestroyIndirectExecutionSetEXT vkDestroyIndirectExecutionSetEXT_ptr = nullptr;
  PFN_vkUpdateIndirectExecutionSetPipelineEXT vkUpdateIndirectExecutionSetPipelineEXT_ptr = nullptr;
  PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR_ptr = nullptr;

  // VK_KHR_performance_query state and function pointers
  bool performanceQuerySupported = false;
  bool performanceQueryEnabled = false;
  bool isPerfQueryActive = false;
  bool perfQueryRecording = false;
  bool perfQuerySubmitted = false;
  bool perfQueryNeedsSubmitInfo = false;
  VkQueryPool perfQueryPool = VK_NULL_HANDLE;
  std::vector<uint32_t> perfQuerySelectedIndices;
  std::vector<VkPerformanceCounterKHR> perfCountersInfo;
  std::vector<VkPerformanceCounterDescriptionKHR> perfCounterDescriptions;
  HardwarePerformanceCounters lastPerfCounters;

  PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR pfnEnumerateQueueFamilyPerformanceQueryCountersKHR_ptr = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR pfnGetQueueFamilyPerformanceQueryPassesKHR_ptr = nullptr;
  PFN_vkAcquireProfilingLockKHR pfnAcquireProfilingLockKHR_ptr = nullptr;
  PFN_vkReleaseProfilingLockKHR pfnReleaseProfilingLockKHR_ptr = nullptr;

  void maybeBeginPerfQuery(VkCommandBuffer cmd);
  void maybeEndPerfQuery(VkCommandBuffer cmd);

  void destroyHeadlessSwapchain();
};
