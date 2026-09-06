#pragma once

#include "IComputeContext.h"
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>

class VulkanContext : public IComputeContext {
public:
  VulkanContext(bool verbose = false, bool debug = false);
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

  // Buffer management
  ComputeBuffer createBuffer(size_t size,
                             const void *host_ptr = nullptr) override;
  void writeBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                   const void *host_ptr) override;
  void readBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                  void *host_ptr) const override;
  void releaseBuffer(ComputeBuffer buffer) override;
  VkDeviceAddress getBufferDeviceAddress(ComputeBuffer buffer) const;

  // Kernel management
  ComputeKernel createKernel(const std::string &file_name,
                             const std::string &kernel_name,
                             uint32_t num_buffer_args) override;
  ComputeKernel createKernelWithSpec(const std::string &file_name,
                                     const std::string &kernel_name,
                                     uint32_t num_buffer_args,
                                     uint32_t spec_id,
                                     uint32_t spec_val);
  ComputeKernel createRTPipeline(const std::string &rgen_path,
                                 const std::string &rmiss_path,
                                 const std::vector<std::string> &rchit_paths,
                                 const std::vector<std::string> &rahit_paths,
                                 const std::vector<std::string> &rint_paths,
                                 uint32_t num_buffer_args) override;
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

  void setExpectedKernelCount(uint32_t count) override;
  void notifyKernelCreated(const std::string &kernel_name) override;
  void setVerbose(bool v) override { verbose = v; }

  VkPhysicalDevice getVulkanPhysicalDevice() const override {
    return physicalDevice;
  }
  VkDevice getVulkanDevice() const override { return device; }
  void *getVulkanContext() const override { return (void *)this; }

  // Vulkan-specific accessors
  VkInstance getInstance() const { return instance; }
  VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
  VkDevice getDevice() const { return device; }
  uint32_t getComputeQueueFamilyIndex() const {
    return computeQueueFamilyIndex;
  }
  VkQueue getComputeQueue() const { return computeQueue; }
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
    return isExtensionEnabled("VK_AMDX_shader_enqueue");
  }
  bool isSERSupported() const {
    return serSupported;
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

private:
  struct VulkanBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceAddress address;
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
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) const;

  VkInstance instance = VK_NULL_HANDLE;
  std::vector<VkPhysicalDevice> physicalDevices;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties properties;

  uint32_t computeQueueFamilyIndex = 0;
  VkQueue computeQueue = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;

  static constexpr size_t kMaxInFlight = 16;
  struct InFlightFrame {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool inUse = false;
  };
  std::array<InFlightFrame, kMaxInFlight> inFlightFrames{};
  size_t currentFrameIndex = 0;

  std::map<ComputeBuffer, VulkanBuffer *> buffers;
  std::map<ComputeKernel, VulkanKernel *> kernels;

  mutable std::vector<DeviceInfo> deviceInfos;
  uint32_t selectedDeviceIndex = 0;
  bool verbose = false;
  bool debug = false;

  uint32_t expectedKernelCount = 0;
  uint32_t createdKernelCount = 0;
  bool subgroupSizeControlSupported = false;

  ComputeKernel createKernelInternal(const std::string &file_name,
                                     const std::string &kernel_name,
                                     uint32_t num_buffer_args,
                                     const uint32_t *spec_id,
                                     const uint32_t *spec_val);
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

  void destroyHeadlessSwapchain();
};
