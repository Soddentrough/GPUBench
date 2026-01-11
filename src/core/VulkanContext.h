#pragma once

#include "IComputeContext.h"
#include <map>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanContext : public IComputeContext {
public:
  VulkanContext(bool verbose = false);
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
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkFence computeFence = VK_NULL_HANDLE;

  std::map<ComputeBuffer, VulkanBuffer *> buffers;
  std::map<ComputeKernel, VulkanKernel *> kernels;

  mutable std::vector<DeviceInfo> deviceInfos;
  uint32_t selectedDeviceIndex = 0;
  bool verbose = false;

  uint32_t expectedKernelCount = 0;
  uint32_t createdKernelCount = 0;
  void printProgressBar(uint32_t current, uint32_t total,
                        const std::string &kernel_name);
};
