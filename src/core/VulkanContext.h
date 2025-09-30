#pragma once

#include "IComputeContext.h"
#include <vulkan/vulkan.h>
#include <vector>

class VulkanContext : public IComputeContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // IComputeContext interface
    ComputeBackend getBackend() const override { return ComputeBackend::Vulkan; }
    const std::vector<DeviceInfo>& getDevices() const override;
    void pickDevice(uint32_t index) override;
    DeviceInfo getCurrentDeviceInfo() const override;
    
    VkPhysicalDevice getVulkanPhysicalDevice() const override { return physicalDevice; }
    VkDevice getVulkanDevice() const override { return device; }
    void* getVulkanContext() const override { return (void*)this; }
    
    // Vulkan-specific accessors
    VkInstance getInstance() const { return instance; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VkDevice getDevice() const { return device; }
    uint32_t getComputeQueueFamilyIndex() const { return computeQueueFamilyIndex; }
    VkQueue getComputeQueue() const { return computeQueue; }
    const VkPhysicalDeviceProperties& getPhysicalDeviceProperties() const { return properties; }

public:
    const std::vector<VkPhysicalDevice>& getPhysicalDevices() const { return physicalDevices; }
    void pickPhysicalDevice(uint32_t index);
private:
    void createInstance();
    void enumeratePhysicalDevices();
    void createDevice();

    VkInstance instance = VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> physicalDevices;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties;
    
    uint32_t computeQueueFamilyIndex = 0;
    VkQueue computeQueue = VK_NULL_HANDLE;
    
    mutable std::vector<DeviceInfo> deviceInfos;
};
