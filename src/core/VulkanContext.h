#pragma once

#include <vulkan/vulkan.h>
#include <vector>

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

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
};
