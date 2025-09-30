#include "VulkanContext.h"
#include <iostream>
#include <vector>
#include <stdexcept>

VulkanContext::VulkanContext() {
    try {
        createInstance();
        enumeratePhysicalDevices();
    } catch (const std::exception& e) {
        std::cerr << "Vulkan initialization failed: " << e.what() << std::endl;
        throw;
    }
}

VulkanContext::~VulkanContext() {
    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

void VulkanContext::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPUBench";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

void VulkanContext::enumeratePhysicalDevices() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    physicalDevices.resize(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());
}

const std::vector<DeviceInfo>& VulkanContext::getDevices() const {
    if (deviceInfos.empty()) {
        for (const auto& device : physicalDevices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(device, &props);
            
            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(device, &memProps);
            
            // Get subgroup properties
            VkPhysicalDeviceSubgroupProperties subgroupProps{};
            subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &subgroupProps;
            vkGetPhysicalDeviceProperties2(device, &props2);
            
            uint64_t vramSize = 0;
            for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
                if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    vramSize = memProps.memoryHeaps[i].size;
                    break;
                }
            }
            
            DeviceInfo info;
            info.name = props.deviceName;
            info.memorySize = vramSize;
            info.computeUnits = subgroupProps.subgroupSize;
            info.maxWorkGroupSize = props.limits.maxComputeWorkGroupInvocations;
            info.maxComputeWorkGroupCountX = props.limits.maxComputeWorkGroupCount[0];
            info.maxComputeWorkGroupCountY = props.limits.maxComputeWorkGroupCount[1];
            info.maxComputeWorkGroupCountZ = props.limits.maxComputeWorkGroupCount[2];
            info.maxComputeSharedMemorySize = props.limits.maxComputeSharedMemorySize;
            info.subgroupSize = subgroupProps.subgroupSize;
            deviceInfos.push_back(info);
        }
    }
    return deviceInfos;
}

void VulkanContext::pickDevice(uint32_t index) {
    pickPhysicalDevice(index);
}

DeviceInfo VulkanContext::getCurrentDeviceInfo() const {
    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("No device selected");
    }
    
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    // Get subgroup properties
    VkPhysicalDeviceSubgroupProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
    
    uint64_t vramSize = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            vramSize = memProps.memoryHeaps[i].size;
            break;
        }
    }
    
    DeviceInfo info;
    info.name = properties.deviceName;
    info.memorySize = vramSize;
    info.computeUnits = subgroupProps.subgroupSize;
    info.maxWorkGroupSize = properties.limits.maxComputeWorkGroupInvocations;
    info.maxComputeWorkGroupCountX = properties.limits.maxComputeWorkGroupCount[0];
    info.maxComputeWorkGroupCountY = properties.limits.maxComputeWorkGroupCount[1];
    info.maxComputeWorkGroupCountZ = properties.limits.maxComputeWorkGroupCount[2];
    info.maxComputeSharedMemorySize = properties.limits.maxComputeSharedMemorySize;
    info.subgroupSize = subgroupProps.subgroupSize;
    return info;
}

void VulkanContext::pickPhysicalDevice(uint32_t index) {
    if (index >= physicalDevices.size()) {
        throw std::runtime_error("invalid device index");
    }
    physicalDevice = physicalDevices[index];
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    createDevice();
}

void VulkanContext::createDevice() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            break;
        }
        i++;
    }

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
}
