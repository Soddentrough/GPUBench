#include "VulkanContext.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <cstring>

void VulkanContext::waitIdle() {
    vkQueueWaitIdle(computeQueue);
}

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
    while (!kernels.empty()) {
        releaseKernel(kernels.begin()->first);
    }
    while (!buffers.empty()) {
        releaseBuffer(buffers.begin()->first);
    }
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
    }
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
    appInfo.apiVersion = VK_API_VERSION_1_3;

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

    std::vector<VkPhysicalDevice> allDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, allDevices.data());

    for (const auto& device : allDevices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        // Filter out software renderers (llvmpipe) and CPUs
        std::string name = props.deviceName;
        if (name.find("llvmpipe") != std::string::npos) continue;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;

        physicalDevices.push_back(device);
    }
}

const std::vector<DeviceInfo>& VulkanContext::getDevices() const {
    if (deviceInfos.empty()) {
        for (const auto& device : physicalDevices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(device, &props);
            
            VkPhysicalDeviceMemoryProperties memProps;
            vkGetPhysicalDeviceMemoryProperties(device, &memProps);
            
            VkPhysicalDeviceSubgroupProperties subgroupProps{};
            subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &subgroupProps;
            vkGetPhysicalDeviceProperties2(device, &props2);
            
            uint64_t vramSize = 0;
            for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
                if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                    vramSize += memProps.memoryHeaps[i].size;
                }
            }
            
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
            
            VkPhysicalDeviceShaderFloat16Int8Features features168{};
            features168.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            features168.pNext = &coopMatrixFeatures;
            
            VkPhysicalDevice8BitStorageFeatures features8bit{};
            features8bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
            features8bit.pNext = &features168;
            
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &features8bit;
            vkGetPhysicalDeviceFeatures2(device, &features2);
            
            DeviceInfo info;
            info.name = props.deviceName;
            info.memorySize = vramSize;
            info.maxWorkGroupSize = props.limits.maxComputeWorkGroupInvocations;
            info.maxComputeWorkGroupCountX = props.limits.maxComputeWorkGroupCount[0];
            info.maxComputeWorkGroupCountY = props.limits.maxComputeWorkGroupCount[1];
            info.maxComputeWorkGroupCountZ = props.limits.maxComputeWorkGroupCount[2];
            info.maxComputeSharedMemorySize = props.limits.maxComputeSharedMemorySize;
            info.subgroupSize = subgroupProps.subgroupSize;
            info.fp64Support = (features2.features.shaderFloat64 == VK_TRUE);
            info.fp16Support = (features168.shaderFloat16 == VK_TRUE);
            info.int8Support = true;
            info.cooperativeMatrixSupport = true;
            info.structuredSparsitySupport = true; 
            info.fp8Support = true; 
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
    
    VkPhysicalDeviceSubgroupProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
    
    uint64_t vramSize = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            vramSize += memProps.memoryHeaps[i].size;
        }
    }
    
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures_curr { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };

    VkPhysicalDeviceShaderFloat16Int8Features features168_curr{};
    features168_curr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    features168_curr.pNext = &coopMatrixFeatures_curr;
    
    VkPhysicalDevice8BitStorageFeatures features8bit_curr{};
    features8bit_curr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    features8bit_curr.pNext = &features168_curr;
    
    VkPhysicalDeviceFeatures2 features2_2{};
    features2_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2_2.pNext = &features8bit_curr;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2_2);

    DeviceInfo info;
    info.name = properties.deviceName;
    info.memorySize = vramSize;
    info.maxWorkGroupSize = properties.limits.maxComputeWorkGroupInvocations;
    info.maxComputeWorkGroupCountX = properties.limits.maxComputeWorkGroupCount[0];
    info.maxComputeWorkGroupCountY = properties.limits.maxComputeWorkGroupCount[1];
    info.maxComputeWorkGroupCountZ = properties.limits.maxComputeWorkGroupCount[2];
    info.maxComputeSharedMemorySize = properties.limits.maxComputeSharedMemorySize;
    info.subgroupSize = subgroupProps.subgroupSize;
    info.fp64Support = (features2_2.features.shaderFloat64 == VK_TRUE);
    info.fp16Support = (features168_curr.shaderFloat16 == VK_TRUE);
    info.int8Support = true;
    info.cooperativeMatrixSupport = true;
    info.fp8Support = true;
    info.fp6Support = false;
    info.fp4Support = true;
    info.int4Support = true;
    info.structuredSparsitySupport = true;
    return info;
}

void VulkanContext::pickPhysicalDevice(uint32_t index) {
    if (index >= physicalDevices.size()) {
        throw std::runtime_error("invalid device index");
    }
    selectedDeviceIndex = index;
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

    // Use features2 chain to enable modern features like FP16 and INT8
    VkPhysicalDeviceFeatures2 features2 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkPhysicalDeviceShaderFloat16Int8Features features168 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
    VkPhysicalDevice16BitStorageFeatures features16Storage { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
    VkPhysicalDevice8BitStorageFeatures features8Storage { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroupSizeFeatures { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };
    
    // Explicitly using the struct names for EXT/KHR features
    struct VkPhysicalDeviceFloat8FeaturesEXT {
        VkStructureType sType;
        void* pNext;
        VkBool32 shaderFloat8;
    } float8Features { (VkStructureType)1000521001 }; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT8_FEATURES_EXT

    struct VkPhysicalDeviceShaderFloatControls2FeaturesKHR {
        VkStructureType sType;
        void* pNext;
        VkBool32 shaderFloatControls2;
    } floatControls2Features { (VkStructureType)1000528001 }; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR

    features2.pNext = &features168;
    features168.pNext = &features16Storage;
    features16Storage.pNext = &features8Storage;
    features8Storage.pNext = &coopMatrixFeatures;
    coopMatrixFeatures.pNext = &subgroupSizeFeatures;
    subgroupSizeFeatures.pNext = &float8Features;
    float8Features.pNext = &floatControls2Features;

    // Query supported features and enable them
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        "VK_EXT_shader_float8",
        "VK_KHR_shader_float_controls2"
    };

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &features2; // Enable all modern features
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.pEnabledFeatures = nullptr; // Must be NULL if pNext contains a VkPhysicalDeviceFeatures2

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

ComputeBuffer VulkanContext::createBuffer(size_t size, const void* host_ptr) {
    auto vulkanBuffer = new VulkanBuffer();

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &vulkanBuffer->buffer) != VK_SUCCESS) {
        delete vulkanBuffer;
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, vulkanBuffer->buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &vulkanBuffer->memory) != VK_SUCCESS) {
        delete vulkanBuffer;
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, vulkanBuffer->buffer, vulkanBuffer->memory, 0);

    buffers[vulkanBuffer] = vulkanBuffer;

    if (host_ptr) {
        writeBuffer(vulkanBuffer, 0, size, host_ptr);
    }

    return vulkanBuffer;
}

void VulkanContext::writeBuffer(ComputeBuffer buffer, size_t offset, size_t size, const void* host_ptr) {
    VulkanBuffer stagingBuffer;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer.buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create staging buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, stagingBuffer.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingBuffer.memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate staging buffer memory!");
    }

    vkBindBufferMemory(device, stagingBuffer.buffer, stagingBuffer.memory, 0);

    void* data;
    vkMapMemory(device, stagingBuffer.memory, 0, size, 0, &data);
    memcpy(data, host_ptr, size);
    vkUnmapMemory(device, stagingBuffer.memory);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = offset;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, stagingBuffer.buffer, buffers.at(buffer)->buffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
    vkFreeMemory(device, stagingBuffer.memory, nullptr);
}

void VulkanContext::readBuffer(ComputeBuffer buffer, size_t offset, size_t size, void* host_ptr) const {
    VulkanBuffer stagingBuffer;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer.buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create staging buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, stagingBuffer.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingBuffer.memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate staging buffer memory!");
    }

    vkBindBufferMemory(device, stagingBuffer.buffer, stagingBuffer.memory, 0);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = offset;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, buffers.at(buffer)->buffer, stagingBuffer.buffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);

    void* data;
    vkMapMemory(device, stagingBuffer.memory, 0, size, 0, &data);
    memcpy(host_ptr, data, size);
    vkUnmapMemory(device, stagingBuffer.memory);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyBuffer(device, stagingBuffer.buffer, nullptr);
    vkFreeMemory(device, stagingBuffer.memory, nullptr);
}

void VulkanContext::releaseBuffer(ComputeBuffer buffer) {
    auto it = buffers.find(buffer);
    if (it != buffers.end()) {
        VulkanBuffer* vulkanBuffer = it->second;
        vkDestroyBuffer(device, vulkanBuffer->buffer, nullptr);
        vkFreeMemory(device, vulkanBuffer->memory, nullptr);
        delete vulkanBuffer;
        buffers.erase(it);
    }
}

ComputeKernel VulkanContext::createKernel(const std::string& file_name, const std::string& kernel_name, uint32_t num_args) {
    std::ifstream file(file_name, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + file_name);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

    auto vulkanKernel = new VulkanKernel();
    if (vkCreateShaderModule(device, &createInfo, nullptr, &vulkanKernel->shaderModule) != VK_SUCCESS) {
        delete vulkanKernel;
        throw std::runtime_error("failed to create shader module!");
    }

    // This is a simplified setup. A real application would inspect the shader for bindings.
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < num_args; ++i) {
        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = i;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(layoutBinding);
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &vulkanKernel->descriptorSetLayout) != VK_SUCCESS) {
        delete vulkanKernel;
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    // Set up push constants for non-buffer arguments (e.g., mode, bufferSize)
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 128; // Allocate 128 bytes for push constants
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &vulkanKernel->descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &vulkanKernel->pipelineLayout) != VK_SUCCESS) {
        delete vulkanKernel;
        throw std::runtime_error("failed to create pipeline layout!");
    }
    
    // Initialize push constant data buffer
    vulkanKernel->pushConstantData.resize(128, 0);

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = vulkanKernel->pipelineLayout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = vulkanKernel->shaderModule;
    pipelineInfo.stage.pName = kernel_name.c_str();

    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vulkanKernel->pipeline);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
        delete vulkanKernel;
        throw std::runtime_error("Failed to create compute pipeline (VkResult: " + std::to_string(result) + "). This may be a driver issue.");
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = num_args;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &vulkanKernel->descriptorPool) != VK_SUCCESS) {
        delete vulkanKernel;
        throw std::runtime_error("failed to create descriptor pool!");
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = vulkanKernel->descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &vulkanKernel->descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &vulkanKernel->descriptorSet) != VK_SUCCESS) {
        delete vulkanKernel;
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    kernels[vulkanKernel] = vulkanKernel;
    return vulkanKernel;
}

void VulkanContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index, ComputeBuffer buffer) {
    auto it = kernels.find(kernel);
    if (it == kernels.end()) {
        throw std::runtime_error("Invalid kernel handle");
    }
    it->second->arg_buffers[arg_index] = buffer;

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffers.at(buffer)->buffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = it->second->descriptorSet;
    descriptorWrite.dstBinding = arg_index;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

void VulkanContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index, size_t arg_size, const void* arg_value) {
    auto it = kernels.find(kernel);
    if (it == kernels.end()) {
        throw std::runtime_error("Invalid kernel handle");
    }
    
    // For Vulkan, non-buffer arguments are passed via push constants
    // Arguments are laid out sequentially in the push constant buffer
    // We map arg_index directly to offset to handle mixed buffer/value args
    size_t offset = arg_index * 4;
    if (offset + arg_size <= it->second->pushConstantData.size()) {
        memcpy(it->second->pushConstantData.data() + offset, arg_value, arg_size);
    }
}

void VulkanContext::dispatch(ComputeKernel kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z) {
    auto it = kernels.find(kernel);
    if (it == kernels.end()) {
        throw std::runtime_error("Invalid kernel handle");
    }
    VulkanKernel* vulkanKernel = it->second;

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanKernel->pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanKernel->pipelineLayout, 0, 1, &vulkanKernel->descriptorSet, 0, nullptr);
    
    // Push constants for non-buffer arguments
    if (!vulkanKernel->pushConstantData.empty()) {
        vkCmdPushConstants(commandBuffer, vulkanKernel->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 
                          static_cast<uint32_t>(vulkanKernel->pushConstantData.size()), 
                          vulkanKernel->pushConstantData.data());
    }
    
    vkCmdDispatch(commandBuffer, grid_x, grid_y, grid_z);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanContext::releaseKernel(ComputeKernel kernel) {
    auto it = kernels.find(kernel);
    if (it != kernels.end()) {
        VulkanKernel* vulkanKernel = it->second;
        vkDestroyPipeline(device, vulkanKernel->pipeline, nullptr);
        vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
        vkDestroyDescriptorPool(device, vulkanKernel->descriptorPool, nullptr);
        delete vulkanKernel;
        kernels.erase(it);
    }
}
