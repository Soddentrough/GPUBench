#include "VulkanContext.h"
#include "utils/ShaderCache.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#ifdef HAVE_SHADERC
#include <shaderc/shaderc.hpp>
#endif

void VulkanContext::waitIdle() { vkQueueWaitIdle(computeQueue); }

VulkanContext::VulkanContext(bool verbose) : verbose(verbose) {
  char *verbose_env = std::getenv("GPUBENCH_VERBOSE");
  if (verbose_env && std::string(verbose_env) == "1") {
    verbose = true;
  }
  try {
    createInstance();
    enumeratePhysicalDevices();
  } catch (const std::exception &e) {
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

  for (const auto &device : allDevices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    // Filter out software renderers (llvmpipe) and CPUs
    std::string name = props.deviceName;
    if (name.find("llvmpipe") != std::string::npos)
      continue;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
      continue;

    physicalDevices.push_back(device);
  }
}

const std::vector<DeviceInfo> &VulkanContext::getDevices() const {
  if (deviceInfos.empty()) {
    for (const auto &device : physicalDevices) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(device, &props);

      VkPhysicalDeviceMemoryProperties memProps;
      vkGetPhysicalDeviceMemoryProperties(device, &memProps);

      VkPhysicalDeviceSubgroupProperties subgroupProps{};
      subgroupProps.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

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

      VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};

      VkPhysicalDeviceShaderFloat16Int8Features features168{};
      features168.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
      features168.pNext = &coopMatrixFeatures;

      VkPhysicalDevice8BitStorageFeatures features8bit{};
      features8bit.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
      features8bit.pNext = &features168;

      VkPhysicalDeviceFeatures2 features2{};
      features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      features2.pNext = &features8bit;
      vkGetPhysicalDeviceFeatures2(device, &features2);

      // Check extensions
      uint32_t extCount;
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
      std::vector<VkExtensionProperties> availableExts(extCount);
      vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount,
                                           availableExts.data());

      auto hasExt = [&](const char *name) {
        for (const auto &ext : availableExts) {
          if (strcmp(ext.extensionName, name) == 0)
            return true;
        }
        return false;
      };

      DeviceInfo info;
      info.name = props.deviceName;
      info.driverVersion = props.driverVersion;

      char uuid_str[33];
      for (int i = 0; i < VK_UUID_SIZE; ++i) {
        sprintf(&uuid_str[i * 2], "%02x", props.pipelineCacheUUID[i]);
      }
      info.driverUUID = std::string(uuid_str);

      info.memorySize = vramSize;
      info.maxWorkGroupSize = props.limits.maxComputeWorkGroupInvocations;
      info.maxComputeWorkGroupCountX = props.limits.maxComputeWorkGroupCount[0];
      info.maxComputeWorkGroupCountY = props.limits.maxComputeWorkGroupCount[1];
      info.maxComputeWorkGroupCountZ = props.limits.maxComputeWorkGroupCount[2];
      info.maxComputeSharedMemorySize = props.limits.maxComputeSharedMemorySize;
      info.subgroupSize = subgroupProps.subgroupSize;
      info.fp64Support = (features2.features.shaderFloat64 == VK_TRUE);
      info.fp16Support = (features168.shaderFloat16 == VK_TRUE);
      info.int8Support =
          true; // Usually supported if 8bit storage/int8 shader is supported
      info.cooperativeMatrixSupport =
          hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
      info.structuredSparsitySupport = true;
      info.fp8Support = hasExt("VK_EXT_shader_float8");
      info.rayTracingSupport =
          hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
          hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME);
      deviceInfos.push_back(info);
    }
  }
  return deviceInfos;
}

void VulkanContext::pickDevice(uint32_t index) { pickPhysicalDevice(index); }

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

  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures_curr{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};

  VkPhysicalDeviceShaderFloat16Int8Features features168_curr{};
  features168_curr.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features168_curr.pNext = &coopMatrixFeatures_curr;

  VkPhysicalDevice8BitStorageFeatures features8bit_curr{};
  features8bit_curr.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
  features8bit_curr.pNext = &features168_curr;

  VkPhysicalDeviceFeatures2 features2_2{};
  features2_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2_2.pNext = &features8bit_curr;
  vkGetPhysicalDeviceFeatures2(physicalDevice, &features2_2);

  // Check extensions
  uint32_t extCount;
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount,
                                       nullptr);
  std::vector<VkExtensionProperties> availableExts(extCount);
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount,
                                       availableExts.data());

  auto hasExt = [&](const char *name) {
    for (const auto &ext : availableExts) {
      if (strcmp(ext.extensionName, name) == 0)
        return true;
    }
    return false;
  };

  DeviceInfo info;
  info.name = properties.deviceName;
  info.memorySize = vramSize;
  info.maxWorkGroupSize = properties.limits.maxComputeWorkGroupInvocations;
  info.maxComputeWorkGroupCountX =
      properties.limits.maxComputeWorkGroupCount[0];
  info.maxComputeWorkGroupCountY =
      properties.limits.maxComputeWorkGroupCount[1];
  info.maxComputeWorkGroupCountZ =
      properties.limits.maxComputeWorkGroupCount[2];
  info.maxComputeSharedMemorySize =
      properties.limits.maxComputeSharedMemorySize;
  info.subgroupSize = subgroupProps.subgroupSize;
  info.fp64Support = (features2_2.features.shaderFloat64 == VK_TRUE);
  info.fp16Support = (features168_curr.shaderFloat16 == VK_TRUE);
  info.int8Support = true;
  info.cooperativeMatrixSupport =
      hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  info.fp8Support = hasExt("VK_EXT_shader_float8");
  info.fp6Support = false;
  info.fp4Support = true;  // Assuming support or emulation
  info.int4Support = true; // Assuming support or emulation
  info.structuredSparsitySupport = true;
  info.rayTracingSupport =
      hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
      hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME);
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
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
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
  VkPhysicalDeviceFeatures2 features2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceShaderFloat16Int8Features features168{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  VkPhysicalDevice16BitStorageFeatures features16Storage{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDevice8BitStorageFeatures features8Storage{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  VkPhysicalDeviceSubgroupSizeControlFeatures subgroupSizeFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};

  VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};

  // Explicitly using the struct names for EXT/KHR features
  struct VkPhysicalDeviceFloat8FeaturesEXT {
    VkStructureType sType;
    void *pNext;
    VkBool32 shaderFloat8;
  } float8Features{(
      VkStructureType)1000521001}; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT8_FEATURES_EXT

  struct VkPhysicalDeviceShaderFloatControls2FeaturesKHR {
    VkStructureType sType;
    void *pNext;
    VkBool32 shaderFloatControls2;
  } floatControls2Features{(
      VkStructureType)1000528001}; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR

  features2.pNext = &features168;
  features168.pNext = &features16Storage;
  features16Storage.pNext = &features8Storage;
  features8Storage.pNext = &coopMatrixFeatures;
  coopMatrixFeatures.pNext = &subgroupSizeFeatures;
  subgroupSizeFeatures.pNext = &asFeatures;
  asFeatures.pNext = &rayQueryFeatures;
  rayQueryFeatures.pNext = &bufferDeviceAddressFeatures;
  bufferDeviceAddressFeatures.pNext = &float8Features;
  float8Features.pNext = &floatControls2Features;

  // Query supported features and enable them
  vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

  const std::vector<const char *> desiredExtensions = {
      VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
      VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
      VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
      VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
      VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
      VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_RAY_QUERY_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
      VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
      "VK_EXT_shader_float8",
      "VK_KHR_shader_float_controls2"};

  // Filter extensions to only request supported ones
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount,
                                       nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount,
                                       availableExtensions.data());

  std::vector<const char *> enabledExtensions;
  for (const auto &extension : desiredExtensions) {
    bool found = false;
    for (const auto &available : availableExtensions) {
      if (strcmp(extension, available.extensionName) == 0) {
        found = true;
        break;
      }
    }
    if (found) {
      enabledExtensions.push_back(extension);
    } else {
      std::cerr << "Warning: Extension " << extension
                << " not supported by device, disabling." << std::endl;
    }
  }

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pNext = &features2; // Enable all modern features
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;
  createInfo.ppEnabledExtensionNames = enabledExtensions.data();
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(enabledExtensions.size());
  createInfo.pEnabledFeatures =
      nullptr; // Must be NULL if pNext contains a VkPhysicalDeviceFeatures2

  if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter,
                                       VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

ComputeBuffer VulkanContext::createBuffer(size_t size, const void *host_ptr) {
  auto vulkanBuffer = new VulkanBuffer();

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  if (getCurrentDeviceInfo().rayTracingSupport) {
    bufferInfo.usage |=
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
  }
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &vulkanBuffer->buffer) !=
      VK_SUCCESS) {
    delete vulkanBuffer;
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, vulkanBuffer->buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VkMemoryAllocateFlagsInfo flagsInfo{};
  if (getCurrentDeviceInfo().rayTracingSupport) {
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocInfo.pNext = &flagsInfo;
  }

  if (vkAllocateMemory(device, &allocInfo, nullptr, &vulkanBuffer->memory) !=
      VK_SUCCESS) {
    delete vulkanBuffer;
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(device, vulkanBuffer->buffer, vulkanBuffer->memory, 0);

  if (getCurrentDeviceInfo().rayTracingSupport) {
    VkBufferDeviceAddressInfo bdaInfo{
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdaInfo.buffer = vulkanBuffer->buffer;
    vulkanBuffer->address = vkGetBufferDeviceAddress(device, &bdaInfo);
  } else {
    vulkanBuffer->address = 0;
  }

  buffers[vulkanBuffer] = vulkanBuffer;

  if (host_ptr) {
    writeBuffer(vulkanBuffer, 0, size, host_ptr);
  }

  return vulkanBuffer;
}

void VulkanContext::writeBuffer(ComputeBuffer buffer, size_t offset,
                                size_t size, const void *host_ptr) {
  VulkanBuffer stagingBuffer;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer.buffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create staging buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, stagingBuffer.buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingBuffer.memory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate staging buffer memory!");
  }

  vkBindBufferMemory(device, stagingBuffer.buffer, stagingBuffer.memory, 0);

  void *data;
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
  vkCmdCopyBuffer(commandBuffer, stagingBuffer.buffer,
                  buffers.at(buffer)->buffer, 1, &copyRegion);

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

void VulkanContext::readBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                               void *host_ptr) const {
  VulkanBuffer stagingBuffer;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer.buffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create staging buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, stagingBuffer.buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingBuffer.memory) !=
      VK_SUCCESS) {
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
  vkCmdCopyBuffer(commandBuffer, buffers.at(buffer)->buffer,
                  stagingBuffer.buffer, 1, &copyRegion);

  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(computeQueue);

  void *data;
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
    VulkanBuffer *vulkanBuffer = it->second;
    vkDestroyBuffer(device, vulkanBuffer->buffer, nullptr);
    vkFreeMemory(device, vulkanBuffer->memory, nullptr);
    delete vulkanBuffer;
    buffers.erase(it);
  }
}

VkDeviceAddress
VulkanContext::getBufferDeviceAddress(ComputeBuffer buffer) const {
  auto it = buffers.find(buffer);
  if (it != buffers.end()) {
    return it->second->address;
  }
  return 0;
}

VkBuffer VulkanContext::getVkBuffer(ComputeBuffer buffer) const {
  auto it = buffers.find(buffer);
  if (it != buffers.end()) {
    return it->second->buffer;
  }
  return VK_NULL_HANDLE;
}

ComputeKernel VulkanContext::createKernel(const std::string &file_name,
                                          const std::string &kernel_name,
                                          uint32_t num_buffer_args) {
  notifyKernelCreated(file_name);
  bool is_glsl = false;
  if (file_name.size() > 5 &&
      file_name.substr(file_name.size() - 5) == ".comp") {
    is_glsl = true;
  }

  std::vector<uint32_t> spirv_code;
  std::string spv_file = file_name;
  if (is_glsl) {
    spv_file = file_name + ".spv";
  }

  bool loaded_from_file = false;
  std::ifstream spv_stream(spv_file, std::ios::ate | std::ios::binary);
  if (spv_stream.is_open()) {
    size_t fileSize = (size_t)spv_stream.tellg();
    if (fileSize > 0 && fileSize % 4 == 0) {
      std::vector<char> buffer(fileSize);
      spv_stream.seekg(0);
      spv_stream.read(buffer.data(), fileSize);
      spv_stream.close();

      spirv_code.resize(fileSize / 4);
      std::memcpy(spirv_code.data(), buffer.data(), fileSize);
      loaded_from_file = true;
      if (verbose) {
        std::cout << "Loaded pre-compiled SPIR-V: " << spv_file << std::endl;
      }
    }
  }

  if (!loaded_from_file) {
#ifdef HAVE_SHADERC
    if (is_glsl) {
      if (utils::ShaderCache::loadVulkanCache(
              file_name, deviceInfos[selectedDeviceIndex], spirv_code)) {
        if (verbose) {
          std::cout << "Loaded Vulkan shader from cache: " << file_name
                    << std::endl;
        }
      } else {
        if (verbose) {
          std::cout << "Compiling Vulkan shader: " << file_name << std::endl;
        }
        std::ifstream file(file_name);
        if (!file.is_open()) {
          throw std::runtime_error("Failed to open shader file: " + file_name);
        }
        std::string source((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                     shaderc_env_version_vulkan_1_3);
        options.SetOptimizationLevel(shaderc_optimization_level_performance);

        shaderc::SpvCompilationResult result = compiler.CompileGlslToSpv(
            source, shaderc_glsl_compute_shader, file_name.c_str(), options);

        if (result.GetCompilationStatus() !=
            shaderc_compilation_status_success) {
          throw std::runtime_error("Failed to compile Vulkan shader " +
                                   file_name + ": " + result.GetErrorMessage());
        }

        spirv_code.assign(result.cbegin(), result.cend());
        utils::ShaderCache::saveVulkanCache(
            file_name, deviceInfos[selectedDeviceIndex], spirv_code);
      }
    } else {
      throw std::runtime_error("Failed to load SPIR-V from " + file_name);
    }
#else
    throw std::runtime_error("Failed to load pre-compiled SPIR-V from " +
                             spv_file + " and shaderc is not available.");
#endif
  }

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = spirv_code.size() * sizeof(uint32_t);
  createInfo.pCode = spirv_code.data();

  auto vulkanKernel = new VulkanKernel();
  vulkanKernel->numBufferDescriptors = num_buffer_args;
  if (vkCreateShaderModule(device, &createInfo, nullptr,
                           &vulkanKernel->shaderModule) != VK_SUCCESS) {
    delete vulkanKernel;
    throw std::runtime_error("failed to create shader module!");
  }

  // This is a simplified setup. A real application would inspect the shader for
  // bindings.
  bool is_rt = (file_name.find("rt_") != std::string::npos);

  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for (uint32_t i = 0; i < num_buffer_args; ++i) {
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = i;
    if (is_rt && i == 0) {
      layoutBinding.descriptorType =
          VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    } else {
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(layoutBinding);
  }

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                  &vulkanKernel->descriptorSetLayout) !=
      VK_SUCCESS) {
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

  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                             &vulkanKernel->pipelineLayout) != VK_SUCCESS) {
    delete vulkanKernel;
    throw std::runtime_error("failed to create pipeline layout!");
  }

  // Initialize push constant data buffer
  vulkanKernel->pushConstantData.resize(128, 0);

  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.layout = vulkanKernel->pipelineLayout;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = vulkanKernel->shaderModule;
  pipelineInfo.stage.pName = kernel_name.c_str();

  VkResult result =
      vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                               nullptr, &vulkanKernel->pipeline);
  if (result != VK_SUCCESS) {
    vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout,
                                 nullptr);
    vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
    delete vulkanKernel;
    throw std::runtime_error("Failed to create compute pipeline (VkResult: " +
                             std::to_string(result) +
                             "). This may be a driver issue.");
  }

  VkDescriptorPoolSize poolSizes[2] = {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[0].descriptorCount = num_buffer_args;
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  poolSizes[1].descriptorCount = 1;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = is_rt ? 2 : 1;
  poolInfo.pPoolSizes = poolSizes;
  poolInfo.maxSets = 1;

  if (vkCreateDescriptorPool(device, &poolInfo, nullptr,
                             &vulkanKernel->descriptorPool) != VK_SUCCESS) {
    delete vulkanKernel;
    throw std::runtime_error("failed to create descriptor pool!");
  }

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = vulkanKernel->descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &vulkanKernel->descriptorSetLayout;

  if (vkAllocateDescriptorSets(device, &allocInfo,
                               &vulkanKernel->descriptorSet) != VK_SUCCESS) {
    delete vulkanKernel;
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  kernels[vulkanKernel] = vulkanKernel;
  return vulkanKernel;
}

void VulkanContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index,
                                 ComputeBuffer buffer) {
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

void VulkanContext::setKernelAS(ComputeKernel kernel, uint32_t arg_index,
                                AccelerationStructure as) {
  auto it = kernels.find(kernel);
  if (it == kernels.end()) {
    throw std::runtime_error("Invalid kernel handle");
  }

  VkAccelerationStructureKHR vkAS = (VkAccelerationStructureKHR)as;
  VkWriteDescriptorSetAccelerationStructureKHR descriptorAS{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descriptorAS.accelerationStructureCount = 1;
  descriptorAS.pAccelerationStructures = &vkAS;

  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.pNext = &descriptorAS;
  descriptorWrite.dstSet = it->second->descriptorSet;
  descriptorWrite.dstBinding = arg_index;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType =
      VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  descriptorWrite.descriptorCount = 1;

  vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
}

void VulkanContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index,
                                 size_t arg_size, const void *arg_value) {
  auto it = kernels.find(kernel);
  if (it == kernels.end()) {
    throw std::runtime_error("Invalid kernel handle");
  }

  // For Vulkan, non-buffer arguments are passed via push constants
  // Arguments are laid out sequentially in the push constant buffer
  // We map arg_index directly to offset to handle mixed buffer/value args
  // Buffer arguments (descriptors) take indices 0 to numBufferDescriptors-1.
  // Push constants start AFTER the descriptor indices.

  if (arg_index < it->second->numBufferDescriptors) {
    if (verbose) {
      std::cerr << "Error: setKernelArg (value) called for index " << arg_index
                << " but it's reserved for a buffer descriptor (numBuffers="
                << it->second->numBufferDescriptors << ")" << std::endl;
    }
    return;
  }

  size_t offset = (arg_index - it->second->numBufferDescriptors) * 4;
  if (offset + arg_size <= it->second->pushConstantData.size()) {
    memcpy(it->second->pushConstantData.data() + offset, arg_value, arg_size);
  }
}

void VulkanContext::dispatch(ComputeKernel kernel, uint32_t grid_x,
                             uint32_t grid_y, uint32_t grid_z, uint32_t block_x,
                             uint32_t block_y, uint32_t block_z) {
  auto it = kernels.find(kernel);
  if (it == kernels.end()) {
    throw std::runtime_error("Invalid kernel handle");
  }
  VulkanKernel *vulkanKernel = it->second;

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
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    vulkanKernel->pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkanKernel->pipelineLayout, 0, 1,
                          &vulkanKernel->descriptorSet, 0, nullptr);

  // Push constants for non-buffer arguments
  if (!vulkanKernel->pushConstantData.empty()) {
    vkCmdPushConstants(
        commandBuffer, vulkanKernel->pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
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
    VulkanKernel *vulkanKernel = it->second;
    vkDestroyPipeline(device, vulkanKernel->pipeline, nullptr);
    vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout,
                                 nullptr);
    vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
    vkDestroyDescriptorPool(device, vulkanKernel->descriptorPool, nullptr);
    delete vulkanKernel;
    kernels.erase(it);
  }
}

void VulkanContext::setExpectedKernelCount(uint32_t count) {
  expectedKernelCount = count;
  createdKernelCount = 0;
  if (verbose && count > 0) {
    std::cout << "Starting setup for " << count << " kernels..." << std::endl;
#ifdef HAVE_SHADERC
    std::cout << "Using compiler: shaderc (Vulkan SPIR-V)" << std::endl;
#endif
  }
}

void VulkanContext::notifyKernelCreated(const std::string &file_name) {
  createdKernelCount++;
  if (!verbose && expectedKernelCount > 0) {
    printProgressBar(createdKernelCount, expectedKernelCount, file_name);
  }
}

void VulkanContext::printProgressBar(uint32_t current, uint32_t total,
                                     const std::string &kernel_name) {
  const int barWidth = 30;
  float progress = static_cast<float>(current) / total;
  int pos = static_cast<int>(barWidth * progress);

  std::string short_name = kernel_name;
  size_t last_slash = kernel_name.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    short_name = kernel_name.substr(last_slash + 1);
  }

  std::cout << "\r\033[K[";
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "#";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << "% Compiling " << short_name
            << (current == total ? "\n" : "") << std::flush;
}
