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

void VulkanContext::waitIdle() {
  if (device == VK_NULL_HANDLE) return;
  for (size_t i = 0; i < kMaxInFlight; ++i) {
    if (inFlightFrames[i].inUse && inFlightFrames[i].fence != VK_NULL_HANDLE) {
      constexpr uint64_t kTimeoutNs = 3'000'000'000ULL;
      VkResult waitResult =
          vkWaitForFences(device, 1, &inFlightFrames[i].fence, VK_TRUE, kTimeoutNs);
      if (waitResult == VK_TIMEOUT) {
        throw std::runtime_error("GPU dispatch timed out (>3 s) in waitIdle()");
      }
      vkResetFences(device, 1, &inFlightFrames[i].fence);
      inFlightFrames[i].inUse = false;
    }
  }
  if (computeQueue != VK_NULL_HANDLE) {
    vkQueueWaitIdle(computeQueue);
  }
}

VulkanContext::VulkanContext(bool verbose, bool debug) : verbose(verbose), debug(debug) {
  char *verbose_env = std::getenv("GPUBENCH_VERBOSE");
  if (verbose_env && std::string(verbose_env) == "1") {
    this->verbose = true;
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
  try {
    waitIdle();
  } catch (...) {
  }
  while (!kernels.empty()) {
    releaseKernel(kernels.begin()->first);
  }
  while (!buffers.empty()) {
    releaseBuffer(buffers.begin()->first);
  }
  if (device != VK_NULL_HANDLE) {
    for (size_t i = 0; i < kMaxInFlight; ++i) {
      if (inFlightFrames[i].fence != VK_NULL_HANDLE) {
        vkDestroyFence(device, inFlightFrames[i].fence, nullptr);
        inFlightFrames[i].fence = VK_NULL_HANDLE;
      }
    }
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

  // Negotiate the instance API version: request up to Vulkan 1.4 but clamp
  // to what the loader/driver stack actually supports (MoltenVK and older
  // drivers may only expose 1.0-1.3). vkEnumerateInstanceVersion does not
  // exist on Vulkan 1.0 loaders, so resolve it dynamically and fall back to
  // 1.0 when it is missing.
  uint32_t apiVersion = VK_API_VERSION_1_0;
  auto pfnEnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
  if (pfnEnumerateInstanceVersion) {
    uint32_t supportedVersion = VK_API_VERSION_1_0;
    if (pfnEnumerateInstanceVersion(&supportedVersion) == VK_SUCCESS) {
      apiVersion = supportedVersion;
    }
  }
  if (apiVersion > VK_API_VERSION_1_4) {
    apiVersion = VK_API_VERSION_1_4;
  }
  appInfo.apiVersion = apiVersion;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  std::vector<const char*> extensions;
#ifdef __APPLE__
  // MoltenVK requires the portability enumeration extension (and its flag)
  // to expose physical devices on macOS.
  extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames =
      extensions.empty() ? nullptr : extensions.data();

  std::vector<const char*> layers;
  if (debug) {
      layers.push_back("VK_LAYER_KHRONOS_validation");
  }
  createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
  createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

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
      info.bf16Support = true;
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
  info.bf16Support = true;
  info.int8Support = true;
  info.cooperativeMatrixSupport =
      hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  info.fp8Support = hasExt("VK_EXT_shader_float8");
  info.fp6Support = false;
  info.fp4Support = false; // No Vulkan FP4 shader type exists; Fp4Bench is
                           // deliberately disabled (see its IsSupported).
  // No Vulkan/SPIR-V cooperative-matrix component type for 4-bit integers
  // exists, so native INT4 rates cannot be measured through Vulkan. The
  // coop_matrix_int4.comp shader actually performs INT8 math.
  info.int4Support = false;
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
  if (physicalDevice == physicalDevices[index] && device != VK_NULL_HANDLE) {
    return; // Already initialized for this device
  }

  if (device != VK_NULL_HANDLE) {
    try {
      waitIdle();
    } catch (...) {
    }
    while (!kernels.empty()) {
      releaseKernel(kernels.begin()->first);
    }
    while (!buffers.empty()) {
      releaseBuffer(buffers.begin()->first);
    }
    for (size_t i = 0; i < kMaxInFlight; ++i) {
      if (inFlightFrames[i].fence != VK_NULL_HANDLE) {
        vkDestroyFence(device, inFlightFrames[i].fence, nullptr);
        inFlightFrames[i].fence = VK_NULL_HANDLE;
      }
      inFlightFrames[i].commandBuffer = VK_NULL_HANDLE;
      inFlightFrames[i].inUse = false;
    }
    currentFrameIndex = 0;
    if (commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, commandPool, nullptr);
      commandPool = VK_NULL_HANDLE;
    }
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
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

  // Query supported extensions first
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, availableExtensions.data());

  auto hasExt = [&](const char *name) {
    for (const auto &available : availableExtensions) {
      if (strcmp(name, available.extensionName) == 0) return true;
    }
    return false;
  };

  // Explicitly using the struct names for EXT/KHR features
  struct VkPhysicalDeviceFloat8FeaturesEXT {
    VkStructureType sType;
    void *pNext;
    VkBool32 shaderFloat8;
  } float8Features{(VkStructureType)1000521001, nullptr, VK_FALSE};

  struct VkPhysicalDeviceShaderFloatControls2FeaturesKHR {
    VkStructureType sType;
    void *pNext;
    VkBool32 shaderFloatControls2;
  } floatControls2Features{(VkStructureType)1000528001, nullptr, VK_FALSE};

  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, nullptr};

  struct VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXTCustom {
    VkStructureType sType;
    void *pNext;
    VkBool32 rayTracingInvocationReorderEXT;
  } serFeatures{(VkStructureType)1000581000, nullptr, VK_FALSE};

  VkPhysicalDeviceRayTracingMaintenance1FeaturesKHR rtMaint1Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MAINTENANCE_1_FEATURES_KHR,
      nullptr, VK_FALSE, VK_FALSE};

  VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT dgcFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT,
      nullptr, VK_FALSE, VK_FALSE};

  VkPhysicalDeviceShaderEnqueueFeaturesAMDX enqueueFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ENQUEUE_FEATURES_AMDX,
      nullptr, VK_FALSE, VK_FALSE};

  VkPhysicalDeviceShaderIntegerDotProductFeatures dotProductFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, nullptr};

  void** currentPNext = (void**)&features2.pNext;
  *currentPNext = &features168; currentPNext = &features168.pNext;
  *currentPNext = &features16Storage; currentPNext = &features16Storage.pNext;
  *currentPNext = &features8Storage; currentPNext = &features8Storage.pNext;
  *currentPNext = &coopMatrixFeatures; currentPNext = &coopMatrixFeatures.pNext;
  *currentPNext = &dotProductFeatures; currentPNext = &dotProductFeatures.pNext;
  *currentPNext = &subgroupSizeFeatures; currentPNext = &subgroupSizeFeatures.pNext;
  *currentPNext = &asFeatures; currentPNext = &asFeatures.pNext;
  *currentPNext = &rayQueryFeatures; currentPNext = &rayQueryFeatures.pNext;
  *currentPNext = &bufferDeviceAddressFeatures; currentPNext = &bufferDeviceAddressFeatures.pNext;

  if (hasExt("VK_EXT_shader_float8")) {
      *currentPNext = &float8Features; currentPNext = &float8Features.pNext;
  }
  if (hasExt("VK_KHR_shader_float_controls2")) {
      *currentPNext = &floatControls2Features; currentPNext = &floatControls2Features.pNext;
  }
  if (hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
      *currentPNext = &rtPipelineFeatures; currentPNext = &rtPipelineFeatures.pNext;
  }
  if (hasExt("VK_EXT_ray_tracing_invocation_reorder")) {
      *currentPNext = &serFeatures; currentPNext = &serFeatures.pNext;
  }
  if (hasExt(VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME)) {
      *currentPNext = &rtMaint1Features; currentPNext = &rtMaint1Features.pNext;
  }
  if (hasExt(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME)) {
      *currentPNext = &dgcFeatures; currentPNext = &dgcFeatures.pNext;
  }
  if (hasExt("VK_AMDX_shader_enqueue")) {
      *currentPNext = &enqueueFeatures; currentPNext = &enqueueFeatures.pNext;
  }
  *currentPNext = nullptr;

  // Query supported features and enable them
  vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

  const std::vector<const char *> desiredExtensions = {
      VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
      VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
      VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
      VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
      VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME,
      VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
      VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_RAY_QUERY_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
      VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME,
      VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME,
      "VK_EXT_shader_float8",
      "VK_KHR_shader_float_controls2",
      "VK_EXT_ray_tracing_invocation_reorder",
      "VK_NV_ray_tracing_invocation_reorder",
      "VK_AMDX_shader_enqueue"};

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
      if (verbose) {
          std::cerr << "Warning: Extension " << extension
                    << " not supported by device, disabling." << std::endl;
      }
    }
  }

  enabledExtensionsSet.clear();
  for (const auto &ext : enabledExtensions) {
    enabledExtensionsSet.insert(ext);
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

  VkCommandBufferAllocateInfo cmdAllocInfo{};
  cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAllocInfo.commandPool = commandPool;
  cmdAllocInfo.commandBufferCount = static_cast<uint32_t>(kMaxInFlight);

  std::vector<VkCommandBuffer> cmdBuffers(kMaxInFlight);
  if (vkAllocateCommandBuffers(device, &cmdAllocInfo, cmdBuffers.data()) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate in-flight command buffers!");
  }

  for (size_t i = 0; i < kMaxInFlight; ++i) {
    inFlightFrames[i].commandBuffer = cmdBuffers[i];
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFrames[i].fence) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create in-flight fence!");
    }
    inFlightFrames[i].inUse = false;
  }
  currentFrameIndex = 0;
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
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
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
  if (file_name.find(".rgen") != std::string::npos ||
      file_name.find(".rmiss") != std::string::npos ||
      file_name.find(".rchit") != std::string::npos) {
    if (verbose) {
      std::cout << "Skipping compute compile for RT shader: " << file_name
                << "\n";
    }
    return nullptr;
  }
  bool is_glsl = false;
  std::string file_ext;
  size_t last_dot = file_name.find_last_of('.');
  if (last_dot != std::string::npos) {
    file_ext = file_name.substr(last_dot);
  }

  if (file_ext == ".comp" || file_ext == ".rgen" || file_ext == ".rmiss" ||
      file_ext == ".rchit" || file_ext == ".rahit" || file_ext == ".rint") {
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

  InFlightFrame &frame = inFlightFrames[currentFrameIndex];
  if (frame.inUse) {
    constexpr uint64_t kTimeoutNs = 3'000'000'000ULL;
    VkResult waitResult =
        vkWaitForFences(device, 1, &frame.fence, VK_TRUE, kTimeoutNs);
    if (waitResult == VK_TIMEOUT) {
      throw std::runtime_error(
          "GPU dispatch timed out (>3 s) — aborting benchmark to prevent amdgpu TDR crash.");
    } else if (waitResult != VK_SUCCESS) {
      throw std::runtime_error("vkWaitForFences failed with result: " +
                               std::to_string(waitResult));
    }
    vkResetFences(device, 1, &frame.fence);
    frame.inUse = false;
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);
  VkPipelineBindPoint bindPoint = vulkanKernel->isRTPipeline
                                      ? VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR
                                      : VK_PIPELINE_BIND_POINT_COMPUTE;

  vkCmdBindPipeline(frame.commandBuffer, bindPoint, vulkanKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, bindPoint,
                          vulkanKernel->pipelineLayout, 0, 1,
                          &vulkanKernel->descriptorSet, 0, nullptr);

  // Push constants for non-buffer arguments
  if (!vulkanKernel->pushConstantData.empty()) {
    VkShaderStageFlags stageFlags = vulkanKernel->isRTPipeline
                                        ? (VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                           VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                           VK_SHADER_STAGE_MISS_BIT_KHR)
                                        : VK_SHADER_STAGE_COMPUTE_BIT;

    vkCmdPushConstants(
        frame.commandBuffer, vulkanKernel->pipelineLayout, stageFlags, 0,
        static_cast<uint32_t>(vulkanKernel->pushConstantData.size()),
        vulkanKernel->pushConstantData.data());
  }

  if (vulkanKernel->isRTPipeline) {
    auto pfnTraceRays =
        (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");
    uint32_t rWidth = (block_x > 1) ? (grid_x * block_x) : grid_x;
    uint32_t rHeight = (block_y > 1) ? (grid_y * block_y) : grid_y;
    uint32_t rDepth = (block_z > 1) ? (grid_z * block_z) : grid_z;
    pfnTraceRays(frame.commandBuffer, &vulkanKernel->rgenRegion,
                 &vulkanKernel->missRegion, &vulkanKernel->hitRegion,
                 &vulkanKernel->callRegion, rWidth, rHeight, rDepth);
  } else {
    vkCmdDispatch(frame.commandBuffer, grid_x, grid_y, grid_z);
  }

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  frame.inUse = true;

  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

void VulkanContext::dispatchIndirect(ComputeKernel kernel_handle,
                                     ComputeBuffer indirectBuffer,
                                     VkDeviceSize offset) {
  auto *vulkanKernel = kernels[kernel_handle];
  if (!vulkanKernel)
    return;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (frame.inUse) {
    vkWaitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &frame.fence);
    frame.inUse = false;
  }

  vkResetCommandBuffer(frame.commandBuffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    vulkanKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkanKernel->pipelineLayout, 0, 1,
                          &vulkanKernel->descriptorSet, 0, nullptr);

  if (!vulkanKernel->pushConstantData.empty()) {
    vkCmdPushConstants(frame.commandBuffer, vulkanKernel->pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       vulkanKernel->pushConstantData.size(),
                       vulkanKernel->pushConstantData.data());
  }

  VkBuffer vkIndirect = getVkBuffer(indirectBuffer);
  vkCmdDispatchIndirect(frame.commandBuffer, vkIndirect, offset);

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  frame.inUse = true;

  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

void VulkanContext::dispatchIndirectSequence(
    ComputeKernel kernel_handle, ComputeBuffer indirectBuffer,
    const std::vector<IndirectBatchEntry> &entries) {
  auto *vulkanKernel = kernels[kernel_handle];
  if (!vulkanKernel || entries.empty())
    return;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (frame.inUse) {
    vkWaitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &frame.fence);
    frame.inUse = false;
  }

  vkResetCommandBuffer(frame.commandBuffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    vulkanKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkanKernel->pipelineLayout, 0, 1,
                          &vulkanKernel->descriptorSet, 0, nullptr);

  VkBuffer vkIndirect = getVkBuffer(indirectBuffer);

  for (const auto &entry : entries) {
    if (!entry.pushConstants.empty()) {
      vkCmdPushConstants(frame.commandBuffer, vulkanKernel->pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         entry.pushConstants.size(),
                         entry.pushConstants.data());
    }
    vkCmdDispatchIndirect(frame.commandBuffer, vkIndirect, entry.offset);
  }

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  frame.inUse = true;

  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

void VulkanContext::dispatchWorkListSequence(
    ComputeBuffer clearBuf1, size_t clearSize1,
    ComputeBuffer clearBuf2, size_t clearSize2,
    ComputeKernel classifyKernel_handle, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    ComputeKernel secondKernel_handle, ComputeBuffer indirectBuffer,
    const std::vector<IndirectBatchEntry> &entries) {
  auto *classifyKernel = kernels[classifyKernel_handle];
  auto *secondKernel = kernels[secondKernel_handle];
  if (!classifyKernel || !secondKernel)
    return;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (frame.inUse) {
    vkWaitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &frame.fence);
    frame.inUse = false;
  }

  vkResetCommandBuffer(frame.commandBuffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

  // 1. Clear queue counters & indirect dispatch commands on the GPU DMA engine
  bool cleared = false;
  if (clearBuf1 && clearSize1 > 0) {
    VkBuffer b1 = getVkBuffer(clearBuf1);
    if (b1) {
      vkCmdFillBuffer(frame.commandBuffer, b1, 0, clearSize1, 0);
      cleared = true;
    }
  }
  if (clearBuf2 && clearSize2 > 0) {
    VkBuffer b2 = getVkBuffer(clearBuf2);
    if (b2) {
      vkCmdFillBuffer(frame.commandBuffer, b2, 0, clearSize2, 0);
      cleared = true;
    }
  }

  if (cleared) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
  }

  // 2. Pass 1: Traversal & classification / stream compaction
  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    classifyKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          classifyKernel->pipelineLayout, 0, 1,
                          &classifyKernel->descriptorSet, 0, nullptr);
  if (!classifyKernel->pushConstantData.empty()) {
    vkCmdPushConstants(frame.commandBuffer, classifyKernel->pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       classifyKernel->pushConstantData.size(),
                       classifyKernel->pushConstantData.data());
  }
  vkCmdDispatch(frame.commandBuffer, grid_x, grid_y, grid_z);

  // 3. Pipeline Barrier: Pass 1 writes -> Pass 2 reads & indirect dispatches
  VkMemoryBarrier passBarrier{};
  passBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  passBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  passBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &passBarrier, 0, nullptr, 0, nullptr);

  // 4. Pass 2: Uniform Indirect Dispatches (DGC / Work Lists)
  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    secondKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          secondKernel->pipelineLayout, 0, 1,
                          &secondKernel->descriptorSet, 0, nullptr);

  VkBuffer vkIndirect = getVkBuffer(indirectBuffer);
  for (const auto &entry : entries) {
    if (!entry.pushConstants.empty()) {
      vkCmdPushConstants(frame.commandBuffer, secondKernel->pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         entry.pushConstants.size(),
                         entry.pushConstants.data());
    }
    vkCmdDispatchIndirect(frame.commandBuffer, vkIndirect, entry.offset);
  }

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  frame.inUse = true;

  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

void VulkanContext::dispatchRayTracingIndirect(ComputeKernel kernel_handle,
                                             ComputeBuffer indirectBuffer,
                                             VkDeviceSize offset) {
  auto *vulkanKernel = kernels[kernel_handle];
  if (!vulkanKernel || !vulkanKernel->isRTPipeline)
    return;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (frame.inUse) {
    vkWaitForFences(device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &frame.fence);
    frame.inUse = false;
  }

  vkResetCommandBuffer(frame.commandBuffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                    vulkanKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer,
                          VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                          vulkanKernel->pipelineLayout, 0, 1,
                          &vulkanKernel->descriptorSet, 0, nullptr);

  if (!vulkanKernel->pushConstantData.empty()) {
    vkCmdPushConstants(
        frame.commandBuffer, vulkanKernel->pipelineLayout,
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_MISS_BIT_KHR,
        0, vulkanKernel->pushConstantData.size(),
        vulkanKernel->pushConstantData.data());
  }

  auto pfnTraceRaysIndirect2 = (PFN_vkCmdTraceRaysIndirect2KHR)vkGetDeviceProcAddr(
      device, "vkCmdTraceRaysIndirect2KHR");
  if (pfnTraceRaysIndirect2) {
    VkDeviceAddress indirectAddress =
        getBufferDeviceAddress(indirectBuffer) + offset;
    pfnTraceRaysIndirect2(frame.commandBuffer, indirectAddress);
  }

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  frame.inUse = true;

  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

void VulkanContext::releaseKernel(ComputeKernel kernel) {
  auto it = kernels.find(kernel);
  if (it != kernels.end()) {
    VulkanKernel *vulkanKernel = it->second;
    vkDestroyPipeline(device, vulkanKernel->pipeline, nullptr);
    vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout,
                                 nullptr);
    if (vulkanKernel->shaderModule) {
      vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
    }
    if (vulkanKernel->sbtBuffer) {
      releaseBuffer(vulkanKernel->sbtBuffer);
    }
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
  // Note: no verbose guard here. This is only called from
  // notifyKernelCreated() in non-verbose mode (matching the OpenCL/ROCm
  // contexts), where the progress bar is the only setup feedback shown.
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

ComputeKernel VulkanContext::createRTPipeline(
    const std::string &rgen_path, const std::string &rmiss_path,
    const std::vector<std::string> &rchit_paths,
    const std::vector<std::string> &rahit_paths,
    const std::vector<std::string> &rint_paths, uint32_t num_buffer_args) {

  auto load_shader = [&](const std::string &path) -> VkShaderModule {
    std::string spv_path = path + ".spv";
    std::ifstream file(spv_path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
      throw std::runtime_error("Failed to open " + spv_path);
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char *)buffer.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create shader module for " +
                               spv_path);
    }
    return shaderModule;
  };

  VulkanKernel *vulkanKernel = new VulkanKernel();
  vulkanKernel->isRTPipeline = true;
  vulkanKernel->numBufferDescriptors = num_buffer_args;

  std::vector<VkShaderModule> modules;
  std::vector<VkPipelineShaderStageCreateInfo> stages;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

  auto add_stage = [&](VkShaderModule module, VkShaderStageFlagBits stage) {
    VkPipelineShaderStageCreateInfo info{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage = stage;
    info.module = module;
    info.pName = "main";
    stages.push_back(info);
  };

  // Raygen
  VkShaderModule rgen_module = load_shader(rgen_path);
  modules.push_back(rgen_module);
  add_stage(rgen_module, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  VkRayTracingShaderGroupCreateInfoKHR group{
      VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = 0;
  group.closestHitShader = VK_SHADER_UNUSED_KHR;
  group.anyHitShader = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;
  groups.push_back(group);

  // Miss
  VkShaderModule rmiss_module = load_shader(rmiss_path);
  modules.push_back(rmiss_module);
  add_stage(rmiss_module, VK_SHADER_STAGE_MISS_BIT_KHR);
  group.generalShader = 1;
  groups.push_back(group);

  // Hit Groups
  size_t num_hit_groups = std::max(rchit_paths.size(), std::max(rahit_paths.size(), rint_paths.size()));
  uint32_t current_stage_index = 2; // 0 is rgen, 1 is rmiss

  for (size_t i = 0; i < rchit_paths.size(); i++) {
    VkShaderModule module = load_shader(rchit_paths[i]);
    modules.push_back(module);
    add_stage(module, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
  }
  uint32_t rahit_stage_start = current_stage_index + rchit_paths.size();

  for (size_t i = 0; i < rahit_paths.size(); i++) {
    VkShaderModule module = load_shader(rahit_paths[i]);
    modules.push_back(module);
    add_stage(module, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
  }
  uint32_t rint_stage_start = rahit_stage_start + rahit_paths.size();

  for (size_t i = 0; i < rint_paths.size(); i++) {
    VkShaderModule module = load_shader(rint_paths[i]);
    modules.push_back(module);
    add_stage(module, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);
  }

  for (size_t i = 0; i < num_hit_groups; i++) {
    VkRayTracingShaderGroupCreateInfoKHR hitGroup{
        VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    if (i < rint_paths.size()) {
      hitGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    } else {
      hitGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    }
    hitGroup.generalShader = VK_SHADER_UNUSED_KHR;
    hitGroup.closestHitShader = (i < rchit_paths.size()) ? (current_stage_index + i) : VK_SHADER_UNUSED_KHR;
    hitGroup.anyHitShader = (i < rahit_paths.size()) ? (rahit_stage_start + i) : VK_SHADER_UNUSED_KHR;
    hitGroup.intersectionShader = (i < rint_paths.size()) ? (rint_stage_start + i) : VK_SHADER_UNUSED_KHR;
    groups.push_back(hitGroup);
  }

  // Descriptors and Layout
  std::vector<VkDescriptorSetLayoutBinding> bindings;
  for (uint32_t i = 0; i < num_buffer_args; i++) {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = i;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    if (i == 0)
      binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                         VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                         VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                         VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                         VK_SHADER_STAGE_MISS_BIT_KHR;
    bindings.push_back(binding);
  }

  VkDescriptorSetLayoutCreateInfo layoutInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();
  vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                              &vulkanKernel->descriptorSetLayout);

  VkPushConstantRange pushConstant{};
  pushConstant.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                            VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                            VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
                            VK_SHADER_STAGE_MISS_BIT_KHR;
  pushConstant.offset = 0;
  pushConstant.size = 128; // Up to 128 bytes
  vulkanKernel->pushConstantData.resize(128);

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &vulkanKernel->descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
  vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                         &vulkanKernel->pipelineLayout);

  // Pipeline execution
  VkRayTracingPipelineCreateInfoKHR pipelineInfo{
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages = stages.data();
  pipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
  pipelineInfo.pGroups = groups.data();
  pipelineInfo.maxPipelineRayRecursionDepth = 1;
  pipelineInfo.layout = vulkanKernel->pipelineLayout;

  auto pfnCreateRays = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(
      device, "vkCreateRayTracingPipelinesKHR");
  if (pfnCreateRays(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo,
                    nullptr, &vulkanKernel->pipeline) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create RT Pipeline");
  }

  // SBT
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 props2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  props2.pNext = &rtProps;
  vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

  uint32_t handleSize = rtProps.shaderGroupHandleSize;
  uint32_t handleSizeAligned =
      (handleSize + rtProps.shaderGroupHandleAlignment - 1) &
      ~(rtProps.shaderGroupHandleAlignment - 1);
  uint32_t groupCount = static_cast<uint32_t>(groups.size());
  uint32_t sbtSize = groupCount * handleSizeAligned;

  std::vector<uint8_t> handles(groupCount * handleSize);
  auto pfnGetHandles =
      (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(
          device, "vkGetRayTracingShaderGroupHandlesKHR");
  pfnGetHandles(device, vulkanKernel->pipeline, 0, groupCount, handles.size(),
                handles.data());

  vulkanKernel->sbtBuffer = createBuffer(sbtSize);
  VkBuffer vkSbt = getVkBuffer(vulkanKernel->sbtBuffer);
  VkDeviceAddress sbtAddr = getBufferDeviceAddress(vulkanKernel->sbtBuffer);

  // Upload handles
  std::vector<uint8_t> sbtData(sbtSize, 0);
  for (uint32_t i = 0; i < groupCount; i++) {
    memcpy(sbtData.data() + i * handleSizeAligned,
           handles.data() + i * handleSize, handleSize);
  }
  writeBuffer(vulkanKernel->sbtBuffer, 0, sbtSize, sbtData.data());

  // Regions
  // 0: rgen, 1: miss, 2..N: closest hit
  vulkanKernel->rgenRegion.deviceAddress = sbtAddr;
  vulkanKernel->rgenRegion.stride = handleSizeAligned;
  vulkanKernel->rgenRegion.size = handleSizeAligned;

  vulkanKernel->missRegion.deviceAddress = sbtAddr + handleSizeAligned;
  vulkanKernel->missRegion.stride = handleSizeAligned;
  vulkanKernel->missRegion.size = handleSizeAligned;

  vulkanKernel->hitRegion.deviceAddress = sbtAddr + 2 * handleSizeAligned;
  vulkanKernel->hitRegion.stride = handleSizeAligned;
  vulkanKernel->hitRegion.size = (groupCount - 2) * handleSizeAligned;

  vulkanKernel->callRegion.deviceAddress = 0;
  vulkanKernel->callRegion.stride = 0;
  vulkanKernel->callRegion.size = 0;

  // Cleanup modules
  for (auto m : modules) {
    vkDestroyShaderModule(device, m, nullptr);
  }

  // Allocate Descriptor Sets
  std::vector<VkDescriptorPoolSize> poolSizes = {
      {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, num_buffer_args - 1}
      // assuming first is AS, rest are buffers
  };
  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = 1;
  vkCreateDescriptorPool(device, &poolInfo, nullptr,
                         &vulkanKernel->descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool = vulkanKernel->descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &vulkanKernel->descriptorSetLayout;
  vkAllocateDescriptorSets(device, &allocInfo, &vulkanKernel->descriptorSet);

  kernels[vulkanKernel] = vulkanKernel;
  return vulkanKernel;
}
