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
      } else if (waitResult != VK_SUCCESS) {
        throw std::runtime_error("vkWaitForFences failed in waitIdle with result: " +
                                 std::to_string(waitResult));
      }
      vkResetFences(device, 1, &inFlightFrames[i].fence);
      inFlightFrames[i].inUse = false;
    }
  }
  if (computeQueue != VK_NULL_HANDLE) {
    VkResult queueResult = vkQueueWaitIdle(computeQueue);
    if (queueResult != VK_SUCCESS) {
      throw std::runtime_error("vkQueueWaitIdle failed in waitIdle with result: " +
                               std::to_string(queueResult));
    }
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
  destroyHeadlessSwapchain();
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

  uint32_t instExtCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &instExtCount, nullptr);
  std::vector<VkExtensionProperties> availableInstExts(instExtCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &instExtCount, availableInstExts.data());
  auto hasInstExt = [&](const char *name) {
    for (const auto &ext : availableInstExts) {
      if (strcmp(name, ext.extensionName) == 0) return true;
    }
    return false;
  };

  headlessSurfaceSupported = false;
  if (hasInstExt(VK_KHR_SURFACE_EXTENSION_NAME) &&
      hasInstExt(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME)) {
    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    extensions.push_back(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
    headlessSurfaceSupported = true;
  }

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

      VkPhysicalDeviceDriverProperties driverProps{};
      driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;

      VkPhysicalDeviceSubgroupProperties subgroupProps{};
      subgroupProps.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
      subgroupProps.pNext = &driverProps;

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
      info.vendorID = props.vendorID;
      info.deviceID = props.deviceID;
      info.apiVersion = props.apiVersion;
      info.driverVersion = props.driverVersion;

      std::string driverVerStr;
      if (props.vendorID == 0x10DE) {
        uint32_t major = (props.driverVersion >> 22) & 0x3FF;
        uint32_t minor = (props.driverVersion >> 14) & 0xFF;
        uint32_t sec = (props.driverVersion >> 6) & 0xFF;
        uint32_t tert = props.driverVersion & 0x3F;
        driverVerStr = std::to_string(major) + "." + std::to_string(minor);
        if (sec > 0 || tert > 0) {
          driverVerStr += "." + std::to_string(sec) + "." + std::to_string(tert);
        }
      } else {
        driverVerStr = std::to_string(VK_VERSION_MAJOR(props.driverVersion)) + "." +
                       std::to_string(VK_VERSION_MINOR(props.driverVersion)) + "." +
                       std::to_string(VK_VERSION_PATCH(props.driverVersion));
      }
      info.driverVersionStr = driverVerStr;
      info.driverName = (driverProps.driverName[0] != '\0') ? driverProps.driverName : "";
      info.driverInfo = (driverProps.driverInfo[0] != '\0') ? driverProps.driverInfo : "";

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

#if defined(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
      VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT serFeatures{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT};
      bool hasSERExt = hasExt(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) ||
                       hasExt("VK_NV_ray_tracing_invocation_reorder");
      if (hasSERExt) {
        VkPhysicalDeviceFeatures2 f2_ser{};
        f2_ser.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2_ser.pNext = &serFeatures;
        vkGetPhysicalDeviceFeatures2(device, &f2_ser);
        info.serSupported = (serFeatures.rayTracingInvocationReorder == VK_TRUE);
      } else {
        info.serSupported = false;
      }
#elif defined(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
      VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeatures{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};
      bool hasSERExt = hasExt(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
      if (hasSERExt) {
        VkPhysicalDeviceFeatures2 f2_ser{};
        f2_ser.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2_ser.pNext = &serFeatures;
        vkGetPhysicalDeviceFeatures2(device, &f2_ser);
        info.serSupported = (serFeatures.rayTracingInvocationReorder == VK_TRUE);
      } else {
        info.serSupported = false;
      }
#else
      info.serSupported = hasExt("VK_EXT_ray_tracing_invocation_reorder") ||
                          hasExt("VK_NV_ray_tracing_invocation_reorder");
#endif
      info.workGraphsSupported = hasExt("VK_AMDX_shader_enqueue") || hasExt("VK_KHR_work_graphs");

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

  VkPhysicalDeviceDriverProperties driverProps{};
  driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;

  VkPhysicalDeviceSubgroupProperties subgroupProps{};
  subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  subgroupProps.pNext = &driverProps;

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
  info.vendorID = properties.vendorID;
  info.deviceID = properties.deviceID;
  info.apiVersion = properties.apiVersion;
  info.driverVersion = properties.driverVersion;

  std::string driverVerStr;
  if (properties.vendorID == 0x10DE) {
    uint32_t major = (properties.driverVersion >> 22) & 0x3FF;
    uint32_t minor = (properties.driverVersion >> 14) & 0xFF;
    uint32_t sec = (properties.driverVersion >> 6) & 0xFF;
    uint32_t tert = properties.driverVersion & 0x3F;
    driverVerStr = std::to_string(major) + "." + std::to_string(minor);
    if (sec > 0 || tert > 0) {
      driverVerStr += "." + std::to_string(sec) + "." + std::to_string(tert);
    }
  } else {
    driverVerStr = std::to_string(VK_VERSION_MAJOR(properties.driverVersion)) + "." +
                   std::to_string(VK_VERSION_MINOR(properties.driverVersion)) + "." +
                   std::to_string(VK_VERSION_PATCH(properties.driverVersion));
  }
  info.driverVersionStr = driverVerStr;
  info.driverName = (driverProps.driverName[0] != '\0') ? driverProps.driverName : "";
  info.driverInfo = (driverProps.driverInfo[0] != '\0') ? driverProps.driverInfo : "";

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

#if defined(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT serFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT};
  bool hasSERExt = hasExt(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME) ||
                   hasExt("VK_NV_ray_tracing_invocation_reorder");
  if (hasSERExt) {
    VkPhysicalDeviceFeatures2 f2_ser{};
    f2_ser.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2_ser.pNext = &serFeatures;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &f2_ser);
    info.serSupported = (serFeatures.rayTracingInvocationReorder == VK_TRUE);
  } else {
    info.serSupported = false;
  }
#elif defined(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};
  bool hasSERExt = hasExt(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME);
  if (hasSERExt) {
    VkPhysicalDeviceFeatures2 f2_ser{};
    f2_ser.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2_ser.pNext = &serFeatures;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &f2_ser);
    info.serSupported = (serFeatures.rayTracingInvocationReorder == VK_TRUE);
  } else {
    info.serSupported = false;
  }
#else
  info.serSupported = hasExt("VK_EXT_ray_tracing_invocation_reorder") ||
                      hasExt("VK_NV_ray_tracing_invocation_reorder");
#endif
  info.workGraphsSupported = hasExt("VK_AMDX_shader_enqueue") || hasExt("VK_KHR_work_graphs");

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
    destroyHeadlessSwapchain();
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

  VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
      nullptr, VK_FALSE};

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
  if (hasExt("VK_KHR_maintenance5")) {
      *currentPNext = &maintenance5Features; currentPNext = &maintenance5Features.pNext;
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
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
      "VK_KHR_maintenance5",
      "VK_EXT_shader_float8",
      "VK_KHR_shader_float_controls2",
      "VK_EXT_ray_tracing_invocation_reorder",
      "VK_NV_ray_tracing_invocation_reorder",
      "VK_AMDX_shader_enqueue"};

  swapchainSupported = hasExt(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

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

  serSupported = hasExt("VK_EXT_ray_tracing_invocation_reorder") &&
                 (serFeatures.rayTracingInvocationReorderEXT == VK_TRUE);

  subgroupSizeControlSupported = hasExt(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) &&
                                 (subgroupSizeFeatures.subgroupSizeControl == VK_TRUE);

  maintenance5Supported = hasExt("VK_KHR_maintenance5") &&
                          (maintenance5Features.maintenance5 == VK_TRUE);

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

  // Load DGC function pointers if extension is enabled
  if (hasExt(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME)) {
    vkGetGeneratedCommandsMemoryRequirementsEXT_ptr =
        (PFN_vkGetGeneratedCommandsMemoryRequirementsEXT)vkGetDeviceProcAddr(
            device, "vkGetGeneratedCommandsMemoryRequirementsEXT");
    vkCmdPreprocessGeneratedCommandsEXT_ptr =
        (PFN_vkCmdPreprocessGeneratedCommandsEXT)vkGetDeviceProcAddr(
            device, "vkCmdPreprocessGeneratedCommandsEXT");
    vkCmdExecuteGeneratedCommandsEXT_ptr =
        (PFN_vkCmdExecuteGeneratedCommandsEXT)vkGetDeviceProcAddr(
            device, "vkCmdExecuteGeneratedCommandsEXT");
    vkCreateIndirectCommandsLayoutEXT_ptr =
        (PFN_vkCreateIndirectCommandsLayoutEXT)vkGetDeviceProcAddr(
            device, "vkCreateIndirectCommandsLayoutEXT");
    vkDestroyIndirectCommandsLayoutEXT_ptr =
        (PFN_vkDestroyIndirectCommandsLayoutEXT)vkGetDeviceProcAddr(
            device, "vkDestroyIndirectCommandsLayoutEXT");
    vkCreateIndirectExecutionSetEXT_ptr =
        (PFN_vkCreateIndirectExecutionSetEXT)vkGetDeviceProcAddr(
            device, "vkCreateIndirectExecutionSetEXT");
    vkDestroyIndirectExecutionSetEXT_ptr =
        (PFN_vkDestroyIndirectExecutionSetEXT)vkGetDeviceProcAddr(
            device, "vkDestroyIndirectExecutionSetEXT");
    vkUpdateIndirectExecutionSetPipelineEXT_ptr =
        (PFN_vkUpdateIndirectExecutionSetPipelineEXT)vkGetDeviceProcAddr(
            device, "vkUpdateIndirectExecutionSetPipelineEXT");

    dgcSupported = (dgcFeatures.deviceGeneratedCommands == VK_TRUE) &&
                   (vkGetGeneratedCommandsMemoryRequirementsEXT_ptr != nullptr) &&
                   (vkCmdPreprocessGeneratedCommandsEXT_ptr != nullptr) &&
                   (vkCmdExecuteGeneratedCommandsEXT_ptr != nullptr) &&
                   (vkCreateIndirectCommandsLayoutEXT_ptr != nullptr) &&
                   (vkDestroyIndirectCommandsLayoutEXT_ptr != nullptr) &&
                   (vkCreateIndirectExecutionSetEXT_ptr != nullptr) &&
                   (vkDestroyIndirectExecutionSetEXT_ptr != nullptr) &&
                   (vkUpdateIndirectExecutionSetPipelineEXT_ptr != nullptr);
  } else {
    dgcSupported = false;
  }
  vkGetBufferDeviceAddressKHR_ptr =
      (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(
          device, "vkGetBufferDeviceAddressKHR");

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
  return createKernelInternal(file_name, kernel_name, num_buffer_args, nullptr, nullptr);
}

ComputeKernel VulkanContext::createKernelWithSpec(const std::string &file_name,
                                                 const std::string &kernel_name,
                                                 uint32_t num_buffer_args,
                                                 uint32_t spec_id,
                                                 uint32_t spec_val) {
  return createKernelInternal(file_name, kernel_name, num_buffer_args, &spec_id, &spec_val);
}

ComputeKernel VulkanContext::createKernelInternal(const std::string &file_name,
                                                 const std::string &kernel_name,
                                                 uint32_t num_buffer_args,
                                                 const uint32_t *spec_id,
                                                 const uint32_t *spec_val) {
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
  bool is_rt = (file_name.find("rt_") != std::string::npos &&
                file_name.find("reset") == std::string::npos &&
                file_name.find("resolve") == std::string::npos);

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

  VkSpecializationMapEntry specEntry{};
  VkSpecializationInfo specInfo{};
  if (spec_id && spec_val) {
    specEntry.constantID = *spec_id;
    specEntry.offset = 0;
    specEntry.size = sizeof(uint32_t);

    specInfo.mapEntryCount = 1;
    specInfo.pMapEntries = &specEntry;
    specInfo.dataSize = sizeof(uint32_t);
    specInfo.pData = spec_val;

    pipelineInfo.stage.pSpecializationInfo = &specInfo;
  }

  VkPipelineCreateFlags2CreateInfoKHR pipeFlags2{
      VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR};
  if (isDGCSupported() && maintenance5Supported) {
    pipeFlags2.flags = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT;
    pipelineInfo.pNext = &pipeFlags2;
  }

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

  VkResult submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  if (submitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatch with result: " +
                             std::to_string(submitRes));
  }
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

  VkResult indSubmitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  if (indSubmitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatchIndirect with result: " +
                             std::to_string(indSubmitRes));
  }
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
  VulkanKernel *lastBound = vulkanKernel;

  for (const auto &entry : entries) {
    VulkanKernel *kToBind = entry.specializedKernel ? kernels[entry.specializedKernel] : vulkanKernel;
    if (kToBind && kToBind != lastBound) {
      vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        kToBind->pipeline);
      vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              kToBind->pipelineLayout, 0, 1,
                              &kToBind->descriptorSet, 0, nullptr);
      lastBound = kToBind;
    }
    if (!entry.pushConstants.empty()) {
      vkCmdPushConstants(frame.commandBuffer, lastBound->pipelineLayout,
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

  VkResult indSeqSubmitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  if (indSeqSubmitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatchIndirectSequence with result: " +
                             std::to_string(indSeqSubmitRes));
  }
  frame.inUse = true;

  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

void VulkanContext::dispatchWorkListSequence(
    ComputeKernel resetKernel_handle,
    ComputeKernel classifyKernel_handle, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    ComputeKernel resolveKernel_handle,
    ComputeKernel secondKernel_handle, ComputeBuffer indirectBuffer,
    const std::vector<IndirectBatchEntry> &entries,
    bool isPingPong,
    const DGCExecutionInfo *dgcInfo,
    uint32_t dgcMode) {
  auto *classifyKernel = kernels[classifyKernel_handle];
  auto *secondKernel = kernels[secondKernel_handle];
  if (!classifyKernel || !secondKernel)
    return;

  auto &frame = inFlightFrames[currentFrameIndex];
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

  vkResetCommandBuffer(frame.commandBuffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

  // 1. Reset queue counters & indirect commands on compute queue (zero transfer bubbles)
  if (resetKernel_handle) {
    auto *resetKernel = kernels[resetKernel_handle];
    if (resetKernel) {
      vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resetKernel->pipeline);
      vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              resetKernel->pipelineLayout, 0, 1,
                              &resetKernel->descriptorSet, 0, nullptr);
      vkCmdDispatch(frame.commandBuffer, 1, 1, 1);

      VkMemoryBarrier resetBarrier{};
      resetBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      resetBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      resetBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &resetBarrier, 0,
                           nullptr, 0, nullptr);
    }
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

  bool useDGC = (dgcInfo != nullptr) && isDGCSupported() && (vkCmdExecuteGeneratedCommandsEXT_ptr != nullptr) && (dgcInfo->layout != VK_NULL_HANDLE);

  // 3. Resolve: Convert queue counters to indirect dispatch commands (32 threads, 1 wave, <0.5 us)
  if (resolveKernel_handle) {
    auto *resolveKernel = kernels[resolveKernel_handle];
    if (resolveKernel) {
      VkMemoryBarrier classifyBarrier{};
      classifyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      classifyBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      classifyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &classifyBarrier, 0,
                           nullptr, 0, nullptr);

      vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resolveKernel->pipeline);
      vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              resolveKernel->pipelineLayout, 0, 1,
                              &resolveKernel->descriptorSet, 0, nullptr);
      uint32_t resetQueue = isPingPong ? 1u : 0xFFFFFFFFu;
      if (useDGC) {
        struct { uint32_t resetQueue; uint32_t dgcMode; uint32_t bounceIndex; } rpcResolve{resetQueue, dgcMode, 0u};
        vkCmdPushConstants(frame.commandBuffer, resolveKernel->pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(rpcResolve), &rpcResolve);
      } else {
        struct { uint32_t resetQueue; uint32_t dgcMode; uint32_t bounceIndex; } rpcResolve{resetQueue, 0u, 0u};
        vkCmdPushConstants(frame.commandBuffer, resolveKernel->pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(rpcResolve), &rpcResolve);
      }
      vkCmdDispatch(frame.commandBuffer, 1, 1, 1);

      VkMemoryBarrier resolveBarrier{};
      resolveBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      resolveBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      resolveBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &resolveBarrier, 0, nullptr, 0, nullptr);
    }
  } else {
    VkMemoryBarrier passBarrier{};
    passBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    passBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    passBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &passBarrier, 0, nullptr, 0, nullptr);
  }

  // 4. Pass 2: Indirect Dispatches (DGC or Legacy Fallback)
  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    secondKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          secondKernel->pipelineLayout, 0, 1,
                          &secondKernel->descriptorSet, 0, nullptr);

  if (useDGC) {
    if (!isPingPong) {
      // Direct DGC execution with dynamic sequence pruning
      VkGeneratedCommandsPipelineInfoEXT pipeInfo{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT};
      pipeInfo.pipeline = secondKernel->pipeline;

      VkGeneratedCommandsInfoEXT genCmds{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT};
      if (dgcInfo->executionSet == VK_NULL_HANDLE) {
        genCmds.pNext = &pipeInfo;
      }
      genCmds.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
      genCmds.indirectExecutionSet = dgcInfo->executionSet;
      genCmds.indirectCommandsLayout = dgcInfo->layout;
      genCmds.indirectAddress = getBufferDeviceAddress(dgcInfo->sequenceBuffer) + dgcInfo->sequenceBufferOffset;
      genCmds.indirectAddressSize = dgcInfo->sequenceBufferSize;
      genCmds.preprocessAddress = getBufferDeviceAddress(dgcInfo->preprocessBuffer);
      genCmds.preprocessSize = dgcInfo->preprocessBufferSize;
      genCmds.maxSequenceCount = dgcInfo->maxSequenceCount;
      genCmds.sequenceCountAddress = dgcInfo->sequenceCountBuffer ?
          (getBufferDeviceAddress(dgcInfo->sequenceCountBuffer) + dgcInfo->sequenceCountBufferOffset) : 0;
      genCmds.maxDrawCount = 0;

      vkCmdExecuteGeneratedCommandsEXT_ptr(frame.commandBuffer, VK_FALSE, &genCmds);
    } else {
      // Multi-bounce ping-pong compaction via DGC
      VulkanKernel *lastBound = secondKernel;
      for (size_t e = 0; e < entries.size(); ++e) {
        const auto &entry = entries[e];
        VulkanKernel *kToBind = entry.specializedKernel ? kernels[entry.specializedKernel] : secondKernel;
        if (kToBind && kToBind != lastBound) {
          vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kToBind->pipeline);
          vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  kToBind->pipelineLayout, 0, 1, &kToBind->descriptorSet, 0, nullptr);
          lastBound = kToBind;
        }

        VkGeneratedCommandsPipelineInfoEXT pipeInfo{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT};
        pipeInfo.pipeline = lastBound->pipeline;

        VkGeneratedCommandsInfoEXT genCmds{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT};
        if (dgcInfo->executionSet == VK_NULL_HANDLE) {
          genCmds.pNext = &pipeInfo;
        }
        genCmds.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
        genCmds.indirectExecutionSet = dgcInfo->executionSet;
        genCmds.indirectCommandsLayout = dgcInfo->layout;
        genCmds.indirectAddress = getBufferDeviceAddress(dgcInfo->sequenceBuffer) + dgcInfo->sequenceBufferOffset + e * sizeof(uint32_t) * 12;
        genCmds.indirectAddressSize = sizeof(uint32_t) * 12;
        genCmds.preprocessAddress = getBufferDeviceAddress(dgcInfo->preprocessBuffer);
        genCmds.preprocessSize = dgcInfo->preprocessBufferSize;
        genCmds.maxSequenceCount = 1;
        genCmds.sequenceCountAddress = dgcInfo->sequenceCountBuffer ?
            (getBufferDeviceAddress(dgcInfo->sequenceCountBuffer) + dgcInfo->sequenceCountBufferOffset) : 0;
        genCmds.maxDrawCount = 0;

        vkCmdExecuteGeneratedCommandsEXT_ptr(frame.commandBuffer, VK_FALSE, &genCmds);

        if (e + 1 < entries.size() && resolveKernel_handle) {
          auto *resolveKernel = kernels[resolveKernel_handle];
          if (resolveKernel) {
            VkMemoryBarrier bounceBarrier{};
            bounceBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            bounceBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bounceBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bounceBarrier, 0, nullptr, 0, nullptr);

            vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resolveKernel->pipeline);
            vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    resolveKernel->pipelineLayout, 0, 1, &resolveKernel->descriptorSet, 0, nullptr);
            uint32_t nextBounce = static_cast<uint32_t>(e + 1);
            uint32_t resetQueue = (nextBounce + 1 < entries.size()) ? static_cast<uint32_t>((nextBounce + 1) % 2) : 0xFFFFFFFFu;
            struct { uint32_t resetQueue; uint32_t dgcMode; uint32_t bounceIndex; } rpcBounce{resetQueue, 3u, nextBounce};
            vkCmdPushConstants(frame.commandBuffer, resolveKernel->pipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpcBounce), &rpcBounce);
            vkCmdDispatch(frame.commandBuffer, 1, 1, 1);

            VkMemoryBarrier resolveBarrier{};
            resolveBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            resolveBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            resolveBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &resolveBarrier, 0, nullptr, 0, nullptr);
          }
        }
      }
    }
  } else {
    // Legacy fallback path: CPU indirect command dispatch loop
    VkBuffer vkIndirect = getVkBuffer(indirectBuffer);
    VulkanKernel *lastBound = secondKernel;
    for (size_t e = 0; e < entries.size(); ++e) {
      const auto &entry = entries[e];
      VulkanKernel *kToBind = entry.specializedKernel ? kernels[entry.specializedKernel] : secondKernel;
      if (kToBind && kToBind != lastBound) {
        vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          kToBind->pipeline);
        vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                kToBind->pipelineLayout, 0, 1,
                                &kToBind->descriptorSet, 0, nullptr);
        lastBound = kToBind;
      }
      if (!entry.pushConstants.empty()) {
        vkCmdPushConstants(frame.commandBuffer, lastBound->pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           entry.pushConstants.size(),
                           entry.pushConstants.data());
      }
      vkCmdDispatchIndirect(frame.commandBuffer, vkIndirect, entry.offset);

      // If ping-pong compaction between bounces, resolve next bounce's indirect command
      if (isPingPong && (e + 1 < entries.size()) && resolveKernel_handle) {
        auto *resolveKernel = kernels[resolveKernel_handle];
        if (resolveKernel) {
          VkMemoryBarrier bounceBarrier{};
          bounceBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
          bounceBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          bounceBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bounceBarrier, 0,
                               nullptr, 0, nullptr);

          vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resolveKernel->pipeline);
          vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  resolveKernel->pipelineLayout, 0, 1,
                                  &resolveKernel->descriptorSet, 0, nullptr);
          uint32_t resetQueue = (e + 2 < entries.size()) ? static_cast<uint32_t>(e % 2) : 0xFFFFFFFFu;
          struct { uint32_t resetQueue; uint32_t dgcMode; uint32_t bounceIndex; } rpcBounce{resetQueue, 0u, 0u};
          vkCmdPushConstants(frame.commandBuffer, resolveKernel->pipelineLayout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(rpcBounce), &rpcBounce);
          vkCmdDispatch(frame.commandBuffer, 1, 1, 1);

          VkMemoryBarrier resolveBarrier{};
          resolveBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
          resolveBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
          resolveBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
          vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                               0, 1, &resolveBarrier, 0, nullptr, 0, nullptr);

          vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            secondKernel->pipeline);
          vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  secondKernel->pipelineLayout, 0, 1,
                                  &secondKernel->descriptorSet, 0, nullptr);
          lastBound = secondKernel;
        }
      }
    }
  }

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkResult wlSubmitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  if (wlSubmitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatchWorkListSequence with result: " +
                             std::to_string(wlSubmitRes));
  }
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

  VkResult rtSubmitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  if (rtSubmitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatchRayTracingIndirect with result: " +
                             std::to_string(rtSubmitRes));
  }
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

void VulkanContext::destroyHeadlessSwapchain() {
  if (headlessSwapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(device, headlessSwapchain, nullptr);
    headlessSwapchain = VK_NULL_HANDLE;
    swapchainImages.clear();
  }
  if (headlessSurface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(instance, headlessSurface, nullptr);
    headlessSurface = VK_NULL_HANDLE;
  }
}

void VulkanContext::enableHeadlessSwapchain() {
  if (headlessSwapchain != VK_NULL_HANDLE) {
    return;
  }
  if (!headlessSurfaceSupported || !swapchainSupported) {
    return;
  }
  if (device == VK_NULL_HANDLE || instance == VK_NULL_HANDLE) {
    return;
  }

  auto pfnCreateHeadlessSurface =
      reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
          vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT"));
  if (!pfnCreateHeadlessSurface) {
    return;
  }

  if (headlessSurface == VK_NULL_HANDLE) {
    VkHeadlessSurfaceCreateInfoEXT surfaceInfo{
        VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT};
    if (pfnCreateHeadlessSurface(instance, &surfaceInfo, nullptr,
                                &headlessSurface) != VK_SUCCESS) {
      return;
    }
  }

  VkBool32 presentSupported = VK_FALSE;
  vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, computeQueueFamilyIndex,
                                       headlessSurface, &presentSupported);
  if (!presentSupported) {
    return;
  }

  VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sci.surface = headlessSurface;
  sci.minImageCount = 2;
  sci.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  sci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  sci.imageExtent = {640, 480};
  sci.imageArrayLayers = 1;
  sci.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  sci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sci.clipped = VK_TRUE;

  if (vkCreateSwapchainKHR(device, &sci, nullptr, &headlessSwapchain) !=
      VK_SUCCESS) {
    headlessSwapchain = VK_NULL_HANDLE;
    return;
  }

  uint32_t imgCount = 0;
  vkGetSwapchainImagesKHR(device, headlessSwapchain, &imgCount, nullptr);
  swapchainImages.resize(imgCount);
  vkGetSwapchainImagesKHR(device, headlessSwapchain, &imgCount,
                          swapchainImages.data());
}

void VulkanContext::presentFrame() {
  if (headlessSwapchain == VK_NULL_HANDLE) {
    enableHeadlessSwapchain();
  }
  if (headlessSwapchain == VK_NULL_HANDLE || computeQueue == VK_NULL_HANDLE) {
    return;
  }

  uint32_t imageIndex = 0;
  VkResult res = vkAcquireNextImageKHR(device, headlessSwapchain, UINT64_MAX,
                                       VK_NULL_HANDLE, VK_NULL_HANDLE,
                                       &imageIndex);
  if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
    return;
  }

  VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  pi.swapchainCount = 1;
  pi.pSwapchains = &headlessSwapchain;
  pi.pImageIndices = &imageIndex;
  vkQueuePresentKHR(computeQueue, &pi);
}

VkPipeline VulkanContext::getVkPipeline(ComputeKernel kernel) const {
  auto it = kernels.find(kernel);
  if (it != kernels.end() && it->second) {
    return it->second->pipeline;
  }
  return VK_NULL_HANDLE;
}

VkPipelineLayout VulkanContext::getVkPipelineLayout(ComputeKernel kernel) const {
  auto it = kernels.find(kernel);
  if (it != kernels.end() && it->second) {
    return it->second->pipelineLayout;
  }
  return VK_NULL_HANDLE;
}

ComputeBuffer VulkanContext::createPreprocessBuffer(size_t size) {
  auto *vulkanBuffer = new VulkanBuffer();

  VkBufferUsageFlags2CreateInfoKHR usage2Info{
      VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR};
  usage2Info.usage = VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT |
                     VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR |
                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.pNext = &usage2Info;
  bufferInfo.size = size;
  bufferInfo.usage = 0;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &vulkanBuffer->buffer) != VK_SUCCESS) {
    delete vulkanBuffer;
    throw std::runtime_error("failed to create preprocess buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, vulkanBuffer->buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VkMemoryAllocateFlagsInfo flagsInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  allocInfo.pNext = &flagsInfo;

  if (vkAllocateMemory(device, &allocInfo, nullptr, &vulkanBuffer->memory) != VK_SUCCESS) {
    delete vulkanBuffer;
    throw std::runtime_error("failed to allocate preprocess buffer memory!");
  }

  vkBindBufferMemory(device, vulkanBuffer->buffer, vulkanBuffer->memory, 0);

  VkBufferDeviceAddressInfo bdaInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bdaInfo.buffer = vulkanBuffer->buffer;
  if (vkGetBufferDeviceAddressKHR_ptr) {
    vulkanBuffer->address = vkGetBufferDeviceAddressKHR_ptr(device, &bdaInfo);
  } else {
    vulkanBuffer->address = vkGetBufferDeviceAddress(device, &bdaInfo);
  }

  buffers[vulkanBuffer] = vulkanBuffer;
  return vulkanBuffer;
}

VkIndirectCommandsLayoutEXT VulkanContext::createIndirectCommandsLayout(
    const VkIndirectCommandsLayoutCreateInfoEXT &createInfo) {
  if (!vkCreateIndirectCommandsLayoutEXT_ptr) {
    throw std::runtime_error("vkCreateIndirectCommandsLayoutEXT is not available");
  }
  VkIndirectCommandsLayoutEXT layout = VK_NULL_HANDLE;
  VkResult res = vkCreateIndirectCommandsLayoutEXT_ptr(device, &createInfo, nullptr, &layout);
  if (res != VK_SUCCESS) {
    throw std::runtime_error("vkCreateIndirectCommandsLayoutEXT failed with result: " +
                             std::to_string(res));
  }
  return layout;
}

void VulkanContext::destroyIndirectCommandsLayout(VkIndirectCommandsLayoutEXT layout) {
  if (layout != VK_NULL_HANDLE && vkDestroyIndirectCommandsLayoutEXT_ptr) {
    vkDestroyIndirectCommandsLayoutEXT_ptr(device, layout, nullptr);
  }
}

VkIndirectExecutionSetEXT VulkanContext::createIndirectExecutionSet(
    const VkIndirectExecutionSetCreateInfoEXT &createInfo) {
  if (!vkCreateIndirectExecutionSetEXT_ptr) {
    throw std::runtime_error("vkCreateIndirectExecutionSetEXT is not available");
  }
  VkIndirectExecutionSetEXT set = VK_NULL_HANDLE;
  VkResult res = vkCreateIndirectExecutionSetEXT_ptr(device, &createInfo, nullptr, &set);
  if (res != VK_SUCCESS) {
    throw std::runtime_error("vkCreateIndirectExecutionSetEXT failed with result: " +
                             std::to_string(res));
  }
  return set;
}

void VulkanContext::updateIndirectExecutionSetPipeline(VkIndirectExecutionSetEXT set,
                                                       uint32_t index,
                                                       ComputeKernel kernel) {
  if (!vkUpdateIndirectExecutionSetPipelineEXT_ptr || set == VK_NULL_HANDLE) {
    return;
  }
  VkWriteIndirectExecutionSetPipelineEXT writeInfo{
      VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT};
  writeInfo.index = index;
  writeInfo.pipeline = getVkPipeline(kernel);
  vkUpdateIndirectExecutionSetPipelineEXT_ptr(device, set, 1, &writeInfo);
}

void VulkanContext::destroyIndirectExecutionSet(VkIndirectExecutionSetEXT set) {
  if (set != VK_NULL_HANDLE && vkDestroyIndirectExecutionSetEXT_ptr) {
    vkDestroyIndirectExecutionSetEXT_ptr(device, set, nullptr);
  }
}

VkDeviceSize VulkanContext::getGeneratedCommandsMemoryRequirements(
    VkIndirectCommandsLayoutEXT layout,
    VkIndirectExecutionSetEXT execSet,
    uint32_t maxSequenceCount,
    ComputeKernel fallbackKernel) {
  if (!vkGetGeneratedCommandsMemoryRequirementsEXT_ptr || layout == VK_NULL_HANDLE) {
    return 0;
  }
  VkGeneratedCommandsPipelineInfoEXT pipelineInfo{
      VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT};
  VkGeneratedCommandsMemoryRequirementsInfoEXT memReqInfo{
      VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT};
  if (execSet == VK_NULL_HANDLE && fallbackKernel != nullptr) {
    pipelineInfo.pipeline = getVkPipeline(fallbackKernel);
    memReqInfo.pNext = &pipelineInfo;
  }
  memReqInfo.indirectExecutionSet = execSet;
  memReqInfo.indirectCommandsLayout = layout;
  memReqInfo.maxSequenceCount = maxSequenceCount;

  VkMemoryRequirements2 memReqs2{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
  vkGetGeneratedCommandsMemoryRequirementsEXT_ptr(device, &memReqInfo, &memReqs2);
  return std::max<VkDeviceSize>(memReqs2.memoryRequirements.size, 256);
}

void VulkanContext::dispatchDGCWorkListSequence(
    ComputeKernel resetKernel_handle,
    ComputeKernel classifyKernel_handle, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    ComputeKernel resolveKernel_handle,
    ComputeKernel secondKernel_handle,
    const DGCExecutionInfo &dgcInfo,
    const void *resolvePc, size_t resolvePcSize) {
  if (!isDGCSupported() || !vkCmdExecuteGeneratedCommandsEXT_ptr) {
    throw std::runtime_error("VK_EXT_device_generated_commands not supported on this device");
  }

  auto *classifyKernel = kernels[classifyKernel_handle];
  auto *secondKernel = kernels[secondKernel_handle];
  if (!classifyKernel || !secondKernel) return;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (frame.inUse) {
    VkResult waitResult = vkWaitForFences(device, 1, &frame.fence, VK_TRUE, 3000000000ULL);
    if (waitResult == VK_TIMEOUT) {
      throw std::runtime_error("GPU dispatch timed out in dispatchDGCWorkListSequence");
    } else if (waitResult != VK_SUCCESS) {
      throw std::runtime_error("vkWaitForFences failed: " + std::to_string(waitResult));
    }
    vkResetFences(device, 1, &frame.fence);
    frame.inUse = false;
  }

  vkResetCommandBuffer(frame.commandBuffer, 0);
  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

  // 1. Reset
  if (resetKernel_handle) {
    auto *resetKernel = kernels[resetKernel_handle];
    if (resetKernel) {
      vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resetKernel->pipeline);
      vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              resetKernel->pipelineLayout, 0, 1, &resetKernel->descriptorSet, 0, nullptr);
      vkCmdDispatch(frame.commandBuffer, 1, 1, 1);

      VkMemoryBarrier resetBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      resetBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      resetBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &resetBarrier, 0, nullptr, 0, nullptr);
    }
  }

  // 2. Classify
  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, classifyKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          classifyKernel->pipelineLayout, 0, 1, &classifyKernel->descriptorSet, 0, nullptr);
  if (!classifyKernel->pushConstantData.empty()) {
    vkCmdPushConstants(frame.commandBuffer, classifyKernel->pipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       classifyKernel->pushConstantData.size(),
                       classifyKernel->pushConstantData.data());
  }
  vkCmdDispatch(frame.commandBuffer, grid_x, grid_y, grid_z);

  // 3. Barrier: Classify -> Resolve
  VkMemoryBarrier classifyBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  classifyBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  classifyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &classifyBarrier, 0, nullptr, 0, nullptr);

  // 4. Resolve: GPU outputs DGC sequence items and dynamic sequence count
  if (resolveKernel_handle) {
    auto *resolveKernel = kernels[resolveKernel_handle];
    if (resolveKernel) {
      vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resolveKernel->pipeline);
      vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              resolveKernel->pipelineLayout, 0, 1, &resolveKernel->descriptorSet, 0, nullptr);
      if (resolvePc && resolvePcSize > 0) {
        vkCmdPushConstants(frame.commandBuffer, resolveKernel->pipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, resolvePcSize, resolvePc);
      }
      vkCmdDispatch(frame.commandBuffer, 1, 1, 1);
    }
  }

  // 5. Barrier: Resolve -> DGC Execution
  VkMemoryBarrier dgcBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  dgcBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  dgcBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(frame.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 1, &dgcBarrier, 0, nullptr, 0, nullptr);

  // 6. Bind base pipeline and descriptor set
  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, secondKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          secondKernel->pipelineLayout, 0, 1, &secondKernel->descriptorSet, 0, nullptr);

  // 7. Execute Generated Commands (DGC)
  VkGeneratedCommandsPipelineInfoEXT pipeInfo{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT};
  pipeInfo.pipeline = secondKernel->pipeline;

  VkGeneratedCommandsInfoEXT genCmds{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT};
  if (dgcInfo.executionSet == VK_NULL_HANDLE) {
    genCmds.pNext = &pipeInfo;
  }
  genCmds.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
  genCmds.indirectExecutionSet = dgcInfo.executionSet;
  genCmds.indirectCommandsLayout = dgcInfo.layout;
  genCmds.indirectAddress = getBufferDeviceAddress(dgcInfo.sequenceBuffer) + dgcInfo.sequenceBufferOffset;
  genCmds.indirectAddressSize = dgcInfo.sequenceBufferSize;
  genCmds.preprocessAddress = getBufferDeviceAddress(dgcInfo.preprocessBuffer);
  genCmds.preprocessSize = dgcInfo.preprocessBufferSize;
  genCmds.maxSequenceCount = dgcInfo.maxSequenceCount;
  genCmds.sequenceCountAddress = dgcInfo.sequenceCountBuffer ?
      (getBufferDeviceAddress(dgcInfo.sequenceCountBuffer) + dgcInfo.sequenceCountBufferOffset) : 0;
  genCmds.maxDrawCount = 0;

  vkCmdExecuteGeneratedCommandsEXT_ptr(frame.commandBuffer, VK_FALSE, &genCmds);

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkResult submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  if (submitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatchDGCWorkListSequence: " +
                             std::to_string(submitRes));
  }
  frame.inUse = true;
  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

void VulkanContext::dispatchDGCSequence(ComputeKernel kernel_handle,
                                       const DGCExecutionInfo &dgcInfo) {
  if (!isDGCSupported() || !vkCmdExecuteGeneratedCommandsEXT_ptr) {
    throw std::runtime_error("VK_EXT_device_generated_commands not supported on this device");
  }

  auto *kernel = kernels[kernel_handle];
  if (!kernel) return;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (frame.inUse) {
    VkResult waitResult = vkWaitForFences(device, 1, &frame.fence, VK_TRUE, 3000000000ULL);
    if (waitResult == VK_TIMEOUT) {
      throw std::runtime_error("GPU dispatch timed out in dispatchDGCSequence");
    } else if (waitResult != VK_SUCCESS) {
      throw std::runtime_error("vkWaitForFences failed: " + std::to_string(waitResult));
    }
    vkResetFences(device, 1, &frame.fence);
    frame.inUse = false;
  }

  vkResetCommandBuffer(frame.commandBuffer, 0);
  VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          kernel->pipelineLayout, 0, 1, &kernel->descriptorSet, 0, nullptr);

  VkGeneratedCommandsPipelineInfoEXT pipeInfo{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT};
  pipeInfo.pipeline = kernel->pipeline;

  VkGeneratedCommandsInfoEXT genCmds{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT};
  if (dgcInfo.executionSet == VK_NULL_HANDLE) {
    genCmds.pNext = &pipeInfo;
  }
  genCmds.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
  genCmds.indirectExecutionSet = dgcInfo.executionSet;
  genCmds.indirectCommandsLayout = dgcInfo.layout;
  genCmds.indirectAddress = getBufferDeviceAddress(dgcInfo.sequenceBuffer) + dgcInfo.sequenceBufferOffset;
  genCmds.indirectAddressSize = dgcInfo.sequenceBufferSize;
  genCmds.preprocessAddress = getBufferDeviceAddress(dgcInfo.preprocessBuffer);
  genCmds.preprocessSize = dgcInfo.preprocessBufferSize;
  genCmds.maxSequenceCount = dgcInfo.maxSequenceCount;
  genCmds.sequenceCountAddress = dgcInfo.sequenceCountBuffer ?
      (getBufferDeviceAddress(dgcInfo.sequenceCountBuffer) + dgcInfo.sequenceCountBufferOffset) : 0;
  genCmds.maxDrawCount = 0;

  vkCmdExecuteGeneratedCommandsEXT_ptr(frame.commandBuffer, VK_FALSE, &genCmds);

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkResult submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  if (submitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatchDGCSequence: " +
                             std::to_string(submitRes));
  }
  frame.inUse = true;
  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

