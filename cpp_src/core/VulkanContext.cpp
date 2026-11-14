#include "VulkanContext.h"
#include "DeviceDatabase.h"
#include "utils/ShaderCache.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef VK_EXT_shader_ocp_microscaling_types
#define VK_EXT_shader_ocp_microscaling_types 1
#define VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME "VK_EXT_shader_ocp_microscaling_types"
static constexpr VkStructureType
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT =
        static_cast<VkStructureType>(1000672000);
typedef struct VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT {
  VkStructureType sType;
  void *pNext;
  VkBool32 shaderFloat4;
  VkBool32 shaderFloat6;
  VkBool32 shaderFloat8UnsignedE8M0;
  VkBool32 shaderMXInt8;
} VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT;
#endif

#ifndef VK_NV_cooperative_vector
#define VK_NV_cooperative_vector 1
#define VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME "VK_NV_cooperative_vector"
static constexpr VkStructureType
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV =
        static_cast<VkStructureType>(1000491000);
typedef struct VkPhysicalDeviceCooperativeVectorFeaturesNV {
  VkStructureType sType;
  void *pNext;
  VkBool32 cooperativeVector;
  VkBool32 cooperativeVectorTraining;
} VkPhysicalDeviceCooperativeVectorFeaturesNV;
#endif

#ifndef VK_NV_cooperative_matrix2
#define VK_NV_cooperative_matrix2 1
#define VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME "VK_NV_cooperative_matrix2"
static constexpr VkStructureType
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV =
        static_cast<VkStructureType>(1000593000);
typedef struct VkPhysicalDeviceCooperativeMatrix2FeaturesNV {
  VkStructureType sType;
  void *pNext;
  VkBool32 cooperativeMatrixWorkgroupScope;
  VkBool32 cooperativeMatrixFlexibleDimensions;
  VkBool32 cooperativeMatrixReductions;
  VkBool32 cooperativeMatrixConversions;
  VkBool32 cooperativeMatrixPerElementOperations;
  VkBool32 cooperativeMatrixTensorAddressing;
  VkBool32 cooperativeMatrixBlockLoads;
} VkPhysicalDeviceCooperativeMatrix2FeaturesNV;
#endif

namespace {
VulkanContext::CoopMatrixTile selectBestSubgroupCoopMatrixTile(
    VkInstance inst, VkPhysicalDevice physDev, VkComponentTypeKHR abType,
    VkComponentTypeKHR cType, uint32_t defaultSubgroupSize, bool wave32Ok) {
  VulkanContext::CoopMatrixTile result{};
  result.subgroupSize = wave32Ok ? 32u : (defaultSubgroupSize > 0 ? defaultSubgroupSize : 32u);
  auto pfnGetCoopProps =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (!pfnGetCoopProps) {
    return result;
  }
  uint32_t propCount = 0;
  if (pfnGetCoopProps(physDev, &propCount, nullptr) != VK_SUCCESS ||
      propCount == 0) {
    return result;
  }
  std::vector<VkCooperativeMatrixPropertiesKHR> props(propCount);
  for (auto &p : props) {
    p = {};
    p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  }
  if (pfnGetCoopProps(physDev, &propCount, props.data()) != VK_SUCCESS) {
    return result;
  }

  uint64_t bestVol = 0;
  for (const auto &p : props) {
    if (p.AType == abType && p.BType == abType && p.CType == cType &&
        p.ResultType == cType && p.scope == VK_SCOPE_SUBGROUP_KHR &&
        p.MSize > 0 && p.NSize > 0 && p.KSize > 0) {
      if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
        result.supported = true;
        result.M = 16;
        result.N = 16;
        result.K = 16;
        return result;
      }
      uint64_t vol = static_cast<uint64_t>(p.MSize) * p.NSize * p.KSize;
      if (!result.supported || vol > bestVol) {
        result.supported = true;
        result.M = p.MSize;
        result.N = p.NSize;
        result.K = p.KSize;
        bestVol = vol;
      }
    }
  }
  return result;
}

bool hasSubgroupCoopMatrix16x16x16(VkInstance inst, VkPhysicalDevice physDev,
                                   VkComponentTypeKHR abType,
                                   VkComponentTypeKHR cType) {
  auto pfnGetCoopProps =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  if (!pfnGetCoopProps) {
    return false;
  }
  uint32_t propCount = 0;
  if (pfnGetCoopProps(physDev, &propCount, nullptr) != VK_SUCCESS ||
      propCount == 0) {
    return false;
  }
  std::vector<VkCooperativeMatrixPropertiesKHR> props(propCount);
  for (auto &p : props) {
    p = {};
    p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
  }
  if (pfnGetCoopProps(physDev, &propCount, props.data()) != VK_SUCCESS) {
    return false;
  }
  for (const auto &p : props) {
    if (p.AType == abType && p.BType == abType && p.CType == cType &&
        p.ResultType == cType && p.scope == VK_SCOPE_SUBGROUP_KHR &&
        p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
      return true;
    }
  }
  return false;
}

bool supportsWave32Compute(VkPhysicalDevice physDev,
                           uint32_t defaultSubgroupSize,
                           bool hasSubgroupSizeCtrlExt) {
  if (defaultSubgroupSize == 32) {
    return true;
  }
  if (!hasSubgroupSizeCtrlExt) {
    return false;
  }
  VkPhysicalDeviceSubgroupSizeControlProperties sgCtrlProps{};
  sgCtrlProps.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
  VkPhysicalDeviceProperties2 p2{};
  p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  p2.pNext = &sgCtrlProps;
  vkGetPhysicalDeviceProperties2(physDev, &p2);
  return (sgCtrlProps.minSubgroupSize <= 32 &&
          sgCtrlProps.maxSubgroupSize >= 32 &&
          (sgCtrlProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0);
}

int scoreDeviceTypePriority(VkPhysicalDeviceType devType) {
  switch (devType) {
  case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
    return 3;
  case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
    return 2;
  case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
    return 1;
  default:
    return 0;
  }
}
} // namespace

#ifdef HAVE_SHADERC
#include <shaderc/shaderc.hpp>
#endif

void VulkanContext::waitIdle() {
  if (device == VK_NULL_HANDLE) return;
  if (m_inBatch) {
    endBatch();
  }
  for (size_t i = 0; i < kMaxInFlight; ++i) {
    if (inFlightFrames[i].inUse && inFlightFrames[i].fence != VK_NULL_HANDLE) {
      constexpr uint64_t kTimeoutNs = 30'000'000'000ULL;
      VkResult waitResult =
          vkWaitForFences(device, 1, &inFlightFrames[i].fence, VK_TRUE, kTimeoutNs);
      if (waitResult == VK_TIMEOUT) {
        throw std::runtime_error("GPU dispatch timed out (>30 s) in waitIdle()");
      } else if (waitResult != VK_SUCCESS) {
        throw std::runtime_error("vkWaitForFences failed in waitIdle with result: " +
                                 std::to_string(waitResult));
      }
      vkResetFences(device, 1, &inFlightFrames[i].fence);
      inFlightFrames[i].inUse = false;
    }
  }
  std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
  if (computeQueue != VK_NULL_HANDLE) {
    VkResult queueResult = vkQueueWaitIdle(computeQueue);
    if (queueResult != VK_SUCCESS) {
      throw std::runtime_error("vkQueueWaitIdle failed in waitIdle with result: " +
                               std::to_string(queueResult));
    }
  }
  if (graphicsQueue != VK_NULL_HANDLE && graphicsQueue != computeQueue) {
    VkResult queueResult = vkQueueWaitIdle(graphicsQueue);
    if (queueResult != VK_SUCCESS) {
      throw std::runtime_error("vkQueueWaitIdle(graphicsQueue) failed in waitIdle with result: " +
                               std::to_string(queueResult));
    }
  }
  if (asyncComputeQueue != VK_NULL_HANDLE &&
      asyncComputeQueue != computeQueue &&
      asyncComputeQueue != graphicsQueue) {
    vkQueueWaitIdle(asyncComputeQueue);
  }
  if (transferQueue != VK_NULL_HANDLE &&
      transferQueue != computeQueue &&
      transferQueue != graphicsQueue &&
      transferQueue != asyncComputeQueue) {
    vkQueueWaitIdle(transferQueue);
  }
  if (secondaryTransferQueue != VK_NULL_HANDLE &&
      secondaryTransferQueue != transferQueue &&
      secondaryTransferQueue != computeQueue &&
      secondaryTransferQueue != graphicsQueue &&
      secondaryTransferQueue != asyncComputeQueue) {
    vkQueueWaitIdle(secondaryTransferQueue);
  }
}

VulkanContext::VulkanContext(
    bool verbose, bool debug,
    const std::vector<const char *> &extraInstanceExtensions)
    : verbose(verbose), debug(debug) {
  for (const char *ext : extraInstanceExtensions) {
    if (ext && ext[0] != '\0') {
      m_extraInstanceExtensions.emplace_back(ext);
    }
  }
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
  clearSharedPreviewImages();
  while (!kernels.empty()) {
    try {
      releaseKernel(kernels.begin()->first);
    } catch (...) {
      kernels.erase(kernels.begin());
    }
  }
  while (!buffers.empty()) {
    try {
      releaseBuffer(buffers.begin()->first);
    } catch (...) {
      buffers.erase(buffers.begin());
    }
  }
  cleanupPipelineCache();
  cleanupStagingBuffer();
  cleanupMemoryBlocks();
  cleanupVmaAllocator();
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
    commandPool = VK_NULL_HANDLE;
  }
  try {
    destroyHeadlessSwapchain();
  } catch (...) {
  }
  if (perfQueryPool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyQueryPool(device, perfQueryPool, nullptr);
    perfQueryPool = VK_NULL_HANDLE;
  }
  if (timestampQueryPool != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    vkDestroyQueryPool(device, timestampQueryPool, nullptr);
    timestampQueryPool = VK_NULL_HANDLE;
  }
  if (device != VK_NULL_HANDLE) {
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
  }
  if (debugMessenger != VK_NULL_HANDLE && instance != VK_NULL_HANDLE && vkDestroyDebugUtilsMessengerEXT_ptr) {
    vkDestroyDebugUtilsMessengerEXT_ptr(instance, debugMessenger, nullptr);
    debugMessenger = VK_NULL_HANDLE;
  }
  if (instance != VK_NULL_HANDLE) {
    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
  }
}

void VulkanContext::createInstance() {
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "GPUBench";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);

  uint32_t supportedVersion = VK_API_VERSION_1_0;
  auto pfnEnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
  if (pfnEnumerateInstanceVersion) {
    pfnEnumerateInstanceVersion(&supportedVersion);
  }
  if (supportedVersion < VK_API_VERSION_1_4) {
    throw std::runtime_error(
        "Vulkan 1.4 loader/runtime is required, but detected instance API " +
        std::to_string(VK_VERSION_MAJOR(supportedVersion)) + "." +
        std::to_string(VK_VERSION_MINOR(supportedVersion)));
  }
  appInfo.apiVersion = VK_API_VERSION_1_4;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

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

  std::vector<const char*> extensions;
  auto pushUniqueExt = [&](const char *extName) {
    for (const char *existing : extensions) {
      if (strcmp(existing, extName) == 0) return;
    }
    extensions.push_back(extName);
  };

  for (const auto &extra : m_extraInstanceExtensions) {
    if (hasInstExt(extra.c_str())) {
      pushUniqueExt(extra.c_str());
    }
  }

#if defined(__APPLE__) || defined(VK_KHR_portability_enumeration)
  if (hasInstExt(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
    pushUniqueExt(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
#endif

  headlessSurfaceSupported = false;
  if (hasInstExt(VK_KHR_SURFACE_EXTENSION_NAME)) {
    pushUniqueExt(VK_KHR_SURFACE_EXTENSION_NAME);
    if (hasInstExt(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME)) {
      pushUniqueExt(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
      headlessSurfaceSupported = true;
    }
  }

  if (debug && hasInstExt(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
    pushUniqueExt(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames =
      extensions.empty() ? nullptr : extensions.data();

  std::vector<const char*> layers;
  if (debug) {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availLayers(layerCount);
    if (layerCount > 0) {
      vkEnumerateInstanceLayerProperties(&layerCount, availLayers.data());
    }
    for (const auto &lp : availLayers) {
      if (strcmp(lp.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        break;
      }
    }
  }
  createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
  createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();

  if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
    throw std::runtime_error("failed to create Vulkan 1.4 instance!");
  }

  if (debug && hasInstExt(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
    vkCreateDebugUtilsMessengerEXT_ptr =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");
    vkDestroyDebugUtilsMessengerEXT_ptr =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");
    if (vkCreateDebugUtilsMessengerEXT_ptr) {
      VkDebugUtilsMessengerCreateInfoEXT dbgCreateInfo{};
      dbgCreateInfo.sType =
          VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      dbgCreateInfo.messageSeverity =
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      dbgCreateInfo.messageType =
          VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      dbgCreateInfo.pfnUserCallback =
          [](VkDebugUtilsMessageSeverityFlagBitsEXT severity,
             VkDebugUtilsMessageTypeFlagsEXT type,
             const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
             void *pUserData) -> VkBool32 {
        (void)severity;
        (void)type;
        (void)pUserData;
        if (pCallbackData && pCallbackData->pMessage) {
          std::cerr << "[Vulkan Validation] " << pCallbackData->pMessage << std::endl;
        }
        return VK_FALSE;
      };
      vkCreateDebugUtilsMessengerEXT_ptr(instance, &dbgCreateInfo, nullptr,
                                         &debugMessenger);
    }
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

  for (const auto &dev : allDevices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev, &props);

    std::string name = props.deviceName;
    if (name.find("llvmpipe") != std::string::npos)
      continue;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
      continue;
    if (props.apiVersion < VK_API_VERSION_1_4) {
      if (verbose) {
        std::cerr << "Skipping pre-Vulkan 1.4 device: " << name << " (API "
                  << VK_VERSION_MAJOR(props.apiVersion) << "."
                  << VK_VERSION_MINOR(props.apiVersion) << ")" << std::endl;
      }
      continue;
    }

    physicalDevices.push_back(dev);
  }

  // Prefer discrete GPUs over integrated/virtual GPUs while preserving deterministic order
  std::stable_sort(physicalDevices.begin(), physicalDevices.end(),
                   [](VkPhysicalDevice a, VkPhysicalDevice b) {
                     VkPhysicalDeviceProperties pa{}, pb{};
                     vkGetPhysicalDeviceProperties(a, &pa);
                     vkGetPhysicalDeviceProperties(b, &pb);
                     return scoreDeviceTypePriority(pa.deviceType) >
                            scoreDeviceTypePriority(pb.deviceType);
                   });

  if (physicalDevices.empty()) {
    throw std::runtime_error("No Vulkan 1.4-capable hardware GPU found!");
  }
}

namespace {
bool queryHardwareSERSupport(VkPhysicalDevice physDev, uint32_t vendorID,
                             bool hasExtEXT, bool hasExtNV) {
  if (!hasExtEXT && !hasExtNV) {
    return false;
  }
  // AMD RDNA architectures (RDNA 2, RDNA 3, RDNA 3.5, RDNA 4) do not have
  // dedicated hardware Shader Execution Reordering (SER). Drivers that expose
  // VK_EXT_ray_tracing_invocation_reorder only accept the SPIR-V capability
  // with VK_RAY_TRACING_INVOCATION_REORDER_MODE_NONE_EXT or software emulation.
  if (vendorID == 0x1002) {
    return false;
  }
#if defined(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
  if (hasExtEXT) {
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT serFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &serFeatures;
    vkGetPhysicalDeviceFeatures2(physDev, &f2);

    VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT serProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &serProps;
    vkGetPhysicalDeviceProperties2(physDev, &p2);

    if (serFeatures.rayTracingInvocationReorder == VK_TRUE &&
        serProps.rayTracingInvocationReorderReorderingHint ==
            VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT) {
      return true;
    }
  }
#endif
#if defined(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME)
  if (hasExtNV) {
    VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV serFeaturesNV{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &serFeaturesNV;
    vkGetPhysicalDeviceFeatures2(physDev, &f2);

    VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV serPropsNV{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV};
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &serPropsNV;
    vkGetPhysicalDeviceProperties2(physDev, &p2);

    if (serFeaturesNV.rayTracingInvocationReorder == VK_TRUE &&
        serPropsNV.rayTracingInvocationReorderReorderingHint ==
            VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV) {
      return true;
    }
  }
#endif
  return false;
}
} // namespace

namespace {
void populateExtendedCapabilities(VkInstance inst, VkPhysicalDevice physDev,
                                  const VkPhysicalDeviceProperties &props,
                                  const VkPhysicalDeviceMemoryProperties &memProps,
                                  const std::vector<VkExtensionProperties> &availableExts,
                                  uint32_t subgroupSize,
                                  DeviceInfo &info) {
  auto hasExt = [&](const char *name) {
    for (const auto &ext : availableExts) {
      if (strcmp(ext.extensionName, name) == 0)
        return true;
    }
    return false;
  };

  // Compute units / SMs discovery
  info.computeUnits = 0;
#ifdef VK_AMD_shader_core_properties2
  if (hasExt(VK_AMD_SHADER_CORE_PROPERTIES_2_EXTENSION_NAME)) {
    VkPhysicalDeviceShaderCoreProperties2AMD amdCore2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD};
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &amdCore2};
    vkGetPhysicalDeviceProperties2(physDev, &p2);
    if (amdCore2.activeComputeUnitCount > 0) {
      info.computeUnits = amdCore2.activeComputeUnitCount;
    }
  }
#endif
#ifdef VK_AMD_shader_core_properties
  if (info.computeUnits == 0 && hasExt(VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME)) {
    VkPhysicalDeviceShaderCorePropertiesAMD amdCore{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD};
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &amdCore};
    vkGetPhysicalDeviceProperties2(physDev, &p2);
    uint32_t totalCu = amdCore.shaderEngineCount *
                       amdCore.shaderArraysPerEngineCount *
                       amdCore.computeUnitsPerShaderArray;
    if (totalCu > 0) {
      info.computeUnits = totalCu;
    }
  }
#endif
#ifdef VK_NV_shader_sm_builtins
  if (info.computeUnits == 0 && hasExt(VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME)) {
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV nvSm{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV};
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &nvSm};
    vkGetPhysicalDeviceProperties2(physDev, &p2);
    if (nvSm.shaderSMCount > 0) {
      info.computeUnits = nvSm.shaderSMCount;
    }
  }
#endif

  // ReBAR (HOST_VISIBLE | DEVICE_LOCAL heap) discovery
  uint64_t maxRebarHeap = 0;
  for (uint32_t t = 0; t < memProps.memoryTypeCount; ++t) {
    constexpr VkMemoryPropertyFlags kRebarFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    if ((memProps.memoryTypes[t].propertyFlags & kRebarFlags) == kRebarFlags) {
      uint32_t hIdx = memProps.memoryTypes[t].heapIndex;
      if (hIdx < memProps.memoryHeapCount) {
        maxRebarHeap = std::max<uint64_t>(maxRebarHeap, memProps.memoryHeaps[hIdx].size);
      }
    }
  }
  info.rebarSizeBytes = maxRebarHeap;
  if (info.isApu) {
    info.rebarSupported = (maxRebarHeap > 0);
  } else {
    // Legacy PCIe BAR1 aperture is <= 256 MB; ReBAR exposes >= 512 MB (typically full VRAM)
    info.rebarSupported = (maxRebarHeap >= 512ULL * 1024ULL * 1024ULL);
  }

  // Dedicated Async Compute (ACE) and DMA Transfer (SDMA) queue families
  uint32_t qCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qCount, nullptr);
  std::vector<VkQueueFamilyProperties> qProps(qCount);
  if (qCount > 0) {
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qCount, qProps.data());
  }
  info.dedicatedAsyncComputeQueue = false;
  info.dedicatedTransferQueue = false;
  for (const auto &qf : qProps) {
    if (qf.queueCount == 0) continue;
    if ((qf.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(qf.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      info.dedicatedAsyncComputeQueue = true;
    }
    if ((qf.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
        !(qf.queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT))) {
      info.dedicatedTransferQueue = true;
    }
  }

  // VK_NV_cooperative_vector
  info.cooperativeVectorSupport = false;
  if (hasExt(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME)) {
    VkPhysicalDeviceCooperativeVectorFeaturesNV coopVecFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
        nullptr, VK_FALSE, VK_FALSE};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                 &coopVecFeat};
    vkGetPhysicalDeviceFeatures2(physDev, &f2);
    info.cooperativeVectorSupport = (coopVecFeat.cooperativeVector == VK_TRUE);
  }

  // VK_EXT_opacity_micromap
  info.opacityMicromapSupport = false;
#ifdef VK_EXT_opacity_micromap
  if (hasExt(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME)) {
    VkPhysicalDeviceOpacityMicromapFeaturesEXT ommFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                 &ommFeat};
    vkGetPhysicalDeviceFeatures2(physDev, &f2);
    info.opacityMicromapSupport = (ommFeat.micromap == VK_TRUE);
  }
#endif

  (void)inst;
  (void)props;
  (void)subgroupSize;
}

std::filesystem::path getPipelineCacheDirectory() {
  std::filesystem::path baseDir;
#ifdef _WIN32
  if (const char *localAppData = std::getenv("LOCALAPPDATA")) {
    baseDir = std::filesystem::path(localAppData) / "gpubench" / "pipeline_cache";
  }
#else
  if (const char *xdgCache = std::getenv("XDG_CACHE_HOME")) {
    baseDir = std::filesystem::path(xdgCache) / "gpubench" / "pipeline_cache";
  } else if (const char *home = std::getenv("HOME")) {
    baseDir = std::filesystem::path(home) / ".cache" / "gpubench" / "pipeline_cache";
  }
#endif
  if (baseDir.empty()) {
    std::error_code ec;
    baseDir = std::filesystem::temp_directory_path(ec) / "gpubench" / "pipeline_cache";
  }
  return baseDir;
}
} // namespace

const std::vector<DeviceInfo> &VulkanContext::getDevices() const {
  if (deviceInfos.empty()) {
    for (const auto &dev : physicalDevices) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(dev, &props);

      VkPhysicalDeviceMemoryProperties memProps;
      vkGetPhysicalDeviceMemoryProperties(dev, &memProps);

      VkPhysicalDeviceDriverProperties driverProps{};
      driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;

      VkPhysicalDeviceSubgroupProperties subgroupProps{};
      subgroupProps.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
      subgroupProps.pNext = &driverProps;

      VkPhysicalDeviceProperties2 props2{};
      props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      props2.pNext = &subgroupProps;
      vkGetPhysicalDeviceProperties2(dev, &props2);

      uint64_t vramSize = 0;
      for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
          vramSize += memProps.memoryHeaps[i].size;
        }
      }

      VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocpFeatures{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT,
          nullptr, VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE};

      VkPhysicalDeviceShaderFloat8FeaturesEXT float8Features{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, &ocpFeatures};

      VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16Features{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR, &float8Features};

      VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatrixFeatures{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &bfloat16Features};

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
      vkGetPhysicalDeviceFeatures2(dev, &features2);

      uint32_t extCount = 0;
      vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
      std::vector<VkExtensionProperties> availableExts(extCount);
      vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount,
                                           availableExts.data());

      auto hasExt = [&](const char *name) {
        for (const auto &ext : availableExts) {
          if (strcmp(ext.extensionName, name) == 0)
            return true;
        }
        return false;
      };

      bool pciExtSupported = hasExt(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME);
      VkPhysicalDevicePCIBusInfoPropertiesEXT pciBusProps{};
      pciBusProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
      if (pciExtSupported) {
        pciBusProps.pNext = (void *)props2.pNext;
        props2.pNext = &pciBusProps;
        vkGetPhysicalDeviceProperties2(dev, &props2);
      }

      DeviceInfo info;
      info.name = props.deviceName;
      info.vendorID = props.vendorID;
      info.deviceID = props.deviceID;
      info.apiVersion = props.apiVersion;
      info.driverVersion = props.driverVersion;
      info.isApu = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
      info.memorySize = vramSize;
      info.dedicatedVramBytes = info.isApu ? 0 : vramSize;

      if (pciExtSupported) {
        char busBuf[32];
        snprintf(busBuf, sizeof(busBuf), "%04x:%02x:%02x.%x",
                 pciBusProps.pciDomain, pciBusProps.pciBus,
                 pciBusProps.pciDevice, pciBusProps.pciFunction);
        info.pcieBusId = busBuf;
      }

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
      for (uint32_t i = 0; i < VK_UUID_SIZE; ++i) {
        std::snprintf(&uuid_str[i * 2], sizeof(uuid_str) - (i * 2), "%02x",
                      props.pipelineCacheUUID[i]);
      }
      info.driverUUID = std::string(uuid_str);

      info.maxWorkGroupSize = props.limits.maxComputeWorkGroupInvocations;
      info.maxComputeWorkGroupCountX = props.limits.maxComputeWorkGroupCount[0];
      info.maxComputeWorkGroupCountY = props.limits.maxComputeWorkGroupCount[1];
      info.maxComputeWorkGroupCountZ = props.limits.maxComputeWorkGroupCount[2];
      info.maxComputeSharedMemorySize = props.limits.maxComputeSharedMemorySize;
      info.subgroupSize = subgroupProps.subgroupSize;
      info.fp64Support = (features2.features.shaderFloat64 == VK_TRUE);
      info.fp16Support = (features168.shaderFloat16 == VK_TRUE);
      bool hasBf16Ext = hasExt("VK_KHR_shader_bfloat16") || hasExt("VK_EXT_shader_bfloat16");
      info.shaderBfloat16DotProductSupport =
          hasBf16Ext && (bfloat16Features.shaderBFloat16DotProduct == VK_TRUE);
      info.shaderBfloat16CoopMatrixSupport =
          hasBf16Ext && (bfloat16Features.shaderBFloat16CooperativeMatrix == VK_TRUE);
      info.bf16Support =
          info.shaderBfloat16DotProductSupport || info.shaderBfloat16CoopMatrixSupport;
      info.int8Support = (features168.shaderInt8 == VK_TRUE);
      info.cooperativeMatrixSupport =
          hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) &&
          (coopMatrixFeatures.cooperativeMatrix == VK_TRUE);
      // Vulkan has no standardized 2:4 sparse cooperative matrix extension yet
      // (VK_NV_cooperative_matrix2 exposes flexible/workgroup dense matrices, not 2:4 sparse MMA).
      info.structuredSparsitySupport = false;
      info.fp8Support = hasExt("VK_EXT_shader_float8") &&
                        (float8Features.shaderFloat8 == VK_TRUE) &&
                        (float8Features.shaderFloat8CooperativeMatrix == VK_TRUE);
      bool hasOcpExt = hasExt(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
      bool wave32Ok = supportsWave32Compute(
          dev, subgroupProps.subgroupSize,
          hasExt(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME));
      info.fp6Support =
          hasOcpExt && (ocpFeatures.shaderFloat6 == VK_TRUE) &&
          info.cooperativeMatrixSupport && wave32Ok &&
          hasSubgroupCoopMatrix16x16x16(
              instance, dev, VulkanContext::kVkComponentTypeFloat6E3M2EXT,
              VK_COMPONENT_TYPE_FLOAT32_KHR);
      info.fp4Support =
          hasOcpExt && (ocpFeatures.shaderFloat4 == VK_TRUE) &&
          info.cooperativeMatrixSupport && wave32Ok &&
          hasSubgroupCoopMatrix16x16x16(
              instance, dev, VulkanContext::kVkComponentTypeFloat4E2M1EXT,
              VK_COMPONENT_TYPE_FLOAT32_KHR);
      info.int4Support = false;
      info.rayTracingSupport =
          hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
          hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME);

      info.serSupported = queryHardwareSERSupport(
          dev, props.vendorID,
          hasExt("VK_EXT_ray_tracing_invocation_reorder"),
          hasExt("VK_NV_ray_tracing_invocation_reorder"));
      info.workGraphsSupported =
          hasExt("VK_AMDX_shader_enqueue") || hasExt("VK_KHR_work_graphs");
      info.dgcSupported = hasExt("VK_EXT_device_generated_commands");
      info.performanceQuerySupported = hasExt(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME);

      populateExtendedCapabilities(instance, dev, props, memProps, availableExts,
                                   subgroupProps.subgroupSize, info);
      DeviceDatabase::enrichDeviceInfo(info);

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
  const auto &allDevs = getDevices();
  if (selectedDeviceIndex < allDevs.size()) {
    DeviceInfo info = allDevs[selectedDeviceIndex];
    info.dgcSupported = dgcSupported;
    info.performanceQuerySupported = performanceQuerySupported;
    info.workGraphsSupported = workGraphsSupported;
    info.cooperativeVectorSupport = cooperativeVectorSupported;
    info.opacityMicromapSupport = opacityMicromapSupported;
    info.structuredSparsitySupport = structuredSparsitySupported;
    return info;
  }
  throw std::runtime_error("Selected device index out of range");
}

VulkanContext::CoopMatrixTile VulkanContext::selectCoopMatrixTile(
    VkComponentTypeKHR abType, VkComponentTypeKHR cType) const {
  if (physicalDevice == VK_NULL_HANDLE) {
    return CoopMatrixTile{};
  }
  VkPhysicalDeviceSubgroupProperties sgProps{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 p2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sgProps};
  vkGetPhysicalDeviceProperties2(physicalDevice, &p2);
  bool wave32Ok = supportsWave32Compute(
      physicalDevice, sgProps.subgroupSize,
      enabledExtensionsSet.count(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) > 0);
  return selectBestSubgroupCoopMatrixTile(
      instance, physicalDevice, abType, cType, sgProps.subgroupSize, wave32Ok);
}

void VulkanContext::pickPhysicalDevice(uint32_t index) {
  if (index >= physicalDevices.size()) {
    throw std::runtime_error("invalid device index");
  }
  if (physicalDevice == physicalDevices[index] && device != VK_NULL_HANDLE) {
    return;
  }

  if (device != VK_NULL_HANDLE) {
    try {
      waitIdle();
    } catch (...) {
    }
    clearSharedPreviewImages();
    while (!kernels.empty()) {
      releaseKernel(kernels.begin()->first);
    }
    while (!buffers.empty()) {
      releaseBuffer(buffers.begin()->first);
    }
    cleanupPipelineCache();
    cleanupStagingBuffer();
    cleanupMemoryBlocks();
    cleanupVmaAllocator();
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
    if (timestampQueryPool != VK_NULL_HANDLE) {
      vkDestroyQueryPool(device, timestampQueryPool, nullptr);
      timestampQueryPool = VK_NULL_HANDLE;
    }
    if (perfQueryPool != VK_NULL_HANDLE) {
      vkDestroyQueryPool(device, perfQueryPool, nullptr);
      perfQueryPool = VK_NULL_HANDLE;
    }
    performanceQuerySupported = false;
    performanceQueryEnabled = false;
    isPerfQueryActive = false;
    perfQueryRecording = false;
    perfQuerySubmitted = false;
    perfQueryNeedsSubmitInfo = false;
    destroyHeadlessSwapchain();
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
    computeQueue = VK_NULL_HANDLE;
    graphicsQueue = VK_NULL_HANDLE;
    presentationQueue = VK_NULL_HANDLE;
    asyncComputeQueue = VK_NULL_HANDLE;
    transferQueue = VK_NULL_HANDLE;
    secondaryTransferQueue = VK_NULL_HANDLE;
  }

  selectedDeviceIndex = index;
  physicalDevice = physicalDevices[index];
  vkGetPhysicalDeviceProperties(physicalDevice, &properties);
  createDevice();
}

void VulkanContext::initVmaAllocator(bool bufferDeviceAddressEnabled,
                                     bool memoryBudgetEnabled) {
  cleanupVmaAllocator();
  if (device == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE) {
    return;
  }

  VmaVulkanFunctions vulkanFunctions{};
  vulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
  vulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_4;
  allocatorInfo.physicalDevice = physicalDevice;
  allocatorInfo.device = device;
  allocatorInfo.instance = instance;
  allocatorInfo.pVulkanFunctions = &vulkanFunctions;
  if (bufferDeviceAddressEnabled) {
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  }
  if (memoryBudgetEnabled) {
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
  }
  if (amdDeviceCoherentMemorySupported) {
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
  }
#ifdef VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT
  if (maintenance5Supported) {
    allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
  }
#endif

  if (vmaCreateAllocator(&allocatorInfo, &m_vmaAllocator) != VK_SUCCESS) {
    throw std::runtime_error("failed to initialize VulkanMemoryAllocator (VMA)!");
  }
}

void VulkanContext::cleanupVmaAllocator() {
  if (m_vmaAllocator != VK_NULL_HANDLE) {
    vmaDestroyAllocator(m_vmaAllocator);
    m_vmaAllocator = VK_NULL_HANDLE;
  }
}

VulkanContext::VmaBudgetSummary VulkanContext::queryVramBudget() const {
  VmaBudgetSummary summary{};
  if (m_vmaAllocator == VK_NULL_HANDLE || physicalDevice == VK_NULL_HANDLE) {
    return summary;
  }
  VkPhysicalDeviceMemoryProperties memProps{};
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
  std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
  vmaGetHeapBudgets(m_vmaAllocator, budgets.data());

  for (uint32_t h = 0; h < memProps.memoryHeapCount; ++h) {
    if (memProps.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      summary.usageBytes += budgets[h].usage;
      summary.budgetBytes += budgets[h].budget;
      summary.available = true;
    }
  }
  return summary;
}

void VulkanContext::initPipelineCache() {
  cleanupPipelineCache();
  if (device == VK_NULL_HANDLE) {
    return;
  }

  char uuidHex[33];
  for (uint32_t i = 0; i < VK_UUID_SIZE; ++i) {
    std::snprintf(&uuidHex[i * 2], sizeof(uuidHex) - (i * 2), "%02x",
                  properties.pipelineCacheUUID[i]);
  }
  char fileName[128];
  std::snprintf(fileName, sizeof(fileName), "vk_pipe_%04x_%04x_%s.bin",
                properties.vendorID, properties.deviceID, uuidHex);

  std::filesystem::path cachePath = getPipelineCacheDirectory() / fileName;
  std::vector<uint8_t> initialData;

  std::ifstream in(cachePath, std::ios::binary | std::ios::ate);
  if (in.is_open()) {
    std::streamsize sz = in.tellg();
    if (sz >= static_cast<std::streamsize>(sizeof(VkPipelineCacheHeaderVersionOne)) &&
        sz <= static_cast<std::streamsize>(256ULL * 1024ULL * 1024ULL)) {
      initialData.resize(static_cast<size_t>(sz));
      in.seekg(0, std::ios::beg);
      if (in.read(reinterpret_cast<char *>(initialData.data()), sz)) {
        VkPipelineCacheHeaderVersionOne hdr{};
        std::memcpy(&hdr, initialData.data(), sizeof(hdr));
        bool validHeader =
            (hdr.headerSize == sizeof(VkPipelineCacheHeaderVersionOne)) &&
            (hdr.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE) &&
            (hdr.vendorID == properties.vendorID) &&
            (hdr.deviceID == properties.deviceID) &&
            (std::memcmp(hdr.pipelineCacheUUID, properties.pipelineCacheUUID,
                         VK_UUID_SIZE) == 0);
        if (!validHeader) {
          initialData.clear();
        }
      } else {
        initialData.clear();
      }
    }
  }

  VkPipelineCacheCreateInfo cacheCI{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  cacheCI.initialDataSize = initialData.size();
  cacheCI.pInitialData = initialData.empty() ? nullptr : initialData.data();

  if (vkCreatePipelineCache(device, &cacheCI, nullptr, &m_pipelineCache) !=
      VK_SUCCESS) {
    // Fallback to empty cache if driver rejected cached blob
    cacheCI.initialDataSize = 0;
    cacheCI.pInitialData = nullptr;
    if (vkCreatePipelineCache(device, &cacheCI, nullptr, &m_pipelineCache) !=
        VK_SUCCESS) {
      m_pipelineCache = VK_NULL_HANDLE;
    }
  }
}

void VulkanContext::flushPipelineCacheToDisk() const {
  if (device == VK_NULL_HANDLE || m_pipelineCache == VK_NULL_HANDLE) {
    return;
  }
  size_t dataSize = 0;
  if (vkGetPipelineCacheData(device, m_pipelineCache, &dataSize, nullptr) !=
          VK_SUCCESS ||
      dataSize < sizeof(VkPipelineCacheHeaderVersionOne)) {
    return;
  }
  std::vector<uint8_t> blob(dataSize);
  if (vkGetPipelineCacheData(device, m_pipelineCache, &dataSize, blob.data()) !=
      VK_SUCCESS) {
    return;
  }

  std::error_code ec;
  std::filesystem::path dir = getPipelineCacheDirectory();
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    return;
  }

  char uuidHex[33];
  for (uint32_t i = 0; i < VK_UUID_SIZE; ++i) {
    std::snprintf(&uuidHex[i * 2], sizeof(uuidHex) - (i * 2), "%02x",
                  properties.pipelineCacheUUID[i]);
  }
  char fileName[128];
  std::snprintf(fileName, sizeof(fileName), "vk_pipe_%04x_%04x_%s.bin",
                properties.vendorID, properties.deviceID, uuidHex);

  std::filesystem::path finalPath = dir / fileName;
  std::filesystem::path tmpPath = dir / (std::string(fileName) + ".tmp");
  {
    std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      return;
    }
    out.write(reinterpret_cast<const char *>(blob.data()),
              static_cast<std::streamsize>(blob.size()));
    if (!out.good()) {
      out.close();
      std::filesystem::remove(tmpPath, ec);
      return;
    }
  }
  std::filesystem::rename(tmpPath, finalPath, ec);
  if (ec) {
    std::filesystem::copy_file(
        tmpPath, finalPath, std::filesystem::copy_options::overwrite_existing, ec);
    std::filesystem::remove(tmpPath, ec);
  }
}

void VulkanContext::cleanupPipelineCache() {
  if (m_pipelineCache != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
    flushPipelineCacheToDisk();
    vkDestroyPipelineCache(device, m_pipelineCache, nullptr);
    m_pipelineCache = VK_NULL_HANDLE;
  }
}

void VulkanContext::createDevice() {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount,
                                           queueFamilies.data());

  computeQueueFamilyIndex = 0;
  graphicsQueueFamilyIndex = 0;
  presentationQueueFamilyIndex = 0;
  asyncComputeQueueFamilyIndex = UINT32_MAX;
  transferQueueFamilyIndex = UINT32_MAX;

  bool foundUniversal = false;
  for (uint32_t q = 0; q < queueFamilyCount; ++q) {
    if ((queueFamilies[q].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        (queueFamilies[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
        queueFamilies[q].queueCount > 0) {
      computeQueueFamilyIndex = q;
      graphicsQueueFamilyIndex = q;
      foundUniversal = true;
      break;
    }
  }
  if (!foundUniversal) {
    for (uint32_t q = 0; q < queueFamilyCount; ++q) {
      if ((queueFamilies[q].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
          queueFamilies[q].queueCount > 0) {
        computeQueueFamilyIndex = q;
        break;
      }
    }
    for (uint32_t q = 0; q < queueFamilyCount; ++q) {
      if ((queueFamilies[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          queueFamilies[q].queueCount > 0) {
        graphicsQueueFamilyIndex = q;
        break;
      }
    }
  }

  // Discover dedicated Async Compute (ACE) and DMA Transfer (SDMA) queue families
  for (uint32_t q = 0; q < queueFamilyCount; ++q) {
    if (queueFamilies[q].queueCount == 0) continue;
    if (asyncComputeQueueFamilyIndex == UINT32_MAX &&
        (queueFamilies[q].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
        !(queueFamilies[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      asyncComputeQueueFamilyIndex = q;
    }
    if (transferQueueFamilyIndex == UINT32_MAX &&
        (queueFamilies[q].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
        !(queueFamilies[q].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT))) {
      transferQueueFamilyIndex = q;
    }
  }

  // Presentation queue selection when GUI shares VkInstance/VkDevice
  presentationQueueFamilyIndex = graphicsQueueFamilyIndex;
  uint32_t presentationQueueSubIndex = 0;
  if (m_presentationSurface != VK_NULL_HANDLE) {
    VkBool32 gfxPresent = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, graphicsQueueFamilyIndex,
                                         m_presentationSurface, &gfxPresent);
    if (gfxPresent == VK_TRUE) {
      presentationQueueFamilyIndex = graphicsQueueFamilyIndex;
      if (queueFamilies[graphicsQueueFamilyIndex].queueCount >= 2) {
        presentationQueueSubIndex = 1;
      }
    } else {
      for (uint32_t q = 0; q < queueFamilyCount; ++q) {
        VkBool32 qPresent = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, q,
                                             m_presentationSurface, &qPresent);
        if (qPresent == VK_TRUE && queueFamilies[q].queueCount > 0) {
          presentationQueueFamilyIndex = q;
          presentationQueueSubIndex = 0;
          break;
        }
      }
    }
  }

  // Deduplicate requested queue counts per family
  std::map<uint32_t, uint32_t> requestedQueues;
  requestedQueues[computeQueueFamilyIndex] =
      std::max(requestedQueues[computeQueueFamilyIndex], 1u);
  requestedQueues[graphicsQueueFamilyIndex] =
      std::max(requestedQueues[graphicsQueueFamilyIndex], 1u);
  if (m_presentationSurface != VK_NULL_HANDLE) {
    requestedQueues[presentationQueueFamilyIndex] = std::max(
        requestedQueues[presentationQueueFamilyIndex], presentationQueueSubIndex + 1u);
  }
  if (asyncComputeQueueFamilyIndex != UINT32_MAX) {
    requestedQueues[asyncComputeQueueFamilyIndex] =
        std::max(requestedQueues[asyncComputeQueueFamilyIndex], 1u);
  }
  if (transferQueueFamilyIndex != UINT32_MAX) {
    requestedQueues[transferQueueFamilyIndex] =
        std::max(requestedQueues[transferQueueFamilyIndex], 2u);
  }

  std::vector<std::vector<float>> queuePrioritiesStorage;
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  queuePrioritiesStorage.reserve(requestedQueues.size());
  for (const auto &[qFamily, reqCount] : requestedQueues) {
    uint32_t clampedCount = std::min(reqCount, queueFamilies[qFamily].queueCount);
    if (clampedCount == 0) continue;
    queuePrioritiesStorage.emplace_back(clampedCount, 1.0f);
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qFamily;
    qci.queueCount = clampedCount;
    qci.pQueuePriorities = queuePrioritiesStorage.back().data();
    queueCreateInfos.push_back(qci);
  }

  // Query supported extensions first
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount,
                                       availableExtensions.data());

  auto hasExt = [&](const char *name) {
    for (const auto &available : availableExtensions) {
      if (strcmp(name, available.extensionName) == 0) return true;
    }
    return false;
  };

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
  VkPhysicalDeviceSynchronization2FeaturesKHR sync2Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR, nullptr, VK_TRUE};
  VkPhysicalDeviceHostQueryResetFeatures hostQueryResetFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};

  VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR, nullptr,
      VK_FALSE, VK_FALSE, VK_FALSE};
  VkPhysicalDeviceShaderFloat8FeaturesEXT float8Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT, nullptr, VK_FALSE};
  VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocpFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT,
      nullptr, VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE};
  VkPhysicalDeviceShaderFloatControls2FeaturesKHR floatControls2Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR, nullptr, VK_FALSE};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, nullptr};
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT serFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT, nullptr, VK_FALSE};
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
  VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR, nullptr, VK_FALSE};
  VkPhysicalDeviceCooperativeVectorFeaturesNV coopVectorFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
      nullptr, VK_FALSE, VK_FALSE};
  VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopMatrix2Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
      nullptr, VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE};
#ifdef VK_EXT_opacity_micromap
  VkPhysicalDeviceOpacityMicromapFeaturesEXT ommFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT,
      nullptr, VK_FALSE, VK_FALSE, VK_FALSE};
#endif

  void **currentPNext = (void **)&features2.pNext;
  *currentPNext = &features168; currentPNext = &features168.pNext;
  *currentPNext = &features16Storage; currentPNext = &features16Storage.pNext;
  *currentPNext = &features8Storage; currentPNext = &features8Storage.pNext;
  if (hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
    *currentPNext = &coopMatrixFeatures; currentPNext = &coopMatrixFeatures.pNext;
  }
  *currentPNext = &dotProductFeatures; currentPNext = &dotProductFeatures.pNext;
  *currentPNext = &subgroupSizeFeatures; currentPNext = &subgroupSizeFeatures.pNext;
  if (hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
    *currentPNext = &asFeatures; currentPNext = &asFeatures.pNext;
  }
  if (hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
    *currentPNext = &rayQueryFeatures; currentPNext = &rayQueryFeatures.pNext;
  }
  *currentPNext = &bufferDeviceAddressFeatures; currentPNext = &bufferDeviceAddressFeatures.pNext;

  if (hasExt("VK_KHR_shader_bfloat16")) {
    *currentPNext = &bfloat16Features; currentPNext = &bfloat16Features.pNext;
  }
  if (hasExt("VK_EXT_shader_float8")) {
    *currentPNext = &float8Features; currentPNext = &float8Features.pNext;
  }
  if (hasExt(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME)) {
    *currentPNext = &ocpFeatures; currentPNext = &ocpFeatures.pNext;
  }
  if (hasExt("VK_KHR_shader_float_controls2")) {
    *currentPNext = &floatControls2Features; currentPNext = &floatControls2Features.pNext;
  }
  if (hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
    *currentPNext = &rtPipelineFeatures; currentPNext = &rtPipelineFeatures.pNext;
  }
  serSupported = queryHardwareSERSupport(
      physicalDevice, properties.vendorID,
      hasExt("VK_EXT_ray_tracing_invocation_reorder"),
      hasExt("VK_NV_ray_tracing_invocation_reorder"));
  if (serSupported &&
      (hasExt("VK_EXT_ray_tracing_invocation_reorder") ||
       hasExt("VK_NV_ray_tracing_invocation_reorder"))) {
    *currentPNext = &serFeatures; currentPNext = &serFeatures.pNext;
  }
  if (hasExt(VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME)) {
    *currentPNext = &rtMaint1Features; currentPNext = &rtMaint1Features.pNext;
  }
  if (hasExt(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME)) {
    *currentPNext = &dgcFeatures; currentPNext = &dgcFeatures.pNext;
  }
  if (hasExt("VK_KHR_maintenance5") || properties.apiVersion >= VK_API_VERSION_1_4) {
    *currentPNext = &maintenance5Features; currentPNext = &maintenance5Features.pNext;
  }
  if (hasExt("VK_AMDX_shader_enqueue")) {
    *currentPNext = &enqueueFeatures; currentPNext = &enqueueFeatures.pNext;
  }
  if (hasExt(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME)) {
    *currentPNext = &coopVectorFeatures; currentPNext = &coopVectorFeatures.pNext;
  }
  if (hasExt(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME)) {
    *currentPNext = &coopMatrix2Features; currentPNext = &coopMatrix2Features.pNext;
  }
#ifdef VK_EXT_opacity_micromap
  if (hasExt(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME)) {
    *currentPNext = &ommFeatures; currentPNext = &ommFeatures.pNext;
  }
#endif
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipelineExecutablePropertiesFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
      nullptr, VK_FALSE};
  if (hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
    *currentPNext = &pipelineExecutablePropertiesFeatures; currentPNext = &pipelineExecutablePropertiesFeatures.pNext;
  }
  *currentPNext = &sync2Features; currentPNext = &sync2Features.pNext;
  VkPhysicalDevicePerformanceQueryFeaturesKHR perfQueryFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR, nullptr, VK_FALSE};
  if (hasExt(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME)) {
    *currentPNext = &perfQueryFeatures; currentPNext = &perfQueryFeatures.pNext;
  }
  VkPhysicalDeviceCoherentMemoryFeaturesAMD amdCoherentFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD, nullptr, VK_FALSE};
  if (hasExt(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME)) {
    *currentPNext = &amdCoherentFeatures; currentPNext = &amdCoherentFeatures.pNext;
  }
  VkPhysicalDeviceVulkanMemoryModelFeatures memoryModelFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, nullptr,
      VK_FALSE, VK_FALSE, VK_FALSE};
  VkPhysicalDeviceMaintenance4Features maintenance4Features{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES, nullptr,
      VK_FALSE};
  *currentPNext = &memoryModelFeatures; currentPNext = &memoryModelFeatures.pNext;
  *currentPNext = &maintenance4Features; currentPNext = &maintenance4Features.pNext;
  *currentPNext = &dynamicRenderingFeatures; currentPNext = &dynamicRenderingFeatures.pNext;
  *currentPNext = &hostQueryResetFeatures; currentPNext = &hostQueryResetFeatures.pNext;
  *currentPNext = nullptr;

  vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
  hostQueryResetSupported = (hostQueryResetFeatures.hostQueryReset == VK_TRUE);

  std::vector<const char *> desiredExtensions = {
      VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
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
      VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
      VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME,
      VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME,
      VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,
      VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME,
      "VK_KHR_maintenance5",
      "VK_KHR_pipeline_library",
      "VK_KHR_shader_bfloat16",
      "VK_EXT_shader_float8",
      VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME,
      "VK_KHR_shader_float_controls2",
      "VK_AMDX_shader_enqueue",
      VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME,
      VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME
#ifdef VK_EXT_opacity_micromap
      , VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME
#endif
  };
  if (serSupported) {
    desiredExtensions.push_back("VK_EXT_ray_tracing_invocation_reorder");
    desiredExtensions.push_back("VK_NV_ray_tracing_invocation_reorder");
  }

  swapchainSupported = hasExt(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  memoryBudgetSupported = hasExt(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  amdDeviceCoherentMemorySupported =
      hasExt(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) &&
      (amdCoherentFeatures.deviceCoherentMemory == VK_TRUE);

  std::vector<const char *> enabledExtensions;
  for (const auto &extension : desiredExtensions) {
    if (hasExt(extension)) {
      enabledExtensions.push_back(extension);
    } else if (verbose) {
      std::cerr << "Warning: Extension " << extension
                << " not supported by device, disabling." << std::endl;
    }
  }

  enabledExtensionsSet.clear();
  for (const auto &ext : enabledExtensions) {
    enabledExtensionsSet.insert(ext);
  }

  subgroupSizeControlSupported =
      (subgroupSizeFeatures.subgroupSizeControl == VK_TRUE);
  maintenance5Supported = (maintenance5Features.maintenance5 == VK_TRUE);
  workGraphsSupported =
      hasExt("VK_AMDX_shader_enqueue") && hasExt("VK_KHR_pipeline_library") &&
      maintenance5Supported && (enqueueFeatures.shaderEnqueue == VK_TRUE);
  cooperativeVectorSupported =
      hasExt(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME) &&
      (coopVectorFeatures.cooperativeVector == VK_TRUE);
#ifdef VK_EXT_opacity_micromap
  opacityMicromapSupported =
      hasExt(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME) &&
      (ommFeatures.micromap == VK_TRUE);
#else
  opacityMicromapSupported = false;
#endif
  structuredSparsitySupported = false;

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pNext = &features2;
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.ppEnabledExtensionNames = enabledExtensions.data();
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(enabledExtensions.size());
  createInfo.pEnabledFeatures = nullptr;

  if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }

  pipelineExecutablePropertiesSupported =
      hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) &&
      (pipelineExecutablePropertiesFeatures.pipelineExecutableInfo == VK_TRUE);
  if (pipelineExecutablePropertiesSupported) {
    vkGetPipelineExecutablePropertiesKHR_ptr =
        (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(
            device, "vkGetPipelineExecutablePropertiesKHR");
    vkGetPipelineExecutableStatisticsKHR_ptr =
        (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(
            device, "vkGetPipelineExecutableStatisticsKHR");
    vkGetPipelineExecutableInternalRepresentationsKHR_ptr =
        (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(
            device, "vkGetPipelineExecutableInternalRepresentationsKHR");
    if (!vkGetPipelineExecutablePropertiesKHR_ptr || !vkGetPipelineExecutableStatisticsKHR_ptr) {
      pipelineExecutablePropertiesSupported = false;
    }
  }

  performanceQuerySupported = hasExt(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME) &&
                              (perfQueryFeatures.performanceCounterQueryPools == VK_TRUE);
  performanceQueryEnabled = false;
  perfQueryPool = VK_NULL_HANDLE;
  perfQuerySelectedIndices.clear();
  perfCountersInfo.clear();
  perfCounterDescriptions.clear();

  if (performanceQuerySupported) {
    pfnEnumerateQueueFamilyPerformanceQueryCountersKHR_ptr =
        (PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)
            vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
    pfnGetQueueFamilyPerformanceQueryPassesKHR_ptr =
        (PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR");
    pfnAcquireProfilingLockKHR_ptr =
        (PFN_vkAcquireProfilingLockKHR)vkGetDeviceProcAddr(device, "vkAcquireProfilingLockKHR");
    pfnReleaseProfilingLockKHR_ptr =
        (PFN_vkReleaseProfilingLockKHR)vkGetDeviceProcAddr(device, "vkReleaseProfilingLockKHR");

    if (pfnEnumerateQueueFamilyPerformanceQueryCountersKHR_ptr &&
        pfnGetQueueFamilyPerformanceQueryPassesKHR_ptr &&
        pfnAcquireProfilingLockKHR_ptr &&
        pfnReleaseProfilingLockKHR_ptr) {
      uint32_t counterCount = 0;
      pfnEnumerateQueueFamilyPerformanceQueryCountersKHR_ptr(
          physicalDevice, computeQueueFamilyIndex, &counterCount, nullptr, nullptr);
      if (counterCount > 0) {
        perfCountersInfo.resize(counterCount);
        perfCounterDescriptions.resize(counterCount);
        for (uint32_t c = 0; c < counterCount; ++c) {
          perfCountersInfo[c].sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR;
          perfCounterDescriptions[c].sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR;
        }
        pfnEnumerateQueueFamilyPerformanceQueryCountersKHR_ptr(
            physicalDevice, computeQueueFamilyIndex, &counterCount,
            perfCountersInfo.data(), perfCounterDescriptions.data());

        for (uint32_t c = 0; c < counterCount; ++c) {
          perfQuerySelectedIndices.push_back(c);
          VkQueryPoolPerformanceCreateInfoKHR testPerfCI{VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR};
          testPerfCI.queueFamilyIndex = computeQueueFamilyIndex;
          testPerfCI.counterIndexCount = static_cast<uint32_t>(perfQuerySelectedIndices.size());
          testPerfCI.pCounterIndices = perfQuerySelectedIndices.data();
          uint32_t passes = 0;
          pfnGetQueueFamilyPerformanceQueryPassesKHR_ptr(physicalDevice, &testPerfCI, &passes);
          if (passes > 1) {
            perfQuerySelectedIndices.pop_back();
          }
        }

        if (!perfQuerySelectedIndices.empty()) {
          VkQueryPoolPerformanceCreateInfoKHR perfPoolCI{VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR};
          perfPoolCI.queueFamilyIndex = computeQueueFamilyIndex;
          perfPoolCI.counterIndexCount = static_cast<uint32_t>(perfQuerySelectedIndices.size());
          perfPoolCI.pCounterIndices = perfQuerySelectedIndices.data();

          VkQueryPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
          poolCI.pNext = &perfPoolCI;
          poolCI.queryType = VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;
          poolCI.queryCount = 1;

          if (vkCreateQueryPool(device, &poolCI, nullptr, &perfQueryPool) == VK_SUCCESS) {
            performanceQueryEnabled = true;
          }
        }
      }
    } else {
      performanceQuerySupported = false;
    }
  }

  if (hasExt(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME)) {
    VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT dgcProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &dgcProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

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
                   ((dgcProperties.supportedIndirectCommandsShaderStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0) &&
                   (vkGetGeneratedCommandsMemoryRequirementsEXT_ptr != nullptr) &&
                   (vkCmdPreprocessGeneratedCommandsEXT_ptr != nullptr) &&
                   (vkCmdExecuteGeneratedCommandsEXT_ptr != nullptr) &&
                   (vkCreateIndirectCommandsLayoutEXT_ptr != nullptr) &&
                   (vkDestroyIndirectCommandsLayoutEXT_ptr != nullptr);

    dgcExecutionSetSupported = dgcSupported &&
                   ((dgcProperties.supportedIndirectCommandsShaderStagesPipelineBinding & VK_SHADER_STAGE_COMPUTE_BIT) != 0) &&
                   (vkCreateIndirectExecutionSetEXT_ptr != nullptr) &&
                   (vkDestroyIndirectExecutionSetEXT_ptr != nullptr) &&
                   (vkUpdateIndirectExecutionSetPipelineEXT_ptr != nullptr);
  } else {
    dgcSupported = false;
    dgcExecutionSetSupported = false;
  }
  vkGetBufferDeviceAddressKHR_ptr =
      (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(
          device, "vkGetBufferDeviceAddressKHR");

  vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
  vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &graphicsQueue);
  if (m_presentationSurface != VK_NULL_HANDLE) {
    uint32_t maxQ = queueFamilies[presentationQueueFamilyIndex].queueCount;
    uint32_t subQ = (presentationQueueSubIndex < maxQ) ? presentationQueueSubIndex : 0u;
    vkGetDeviceQueue(device, presentationQueueFamilyIndex, subQ, &presentationQueue);
  } else {
    presentationQueue = VK_NULL_HANDLE;
  }
  if (asyncComputeQueueFamilyIndex != UINT32_MAX) {
    vkGetDeviceQueue(device, asyncComputeQueueFamilyIndex, 0, &asyncComputeQueue);
  } else {
    asyncComputeQueue = VK_NULL_HANDLE;
  }
  if (transferQueueFamilyIndex != UINT32_MAX) {
    vkGetDeviceQueue(device, transferQueueFamilyIndex, 0, &transferQueue);
    if (queueFamilies[transferQueueFamilyIndex].queueCount >= 2u) {
      vkGetDeviceQueue(device, transferQueueFamilyIndex, 1,
                       &secondaryTransferQueue);
    } else {
      secondaryTransferQueue = VK_NULL_HANDLE;
    }
  } else {
    transferQueue = VK_NULL_HANDLE;
    secondaryTransferQueue = VK_NULL_HANDLE;
  }

  VkQueryPoolCreateInfo qpInfo{};
  qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpInfo.queryCount = kMaxTimestamps;
  if (vkCreateQueryPool(device, &qpInfo, nullptr, &timestampQueryPool) == VK_SUCCESS) {
    timingSupported = (properties.limits.timestampPeriod > 0.0f);
    timestampPeriod = static_cast<double>(properties.limits.timestampPeriod);
    if (hostQueryResetSupported) {
      vkResetQueryPool(device, timestampQueryPool, 0, kMaxTimestamps);
    }
  } else {
    timingSupported = false;
  }

  vkCmdPipelineBarrier2KHR_ptr = (PFN_vkCmdPipelineBarrier2KHR)vkGetDeviceProcAddr(device, "vkCmdPipelineBarrier2");
  if (!vkCmdPipelineBarrier2KHR_ptr) {
    vkCmdPipelineBarrier2KHR_ptr = (PFN_vkCmdPipelineBarrier2KHR)vkGetDeviceProcAddr(device, "vkCmdPipelineBarrier2KHR");
  }
  sync2Supported = (sync2Features.synchronization2 == VK_TRUE) &&
                   (vkCmdPipelineBarrier2KHR_ptr != nullptr);

  vkCmdBeginRendering_ptr = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdBeginRendering");
  if (!vkCmdBeginRendering_ptr) {
    vkCmdBeginRendering_ptr = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR");
  }
  vkCmdEndRendering_ptr = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdEndRendering");
  if (!vkCmdEndRendering_ptr) {
    vkCmdEndRendering_ptr = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR");
  }
  dynamicRenderingSupported = (dynamicRenderingFeatures.dynamicRendering == VK_TRUE) &&
                              (vkCmdBeginRendering_ptr != nullptr) &&
                              (vkCmdEndRendering_ptr != nullptr);

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

  initVmaAllocator(bufferDeviceAddressFeatures.bufferDeviceAddress == VK_TRUE,
                   memoryBudgetSupported);
  initPipelineCache();
  initStagingBuffer();
}

void VulkanContext::startTiming(VkCommandBuffer cmd) {
  if (!timingSupported || timestampQueryPool == VK_NULL_HANDLE) {
    return;
  }
  if (cmd == VK_NULL_HANDLE) {
    waitIdle();
    m_activeTimingPairs.clear();
    currentQueryIndex = 0;
    if (hostQueryResetSupported) {
      vkResetQueryPool(device, timestampQueryPool, 0, kMaxTimestamps);
    }
    isTimingActive = true;
    timingRecording = false;
    timingSubmitted = false;
    timingSubmittedFence = VK_NULL_HANDLE;
  } else {
    if (!isTimingActive) {
      return;
    }
    if (currentQueryIndex + 2 > kMaxTimestamps) return;
    uint32_t startIdx = currentQueryIndex++;
    uint32_t stopIdx = currentQueryIndex++;
    if (!hostQueryResetSupported) {
      vkCmdResetQueryPool(cmd, timestampQueryPool, startIdx, 2);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueryPool, startIdx);
    m_activeTimingPairs.push_back({startIdx, stopIdx});
    timingRecording = true;
  }
}

double VulkanContext::stopTiming(VkCommandBuffer cmd) {
  if (!isTimingActive || !timingSupported || timestampQueryPool == VK_NULL_HANDLE) {
    isTimingActive = false;
    return 0.0;
  }

  if (cmd != VK_NULL_HANDLE) {
    if (!m_activeTimingPairs.empty()) {
      uint32_t stopIdx = m_activeTimingPairs.back().stopQuery;
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueryPool, stopIdx);
      timingRecording = false;
      timingSubmitted = true;
    }
    return 0.0;
  }

  if (timingRecording && m_inBatch && m_batchCmdRecording) {
    endBatch();
  }

  waitIdle();
  isTimingActive = false;

  if (m_activeTimingPairs.empty() || !timingSubmitted) {
    return 0.0;
  }

  double totalElapsedMs = 0.0;
  for (const auto &pair : m_activeTimingPairs) {
    uint64_t timestamps[2] = {0, 0};
    VkResult res = vkGetQueryPoolResults(
        device, timestampQueryPool, pair.startQuery, 2, sizeof(timestamps), timestamps,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (res == VK_SUCCESS && timestamps[1] >= timestamps[0]) {
      double elapsedNs = static_cast<double>(timestamps[1] - timestamps[0]) * timestampPeriod;
      totalElapsedMs += elapsedNs * 1e-6;
    }
  }

  m_activeTimingPairs.clear();
  return totalElapsedMs;
}

void VulkanContext::maybeBeginTiming(VkCommandBuffer cmd) {
  if (isTimingActive && !timingRecording && timestampQueryPool != VK_NULL_HANDLE) {
    if (currentQueryIndex + 2 > kMaxTimestamps) {
      return; // Query pool capacity safety guard
    }
    uint32_t startIdx = currentQueryIndex++;
    uint32_t stopIdx = currentQueryIndex++;
    if (!hostQueryResetSupported) {
      vkCmdResetQueryPool(cmd, timestampQueryPool, startIdx, 2);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueryPool, startIdx);
    m_activeTimingPairs.push_back({startIdx, stopIdx});
    timingRecording = true;
  }
}

void VulkanContext::maybeEndTiming(VkCommandBuffer cmd) {
  if (isTimingActive && timingRecording && timestampQueryPool != VK_NULL_HANDLE) {
    if (!m_activeTimingPairs.empty()) {
      uint32_t stopIdx = m_activeTimingPairs.back().stopQuery;
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueryPool, stopIdx);
      timingRecording = false;
      timingSubmitted = true;
      timingSubmittedFence = inFlightFrames[currentFrameIndex].fence;
    }
  }
}

void VulkanContext::startPerformanceQuery() {
  if (!performanceQueryEnabled || perfQueryPool == VK_NULL_HANDLE || !pfnAcquireProfilingLockKHR_ptr) {
    return;
  }
  if (isPerfQueryActive) return;

  waitIdle();
  VkAcquireProfilingLockInfoKHR lockInfo{VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR};
  lockInfo.timeout = 2000000000ULL; // 2 seconds
  VkResult lockRes = pfnAcquireProfilingLockKHR_ptr(device, &lockInfo);
  if (lockRes != VK_SUCCESS) {
    if (verbose) {
      std::cerr << "Warning: vkAcquireProfilingLockKHR returned " << lockRes << std::endl;
    }
    return;
  }

  isPerfQueryActive = true;
  perfQueryRecording = false;
  perfQuerySubmitted = false;
  perfQueryNeedsSubmitInfo = false;
}

HardwarePerformanceCounters VulkanContext::stopPerformanceQuery() {
  if (!isPerfQueryActive) {
    return HardwarePerformanceCounters{};
  }
  waitIdle();
  isPerfQueryActive = false;

  HardwarePerformanceCounters counters;
  counters.available = false;

  if (perfQuerySubmitted && perfQueryPool != VK_NULL_HANDLE) {
    std::vector<VkPerformanceCounterResultKHR> results(perfQuerySelectedIndices.size());
    VkResult qRes = vkGetQueryPoolResults(
        device, perfQueryPool, 0, 1,
        results.size() * sizeof(VkPerformanceCounterResultKHR),
        results.data(),
        sizeof(VkPerformanceCounterResultKHR) * perfQuerySelectedIndices.size(),
        VK_QUERY_RESULT_WAIT_BIT);
    if (qRes == VK_SUCCESS) {
      counters.available = true;
      for (size_t i = 0; i < perfQuerySelectedIndices.size(); ++i) {
        uint32_t cIdx = perfQuerySelectedIndices[i];
        const auto &desc = perfCounterDescriptions[cIdx];
        const auto &info = perfCountersInfo[cIdx];
        double val = 0.0;
        if (info.storage == VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR) {
          val = static_cast<double>(results[i].float32);
        } else if (info.storage == VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR) {
          val = results[i].float64;
        } else if (info.storage == VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR) {
          val = static_cast<double>(results[i].uint32);
        } else if (info.storage == VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR) {
          val = static_cast<double>(results[i].uint64);
        } else if (info.storage == VK_PERFORMANCE_COUNTER_STORAGE_INT32_KHR) {
          val = static_cast<double>(results[i].int32);
        } else if (info.storage == VK_PERFORMANCE_COUNTER_STORAGE_INT64_KHR) {
          val = static_cast<double>(results[i].int64);
        }

        std::string name(desc.name);
        counters.rawCounters[name] = val;

        if (name == "GPU active cycles") {
          counters.gpuActiveCycles = static_cast<uint64_t>(val);
        } else if (name == "Waves") {
          counters.waves = static_cast<uint64_t>(val);
        } else if (name == "VALU Instructions") {
          counters.valuInstructions = static_cast<uint64_t>(val);
        } else if (name == "SALU Instructions") {
          counters.saluInstructions = static_cast<uint64_t>(val);
        } else if (name == "VMEM Load Instructions") {
          counters.vmemLoadInstructions = static_cast<uint64_t>(val);
        } else if (name == "SMEM Load Instructions") {
          counters.smemLoadInstructions = static_cast<uint64_t>(val);
        } else if (name == "VMEM Store Instructions") {
          counters.vmemStoreInstructions = static_cast<uint64_t>(val);
        } else if (name == "LDS Instructions") {
          counters.ldsInstructions = static_cast<uint64_t>(val);
        } else if (name == "GDS Instructions") {
          counters.gdsInstructions = static_cast<uint64_t>(val);
        } else if (name == "VALU Busy") {
          counters.valuBusyPct = static_cast<float>(val);
        } else if (name == "SALU Busy") {
          counters.saluBusyPct = static_cast<float>(val);
        } else if (name == "VRAM read size") {
          counters.vramReadBytes = static_cast<uint64_t>(val);
        } else if (name == "VRAM write size") {
          counters.vramWriteBytes = static_cast<uint64_t>(val);
        } else if (name == "L0 cache hit ratio") {
          counters.l0CacheHitRatio = !std::isnan(val) ? static_cast<float>(val) : 0.0f;
        } else if (name == "L1 cache hit ratio") {
          counters.l1CacheHitRatio = !std::isnan(val) ? static_cast<float>(val) : 0.0f;
        } else if (name == "L2 cache hit ratio") {
          counters.l2CacheHitRatio = !std::isnan(val) ? static_cast<float>(val) : 0.0f;
        }
      }
    }
  }

  if (pfnReleaseProfilingLockKHR_ptr) {
    pfnReleaseProfilingLockKHR_ptr(device);
  }
  perfQuerySubmitted = false;
  perfQueryRecording = false;
  perfQueryNeedsSubmitInfo = false;
  lastPerfCounters = counters;
  return counters;
}

void VulkanContext::maybeBeginPerfQuery(VkCommandBuffer cmd) {
  if (isPerfQueryActive && !perfQueryRecording && !perfQuerySubmitted && perfQueryPool != VK_NULL_HANDLE) {
    vkCmdResetQueryPool(cmd, perfQueryPool, 0, 1);
    vkCmdBeginQuery(cmd, perfQueryPool, 0, 0);
    perfQueryRecording = true;
    perfQueryNeedsSubmitInfo = true;
  }
}

void VulkanContext::maybeEndPerfQuery(VkCommandBuffer cmd) {
  if (isPerfQueryActive && perfQueryRecording && perfQueryPool != VK_NULL_HANDLE) {
    vkCmdEndQuery(cmd, perfQueryPool, 0);
    perfQueryRecording = false;
    perfQuerySubmitted = true;
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

// Staging Buffer Pool Implementation (VMA-backed)
void VulkanContext::initStagingBuffer() {
  std::lock_guard<std::mutex> lock(stagingMutex);
  if (device == VK_NULL_HANDLE || commandPool == VK_NULL_HANDLE ||
      m_vmaAllocator == VK_NULL_HANDLE) {
    return;
  }

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = kStagingBufferSize;
  bufferInfo.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo vmaAllocCI{};
  vmaAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  vmaAllocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
  vmaAllocCI.requiredFlags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VmaAllocationInfo vmaOut{};
  if (vmaCreateBuffer(m_vmaAllocator, &bufferInfo, &vmaAllocCI, &stagingBuffer,
                      &stagingVmaAllocation, &vmaOut) != VK_SUCCESS ||
      !vmaOut.pMappedData) {
    stagingBuffer = VK_NULL_HANDLE;
    stagingVmaAllocation = VK_NULL_HANDLE;
    throw std::runtime_error("failed to create VMA persistent staging buffer!");
  }
  stagingMappedPtr = vmaOut.pMappedData;

  VkCommandBufferAllocateInfo cmdAllocInfo{};
  cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAllocInfo.commandPool = commandPool;
  cmdAllocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &stagingCmdBuffer) !=
      VK_SUCCESS) {
    vmaDestroyBuffer(m_vmaAllocator, stagingBuffer, stagingVmaAllocation);
    stagingBuffer = VK_NULL_HANDLE;
    stagingVmaAllocation = VK_NULL_HANDLE;
    stagingMappedPtr = nullptr;
    throw std::runtime_error("failed to allocate staging command buffer!");
  }

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (vkCreateFence(device, &fenceInfo, nullptr, &stagingFence) != VK_SUCCESS) {
    vkFreeCommandBuffers(device, commandPool, 1, &stagingCmdBuffer);
    stagingCmdBuffer = VK_NULL_HANDLE;
    vmaDestroyBuffer(m_vmaAllocator, stagingBuffer, stagingVmaAllocation);
    stagingBuffer = VK_NULL_HANDLE;
    stagingVmaAllocation = VK_NULL_HANDLE;
    stagingMappedPtr = nullptr;
    throw std::runtime_error("failed to create staging fence!");
  }
}

void VulkanContext::cleanupStagingBuffer() {
  std::lock_guard<std::mutex> lock(stagingMutex);
  if (device == VK_NULL_HANDLE) return;

  if (stagingFence != VK_NULL_HANDLE) {
    vkDestroyFence(device, stagingFence, nullptr);
    stagingFence = VK_NULL_HANDLE;
  }
  if (stagingCmdBuffer != VK_NULL_HANDLE && commandPool != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(device, commandPool, 1, &stagingCmdBuffer);
    stagingCmdBuffer = VK_NULL_HANDLE;
  }
  if (stagingVmaAllocation != VK_NULL_HANDLE && m_vmaAllocator != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_vmaAllocator, stagingBuffer, stagingVmaAllocation);
    stagingBuffer = VK_NULL_HANDLE;
    stagingVmaAllocation = VK_NULL_HANDLE;
    stagingMappedPtr = nullptr;
  } else {
    if (stagingMappedPtr != nullptr && stagingMemory != VK_NULL_HANDLE) {
      vkUnmapMemory(device, stagingMemory);
      stagingMappedPtr = nullptr;
    }
    if (stagingBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, stagingBuffer, nullptr);
      stagingBuffer = VK_NULL_HANDLE;
    }
    if (stagingMemory != VK_NULL_HANDLE) {
      vkFreeMemory(device, stagingMemory, nullptr);
      stagingMemory = VK_NULL_HANDLE;
    }
  }
}

// Retained legacy helpers
bool VulkanContext::allocateSuballocatedBuffer(VkDeviceSize size,
                                               VkDeviceSize alignment,
                                               uint32_t memoryTypeIndex,
                                               bool needDeviceAddress,
                                               VulkanBuffer *outBuf) {
  if (!outBuf) return false;

  auto res = m_suballocator.allocate(size, alignment, memoryTypeIndex, needDeviceAddress);
  if (res.success) {
    outBuf->memory = m_blockMemories[res.blockIndex];
    outBuf->offset = res.offset;
    outBuf->isSuballocated = true;
    outBuf->blockIndex = res.blockIndex;
    return true;
  }

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = kDefaultBlockSize;
  allocInfo.memoryTypeIndex = memoryTypeIndex;

  VkMemoryAllocateFlagsInfo flagsInfo{};
  if (needDeviceAddress) {
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocInfo.pNext = &flagsInfo;
  }

  VkDeviceMemory blockMemory = VK_NULL_HANDLE;
  if (vkAllocateMemory(device, &allocInfo, nullptr, &blockMemory) != VK_SUCCESS) {
    return false;
  }

  auto regRes = m_suballocator.registerNewBlock(size, kDefaultBlockSize, memoryTypeIndex, needDeviceAddress);
  m_blockMemories.push_back(blockMemory);

  outBuf->memory = blockMemory;
  outBuf->offset = regRes.offset;
  outBuf->isSuballocated = true;
  outBuf->blockIndex = regRes.blockIndex;
  return true;
}

void VulkanContext::freeSuballocatedBuffer(const VulkanBuffer *buf) {
  if (!buf || !buf->isSuballocated) {
    return;
  }
  m_suballocator.free(buf->blockIndex, buf->offset);
}

void VulkanContext::cleanupMemoryBlocks() {
  std::lock_guard<std::mutex> lock(suballocatorMutex);
  if (device != VK_NULL_HANDLE) {
    for (auto &mem : m_blockMemories) {
      if (mem != VK_NULL_HANDLE) {
        vkFreeMemory(device, mem, nullptr);
        mem = VK_NULL_HANDLE;
      }
    }
  }
  m_blockMemories.clear();
  m_suballocator.clear();
}

ComputeBuffer VulkanContext::createBuffer(size_t size, const void *host_ptr) {
  if (m_vmaAllocator == VK_NULL_HANDLE) {
    throw std::runtime_error("VMA allocator not initialized in createBuffer!");
  }
  auto *vulkanBuffer = new VulkanBuffer();
  VkDeviceSize allocByteSize = std::max<VkDeviceSize>(size, 16);

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = allocByteSize;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  if (enabledExtensionsSet.count(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) > 0) {
    bufferInfo.usage |=
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
  }
  if (enabledExtensionsSet.count(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) > 0) {
    bufferInfo.usage |= VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR;
  }
#ifdef VK_EXT_opacity_micromap
  if (opacityMicromapSupported) {
    bufferInfo.usage |=
        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT |
        VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT;
  }
#endif
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo vmaAllocCI{};
  vmaAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VmaAllocationInfo vmaOut{};
  if (vmaCreateBuffer(m_vmaAllocator, &bufferInfo, &vmaAllocCI,
                      &vulkanBuffer->buffer, &vulkanBuffer->vmaAllocation,
                      &vmaOut) != VK_SUCCESS) {
    delete vulkanBuffer;
    throw std::runtime_error("failed to allocate VMA device buffer!");
  }

  vulkanBuffer->size = allocByteSize;
  vulkanBuffer->mappedPtr = vmaOut.pMappedData;

  VkBufferDeviceAddressInfo bdaInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bdaInfo.buffer = vulkanBuffer->buffer;
  vulkanBuffer->address = vkGetBufferDeviceAddress(device, &bdaInfo);

  buffers[vulkanBuffer] = vulkanBuffer;

  if (host_ptr && size > 0) {
    writeBuffer(vulkanBuffer, 0, size, host_ptr);
  }

  return vulkanBuffer;
}

ComputeBuffer VulkanContext::createPinnedHostBuffer(size_t size, bool forUpload,
                                                    void **outMappedPtr) {
  if (m_vmaAllocator == VK_NULL_HANDLE) {
    throw std::runtime_error("VMA allocator not initialized in createPinnedHostBuffer!");
  }
  auto *vulkanBuffer = new VulkanBuffer();
  VkDeviceSize allocByteSize = std::max<VkDeviceSize>(size, 16);

  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = allocByteSize;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo vmaAllocCI{};
  vmaAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  vmaAllocCI.flags =
      VMA_ALLOCATION_CREATE_MAPPED_BIT |
      (forUpload ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                 : VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
  vmaAllocCI.requiredFlags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (!forUpload) {
    vmaAllocCI.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }

  VmaAllocationInfo vmaOut{};
  if (vmaCreateBuffer(m_vmaAllocator, &bufferInfo, &vmaAllocCI,
                      &vulkanBuffer->buffer, &vulkanBuffer->vmaAllocation,
                      &vmaOut) != VK_SUCCESS ||
      !vmaOut.pMappedData) {
    delete vulkanBuffer;
    throw std::runtime_error("failed to allocate VMA pinned host buffer!");
  }

  vulkanBuffer->size = allocByteSize;
  vulkanBuffer->mappedPtr = vmaOut.pMappedData;
  VkBufferDeviceAddressInfo bdaInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bdaInfo.buffer = vulkanBuffer->buffer;
  vulkanBuffer->address = vkGetBufferDeviceAddress(device, &bdaInfo);

  buffers[vulkanBuffer] = vulkanBuffer;
  if (outMappedPtr) {
    *outMappedPtr = vulkanBuffer->mappedPtr;
  }
  return vulkanBuffer;
}

ComputeBuffer VulkanContext::createRebarBuffer(size_t size, void **outMappedPtr) {
  if (m_vmaAllocator == VK_NULL_HANDLE) {
    return nullptr;
  }
  auto *vulkanBuffer = new VulkanBuffer();
  VkDeviceSize allocByteSize = std::max<VkDeviceSize>(size, 16);

  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = allocByteSize;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo vmaAllocCI{};
  vmaAllocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  vmaAllocCI.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  vmaAllocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  VmaAllocationInfo vmaOut{};
  if (vmaCreateBuffer(m_vmaAllocator, &bufferInfo, &vmaAllocCI,
                      &vulkanBuffer->buffer, &vulkanBuffer->vmaAllocation,
                      &vmaOut) != VK_SUCCESS ||
      !vmaOut.pMappedData) {
    delete vulkanBuffer;
    return nullptr;
  }

  vulkanBuffer->size = allocByteSize;
  vulkanBuffer->mappedPtr = vmaOut.pMappedData;
  VkBufferDeviceAddressInfo bdaInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bdaInfo.buffer = vulkanBuffer->buffer;
  vulkanBuffer->address = vkGetBufferDeviceAddress(device, &bdaInfo);

  buffers[vulkanBuffer] = vulkanBuffer;
  if (outMappedPtr) {
    *outMappedPtr = vulkanBuffer->mappedPtr;
  }
  return vulkanBuffer;
}

void *VulkanContext::getBufferMappedPtr(ComputeBuffer buffer) const {
  auto it = buffers.find(buffer);
  if (it != buffers.end() && it->second) {
    return it->second->mappedPtr;
  }
  return nullptr;
}

void VulkanContext::writeBuffer(ComputeBuffer buffer, size_t offset,
                                size_t size, const void *host_ptr) {
  if (size == 0) return;
  auto it = buffers.find(buffer);
  if (it == buffers.end() || !it->second) {
    throw std::runtime_error("Invalid buffer handle in writeBuffer");
  }
  VkBuffer dstBuffer = it->second->buffer;

  if (m_inBatch && m_batchCmdRecording) {
    endBatch();
  }

  std::lock_guard<std::mutex> lock(stagingMutex);
  if (stagingBuffer == VK_NULL_HANDLE || !stagingMappedPtr || stagingCmdBuffer == VK_NULL_HANDLE) {
    throw std::runtime_error("Staging buffer not initialized in writeBuffer");
  }

  const uint8_t *srcBytes = static_cast<const uint8_t *>(host_ptr);
  size_t bytesWritten = 0;

  while (bytesWritten < size) {
    size_t curChunk = std::min(size - bytesWritten, kStagingBufferSize);
    memcpy(stagingMappedPtr, srcBytes + bytesWritten, curChunk);

    vkResetCommandBuffer(stagingCmdBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(stagingCmdBuffer, &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin staging command buffer in writeBuffer");
    }

    cmdPipelineMemoryBarrier2(
        stagingCmdBuffer,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = offset + bytesWritten;
    copyRegion.size = curChunk;
    vkCmdCopyBuffer(stagingCmdBuffer, stagingBuffer, dstBuffer, 1, &copyRegion);

    cmdPipelineMemoryBarrier2(
        stagingCmdBuffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT);

    if (vkEndCommandBuffer(stagingCmdBuffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to end staging command buffer in writeBuffer");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &stagingCmdBuffer;

    if (stagingFence != VK_NULL_HANDLE) {
      vkResetFences(device, 1, &stagingFence);
      VkResult submitRes;
      {
        std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
        submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, stagingFence);
      }
      if (submitRes != VK_SUCCESS) {
        throw std::runtime_error("vkQueueSubmit failed in writeBuffer with result: " +
                                 std::to_string(submitRes));
      }
      VkResult waitRes = vkWaitForFences(device, 1, &stagingFence, VK_TRUE, UINT64_MAX);
      if (waitRes != VK_SUCCESS) {
        throw std::runtime_error("vkWaitForFences failed in writeBuffer with result: " +
                                 std::to_string(waitRes));
      }
    } else {
      std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
      VkResult submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
      if (submitRes != VK_SUCCESS) {
        throw std::runtime_error("vkQueueSubmit failed in writeBuffer with result: " +
                                 std::to_string(submitRes));
      }
      VkResult waitRes = vkQueueWaitIdle(computeQueue);
      if (waitRes != VK_SUCCESS) {
        throw std::runtime_error("vkQueueWaitIdle failed in writeBuffer with result: " +
                                 std::to_string(waitRes));
      }
    }

    bytesWritten += curChunk;
  }
}

void VulkanContext::readBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                               void *host_ptr) const {
  if (size == 0) return;
  auto it = buffers.find(buffer);
  if (it == buffers.end() || !it->second) {
    throw std::runtime_error("Invalid buffer handle in readBuffer");
  }
  VkBuffer srcBuffer = it->second->buffer;

  auto *mutableThis = const_cast<VulkanContext *>(this);
  if (m_inBatch && m_batchCmdRecording) {
    mutableThis->endBatch();
  }

  std::lock_guard<std::mutex> lock(stagingMutex);
  if (stagingBuffer == VK_NULL_HANDLE || !stagingMappedPtr || stagingCmdBuffer == VK_NULL_HANDLE) {
    throw std::runtime_error("Staging buffer not initialized in readBuffer");
  }

  uint8_t *dstBytes = static_cast<uint8_t *>(host_ptr);
  size_t bytesRead = 0;

  while (bytesRead < size) {
    size_t curChunk = std::min(size - bytesRead, kStagingBufferSize);

    vkResetCommandBuffer(stagingCmdBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(stagingCmdBuffer, &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin staging command buffer in readBuffer");
    }

    mutableThis->cmdPipelineMemoryBarrier2(
        stagingCmdBuffer,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = offset + bytesRead;
    copyRegion.dstOffset = 0;
    copyRegion.size = curChunk;
    vkCmdCopyBuffer(stagingCmdBuffer, srcBuffer, stagingBuffer, 1, &copyRegion);

    mutableThis->cmdPipelineMemoryBarrier2(
        stagingCmdBuffer,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT);

    if (vkEndCommandBuffer(stagingCmdBuffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to end staging command buffer in readBuffer");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &stagingCmdBuffer;

    if (stagingFence != VK_NULL_HANDLE) {
      vkResetFences(device, 1, &stagingFence);
      VkResult submitRes;
      {
        std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
        submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, stagingFence);
      }
      if (submitRes != VK_SUCCESS) {
        throw std::runtime_error("vkQueueSubmit failed in readBuffer with result: " +
                                 std::to_string(submitRes));
      }
      VkResult waitRes = vkWaitForFences(device, 1, &stagingFence, VK_TRUE, UINT64_MAX);
      if (waitRes != VK_SUCCESS) {
        throw std::runtime_error("vkWaitForFences failed in readBuffer with result: " +
                                 std::to_string(waitRes));
      }
    } else {
      std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
      VkResult submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
      if (submitRes != VK_SUCCESS) {
        throw std::runtime_error("vkQueueSubmit failed in readBuffer with result: " +
                                 std::to_string(submitRes));
      }
      VkResult waitRes = vkQueueWaitIdle(computeQueue);
      if (waitRes != VK_SUCCESS) {
        throw std::runtime_error("vkQueueWaitIdle failed in readBuffer with result: " +
                                 std::to_string(waitRes));
      }
    }

    memcpy(dstBytes + bytesRead, stagingMappedPtr, curChunk);
    bytesRead += curChunk;
  }
}

void VulkanContext::releaseBuffer(ComputeBuffer buffer) {
  auto it = buffers.find(buffer);
  if (it != buffers.end()) {
    VulkanBuffer *vulkanBuffer = it->second;
    if (vulkanBuffer) {
      if (vulkanBuffer->vmaAllocation != VK_NULL_HANDLE &&
          m_vmaAllocator != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_vmaAllocator, vulkanBuffer->buffer,
                         vulkanBuffer->vmaAllocation);
        vulkanBuffer->buffer = VK_NULL_HANDLE;
        vulkanBuffer->vmaAllocation = VK_NULL_HANDLE;
      } else {
        if (vulkanBuffer->buffer != VK_NULL_HANDLE) {
          vkDestroyBuffer(device, vulkanBuffer->buffer, nullptr);
          vulkanBuffer->buffer = VK_NULL_HANDLE;
        }
        if (vulkanBuffer->isSuballocated) {
          std::lock_guard<std::mutex> lock(suballocatorMutex);
          freeSuballocatedBuffer(vulkanBuffer);
        } else if (vulkanBuffer->memory != VK_NULL_HANDLE) {
          vkFreeMemory(device, vulkanBuffer->memory, nullptr);
          vulkanBuffer->memory = VK_NULL_HANDLE;
        }
      }
      delete vulkanBuffer;
    }
    buffers.erase(it);
  }
}

VkDeviceAddress
VulkanContext::getBufferDeviceAddress(ComputeBuffer buffer) const {
  auto it = buffers.find(buffer);
  if (it != buffers.end() && it->second) {
    return it->second->address;
  }
  return 0;
}

VkBuffer VulkanContext::getVkBuffer(ComputeBuffer buffer) const {
  auto it = buffers.find(buffer);
  if (it != buffers.end() && it->second) {
    return it->second->buffer;
  }
  return VK_NULL_HANDLE;
}

ComputeKernel VulkanContext::createKernel(const std::string &file_name,
                                          const std::string &kernel_name,
                                          uint32_t num_buffer_args) {
  return createKernelInternal(file_name, kernel_name, num_buffer_args, nullptr,
                              nullptr, 0);
}

ComputeKernel VulkanContext::createKernelWithSpec(const std::string &file_name,
                                                  const std::string &kernel_name,
                                                  uint32_t num_buffer_args,
                                                  uint32_t spec_id,
                                                  uint32_t spec_val) {
  return createKernelInternal(file_name, kernel_name, num_buffer_args, &spec_id,
                              &spec_val, 1);
}

ComputeKernel VulkanContext::createKernelWithSpecConstants(
    const std::string &file_name, const std::string &kernel_name,
    uint32_t num_buffer_args, const std::vector<uint32_t> &spec_uints) {
  if (spec_uints.empty()) {
    return createKernelInternal(file_name, kernel_name, num_buffer_args,
                                nullptr, nullptr, 0);
  }
  std::vector<uint32_t> ids(spec_uints.size());
  for (uint32_t i = 0; i < static_cast<uint32_t>(spec_uints.size()); ++i) {
    ids[i] = i;
  }
  return createKernelInternal(file_name, kernel_name, num_buffer_args,
                              ids.data(), spec_uints.data(),
                              static_cast<uint32_t>(spec_uints.size()));
}

ComputeKernel VulkanContext::createKernelInternal(const std::string &file_name,
                                                  const std::string &kernel_name,
                                                  uint32_t num_buffer_args,
                                                  const uint32_t *spec_ids,
                                                  const uint32_t *spec_vals,
                                                  uint32_t spec_count) {
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

#if defined(shaderc_env_version_vulkan_1_4)
        options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                     shaderc_env_version_vulkan_1_4);
#else
        options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                     shaderc_env_version_vulkan_1_3);
#endif
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

  bool has_accel_struct = false;
  uint32_t spirv_local_size_x = 0;
  if (spirv_code.size() >= 5 && spirv_code[0] == 0x07230203) {
    size_t spvIdx = 5;
    while (spvIdx < spirv_code.size()) {
      uint32_t instrWord = spirv_code[spvIdx];
      uint16_t opcode = static_cast<uint16_t>(instrWord & 0xFFFF);
      uint16_t length = static_cast<uint16_t>((instrWord >> 16) & 0xFFFF);
      if (length == 0 || spvIdx + length > spirv_code.size()) break;
      if (opcode == 16 /* OpExecutionMode */ && length >= 6 &&
          spirv_code[spvIdx + 2] == 17 /* ExecutionModeLocalSize */) {
        spirv_local_size_x = spirv_code[spvIdx + 3];
      } else if (opcode == 5341 /* OpTypeAccelerationStructureKHR / NV */) {
        has_accel_struct = true;
      } else if (opcode == 54 /* OpFunction */) {
        break;
      }
      spvIdx += length;
    }
  }

  bool is_rt = has_accel_struct ||
               (file_name.find("rt_") != std::string::npos &&
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
    vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
    delete vulkanKernel;
    throw std::runtime_error("failed to create descriptor set layout!");
  }

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = 128;

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &vulkanKernel->descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                             &vulkanKernel->pipelineLayout) != VK_SUCCESS) {
    vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
    delete vulkanKernel;
    throw std::runtime_error("failed to create pipeline layout!");
  }

  vulkanKernel->pushConstantData.resize(128, 0);

  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  if (pipelineExecutablePropertiesSupported) {
    pipelineInfo.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    if (file_name.find("bf16.comp") != std::string::npos) {
      pipelineInfo.flags |= VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    }
  }
  pipelineInfo.layout = vulkanKernel->pipelineLayout;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = vulkanKernel->shaderModule;
  pipelineInfo.stage.pName = kernel_name.c_str();

  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSubgroupInfo{};
  reqSubgroupInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
  reqSubgroupInfo.requiredSubgroupSize = 32;
  if (subgroupSizeControlSupported) {
    VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroupSizeProps};
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
    if (subgroupSizeProps.minSubgroupSize <= 32 &&
        subgroupSizeProps.maxSubgroupSize >= 32 &&
        (subgroupSizeProps.requiredSubgroupSizeStages &
         VK_SHADER_STAGE_COMPUTE_BIT)) {
      reqSubgroupInfo.pNext = (void *)pipelineInfo.stage.pNext;
      pipelineInfo.stage.pNext = &reqSubgroupInfo;
      if (spirv_local_size_x > 0 && (spirv_local_size_x % 32u == 0u)) {
        pipelineInfo.stage.flags |=
            VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
      }
    }
  }

  std::vector<VkSpecializationMapEntry> specEntries;
  VkSpecializationInfo specInfo{};
  if (spec_ids && spec_vals && spec_count > 0) {
    specEntries.resize(spec_count);
    for (uint32_t s = 0; s < spec_count; ++s) {
      specEntries[s].constantID = spec_ids[s];
      specEntries[s].offset = s * static_cast<uint32_t>(sizeof(uint32_t));
      specEntries[s].size = sizeof(uint32_t);
    }
    specInfo.mapEntryCount = spec_count;
    specInfo.pMapEntries = specEntries.data();
    specInfo.dataSize = spec_count * sizeof(uint32_t);
    specInfo.pData = spec_vals;
    pipelineInfo.stage.pSpecializationInfo = &specInfo;
  }

  VkPipelineCreateFlags2CreateInfoKHR pipeFlags2{
      VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR};
  if (isDGCSupported() && maintenance5Supported) {
    pipeFlags2.flags = static_cast<VkPipelineCreateFlags2KHR>(pipelineInfo.flags) |
                       VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT;
    pipelineInfo.pNext = &pipeFlags2;
  }

  VkResult result =
      vkCreateComputePipelines(device, m_pipelineCache, 1, &pipelineInfo,
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
  poolSizes[0].descriptorCount = std::max(1u, num_buffer_args);
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  poolSizes[1].descriptorCount = 1;

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = is_rt ? 2 : 1;
  poolInfo.pPoolSizes = poolSizes;
  poolInfo.maxSets = 1;

  if (vkCreateDescriptorPool(device, &poolInfo, nullptr,
                             &vulkanKernel->descriptorPool) != VK_SUCCESS) {
    vkDestroyPipeline(device, vulkanKernel->pipeline, nullptr);
    vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
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
    vkDestroyDescriptorPool(device, vulkanKernel->descriptorPool, nullptr);
    vkDestroyPipeline(device, vulkanKernel->pipeline, nullptr);
    vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
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

  if (arg_index < it->second->numBufferDescriptors) {
    throw std::runtime_error("setKernelArg (value) called for index " +
                             std::to_string(arg_index) +
                             " but it is reserved for a buffer descriptor (numBuffers=" +
                             std::to_string(it->second->numBufferDescriptors) + ")");
  }

  size_t offset = (arg_index - it->second->numBufferDescriptors) * 4;
  if (offset + arg_size > it->second->pushConstantData.size()) {
    throw std::runtime_error("setKernelArg push constant overflow: offset " +
                             std::to_string(offset) + " + size " +
                             std::to_string(arg_size) + " exceeds capacity " +
                             std::to_string(it->second->pushConstantData.size()));
  }
  memcpy(it->second->pushConstantData.data() + offset, arg_value, arg_size);
}

void VulkanContext::beginBatch(uint32_t /*expected_dispatches*/,
                               bool inter_dispatch_barrier) {
  if (m_inBatch) {
    endBatch();
  }
  m_inBatch = true;
  m_batchCmdRecording = false;
  m_batchInterDispatchBarrier = inter_dispatch_barrier;
  m_batchDispatchCount = 0;
}

void VulkanContext::endBatch() {
  if (!m_inBatch) return;
  if (m_batchCmdRecording) {
    InFlightFrame &frame = inFlightFrames[currentFrameIndex];
    maybeEndPerfQuery(frame.commandBuffer);
    maybeEndTiming(frame.commandBuffer);
    vkEndCommandBuffer(frame.commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &frame.commandBuffer;

    VkPerformanceQuerySubmitInfoKHR perfSubmitInfo{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
    perfSubmitInfo.counterPassIndex = 0;
    if (perfQueryNeedsSubmitInfo) {
      perfSubmitInfo.pNext = submitInfo.pNext;
      submitInfo.pNext = &perfSubmitInfo;
      perfQueryNeedsSubmitInfo = false;
    }

    VkResult submitRes;
    {
      std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
      submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
    }
    if (submitRes != VK_SUCCESS) {
      throw std::runtime_error("vkQueueSubmit failed in endBatch with result: " +
                               std::to_string(submitRes));
    }
    frame.inUse = true;
    currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
    m_batchCmdRecording = false;
  }
  m_inBatch = false;
  m_batchInterDispatchBarrier = false;
  m_batchDispatchCount = 0;
}

void VulkanContext::dispatch(ComputeKernel kernel, uint32_t grid_x,
                             uint32_t grid_y, uint32_t grid_z, uint32_t block_x,
                             uint32_t block_y, uint32_t block_z) {
  auto it = kernels.find(kernel);
  if (it == kernels.end()) {
    throw std::runtime_error("Invalid kernel handle");
  }
  VulkanKernel *vulkanKernel = it->second;
  lastDispatchedKernel = kernel;

  if (m_inBatch) {
    InFlightFrame &frame = inFlightFrames[currentFrameIndex];
    if (!m_batchCmdRecording) {
      if (frame.inUse) {
        constexpr uint64_t kTimeoutNs = 30'000'000'000ULL;
        VkResult waitResult =
            vkWaitForFences(device, 1, &frame.fence, VK_TRUE, kTimeoutNs);
        if (waitResult == VK_TIMEOUT) {
          throw std::runtime_error(
              "GPU dispatch timed out (>30 s) — aborting benchmark to prevent amdgpu TDR crash.");
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
      maybeBeginPerfQuery(frame.commandBuffer);
      maybeBeginTiming(frame.commandBuffer);
      m_batchCmdRecording = true;
      m_batchDispatchCount = 0;
    }

    VkPipelineBindPoint bindPoint = vulkanKernel->isRTPipeline
                                        ? VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR
                                        : VK_PIPELINE_BIND_POINT_COMPUTE;

    // Only emit inter-dispatch memory barrier for stateful/dependent kernels
    if (m_batchDispatchCount > 0 && m_batchInterDispatchBarrier) {
      VkPipelineStageFlags2 stageMask =
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
          (vulkanKernel->isRTPipeline
               ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
               : 0);
      cmdPipelineMemoryBarrier2(
          frame.commandBuffer, stageMask, VK_ACCESS_2_SHADER_WRITE_BIT,
          stageMask, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    }

    vkCmdBindPipeline(frame.commandBuffer, bindPoint, vulkanKernel->pipeline);
    vkCmdBindDescriptorSets(frame.commandBuffer, bindPoint,
                            vulkanKernel->pipelineLayout, 0, 1,
                            &vulkanKernel->descriptorSet, 0, nullptr);

    if (!vulkanKernel->pushConstantData.empty()) {
      VkShaderStageFlags stageFlags = vulkanKernel->isRTPipeline
                                          ? (VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                             VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                             VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                             VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
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

    m_batchDispatchCount++;
    return;
  }

  InFlightFrame &frame = inFlightFrames[currentFrameIndex];
  if (frame.inUse) {
    constexpr uint64_t kTimeoutNs = 30'000'000'000ULL;
    VkResult waitResult =
        vkWaitForFences(device, 1, &frame.fence, VK_TRUE, kTimeoutNs);
    if (waitResult == VK_TIMEOUT) {
      throw std::runtime_error(
          "GPU dispatch timed out (>30 s) — aborting benchmark to prevent amdgpu TDR crash.");
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
  maybeBeginPerfQuery(frame.commandBuffer);
  maybeBeginTiming(frame.commandBuffer);
  VkPipelineBindPoint bindPoint = vulkanKernel->isRTPipeline
                                      ? VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR
                                      : VK_PIPELINE_BIND_POINT_COMPUTE;

  VkPipelineStageFlags2 unbatchedStageMask =
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
      (vulkanKernel->isRTPipeline
           ? VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR
           : 0);
  cmdPipelineMemoryBarrier2(
      frame.commandBuffer, unbatchedStageMask, VK_ACCESS_2_SHADER_WRITE_BIT,
      unbatchedStageMask,
      VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

  vkCmdBindPipeline(frame.commandBuffer, bindPoint, vulkanKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, bindPoint,
                           vulkanKernel->pipelineLayout, 0, 1,
                           &vulkanKernel->descriptorSet, 0, nullptr);

  if (!vulkanKernel->pushConstantData.empty()) {
    VkShaderStageFlags stageFlags = vulkanKernel->isRTPipeline
                                        ? (VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                           VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                           VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
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

  maybeEndPerfQuery(frame.commandBuffer);
  maybeEndTiming(frame.commandBuffer);
  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkPerformanceQuerySubmitInfoKHR perfSubmitInfo{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
  perfSubmitInfo.counterPassIndex = 0;
  if (perfQueryNeedsSubmitInfo) {
    perfSubmitInfo.pNext = submitInfo.pNext;
    submitInfo.pNext = &perfSubmitInfo;
    perfQueryNeedsSubmitInfo = false;
  }

  VkResult submitRes;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  }
  if (submitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatch with result: " +
                             std::to_string(submitRes));
  }
  frame.inUse = true;

  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

KernelResourceUsage VulkanContext::getKernelResourceUsage(ComputeKernel kernel) const {
  KernelResourceUsage usage;
  if (!kernel || !pipelineExecutablePropertiesSupported ||
      !vkGetPipelineExecutablePropertiesKHR_ptr || !vkGetPipelineExecutableStatisticsKHR_ptr) {
    return usage;
  }
  VulkanKernel *vkKernel = getKernel(kernel);
  if (!vkKernel || vkKernel->pipeline == VK_NULL_HANDLE) {
    return usage;
  }

  VkPipelineInfoKHR pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
  pipelineInfo.pipeline = vkKernel->pipeline;

  uint32_t executableCount = 0;
  if (vkGetPipelineExecutablePropertiesKHR_ptr(device, &pipelineInfo, &executableCount, nullptr) != VK_SUCCESS ||
      executableCount == 0) {
    return usage;
  }

  std::vector<VkPipelineExecutablePropertiesKHR> props(executableCount);
  for (auto &p : props) {
    p.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    p.pNext = nullptr;
  }
  if (vkGetPipelineExecutablePropertiesKHR_ptr(device, &pipelineInfo, &executableCount, props.data()) != VK_SUCCESS) {
    return usage;
  }

  VkPipelineExecutableInfoKHR execInfo{};
  execInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
  execInfo.pipeline = vkKernel->pipeline;
  execInfo.executableIndex = 0;

  uint32_t statCount = 0;
  if (vkGetPipelineExecutableStatisticsKHR_ptr(device, &execInfo, &statCount, nullptr) == VK_SUCCESS && statCount > 0) {
    std::vector<VkPipelineExecutableStatisticKHR> stats(statCount);
    for (auto &s : stats) {
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
      s.pNext = nullptr;
    }
    if (vkGetPipelineExecutableStatisticsKHR_ptr(device, &execInfo, &statCount, stats.data()) == VK_SUCCESS) {
      usage.available = true;
      VkPhysicalDeviceDriverProperties driverProps{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
      VkPhysicalDeviceProperties2 drvProps2{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driverProps};
      vkGetPhysicalDeviceProperties2(physicalDevice, &drvProps2);
      std::string drvName = driverProps.driverName;
      std::string drvInfo = driverProps.driverInfo;
      if (drvName == "radv" || drvName.find("radv") != std::string::npos ||
          drvName.find("RADV") != std::string::npos) {
        usage.compilerNotes = "Mesa RADV (ACO)";
      } else if (!drvName.empty()) {
        usage.compilerNotes =
            drvName + (drvInfo.empty() ? "" : " (" + drvInfo + ")");
      } else {
        usage.compilerNotes = "Vulkan Pipeline Compiler";
      }
      for (const auto &s : stats) {
        std::string name = s.name;
        std::string lname = name;
        std::transform(lname.begin(), lname.end(), lname.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        uint64_t val = 0;
        if (s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) {
          val = s.value.u64;
        } else if (s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR) {
          val = static_cast<uint64_t>(s.value.i64);
        } else if (s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR) {
          val = static_cast<uint64_t>(s.value.f64);
        }

        if (lname == "vgprs" || lname == "numusedvgprs" ||
            (lname.find("vgpr") != std::string::npos &&
             lname.find("spill") == std::string::npos &&
             lname.find("pre-sched") == std::string::npos &&
             lname.find("available") == std::string::npos &&
             lname.find("physical") == std::string::npos)) {
          usage.vgprCount = static_cast<uint32_t>(val);
        } else if (lname == "sgprs" || lname == "numusedsgprs" ||
                   (lname.find("sgpr") != std::string::npos &&
                    lname.find("spill") == std::string::npos &&
                    lname.find("pre-sched") == std::string::npos &&
                    lname.find("available") == std::string::npos &&
                    lname.find("physical") == std::string::npos)) {
          usage.sgprCount = static_cast<uint32_t>(val);
        } else if (lname.find("scratch") != std::string::npos ||
                   lname.find("spill") != std::string::npos) {
          usage.scratchSizeBytes = std::max(usage.scratchSizeBytes, static_cast<uint32_t>(val));
        } else if (lname.find("lds") != std::string::npos &&
                   lname.find("available") == std::string::npos) {
          usage.ldsSizeBytes = static_cast<uint32_t>(val);
        } else if (lname.find("code") != std::string::npos ||
                   lname.find("binary") != std::string::npos) {
          usage.codeSizeBytes = static_cast<uint32_t>(val);
        } else if (lname.find("subgroups per simd") != std::string::npos ||
                   lname.find("waves per simd") != std::string::npos ||
                   lname.find("occupancy") != std::string::npos) {
          usage.maxWavesPerSimd = static_cast<uint32_t>(val);
        }
      }
      if (usage.maxWavesPerSimd == 0 && usage.vgprCount > 0 &&
          properties.vendorID == 0x1002) {
        const std::string devNameStr(properties.deviceName);
        const auto &prof = DeviceDatabase::lookup(
            properties.vendorID, properties.deviceID, devNameStr);
        if (prof.archFamily == "RDNA 3" || prof.archFamily == "RDNA 3.5" ||
            prof.archFamily == "RDNA 4") {
          usage.maxWavesPerSimd = std::min(16u, 1536u / usage.vgprCount);
        } else if (prof.archFamily == "RDNA 2" || prof.archFamily == "RDNA 1") {
          usage.maxWavesPerSimd = std::min(16u, 1024u / usage.vgprCount);
        } else if (prof.archFamily.find("CDNA") != std::string::npos ||
                   prof.archFamily.find("GCN") != std::string::npos) {
          usage.maxWavesPerSimd = std::min(8u, 512u / usage.vgprCount);
        }
      }
    }
  }
  return usage;
}

bool VulkanContext::isKernelBf16Emulated(ComputeKernel kernel) const {
  if (!pipelineExecutablePropertiesSupported ||
      !vkGetPipelineExecutableInternalRepresentationsKHR_ptr ||
      device == VK_NULL_HANDLE || !kernel) {
    return false;
  }
  const auto *vkKernel = static_cast<const VulkanKernel *>(kernel);
  if (!vkKernel || vkKernel->pipeline == VK_NULL_HANDLE) {
    return false;
  }

  VkPipelineExecutableInfoKHR execInfo{};
  execInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
  execInfo.pipeline = vkKernel->pipeline;
  execInfo.executableIndex = 0;

  uint32_t repCount = 0;
  if (vkGetPipelineExecutableInternalRepresentationsKHR_ptr(
          device, &execInfo, &repCount, nullptr) != VK_SUCCESS ||
      repCount == 0) {
    return false;
  }

  std::vector<VkPipelineExecutableInternalRepresentationKHR> reps(repCount);
  for (auto &r : reps) {
    r.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
    r.pNext = nullptr;
    r.dataSize = 0;
    r.pData = nullptr;
  }
  if (vkGetPipelineExecutableInternalRepresentationsKHR_ptr(
          device, &execInfo, &repCount, reps.data()) != VK_SUCCESS) {
    return false;
  }

  std::vector<std::vector<char>> buffers(repCount);
  for (uint32_t i = 0; i < repCount; ++i) {
    if (reps[i].dataSize > 0) {
      buffers[i].resize(reps[i].dataSize);
      reps[i].pData = buffers[i].data();
    }
  }
  if (vkGetPipelineExecutableInternalRepresentationsKHR_ptr(
          device, &execInfo, &repCount, reps.data()) != VK_SUCCESS) {
    return false;
  }

  for (uint32_t i = 0; i < repCount; ++i) {
    if (reps[i].isText != VK_TRUE || buffers[i].empty()) {
      continue;
    }
    std::string repName = reps[i].name;
    std::transform(repName.begin(), repName.end(), repName.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // Skip intermediate IR dumps (LLVM IR / SPIR-V / NIR) and inspect final hardware ISA
    if (repName.find("ir") != std::string::npos ||
        repName.find("nir") != std::string::npos ||
        repName.find("spirv") != std::string::npos ||
        repName.find("amdil") != std::string::npos) {
      continue;
    }

    std::string text(buffers[i].data(),
                     strnlen(buffers[i].data(), buffers[i].size()));
    std::istringstream iss(text);
    std::string line;
    uint32_t valuCount = 0;
    uint32_t nativeBf16Count = 0;

    while (std::getline(iss, line)) {
      size_t start = line.find_first_not_of(" \t\r");
      if (start == std::string::npos || line[start] == ';' ||
          line[start] == '#' ||
          (start + 1 < line.size() && line[start] == '/' &&
           line[start + 1] == '/')) {
        continue;
      }
      size_t end = line.find_first_of(" \t\r", start);
      std::string mnemonic = line.substr(start, end - start);
      if (mnemonic.empty() || mnemonic.back() == ':') {
        continue;
      }
      std::transform(mnemonic.begin(), mnemonic.end(), mnemonic.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (mnemonic.rfind("v_", 0) == 0) {
        ++valuCount;
        if (mnemonic.find("bf16") != std::string::npos) {
          ++nativeBf16Count;
        }
      }
    }

    if (properties.vendorID == 0x1002 && valuCount >= 8) {
      return (nativeBf16Count == 0);
    }
  }

  return false;
}

void VulkanContext::dispatchIndirect(ComputeKernel kernel_handle,
                                     ComputeBuffer indirectBuffer,
                                     VkDeviceSize offset) {
  auto *vulkanKernel = getKernel(kernel_handle);
  if (!vulkanKernel)
    return;
  lastDispatchedKernel = kernel_handle;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (!m_inBatch || !m_batchCmdRecording) {
    if (frame.inUse) {
      constexpr uint64_t kTimeoutNs = 30'000'000'000ULL;
      VkResult waitResult =
          vkWaitForFences(device, 1, &frame.fence, VK_TRUE, kTimeoutNs);
      if (waitResult == VK_TIMEOUT) {
        throw std::runtime_error(
            "GPU dispatch timed out (>30 s) — aborting benchmark to prevent amdgpu TDR crash.");
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
    maybeBeginPerfQuery(frame.commandBuffer);
    maybeBeginTiming(frame.commandBuffer);
    if (m_inBatch) {
      m_batchCmdRecording = true;
      m_batchDispatchCount = 0;
    } else {
      cmdPipelineMemoryBarrier2(
          frame.commandBuffer,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
              VK_ACCESS_2_SHADER_READ_BIT |
              VK_ACCESS_2_SHADER_WRITE_BIT);
    }
  } else if (m_batchDispatchCount > 0 && m_batchInterDispatchBarrier) {
    cmdPipelineMemoryBarrier2(
        frame.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_2_SHADER_READ_BIT |
            VK_ACCESS_2_SHADER_WRITE_BIT);
  }

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

  if (m_inBatch) {
    m_batchDispatchCount++;
    return;
  }

  maybeEndPerfQuery(frame.commandBuffer);
  maybeEndTiming(frame.commandBuffer);
  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkPerformanceQuerySubmitInfoKHR perfSubmitInfo{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
  perfSubmitInfo.counterPassIndex = 0;
  if (perfQueryNeedsSubmitInfo) {
    perfSubmitInfo.pNext = submitInfo.pNext;
    submitInfo.pNext = &perfSubmitInfo;
    perfQueryNeedsSubmitInfo = false;
  }

  VkResult indSubmitRes = VK_SUCCESS;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    indSubmitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  }
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
  auto *vulkanKernel = getKernel(kernel_handle);
  if (!vulkanKernel || entries.empty())
    return;
  lastDispatchedKernel = kernel_handle;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (!m_inBatch || !m_batchCmdRecording) {
    if (frame.inUse) {
      constexpr uint64_t kTimeoutNs = 30'000'000'000ULL;
      VkResult waitResult =
          vkWaitForFences(device, 1, &frame.fence, VK_TRUE, kTimeoutNs);
      if (waitResult == VK_TIMEOUT) {
        throw std::runtime_error(
            "GPU dispatch timed out (>30 s) — aborting benchmark to prevent amdgpu TDR crash.");
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
    maybeBeginPerfQuery(frame.commandBuffer);
    maybeBeginTiming(frame.commandBuffer);
    if (m_inBatch) {
      m_batchCmdRecording = true;
      m_batchDispatchCount = 0;
    }
  } else if (m_batchDispatchCount > 0 && m_batchInterDispatchBarrier) {
    cmdPipelineMemoryBarrier2(
        frame.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_2_SHADER_READ_BIT |
            VK_ACCESS_2_SHADER_WRITE_BIT);
  }

  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    vulkanKernel->pipeline);
  vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkanKernel->pipelineLayout, 0, 1,
                          &vulkanKernel->descriptorSet, 0, nullptr);

  VkBuffer vkIndirect = getVkBuffer(indirectBuffer);
  VulkanKernel *lastBound = vulkanKernel;

  for (const auto &entry : entries) {
    VulkanKernel *kToBind = entry.specializedKernel ? getKernel(entry.specializedKernel) : vulkanKernel;
    if (!kToBind) kToBind = vulkanKernel;
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

  if (m_inBatch) {
    m_batchDispatchCount++;
    return;
  }

  maybeEndPerfQuery(frame.commandBuffer);
  maybeEndTiming(frame.commandBuffer);
  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkPerformanceQuerySubmitInfoKHR perfSubmitInfo{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
  perfSubmitInfo.counterPassIndex = 0;
  if (perfQueryNeedsSubmitInfo) {
    perfSubmitInfo.pNext = submitInfo.pNext;
    submitInfo.pNext = &perfSubmitInfo;
    perfQueryNeedsSubmitInfo = false;
  }

  VkResult indSeqSubmitRes = VK_SUCCESS;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    indSeqSubmitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  }
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
  auto *classifyKernel = getKernel(classifyKernel_handle);
  auto *secondKernel = getKernel(secondKernel_handle);
  if (!classifyKernel || !secondKernel)
    return;
  lastDispatchedKernel = secondKernel_handle;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (!m_inBatch || !m_batchCmdRecording) {
    if (frame.inUse) {
      constexpr uint64_t kTimeoutNs = 30'000'000'000ULL;
      VkResult waitResult =
          vkWaitForFences(device, 1, &frame.fence, VK_TRUE, kTimeoutNs);
      if (waitResult == VK_TIMEOUT) {
        throw std::runtime_error(
            "GPU dispatch timed out (>30 s) — aborting benchmark to prevent amdgpu TDR crash.");
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
    maybeBeginPerfQuery(frame.commandBuffer);
    maybeBeginTiming(frame.commandBuffer);
    if (m_inBatch) {
      m_batchCmdRecording = true;
      m_batchDispatchCount = 0;
    }
  } else if (m_batchDispatchCount > 0) {
    cmdPipelineMemoryBarrier2(
        frame.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT |
            VK_ACCESS_2_SHADER_WRITE_BIT);
  }

  // 1. Reset queue counters & indirect commands on compute queue (zero transfer bubbles)
  // Note: resolveKernel resets queueCounters[0..15] = 0 after snapshotting on every pass,
  // so subsequent iterations in a batch do not need a redundant resetKernel dispatch + barrier.
  if (resetKernel_handle && (!m_inBatch || m_batchDispatchCount == 0 || !resolveKernel_handle)) {
    auto *resetKernel = getKernel(resetKernel_handle);
    if (resetKernel) {
      vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resetKernel->pipeline);
      vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              resetKernel->pipelineLayout, 0, 1,
                              &resetKernel->descriptorSet, 0, nullptr);
      vkCmdDispatch(frame.commandBuffer, 1, 1, 1);

      cmdPipelineMemoryBarrier2(
          frame.commandBuffer,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
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
    auto *resolveKernel = getKernel(resolveKernel_handle);
    if (resolveKernel) {
      cmdPipelineMemoryBarrier2(
          frame.commandBuffer,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

      vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resolveKernel->pipeline);
      vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              resolveKernel->pipelineLayout, 0, 1,
                              &resolveKernel->descriptorSet, 0, nullptr);
      uint32_t resetQueue = (isPingPong && entries.size() > 1) ? 1u : 0xFFFFFFFFu;
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

      cmdPipelineMemoryBarrier2(
          frame.commandBuffer,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
          VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    }
  } else {
    cmdPipelineMemoryBarrier2(
        frame.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
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
      uint32_t maxSeq = dgcInfo->maxSequenceCount;
      if (dgcMode == 2u || dgcMode == 3u || dgcMode == 5u || dgcMode == 7u) {
        maxSeq = std::min(maxSeq, 1u);
      } else if (dgcMode == 4u) {
        maxSeq = std::min(maxSeq, 3u);
      } else if (dgcMode == 1u || dgcMode == 6u) {
        maxSeq = std::min(maxSeq, 8u);
      }
      genCmds.maxSequenceCount = maxSeq;
      genCmds.sequenceCountAddress = dgcInfo->sequenceCountBuffer ?
          (getBufferDeviceAddress(dgcInfo->sequenceCountBuffer) + dgcInfo->sequenceCountBufferOffset) : 0;
      genCmds.maxDrawCount = 0;

      vkCmdExecuteGeneratedCommandsEXT_ptr(frame.commandBuffer, VK_FALSE, &genCmds);
    } else {
      // Multi-bounce ping-pong compaction via DGC
      VulkanKernel *lastBound = secondKernel;
      for (size_t e = 0; e < entries.size(); ++e) {
        const auto &entry = entries[e];
        VulkanKernel *kToBind = entry.specializedKernel ? getKernel(entry.specializedKernel) : secondKernel;
        if (!kToBind) kToBind = secondKernel;
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
        genCmds.indirectAddress = getBufferDeviceAddress(dgcInfo->sequenceBuffer) + dgcInfo->sequenceBufferOffset;
        genCmds.indirectAddressSize = sizeof(uint32_t) * 12;
        genCmds.preprocessAddress = getBufferDeviceAddress(dgcInfo->preprocessBuffer);
        genCmds.preprocessSize = dgcInfo->preprocessBufferSize;
        genCmds.maxSequenceCount = 1;
        genCmds.sequenceCountAddress = dgcInfo->sequenceCountBuffer ?
            (getBufferDeviceAddress(dgcInfo->sequenceCountBuffer) + dgcInfo->sequenceCountBufferOffset) : 0;
        genCmds.maxDrawCount = 0;

        vkCmdExecuteGeneratedCommandsEXT_ptr(frame.commandBuffer, VK_FALSE, &genCmds);

        if (e + 1 < entries.size() && resolveKernel_handle) {
          auto *resolveKernel = getKernel(resolveKernel_handle);
          if (resolveKernel) {
            cmdPipelineMemoryBarrier2(
                frame.commandBuffer,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

            vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resolveKernel->pipeline);
            vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    resolveKernel->pipelineLayout, 0, 1, &resolveKernel->descriptorSet, 0, nullptr);
            uint32_t nextBounce = static_cast<uint32_t>(e + 1);
            uint32_t resetQueue = (nextBounce + 1 < entries.size()) ? static_cast<uint32_t>((nextBounce + 1) % 2) : 0xFFFFFFFFu;
            struct { uint32_t resetQueue; uint32_t dgcMode; uint32_t bounceIndex; } rpcBounce{resetQueue, 3u, nextBounce};
            vkCmdPushConstants(frame.commandBuffer, resolveKernel->pipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpcBounce), &rpcBounce);
            vkCmdDispatch(frame.commandBuffer, 1, 1, 1);
            lastBound = nullptr;

            cmdPipelineMemoryBarrier2(
                frame.commandBuffer,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
          }
        }
      }
    }
  } else if (!useDGC) {
    // Legacy fallback path: CPU indirect command dispatch loop
    VkBuffer vkIndirect = getVkBuffer(indirectBuffer);
    VulkanKernel *lastBound = secondKernel;
    for (size_t e = 0; e < entries.size(); ++e) {
      const auto &entry = entries[e];
      VulkanKernel *kToBind = entry.specializedKernel ? getKernel(entry.specializedKernel) : secondKernel;
      if (!kToBind) kToBind = secondKernel;
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
        auto *resolveKernel = getKernel(resolveKernel_handle);
        if (resolveKernel) {
          cmdPipelineMemoryBarrier2(
              frame.commandBuffer,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

          vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resolveKernel->pipeline);
          vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  resolveKernel->pipelineLayout, 0, 1,
                                  &resolveKernel->descriptorSet, 0, nullptr);
          uint32_t nextBounce = static_cast<uint32_t>(e + 1);
          uint32_t resetQueue = (e + 2 < entries.size()) ? static_cast<uint32_t>(e % 2) : 0xFFFFFFFFu;
          struct { uint32_t resetQueue; uint32_t dgcMode; uint32_t bounceIndex; } rpcBounce{resetQueue, 0u, nextBounce};
          vkCmdPushConstants(frame.commandBuffer, resolveKernel->pipelineLayout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(rpcBounce), &rpcBounce);
          vkCmdDispatch(frame.commandBuffer, 1, 1, 1);

          cmdPipelineMemoryBarrier2(
              frame.commandBuffer,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
              VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

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

  if (m_inBatch) {
    m_batchDispatchCount++;
    return;
  }

  maybeEndPerfQuery(frame.commandBuffer);
  maybeEndTiming(frame.commandBuffer);
  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkPerformanceQuerySubmitInfoKHR perfSubmitInfo{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
  perfSubmitInfo.counterPassIndex = 0;
  if (perfQueryNeedsSubmitInfo) {
    perfSubmitInfo.pNext = submitInfo.pNext;
    submitInfo.pNext = &perfSubmitInfo;
    perfQueryNeedsSubmitInfo = false;
  }

  VkResult wlSubmitRes = VK_SUCCESS;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    wlSubmitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  }
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
  auto *vulkanKernel = getKernel(kernel_handle);
  if (!vulkanKernel || !vulkanKernel->isRTPipeline)
    return;
  lastDispatchedKernel = kernel_handle;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (!m_inBatch || !m_batchCmdRecording) {
    if (frame.inUse) {
      constexpr uint64_t kTimeoutNs = 30'000'000'000ULL;
      VkResult waitResult =
          vkWaitForFences(device, 1, &frame.fence, VK_TRUE, kTimeoutNs);
      if (waitResult == VK_TIMEOUT) {
        throw std::runtime_error(
            "GPU dispatch timed out (>30 s) — aborting benchmark to prevent amdgpu TDR crash.");
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
    maybeBeginPerfQuery(frame.commandBuffer);
    maybeBeginTiming(frame.commandBuffer);
    if (m_inBatch) {
      m_batchCmdRecording = true;
      m_batchDispatchCount = 0;
    }
  } else if (m_batchDispatchCount > 0 && m_batchInterDispatchBarrier) {
    cmdPipelineMemoryBarrier2(
        frame.commandBuffer,
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_2_SHADER_READ_BIT |
            VK_ACCESS_2_SHADER_WRITE_BIT);
  }

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
            VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR |
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

  if (m_inBatch) {
    m_batchDispatchCount++;
    return;
  }

  maybeEndPerfQuery(frame.commandBuffer);
  maybeEndTiming(frame.commandBuffer);
  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkPerformanceQuerySubmitInfoKHR perfSubmitInfo{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
  perfSubmitInfo.counterPassIndex = 0;
  if (perfQueryNeedsSubmitInfo) {
    perfSubmitInfo.pNext = submitInfo.pNext;
    submitInfo.pNext = &perfSubmitInfo;
    perfQueryNeedsSubmitInfo = false;
  }

  VkResult rtSubmitRes = VK_SUCCESS;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    rtSubmitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  }
  if (rtSubmitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatchRayTracingIndirect with result: " +
                             std::to_string(rtSubmitRes));
  }
  frame.inUse = true;

  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

VulkanContext::VulkanKernel *VulkanContext::getKernel(ComputeKernel handle) const {
  if (!handle) return nullptr;
  auto it = kernels.find(handle);
  return (it != kernels.end()) ? it->second : nullptr;
}

void VulkanContext::releaseKernel(ComputeKernel kernel) {
  if (lastDispatchedKernel == kernel) {
    lastDispatchedKernel = nullptr;
  }
  auto it = kernels.find(kernel);
  if (it != kernels.end()) {
    VulkanKernel *vulkanKernel = it->second;
    if (vulkanKernel) {
      if (vulkanKernel->pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, vulkanKernel->pipeline, nullptr);
        vulkanKernel->pipeline = VK_NULL_HANDLE;
      }
      if (vulkanKernel->pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
        vulkanKernel->pipelineLayout = VK_NULL_HANDLE;
      }
      if (vulkanKernel->descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout,
                                     nullptr);
        vulkanKernel->descriptorSetLayout = VK_NULL_HANDLE;
      }
      if (vulkanKernel->shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, vulkanKernel->shaderModule, nullptr);
        vulkanKernel->shaderModule = VK_NULL_HANDLE;
      }
      if (vulkanKernel->sbtBuffer) {
        releaseBuffer(vulkanKernel->sbtBuffer);
        vulkanKernel->sbtBuffer = nullptr;
      }
      if (vulkanKernel->descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, vulkanKernel->descriptorPool, nullptr);
        vulkanKernel->descriptorPool = VK_NULL_HANDLE;
      }
      delete vulkanKernel;
    }
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
  if (!quiet && !verbose && expectedKernelCount > 0) {
    printProgressBar(createdKernelCount, expectedKernelCount, file_name);
  }
}

void VulkanContext::printProgressBar(uint32_t current, uint32_t total,
                                     const std::string &kernel_name) {
  const int barWidth = 28;
  float progress = std::clamp(static_cast<float>(current) / std::max(1u, total), 0.0f, 1.0f);
  int pos = std::clamp(static_cast<int>(barWidth * progress), 0, barWidth);

  std::string short_name = kernel_name;
  size_t last_slash = kernel_name.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    short_name = kernel_name.substr(last_slash + 1);
  }

  std::cout << "\r\033[K  Progress: [";
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "━";
    else if (i == pos && pos < barWidth)
      std::cout << "╸";
    else
      std::cout << " ";
  }
  std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% (" << std::min(current, total) << "/" << total
            << ") Compiling " << short_name
            << (current >= total ? "\n" : "") << std::flush;
}

ComputeKernel VulkanContext::createRTPipeline(
    const std::string &rgen_path, const std::string &rmiss_path,
    const std::vector<std::string> &rchit_paths,
    const std::vector<std::string> &rahit_paths,
    const std::vector<std::string> &rint_paths, uint32_t num_buffer_args,
    uint32_t spec_const0, bool has_spec_const0) {

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

  auto cleanupOnError = [&]() {
    for (auto m : modules) {
      if (m != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, m, nullptr);
      }
    }
    if (vulkanKernel->sbtBuffer) {
      releaseBuffer(vulkanKernel->sbtBuffer);
      vulkanKernel->sbtBuffer = nullptr;
    }
    if (vulkanKernel->pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, vulkanKernel->pipeline, nullptr);
    }
    if (vulkanKernel->pipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, vulkanKernel->pipelineLayout, nullptr);
    }
    if (vulkanKernel->descriptorPool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device, vulkanKernel->descriptorPool, nullptr);
    }
    if (vulkanKernel->descriptorSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, vulkanKernel->descriptorSetLayout, nullptr);
    }
    delete vulkanKernel;
  };

  VkSpecializationMapEntry specEntry{};
  specEntry.constantID = 0;
  specEntry.offset = 0;
  specEntry.size = sizeof(uint32_t);

  VkSpecializationInfo specInfo{};
  specInfo.mapEntryCount = 1;
  specInfo.pMapEntries = &specEntry;
  specInfo.dataSize = sizeof(uint32_t);
  specInfo.pData = &spec_const0;

  auto add_stage = [&](VkShaderModule module, VkShaderStageFlagBits stage) {
    VkPipelineShaderStageCreateInfo info{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage = stage;
    info.module = module;
    info.pName = "main";
    if (has_spec_const0) {
      info.pSpecializationInfo = &specInfo;
    }
    stages.push_back(info);
  };

  try {
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
  } catch (...) {
    cleanupOnError();
    throw;
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
  if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
                                  &vulkanKernel->descriptorSetLayout) != VK_SUCCESS) {
    cleanupOnError();
    throw std::runtime_error("Failed to create RT descriptor set layout");
  }

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
  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                             &vulkanKernel->pipelineLayout) != VK_SUCCESS) {
    cleanupOnError();
    throw std::runtime_error("Failed to create RT pipeline layout");
  }

  // Pipeline execution
  VkRayTracingPipelineCreateInfoKHR pipelineInfo{
      VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  if (pipelineExecutablePropertiesSupported) {
    pipelineInfo.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  }
  if (opacityMicromapSupported) {
    pipelineInfo.flags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
  }
  pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
  pipelineInfo.pStages = stages.data();
  pipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
  pipelineInfo.pGroups = groups.data();
  pipelineInfo.maxPipelineRayRecursionDepth = 1;
  pipelineInfo.layout = vulkanKernel->pipelineLayout;

  auto pfnCreateRays = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(
      device, "vkCreateRayTracingPipelinesKHR");
  if (!pfnCreateRays ||
      pfnCreateRays(device, VK_NULL_HANDLE, m_pipelineCache, 1, &pipelineInfo,
                    nullptr, &vulkanKernel->pipeline) != VK_SUCCESS) {
    cleanupOnError();
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
  uint32_t handleAlign =
      rtProps.shaderGroupHandleAlignment > 0 ? rtProps.shaderGroupHandleAlignment : 1u;
  uint32_t baseAlign =
      rtProps.shaderGroupBaseAlignment > 0 ? rtProps.shaderGroupBaseAlignment : handleAlign;
  uint32_t handleSizeAligned = (handleSize + handleAlign - 1u) & ~(handleAlign - 1u);
  uint32_t groupCount = static_cast<uint32_t>(groups.size());
  uint32_t hitGroupCount = (groupCount > 2u) ? (groupCount - 2u) : 0u;

  uint32_t rgenOffset = 0u;
  uint32_t missOffset = (rgenOffset + handleSizeAligned + baseAlign - 1u) & ~(baseAlign - 1u);
  uint32_t hitOffset = (missOffset + handleSizeAligned + baseAlign - 1u) & ~(baseAlign - 1u);
  uint32_t sbtPayloadSize = hitOffset + hitGroupCount * handleSizeAligned;

  std::vector<uint8_t> handles(groupCount * handleSize);
  auto pfnGetHandles =
      (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(
          device, "vkGetRayTracingShaderGroupHandlesKHR");
  if (!pfnGetHandles ||
      pfnGetHandles(device, vulkanKernel->pipeline, 0, groupCount, handles.size(),
                    handles.data()) != VK_SUCCESS) {
    cleanupOnError();
    throw std::runtime_error("Failed to get RT shader group handles");
  }

  vulkanKernel->sbtBuffer = createBuffer(sbtPayloadSize + baseAlign);
  VkDeviceAddress rawSbtAddr = getBufferDeviceAddress(vulkanKernel->sbtBuffer);
  VkDeviceAddress sbtAddr =
      (rawSbtAddr + static_cast<VkDeviceAddress>(baseAlign) - 1ULL) &
      ~(static_cast<VkDeviceAddress>(baseAlign) - 1ULL);
  size_t sbtPadOffset = static_cast<size_t>(sbtAddr - rawSbtAddr);

  // Upload handles at baseAlign-aligned region boundaries
  std::vector<uint8_t> sbtData(sbtPayloadSize, 0);
  if (groupCount > 0u) {
    memcpy(sbtData.data() + rgenOffset, handles.data(), handleSize);
  }
  if (groupCount > 1u) {
    memcpy(sbtData.data() + missOffset, handles.data() + handleSize, handleSize);
  }
  for (uint32_t i = 2u; i < groupCount; ++i) {
    memcpy(sbtData.data() + hitOffset + (i - 2u) * handleSizeAligned,
           handles.data() + i * handleSize, handleSize);
  }
  writeBuffer(vulkanKernel->sbtBuffer, sbtPadOffset, sbtPayloadSize, sbtData.data());

  // Regions (each deviceAddress aligned to shaderGroupBaseAlignment)
  // 0: rgen, 1: miss, 2..N: hit groups
  vulkanKernel->rgenRegion.deviceAddress = sbtAddr + rgenOffset;
  vulkanKernel->rgenRegion.stride = handleSizeAligned;
  vulkanKernel->rgenRegion.size = handleSizeAligned;

  vulkanKernel->missRegion.deviceAddress = sbtAddr + missOffset;
  vulkanKernel->missRegion.stride = handleSizeAligned;
  vulkanKernel->missRegion.size = handleSizeAligned;

  if (hitGroupCount > 0u) {
    vulkanKernel->hitRegion.deviceAddress = sbtAddr + hitOffset;
    vulkanKernel->hitRegion.stride = handleSizeAligned;
    vulkanKernel->hitRegion.size = hitGroupCount * handleSizeAligned;
  } else {
    vulkanKernel->hitRegion.deviceAddress = 0;
    vulkanKernel->hitRegion.stride = 0;
    vulkanKernel->hitRegion.size = 0;
  }

  vulkanKernel->callRegion.deviceAddress = 0;
  vulkanKernel->callRegion.stride = 0;
  vulkanKernel->callRegion.size = 0;

  // Cleanup modules
  for (auto m : modules) {
    vkDestroyShaderModule(device, m, nullptr);
  }
  modules.clear();

  // Allocate Descriptor Sets
  std::vector<VkDescriptorPoolSize> poolSizes = {
      {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1}};
  if (num_buffer_args > 1) {
    poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, num_buffer_args - 1});
  }
  VkDescriptorPoolCreateInfo poolInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = 1;
  if (vkCreateDescriptorPool(device, &poolInfo, nullptr,
                             &vulkanKernel->descriptorPool) != VK_SUCCESS) {
    cleanupOnError();
    throw std::runtime_error("Failed to create RT descriptor pool");
  }

  VkDescriptorSetAllocateInfo allocInfo{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool = vulkanKernel->descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &vulkanKernel->descriptorSetLayout;
  if (vkAllocateDescriptorSets(device, &allocInfo, &vulkanKernel->descriptorSet) != VK_SUCCESS) {
    cleanupOnError();
    throw std::runtime_error("Failed to allocate RT descriptor sets");
  }

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
  std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
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
  bufferInfo.size = std::max<size_t>(size, 256);
  bufferInfo.usage = 0;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  bool allocatedViaVma = false;
  if (m_vmaAllocator != VK_NULL_HANDLE) {
    VmaAllocationCreateInfo vmaAllocCi{};
    vmaAllocCi.usage = maintenance5Supported
                           ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
                           : VMA_MEMORY_USAGE_UNKNOWN;
    vmaAllocCi.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VmaAllocationInfo vmaInfo{};
    if (vmaCreateBuffer(m_vmaAllocator, &bufferInfo, &vmaAllocCi,
                        &vulkanBuffer->buffer, &vulkanBuffer->vmaAllocation,
                        &vmaInfo) == VK_SUCCESS) {
      vulkanBuffer->size = vmaInfo.size;
      allocatedViaVma = true;
    }
  }
  if (!allocatedViaVma) {
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &vulkanBuffer->buffer) != VK_SUCCESS) {
      delete vulkanBuffer;
      throw std::runtime_error("failed to create preprocess buffer!");
    }
    VkMemoryRequirements memRequirements{};
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
      vkDestroyBuffer(device, vulkanBuffer->buffer, nullptr);
      delete vulkanBuffer;
      throw std::runtime_error("failed to allocate preprocess buffer memory!");
    }
    if (vkBindBufferMemory(device, vulkanBuffer->buffer, vulkanBuffer->memory, 0) != VK_SUCCESS) {
      vkDestroyBuffer(device, vulkanBuffer->buffer, nullptr);
      vkFreeMemory(device, vulkanBuffer->memory, nullptr);
      delete vulkanBuffer;
      throw std::runtime_error("failed to bind preprocess buffer memory!");
    }
    vulkanBuffer->size = memRequirements.size;
  }

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

  auto *classifyKernel = getKernel(classifyKernel_handle);
  auto *secondKernel = getKernel(secondKernel_handle);
  if (!classifyKernel || !secondKernel) return;
  lastDispatchedKernel = secondKernel_handle;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (!m_inBatch || !m_batchCmdRecording) {
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
    maybeBeginPerfQuery(frame.commandBuffer);
    maybeBeginTiming(frame.commandBuffer);
    if (m_inBatch) {
      m_batchCmdRecording = true;
      m_batchDispatchCount = 0;
    }
  } else if (m_batchDispatchCount > 0) {
    cmdPipelineMemoryBarrier2(
        frame.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT |
            VK_ACCESS_2_SHADER_WRITE_BIT);
  }

  // 1. Reset
  if (resetKernel_handle) {
    auto *resetKernel = getKernel(resetKernel_handle);
    if (resetKernel) {
      vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, resetKernel->pipeline);
      vkCmdBindDescriptorSets(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              resetKernel->pipelineLayout, 0, 1, &resetKernel->descriptorSet, 0, nullptr);
      vkCmdDispatch(frame.commandBuffer, 1, 1, 1);

      cmdPipelineMemoryBarrier2(
          frame.commandBuffer,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
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
  cmdPipelineMemoryBarrier2(
      frame.commandBuffer,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

  // 4. Resolve: GPU outputs DGC sequence items and dynamic sequence count
  if (resolveKernel_handle) {
    auto *resolveKernel = getKernel(resolveKernel_handle);
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
  cmdPipelineMemoryBarrier2(
      frame.commandBuffer,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);

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

  if (m_inBatch) {
    m_batchDispatchCount++;
    return;
  }

  maybeEndPerfQuery(frame.commandBuffer);
  maybeEndTiming(frame.commandBuffer);
  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkPerformanceQuerySubmitInfoKHR perfSubmitInfo{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
  perfSubmitInfo.counterPassIndex = 0;
  if (perfQueryNeedsSubmitInfo) {
    perfSubmitInfo.pNext = submitInfo.pNext;
    submitInfo.pNext = &perfSubmitInfo;
    perfQueryNeedsSubmitInfo = false;
  }

  VkResult submitRes = VK_SUCCESS;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  }
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

  auto *kernel = getKernel(kernel_handle);
  if (!kernel) return;
  lastDispatchedKernel = kernel_handle;

  auto &frame = inFlightFrames[currentFrameIndex];
  if (!m_inBatch || !m_batchCmdRecording) {
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
    maybeBeginPerfQuery(frame.commandBuffer);
    maybeBeginTiming(frame.commandBuffer);
    if (m_inBatch) {
      m_batchCmdRecording = true;
      m_batchDispatchCount = 0;
    }
  } else if (m_batchDispatchCount > 0 && m_batchInterDispatchBarrier) {
    cmdPipelineMemoryBarrier2(
        frame.commandBuffer,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT |
            VK_ACCESS_2_SHADER_WRITE_BIT);
  }

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

  if (m_inBatch) {
    m_batchDispatchCount++;
    return;
  }

  maybeEndPerfQuery(frame.commandBuffer);
  maybeEndTiming(frame.commandBuffer);
  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  VkPerformanceQuerySubmitInfoKHR perfSubmitInfo{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};
  perfSubmitInfo.counterPassIndex = 0;
  if (perfQueryNeedsSubmitInfo) {
    perfSubmitInfo.pNext = submitInfo.pNext;
    submitInfo.pNext = &perfSubmitInfo;
    perfQueryNeedsSubmitInfo = false;
  }

  VkResult submitRes = VK_SUCCESS;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    submitRes = vkQueueSubmit(computeQueue, 1, &submitInfo, frame.fence);
  }
  if (submitRes != VK_SUCCESS) {
    throw std::runtime_error("vkQueueSubmit failed in dispatchDGCSequence: " +
                             std::to_string(submitRes));
  }
  frame.inUse = true;
  currentFrameIndex = (currentFrameIndex + 1) % kMaxInFlight;
}

void VulkanContext::cmdPipelineMemoryBarrier2(
    VkCommandBuffer cmd,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask) {
  if (sync2Supported && vkCmdPipelineBarrier2KHR_ptr) {
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2KHR_ptr(cmd, &depInfo);
  } else {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = static_cast<VkAccessFlags>(srcAccessMask);
    barrier.dstAccessMask = static_cast<VkAccessFlags>(dstAccessMask);
    vkCmdPipelineBarrier(
        cmd,
        static_cast<VkPipelineStageFlags>(srcStageMask),
        static_cast<VkPipelineStageFlags>(dstStageMask),
        0, 1, &barrier, 0, nullptr, 0, nullptr);
  }
}

bool VulkanContext::captureBufferToSharedPreviewImage(
    const std::string &key, ComputeBuffer hdrBufA, ComputeBuffer hdrBufB,
    uint32_t width, uint32_t height, uint32_t mode,
    const std::string &kernelDir) {
  if (device == VK_NULL_HANDLE || m_vmaAllocator == VK_NULL_HANDLE ||
      !hdrBufA || width == 0 || height == 0) {
    return false;
  }

  VkBuffer vkBufA = getVkBuffer(hdrBufA);
  VkBuffer vkBufB = hdrBufB ? getVkBuffer(hdrBufB) : vkBufA;
  if (vkBufA == VK_NULL_HANDLE) {
    return false;
  }

  std::vector<std::string> candidatePaths = {
      kernelDir + "/vulkan/tonemap_preview.comp.spv",
      kernelDir + "/tonemap_preview.comp.spv",
      kernelDir + "/vulkan/tonemap_preview.spv",
      kernelDir + "/tonemap_preview.spv",
      "kernels/vulkan/tonemap_preview.comp.spv",
      "kernels/vulkan/tonemap_preview.spv",
  };
  std::string spvPath;
  for (const auto &cand : candidatePaths) {
    std::ifstream probe(cand, std::ios::binary);
    if (probe.is_open()) {
      spvPath = cand;
      break;
    }
  }
  if (spvPath.empty()) {
    return false;
  }

  std::ifstream spvFile(spvPath, std::ios::ate | std::ios::binary);
  if (!spvFile.is_open()) {
    return false;
  }
  size_t codeBytes = static_cast<size_t>(spvFile.tellg());
  if (codeBytes == 0 || (codeBytes % 4) != 0) {
    return false;
  }
  std::vector<uint32_t> spvWords(codeBytes / 4);
  spvFile.seekg(0);
  spvFile.read(reinterpret_cast<char *>(spvWords.data()), codeBytes);
  spvFile.close();

  SharedPreviewImage preview{};
  {
    std::lock_guard<std::mutex> plock(m_sharedPreviewMutex);
    auto it = m_sharedPreviewImages.find(key);
    if (it != m_sharedPreviewImages.end()) {
      if (it->second.width == width && it->second.height == height &&
          it->second.image != VK_NULL_HANDLE &&
          it->second.view != VK_NULL_HANDLE) {
        preview = it->second;
      } else {
        waitIdle();
        if (it->second.view != VK_NULL_HANDLE) {
          vkDestroyImageView(device, it->second.view, nullptr);
        }
        if (it->second.image != VK_NULL_HANDLE &&
            it->second.allocation != VK_NULL_HANDLE) {
          vmaDestroyImage(m_vmaAllocator, it->second.image,
                          it->second.allocation);
        }
        m_sharedPreviewImages.erase(it);
      }
    }
  }

  if (preview.image == VK_NULL_HANDLE) {
    VkImageCreateInfo imgCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgCi.imageType = VK_IMAGE_TYPE_2D;
    imgCi.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgCi.extent = {width, height, 1};
    imgCi.mipLevels = 1;
    imgCi.arrayLayers = 1;
    imgCi.samples = VK_SAMPLE_COUNT_1_BIT;
    imgCi.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCi.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo vmaCi{};
    vmaCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(m_vmaAllocator, &imgCi, &vmaCi, &preview.image,
                       &preview.allocation, nullptr) != VK_SUCCESS) {
      return false;
    }

    VkImageViewCreateInfo viewCi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewCi.image = preview.image;
    viewCi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCi.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewCi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCi.subresourceRange.baseMipLevel = 0;
    viewCi.subresourceRange.levelCount = 1;
    viewCi.subresourceRange.baseArrayLayer = 0;
    viewCi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &viewCi, nullptr, &preview.view) !=
        VK_SUCCESS) {
      vmaDestroyImage(m_vmaAllocator, preview.image, preview.allocation);
      return false;
    }
    preview.width = width;
    preview.height = height;
  }

  VkShaderModuleCreateInfo smCi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smCi.codeSize = codeBytes;
  smCi.pCode = spvWords.data();
  VkShaderModule shaderMod = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &smCi, nullptr, &shaderMod) != VK_SUCCESS) {
    return false;
  }

  VkDescriptorSetLayoutBinding bindings[3]{};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  bindings[2].binding = 2;
  bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  bindings[2].descriptorCount = 1;
  bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslCi{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslCi.bindingCount = 3;
  dslCi.pBindings = bindings;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  if (vkCreateDescriptorSetLayout(device, &dslCi, nullptr, &dsl) !=
      VK_SUCCESS) {
    vkDestroyShaderModule(device, shaderMod, nullptr);
    return false;
  }

  VkPushConstantRange pcRange{};
  pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcRange.offset = 0;
  pcRange.size = sizeof(uint32_t) * 3;

  VkPipelineLayoutCreateInfo plCi{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plCi.setLayoutCount = 1;
  plCi.pSetLayouts = &dsl;
  plCi.pushConstantRangeCount = 1;
  plCi.pPushConstantRanges = &pcRange;
  VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
  if (vkCreatePipelineLayout(device, &plCi, nullptr, &pipeLayout) !=
      VK_SUCCESS) {
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shaderMod, nullptr);
    return false;
  }

  VkComputePipelineCreateInfo pipeCi{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipeCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeCi.stage.module = shaderMod;
  pipeCi.stage.pName = "main";
  pipeCi.layout = pipeLayout;
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(device, m_pipelineCache, 1, &pipeCi, nullptr,
                               &pipeline) != VK_SUCCESS) {
    vkDestroyPipelineLayout(device, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shaderMod, nullptr);
    return false;
  }

  VkDescriptorPoolSize poolSizes[2]{
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
  };
  VkDescriptorPoolCreateInfo poolCi{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolCi.maxSets = 1;
  poolCi.poolSizeCount = 2;
  poolCi.pPoolSizes = poolSizes;
  VkDescriptorPool descPool = VK_NULL_HANDLE;
  if (vkCreateDescriptorPool(device, &poolCi, nullptr, &descPool) !=
      VK_SUCCESS) {
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shaderMod, nullptr);
    return false;
  }

  VkDescriptorSetAllocateInfo dsAi{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsAi.descriptorPool = descPool;
  dsAi.descriptorSetCount = 1;
  dsAi.pSetLayouts = &dsl;
  VkDescriptorSet descSet = VK_NULL_HANDLE;
  vkAllocateDescriptorSets(device, &dsAi, &descSet);

  VkDescriptorBufferInfo bufInfoA{vkBufA, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo bufInfoB{vkBufB, 0, VK_WHOLE_SIZE};
  VkDescriptorImageInfo imgInfo{VK_NULL_HANDLE, preview.view,
                                VK_IMAGE_LAYOUT_GENERAL};

  VkWriteDescriptorSet writes[3]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = descSet;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &bufInfoA;

  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = descSet;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &bufInfoB;

  writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[2].dstSet = descSet;
  writes[2].dstBinding = 2;
  writes[2].descriptorCount = 1;
  writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writes[2].pImageInfo = &imgInfo;
  vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

  waitIdle();
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo cmdAi{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdAi.commandPool = commandPool;
  cmdAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAi.commandBufferCount = 1;
  vkAllocateCommandBuffers(device, &cmdAi, &cmd);

  VkCommandBufferBeginInfo cmdBi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cmdBi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &cmdBi);

  VkImageMemoryBarrier toGen{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  toGen.srcAccessMask = 0;
  toGen.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  toGen.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  toGen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  toGen.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGen.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toGen.image = preview.image;
  toGen.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &toGen);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1,
                          &descSet, 0, nullptr);
  uint32_t pcVals[3] = {width, height, mode};
  vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(pcVals), pcVals);
  vkCmdDispatch(cmd, (width + 15u) / 16u, (height + 15u) / 16u, 1u);

  VkImageMemoryBarrier toRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toRead.image = preview.image;
  toRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &toRead);

  vkEndCommandBuffer(cmd);

  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fenceCi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(device, &fenceCi, nullptr, &fence);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
  }
  vkWaitForFences(device, 1, &fence, VK_TRUE, 30'000'000'000ULL);

  vkDestroyFence(device, fence, nullptr);
  vkFreeCommandBuffers(device, commandPool, 1, &cmd);
  vkDestroyDescriptorPool(device, descPool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, dsl, nullptr);
  vkDestroyShaderModule(device, shaderMod, nullptr);

  {
    std::lock_guard<std::mutex> plock(m_sharedPreviewMutex);
    preview.generation = ++m_previewGenerationCounter;
    m_sharedPreviewImages[key] = preview;
  }
  return true;
}

bool VulkanContext::uploadRgbToSharedPreviewImage(const std::string &key,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  const uint8_t *rgbPixels) {
  if (device == VK_NULL_HANDLE || m_vmaAllocator == VK_NULL_HANDLE ||
      computeQueue == VK_NULL_HANDLE || !rgbPixels || width == 0 ||
      height == 0) {
    return false;
  }

  SharedPreviewImage preview{};
  SharedPreviewImage stale{};
  bool hasStale = false;
  bool reusedExisting = false;
  {
    std::lock_guard<std::mutex> plock(m_sharedPreviewMutex);
    auto it = m_sharedPreviewImages.find(key);
    if (it != m_sharedPreviewImages.end()) {
      if (it->second.width == width && it->second.height == height &&
          it->second.image != VK_NULL_HANDLE &&
          it->second.view != VK_NULL_HANDLE) {
        preview = it->second;
        reusedExisting = true;
      } else {
        stale = it->second;
        hasStale = true;
        m_sharedPreviewImages.erase(it);
      }
    }
  }

  if (hasStale) {
    waitIdle();
    if (stale.view != VK_NULL_HANDLE) {
      vkDestroyImageView(device, stale.view, nullptr);
    }
    if (stale.image != VK_NULL_HANDLE && stale.allocation != VK_NULL_HANDLE) {
      vmaDestroyImage(m_vmaAllocator, stale.image, stale.allocation);
    }
  } else if (reusedExisting) {
    waitIdle();
  }

  if (preview.image == VK_NULL_HANDLE) {
    VkImageCreateInfo imgCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgCi.imageType = VK_IMAGE_TYPE_2D;
    imgCi.format = VK_FORMAT_R8G8B8A8_UNORM;
    imgCi.extent = {width, height, 1};
    imgCi.mipLevels = 1;
    imgCi.arrayLayers = 1;
    imgCi.samples = VK_SAMPLE_COUNT_1_BIT;
    imgCi.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCi.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo vmaCi{};
    vmaCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(m_vmaAllocator, &imgCi, &vmaCi, &preview.image,
                       &preview.allocation, nullptr) != VK_SUCCESS) {
      return false;
    }

    VkImageViewCreateInfo viewCi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewCi.image = preview.image;
    viewCi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCi.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewCi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCi.subresourceRange.baseMipLevel = 0;
    viewCi.subresourceRange.levelCount = 1;
    viewCi.subresourceRange.baseArrayLayer = 0;
    viewCi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device, &viewCi, nullptr, &preview.view) !=
        VK_SUCCESS) {
      vmaDestroyImage(m_vmaAllocator, preview.image, preview.allocation);
      return false;
    }
    preview.width = width;
    preview.height = height;
  }

  auto cleanupNewPreviewOnFail = [&]() {
    if (!reusedExisting) {
      if (preview.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, preview.view, nullptr);
      }
      if (preview.image != VK_NULL_HANDLE &&
          preview.allocation != VK_NULL_HANDLE) {
        vmaDestroyImage(m_vmaAllocator, preview.image, preview.allocation);
      }
    }
  };

  const size_t pixelCount = static_cast<size_t>(width) * height;
  const VkDeviceSize stagingBytes = static_cast<VkDeviceSize>(pixelCount) * 4;

  VkBufferCreateInfo bufCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufCi.size = stagingBytes;
  bufCi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bufCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo stagingAllocCi{};
  stagingAllocCi.usage = VMA_MEMORY_USAGE_AUTO;
  stagingAllocCi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                         VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VkBuffer stagingBuf = VK_NULL_HANDLE;
  VmaAllocation stagingAlloc = VK_NULL_HANDLE;
  VmaAllocationInfo stagingInfo{};
  if (vmaCreateBuffer(m_vmaAllocator, &bufCi, &stagingAllocCi, &stagingBuf,
                      &stagingAlloc, &stagingInfo) != VK_SUCCESS ||
      !stagingInfo.pMappedData) {
    if (stagingBuf != VK_NULL_HANDLE && stagingAlloc != VK_NULL_HANDLE) {
      vmaDestroyBuffer(m_vmaAllocator, stagingBuf, stagingAlloc);
    }
    cleanupNewPreviewOnFail();
    return false;
  }

  uint8_t *dst = static_cast<uint8_t *>(stagingInfo.pMappedData);
  for (size_t i = 0; i < pixelCount; ++i) {
    dst[i * 4 + 0] = rgbPixels[i * 3 + 0];
    dst[i * 4 + 1] = rgbPixels[i * 3 + 1];
    dst[i * 4 + 2] = rgbPixels[i * 3 + 2];
    dst[i * 4 + 3] = 255u;
  }
  vmaFlushAllocation(m_vmaAllocator, stagingAlloc, 0, stagingBytes);

  VkCommandPool transientPool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolCi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  poolCi.queueFamilyIndex = computeQueueFamilyIndex;
  if (vkCreateCommandPool(device, &poolCi, nullptr, &transientPool) !=
      VK_SUCCESS) {
    vmaDestroyBuffer(m_vmaAllocator, stagingBuf, stagingAlloc);
    cleanupNewPreviewOnFail();
    return false;
  }

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo cmdAi{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdAi.commandPool = transientPool;
  cmdAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAi.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(device, &cmdAi, &cmd) != VK_SUCCESS) {
    vkDestroyCommandPool(device, transientPool, nullptr);
    vmaDestroyBuffer(m_vmaAllocator, stagingBuf, stagingAlloc);
    cleanupNewPreviewOnFail();
    return false;
  }

  VkCommandBufferBeginInfo cmdBi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cmdBi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &cmdBi);

  VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toTransfer.image = preview.image;
  toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  toTransfer.srcAccessMask = 0;
  toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                       1, &toTransfer);

  VkBufferImageCopy copyRegion{};
  copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  copyRegion.imageExtent = {width, height, 1};
  vkCmdCopyBufferToImage(cmd, stagingBuf, preview.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

  VkImageMemoryBarrier toRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  toRead.image = preview.image;
  toRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &toRead);

  vkEndCommandBuffer(cmd);

  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fenceCi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(device, &fenceCi, nullptr, &fence) != VK_SUCCESS) {
    vkDestroyCommandPool(device, transientPool, nullptr);
    vmaDestroyBuffer(m_vmaAllocator, stagingBuf, stagingAlloc);
    cleanupNewPreviewOnFail();
    return false;
  }

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  VkResult subRes = VK_SUCCESS;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_queueMutex);
    subRes = vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
  }
  bool ok = (subRes == VK_SUCCESS) &&
            (vkWaitForFences(device, 1, &fence, VK_TRUE, 30'000'000'000ULL) ==
             VK_SUCCESS);

  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, transientPool, nullptr);
  vmaDestroyBuffer(m_vmaAllocator, stagingBuf, stagingAlloc);

  if (!ok) {
    cleanupNewPreviewOnFail();
    return false;
  }

  {
    std::lock_guard<std::mutex> plock(m_sharedPreviewMutex);
    preview.generation = ++m_previewGenerationCounter;
    m_sharedPreviewImages[key] = preview;
  }
  return true;
}

bool VulkanContext::getSharedPreviewImage(const std::string &key,
                                          SharedPreviewImage &outInfo) const {
  std::lock_guard<std::mutex> plock(m_sharedPreviewMutex);
  auto it = m_sharedPreviewImages.find(key);
  if (it == m_sharedPreviewImages.end()) {
    return false;
  }
  outInfo = it->second;
  return true;
}

void VulkanContext::clearSharedPreviewImages() {
  std::lock_guard<std::mutex> plock(m_sharedPreviewMutex);
  if (device != VK_NULL_HANDLE) {
    for (auto &kv : m_sharedPreviewImages) {
      if (kv.second.view != VK_NULL_HANDLE) {
        vkDestroyImageView(device, kv.second.view, nullptr);
      }
      if (kv.second.image != VK_NULL_HANDLE &&
          kv.second.allocation != VK_NULL_HANDLE &&
          m_vmaAllocator != VK_NULL_HANDLE) {
        vmaDestroyImage(m_vmaAllocator, kv.second.image, kv.second.allocation);
      }
    }
  }
  m_sharedPreviewImages.clear();
  m_inMemoryJsons.clear();
}

void VulkanContext::storeInMemoryJson(const std::string &key,
                                      const std::string &jsonContent) {
  std::lock_guard<std::mutex> plock(m_sharedPreviewMutex);
  m_inMemoryJsons[key] = jsonContent;
}

bool VulkanContext::getInMemoryJson(const std::string &key,
                                    std::string &outJsonContent) const {
  std::lock_guard<std::mutex> plock(m_sharedPreviewMutex);
  auto it = m_inMemoryJsons.find(key);
  if (it == m_inMemoryJsons.end()) {
    return false;
  }
  outJsonContent = it->second;
  return true;
}


