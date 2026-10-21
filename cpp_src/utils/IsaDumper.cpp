#include "utils/IsaDumper.h"
#include "core/DeviceDatabase.h"
#include "utils/KernelPath.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace fs = std::filesystem;

namespace {

struct KernelArtifact {
  std::string benchmarkName;
  std::string configName;
  int configIndex = 0;
  std::string backend; // "vulkan", "rocm", "opencl"
  fs::path filePath;
  std::string rocmSymbol;
  std::string skipReason;
};

static std::string normalizeKey(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c != ' ' && c != '_' && c != '-' && c != '(' && c != ')' && c != '&') {
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return out;
}

struct BenchmarkMapping {
  std::string benchName;
  std::vector<std::string> aliases;
  int configIndex;
  std::string configLabel;
  std::string vulkanSpv;
  std::string rocmCo;
  std::string openclCl;
  std::string rocmSymbol = "";
};

static std::vector<BenchmarkMapping> getBenchmarkMappings() {
  return {
      {"BF16", {"bf16"}, 0, "BF16 (Vector)", "bf16.comp.spv", "bf16.co", ""},
      {"BF16", {"bf16", "bf16matrix", "coopmatrixbf16"}, 1, "BF16 (Matrix)", "coop_matrix_bf16.comp.spv", "bf16_matrix.co", ""},
      {"FP16", {"fp16"}, 0, "FP16 (Vector)", "fp16.comp.spv", "fp16.co", "fp16.cl"},
      {"FP16", {"fp16", "fp16matrix", "coopmatrixfp16"}, 1, "FP16 (Matrix)", "coop_matrix_fp16.comp.spv", "fp16_matrix.co", ""},
      {"FP32", {"fp32"}, 0, "FP32 (Vector)", "fp32.comp.spv", "fp32.co", "fp32.cl"},
      {"FP64", {"fp64"}, 0, "FP64 (Vector)", "fp64.comp.spv", "fp64.co", "fp64.cl"},
      {"FP8", {"fp8", "coopmatrixfp8"}, 0, "FP8 (Matrix)", "coop_matrix_fp8.comp.spv", "fp8_matrix.co", ""},
      {"FP6", {"fp6"}, 0, "FP6 (Vector)", "", "", ""},
      {"FP4", {"fp4"}, 0, "FP4 (Vector)", "", "", ""},
      {"INT8", {"int8"}, 0, "INT8 (Vector)", "int8.comp.spv", "int8.co", "int8.cl"},
      {"INT8", {"int8", "int8matrix", "coopmatrixint8"}, 1, "INT8 (Matrix)", "coop_matrix_int8.comp.spv", "int8_matrix.co", ""},
      {"INT4", {"int4", "int4matrix"}, 0, "INT4 (Matrix)", "", "int4_matrix.co", ""},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, 0, "FP32 ILP4", "dual_issue_ilp4.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_ilp4"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, 1, "FP32 ILP8", "dual_issue_ilp8.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_ilp8"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, 2, "FP32 ILP16", "dual_issue_ilp16.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_ilp16"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, 3, "INT32 ILP4", "dual_issue_int32_4.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_int32_4"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, 4, "INT32 ILP8", "dual_issue_int32_8.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_int32_8"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, 5, "INT32 ILP16", "dual_issue_int32.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_int32"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, 6, "Mixed FP32+INT32", "dual_issue_mixed.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_mixed"},
      {"LDS Bank Conflicts", {"lds", "ldsbankconflicts"}, 0, "LDS Bank Conflicts", "lds_bank_conflicts.comp.spv", "lds_bank_conflicts.co", "lds_bank_conflicts.cl"},
      {"In-Shader Indirect Synthesis", {"indirect", "inshaderindirect"}, 0, "Direct Dispatch", "direct_worker.comp.spv", "", ""},
      {"In-Shader Indirect Synthesis", {"indirect", "inshaderindirect"}, 1, "Indirect Classify", "indirect_classify.comp.spv", "", ""},
      {"In-Shader Indirect Synthesis", {"indirect", "inshaderindirect"}, 2, "Indirect Worker", "indirect_worker.comp.spv", "", ""},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, 0, "MemBW 128", "membw_128.comp.spv", "membw_128.co", "membw_128.cl"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, 1, "MemBW 256", "membw_256.comp.spv", "membw_256.co", "membw_256.cl"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, 2, "MemBW 1024", "membw_1024.comp.spv", "membw_1024.co", "membw_1024.cl"},
      {"L0 Vector Cache Latency", {"l0cache", "l0cachelatency", "cache"}, 0, "L0 Cache Latency", "l0_cache_latency.comp.spv", "l0_cache_latency.co", "l0_cache_latency.cl"},
      {"Cache Latency Curve", {"cachelatencycurve", "cachecurve"}, 0, "Cache Latency Curve", "cache_latency_curve.comp.spv", "cache_latency_curve.co", "cache_latency_curve.cl"},
      {"Pixel Fill Rate", {"pixelfill", "pixelfillrate", "raster", "rop"}, 0, "Pixel Fill Rate (Vertex + Fragment)", "pixel_fill.frag.spv", "", ""},
      {"RayRawTraversal", {"rayrawtraversal", "rawtraversal"}, 0, "Raw Traversal", "rt_raw_traversal.comp.spv", "", ""},
      {"RayIntersect", {"rayintersect"}, 0, "Ray Intersect", "rt_benchmark.comp.spv", "", ""},
      {"RayAnyHit", {"rayanyhit"}, 0, "Ray Any-Hit (RTP)", "rayanyhit_pipeline.rgen.spv", "", ""},
      {"RayDivergence", {"raydivergence"}, 0, "Ray Divergence (RTP)", "raydiv_pipeline.rgen.spv", "", ""},
      {"RayProcedural", {"rayprocedural"}, 0, "Ray Procedural AABB (RTP)", "rayprocedural.rgen.spv", "", ""},
      {"RayPayload", {"raypayload"}, 0, "Ray Payload 16B (RTP)", "raypayload_16b.rgen.spv", "", ""},
      {"RayPayload", {"raypayload"}, 1, "Ray Payload 128B (RTP)", "raypayload_128b.rgen.spv", "", ""},
      {"RayPayload", {"raypayload"}, 2, "Ray Payload 256B (RTP)", "raypayload_256b.rgen.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 0, "RT Scheduling — Compute Megakernel", "rt_scheduling_traditional_megakernel.comp.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 1, "RT Scheduling — DGC Classify", "rt_scheduling_device_generated_commands_classify.comp.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 2, "RT Scheduling — DGC Material", "rt_scheduling_device_generated_commands_material.comp.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 3, "RT Scheduling — DGC Bounce", "rt_scheduling_device_generated_commands_bounce.comp.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 4, "RT Scheduling — DGC Shadow", "rt_scheduling_device_generated_commands_shadow.comp.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 5, "RT Scheduling — DGC Multi-Light", "rt_scheduling_device_generated_commands_light.comp.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 6, "RT Scheduling — Persistent Threads", "rt_scheduling_persistent.comp.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 7, "RT Scheduling — RTP Baseline", "rt_scheduling.rgen.spv", "", ""},
      {"RayScheduling", {"rayscheduling", "rtscheduling"}, 8, "RT Scheduling — RTP + SER", "rt_scheduling_ser.rgen.spv", "", ""},
  };
}

static void printIsaTextSummary(const std::string &isaText, bool fullIsa) {
  std::istringstream iss(isaText);
  std::vector<std::string> lines;
  std::string line;
  std::map<std::string, int> opcodeHistogram;

  while (std::getline(iss, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.find("s_code_end") != std::string::npos)
      continue;
    lines.push_back(line);

    if (!line.empty() && (line[0] == '\t' || line[0] == ' ')) {
      size_t start = line.find_first_not_of(" \t");
      if (start != std::string::npos && line[start] != ';' && line[start] != '/' &&
          line[start] != '(') {
        size_t end = line.find_first_of(" \t", start);
        std::string mnemonic = line.substr(start, end - start);
        if (!mnemonic.empty() && mnemonic.back() != ':') {
          opcodeHistogram[mnemonic]++;
        }
      }
    }
  }

  if (!opcodeHistogram.empty()) {
    std::vector<std::pair<std::string, int>> sortedOps(opcodeHistogram.begin(),
                                                       opcodeHistogram.end());
    std::sort(sortedOps.begin(), sortedOps.end(),
              [](const auto &a, const auto &b) { return b.second < a.second; });

    std::cout << "  Instruction Mix (Top Opcodes):\n    ";
    int printed = 0;
    for (const auto &[op, count] : sortedOps) {
      if (op == "s_wait_alu" || op == "s_delay_alu" || op == "s_nop")
        continue;
      if (printed > 0 && printed % 3 == 0)
        std::cout << "\n    ";
      std::cout << std::left << std::setw(32) << (op + ": " + std::to_string(count));
      if (++printed >= 12)
        break;
    }
    std::cout << "\n\n";
  }

  if (fullIsa) {
    std::cout << "  Full Disassembly (" << lines.size() << " lines):\n";
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n";
    for (const auto &l : lines) {
      std::cout << "  " << l << "\n";
    }
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n\n";
    return;
  }

  // Locate hot loop: look for backward branch (s_cbranch_* jumping to an earlier label)
  std::map<std::string, int> labelLines;
  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    std::string trimmed = lines[i];
    while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\t'))
      trimmed.pop_back();
    if (!trimmed.empty() && trimmed.back() == ':') {
      if (trimmed[0] != '\t' && trimmed[0] != ' ') {
        labelLines[trimmed.substr(0, trimmed.size() - 1)] = i;
      }
      size_t lt = trimmed.find('<');
      size_t gt = trimmed.find('>', lt != std::string::npos ? lt : 0);
      if (lt != std::string::npos && gt != std::string::npos && gt > lt) {
        labelLines[trimmed.substr(lt, gt - lt + 1)] = i;
      }
    }
  }

  int loopStartIdx = -1;
  int loopEndIdx = -1;
  int maxLoopSpan = 0;
  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    if (lines[i].find("s_cbranch_") != std::string::npos) {
      // Extract target label token after s_cbranch_*
      size_t pos = lines[i].find("s_cbranch_");
      size_t spacePos = lines[i].find_first_of(" \t", pos);
      if (spacePos != std::string::npos) {
        size_t targetStart = lines[i].find_first_not_of(" \t", spacePos);
        if (targetStart != std::string::npos) {
          size_t targetEnd = lines[i].find_first_of(" \t;/", targetStart);
          std::string targetLabel = lines[i].substr(targetStart, targetEnd - targetStart);
          auto it = labelLines.find(targetLabel);
          if (it == labelLines.end()) {
            size_t lt = lines[i].find('<', targetEnd);
            size_t gt = lines[i].find('>', lt != std::string::npos ? lt : 0);
            if (lt != std::string::npos && gt != std::string::npos && gt > lt) {
              it = labelLines.find(lines[i].substr(lt, gt - lt + 1));
            }
          }
          if (it != labelLines.end() && it->second < i) {
            int span = i - it->second + 1;
            if (span >= maxLoopSpan) {
              maxLoopSpan = span;
              loopStartIdx = it->second;
              loopEndIdx = i;
            }
          }
        }
      }
    }
  }
  // Fallback for numeric relative offsets (e.g. llvm-objdump without symbol labels)
  if (loopStartIdx == -1) {
    for (int i = static_cast<int>(lines.size()) - 1; i >= 0; --i) {
      if (lines[i].find("s_cbranch_") != std::string::npos) {
        loopEndIdx = i;
        loopStartIdx = std::max(0, i - 80);
        break;
      }
    }
  }

  if (loopStartIdx != -1 && loopEndIdx != -1) {
    int totalLoopLines = loopEndIdx - loopStartIdx + 1;
    std::cout << "  Hot Loop Disassembly (lines " << (loopStartIdx + 1) << "–" << (loopEndIdx + 1)
              << ", " << totalLoopLines << " instructions — pass --full-isa for entire shader):\n";
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n";
    const int maxPreview = 110;
    if (totalLoopLines <= maxPreview) {
      for (int i = loopStartIdx; i <= loopEndIdx; ++i) {
        std::cout << "  " << lines[i] << "\n";
      }
    } else {
      for (int i = loopStartIdx; i < loopStartIdx + 50; ++i) {
        std::cout << "  " << lines[i] << "\n";
      }
      std::cout << "  \t... (" << (totalLoopLines - 70)
                << " instructions omitted; use --full-isa to view all) ...\n";
      for (int i = loopEndIdx - 19; i <= loopEndIdx; ++i) {
        std::cout << "  " << lines[i] << "\n";
      }
    }
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n\n";
  } else {
    std::cout << "  Disassembly Preview (" << lines.size() << " total lines):\n";
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n";
    int limit = std::min(static_cast<int>(lines.size()), 80);
    for (int i = 0; i < limit; ++i) {
      std::cout << "  " << lines[i] << "\n";
    }
    if (static_cast<int>(lines.size()) > limit) {
      std::cout << "  \t... (" << (lines.size() - limit)
                << " lines omitted; use --full-isa to view all) ...\n";
    }
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n\n";
  }
}

#ifdef HAVE_VULKAN
class NativeVulkanIsaInspector {
public:
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  std::string deviceName;
  std::string driverInfo;
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  bool subgroupSize32Supported = false;
  bool rtPipelineSupported = false;
  bool rayQuerySupported = false;
  bool serSupported = false;
  bool coopMatrixSupported = false;
  bool bf16DotProductSupported = false;
  bool bf16CoopMatrixSupported = false;
  bool fp8CoopMatrixSupported = false;
  bool fp16Supported = false;
  bool fp64Supported = false;
  bool int8Supported = false;

  PFN_vkGetPipelineExecutablePropertiesKHR pfnGetProps = nullptr;
  PFN_vkGetPipelineExecutableStatisticsKHR pfnGetStats = nullptr;
  PFN_vkGetPipelineExecutableInternalRepresentationsKHR pfnGetIrs = nullptr;
  PFN_vkCreateRayTracingPipelinesKHR pfnCreateRTPipelines = nullptr;

  bool init(uint32_t deviceIdx) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPUBench ISA Inspector";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "GPUBench";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) {
      std::cerr << "Error: Failed to create Vulkan instance for ISA inspection.\n";
      return false;
    }

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    if (devCount == 0) {
      std::cerr << "Error: No Vulkan physical devices found.\n";
      return false;
    }
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, devs.data());
    if (deviceIdx >= devCount) {
      deviceIdx = 0;
    }
    physicalDevice = devs[deviceIdx];

    VkPhysicalDeviceSubgroupSizeControlProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    VkPhysicalDeviceDriverProperties driverProps{};
    driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    driverProps.pNext = &subgroupProps;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &driverProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
    deviceName = props2.properties.deviceName;
    vendorId = props2.properties.vendorID;
    deviceId = props2.properties.deviceID;
    driverInfo = std::string(driverProps.driverName) + " (" + driverProps.driverInfo + ")";

    // Enumerate supported device extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availExts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, availExts.data());

    auto hasExt = [&](const char *name) {
      for (const auto &e : availExts) {
        if (std::strcmp(e.extensionName, name) == 0)
          return true;
      }
      return false;
    };

    if (!hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
      std::cerr << "Error: Active Vulkan driver does not support "
                << VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME << ".\n";
      return false;
    }

    std::vector<const char *> enabledExts = {
        VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
    const char *optionalExts[] = {
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        "VK_KHR_shader_bfloat16",
        "VK_KHR_cooperative_matrix",
        "VK_EXT_shader_float8",
        "VK_KHR_shader_float16_int8",
        "VK_KHR_16bit_storage",
        "VK_KHR_8bit_storage",
        "VK_KHR_shader_subgroup_extended_types",
        "VK_KHR_shader_integer_dot_product",
        "VK_KHR_buffer_device_address",
        "VK_KHR_shader_atomic_int64",
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_query",
        "VK_KHR_ray_tracing_pipeline",
        "VK_EXT_ray_tracing_invocation_reorder",
        "VK_NV_ray_tracing_invocation_reorder",
        "VK_KHR_deferred_host_operations",
        "VK_KHR_spirv_1_4",
        "VK_KHR_shader_float_controls",
        "VK_KHR_dynamic_rendering"};
    for (const char *opt : optionalExts) {
      if (hasExt(opt)) {
        enabledExts.push_back(opt);
      }
    }

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR execFeatures{};
    execFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;

    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16Features{};
    bf16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    execFeatures.pNext = &bf16Features;

    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8Features{};
    fp8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    bf16Features.pNext = &fp8Features;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures{};
    coopFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    fp8Features.pNext = &coopFeatures;

    VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{};
    rqFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    coopFeatures.pNext = &rqFeatures;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtpFeatures{};
    rtpFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rqFeatures.pNext = &rtpFeatures;

    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT serFeatures{};
    serFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT;
    rtpFeatures.pNext = &serFeatures;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    serFeatures.pNext = &asFeatures;

    VkPhysicalDeviceVulkan11Features v11Features{};
    v11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    asFeatures.pNext = &v11Features;

    VkPhysicalDeviceVulkan12Features v12Features{};
    v12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v11Features.pNext = &v12Features;

    VkPhysicalDeviceVulkan13Features v13Features{};
    v13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v12Features.pNext = &v13Features;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &execFeatures;

    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    execFeatures.pipelineExecutableInfo = VK_TRUE;

    subgroupSize32Supported =
        (v13Features.subgroupSizeControl == VK_TRUE) &&
        (subgroupProps.minSubgroupSize <= 32 && subgroupProps.maxSubgroupSize >= 32) &&
        ((subgroupProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0);
    rtPipelineSupported = hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
                          (rtpFeatures.rayTracingPipeline == VK_TRUE);
    rayQuerySupported = hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                        hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                        (rqFeatures.rayQuery == VK_TRUE);
    serSupported = (hasExt("VK_EXT_ray_tracing_invocation_reorder") ||
                    hasExt("VK_NV_ray_tracing_invocation_reorder")) &&
                   (serFeatures.rayTracingInvocationReorder == VK_TRUE);
    coopMatrixSupported = hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) &&
                          (coopFeatures.cooperativeMatrix == VK_TRUE);
    bool hasBf16 = hasExt("VK_KHR_shader_bfloat16") || hasExt("VK_EXT_shader_bfloat16");
    bf16DotProductSupported = hasBf16 && (bf16Features.shaderBFloat16DotProduct == VK_TRUE);
    bf16CoopMatrixSupported = hasBf16 && coopMatrixSupported &&
                              (bf16Features.shaderBFloat16CooperativeMatrix == VK_TRUE);
    fp8CoopMatrixSupported = hasExt("VK_EXT_shader_float8") && coopMatrixSupported &&
                             (fp8Features.shaderFloat8 == VK_TRUE) &&
                             (fp8Features.shaderFloat8CooperativeMatrix == VK_TRUE);
    fp16Supported = (v12Features.shaderFloat16 == VK_TRUE);
    fp64Supported = (features2.features.shaderFloat64 == VK_TRUE);
    int8Supported = (v12Features.shaderInt8 == VK_TRUE);

    float qPrio = 1.0f;
    VkDeviceQueueCreateInfo qInfo{};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = 0;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &qPrio;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.pNext = &features2;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
    devInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
    devInfo.ppEnabledExtensionNames = enabledExts.data();

    if (vkCreateDevice(physicalDevice, &devInfo, nullptr, &device) != VK_SUCCESS) {
      std::cerr << "Error: Failed to create logical VkDevice for ISA inspection.\n";
      return false;
    }

    pfnGetProps = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
    pfnGetStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR"));
    pfnGetIrs = reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutableInternalRepresentationsKHR"));
    pfnCreateRTPipelines = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));

    return pfnGetProps && pfnGetStats && pfnGetIrs;
  }

  ~NativeVulkanIsaInspector() {
    if (device != VK_NULL_HANDLE)
      vkDestroyDevice(device, nullptr);
    if (instance != VK_NULL_HANDLE)
      vkDestroyInstance(instance, nullptr);
  }

  VkShaderModule loadSpvModule(const fs::path &spvPath, std::vector<uint32_t> *outWords = nullptr) {
    std::ifstream ifs(spvPath, std::ios::binary | std::ios::ate);
    if (!ifs) {
      return VK_NULL_HANDLE;
    }
    size_t fileSize = static_cast<size_t>(ifs.tellg());
    if (fileSize == 0 || fileSize % 4 != 0) {
      return VK_NULL_HANDLE;
    }
    ifs.seekg(0, std::ios::beg);
    std::vector<uint32_t> spvCode(fileSize / sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(spvCode.data()), static_cast<std::streamsize>(fileSize));
    if (outWords) {
      *outWords = spvCode;
    }

    VkShaderModuleCreateInfo modInfo{};
    modInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    modInfo.codeSize = fileSize;
    modInfo.pCode = spvCode.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &modInfo, nullptr, &mod) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    return mod;
  }

  static bool spirvHasAccelerationStructure(const std::vector<uint32_t> &spvCode) {
    if (spvCode.size() >= 5 && spvCode[0] == 0x07230203) {
      size_t idx = 5;
      while (idx < spvCode.size()) {
        uint32_t word = spvCode[idx];
        uint16_t opcode = static_cast<uint16_t>(word & 0xFFFF);
        uint16_t len = static_cast<uint16_t>((word >> 16) & 0xFFFF);
        if (len == 0 || idx + len > spvCode.size())
          break;
        if (opcode == 5341 /* OpTypeAccelerationStructureKHR */) {
          return true;
        }
        idx += len;
      }
    }
    return false;
  }

  void printPipelineExecutables(VkPipeline pipeline, const KernelArtifact &art, bool fullIsa) {
    std::cout << "━━━ [vulkan] " << art.configName
              << " (" << art.filePath.filename().string() << ") ━━━\n";

    VkPipelineInfoKHR pInfo{};
    pInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
    pInfo.pipeline = pipeline;

    uint32_t execCount = 0;
    pfnGetProps(device, &pInfo, &execCount, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> execProps(execCount);
    for (auto &ep : execProps) {
      ep.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    }
    pfnGetProps(device, &pInfo, &execCount, execProps.data());

    for (uint32_t e = 0; e < execCount; ++e) {
      VkPipelineExecutableInfoKHR execInfo{};
      execInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
      execInfo.pipeline = pipeline;
      execInfo.executableIndex = e;

      uint32_t statCount = 0;
      pfnGetStats(device, &execInfo, &statCount, nullptr);
      std::vector<VkPipelineExecutableStatisticKHR> stats(statCount);
      for (auto &s : stats) {
        s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
      }
      pfnGetStats(device, &execInfo, &statCount, stats.data());

      std::cout << "  Executable [" << e << "] : " << execProps[e].name << " — "
                << execProps[e].description
                << " (Subgroup Size: " << execProps[e].subgroupSize << ")\n";
      std::cout << "  Hardware Statistics (VK_KHR_pipeline_executable_properties):\n    ";
      int col = 0;
      for (const auto &st : stats) {
        if (std::strcmp(st.name, "Driver pipeline hash") == 0 ||
            std::strcmp(st.name, "Hash") == 0) {
          continue;
        }
        std::ostringstream valStream;
        switch (st.format) {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
          valStream << (st.value.b32 ? "true" : "false");
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
          valStream << st.value.i64;
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
          valStream << st.value.u64;
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
          valStream << st.value.f64;
          break;
        default:
          break;
        }
        if (col > 0 && col % 3 == 0)
          std::cout << "\n    ";
        std::cout << std::left << std::setw(32)
                  << (std::string(st.name) + ": " + valStream.str());
        ++col;
      }
      std::cout << "\n\n";

      uint32_t irCount = 0;
      pfnGetIrs(device, &execInfo, &irCount, nullptr);
      std::vector<VkPipelineExecutableInternalRepresentationKHR> irs(irCount);
      for (auto &ir : irs) {
        ir.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        ir.pData = nullptr;
      }
      pfnGetIrs(device, &execInfo, &irCount, irs.data());

      std::vector<std::vector<char>> irBuffers(irCount);
      for (uint32_t i = 0; i < irCount; ++i) {
        irBuffers[i].resize(irs[i].dataSize);
        irs[i].pData = irBuffers[i].data();
      }
      pfnGetIrs(device, &execInfo, &irCount, irs.data());

      int chosenIr = -1;
      for (uint32_t i = 0; i < irCount; ++i) {
        std::string irName = irs[i].name;
        for (char &c : irName)
          c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (irName.find("disassembly") != std::string::npos ||
            irName.find("isa") != std::string::npos ||
            irName.find("assembly") != std::string::npos) {
          chosenIr = static_cast<int>(i);
          break;
        }
      }
      if (chosenIr == -1 && irCount > 0) {
        chosenIr = static_cast<int>(irCount) - 1;
      }

      if (chosenIr != -1 && irs[chosenIr].isText && irs[chosenIr].pData) {
        std::string isaStr(static_cast<const char *>(irs[chosenIr].pData));
        printIsaTextSummary(isaStr, fullIsa);
      }
    }
  }

  bool inspectRasterSpv(const KernelArtifact &art, bool fullIsa) {
    fs::path dir = art.filePath.parent_path();
    fs::path vertPath = dir / "pixel_fill.vert.spv";
    fs::path fragPath = dir / "pixel_fill.frag.spv";
    std::string fname = art.filePath.filename().string();
    if (fname.find(".vert.spv") != std::string::npos) {
      vertPath = art.filePath;
    } else if (fname.find(".frag.spv") != std::string::npos) {
      fragPath = art.filePath;
    }

    VkShaderModule vertMod = loadSpvModule(vertPath);
    VkShaderModule fragMod = loadSpvModule(fragPath);
    if (!vertMod || !fragMod) {
      if (vertMod) vkDestroyShaderModule(device, vertMod, nullptr);
      if (fragMod) vkDestroyShaderModule(device, fragMod, nullptr);
      std::cerr << "  [Error] Failed to load vertex/fragment SPIR-V modules for " << art.filePath << "\n";
      return false;
    }

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = 16;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{0.0f, 0.0f, 3840.0f, 2160.0f, 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {3840, 2160}};
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                     VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline);
    if (res == VK_SUCCESS && pipeline != VK_NULL_HANDLE) {
      printPipelineExecutables(pipeline, art, fullIsa);
      vkDestroyPipeline(device, pipeline, nullptr);
    } else {
      std::cerr << "  [Error] vkCreateGraphicsPipelines failed for " << art.filePath
                << " (VkResult=" << res << ")\n";
    }

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
    return res == VK_SUCCESS;
  }

  bool inspectRayTracingSpv(const KernelArtifact &art, bool fullIsa) {
    if (!rtPipelineSupported || !pfnCreateRTPipelines) {
      std::cerr << "  [Error] VK_KHR_ray_tracing_pipeline not supported on this device.\n";
      return false;
    }

    fs::path dir = art.filePath.parent_path();
    std::string fname = art.filePath.filename().string();
    std::string prefix = fname;
    size_t dotPos = prefix.find('.');
    if (dotPos != std::string::npos) {
      prefix = prefix.substr(0, dotPos);
    }
    std::string basePrefix = prefix;
    if (prefix == "rt_scheduling_ser" || prefix == "raydiv_pipeline_ser") {
      if (!serSupported) {
        std::cout << "━━━ [vulkan] " << art.configName
                  << " (" << art.filePath.filename().string() << ") ━━━\n"
                  << "  [Skipped] Hardware SER (VK_EXT_ray_tracing_invocation_reorder) is not "
                     "exposed by the active Vulkan driver (" << driverInfo << ").\n\n";
        return true;
      }
      basePrefix = (prefix == "rt_scheduling_ser") ? "rt_scheduling" : "raydiv_pipeline";
    } else if (prefix.rfind("raydiv_pipeline_", 0) == 0) {
      basePrefix = "raydiv_pipeline";
    }

    fs::path rgenPath = dir / (prefix + ".rgen.spv");
    if (!fs::exists(rgenPath)) {
      rgenPath = dir / (basePrefix + ".rgen.spv");
    }
    fs::path rmissPath = dir / (basePrefix + ".rmiss.spv");
    fs::path rchitPath = dir / (prefix + ".rchit.spv");
    if (!fs::exists(rchitPath)) {
      rchitPath = dir / (basePrefix + ".rchit.spv");
    }
    if (!fs::exists(rchitPath)) {
      rchitPath = dir / (basePrefix + "_a.rchit.spv");
    }
    fs::path rahitPath = dir / (basePrefix + ".rahit.spv");
    fs::path rintPath = dir / (basePrefix + ".rint.spv");

    std::vector<VkShaderModule> modules;
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

    auto addStage = [&](const fs::path &p, VkShaderStageFlagBits stageBit) -> int {
      if (!fs::exists(p))
        return -1;
      VkShaderModule m = loadSpvModule(p);
      if (!m)
        return -1;
      modules.push_back(m);
      VkPipelineShaderStageCreateInfo st{};
      st.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      st.stage = stageBit;
      st.module = m;
      st.pName = "main";
      stages.push_back(st);
      return static_cast<int>(stages.size()) - 1;
    };

    int rgenIdx = addStage(rgenPath, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    int rmissIdx = addStage(rmissPath, VK_SHADER_STAGE_MISS_BIT_KHR);
    int rchitIdx = addStage(rchitPath, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    int rahitIdx = addStage(rahitPath, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
    int rintIdx = addStage(rintPath, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);

    if (rgenIdx >= 0) {
      VkRayTracingShaderGroupCreateInfoKHR g{};
      g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      g.generalShader = static_cast<uint32_t>(rgenIdx);
      g.closestHitShader = VK_SHADER_UNUSED_KHR;
      g.anyHitShader = VK_SHADER_UNUSED_KHR;
      g.intersectionShader = VK_SHADER_UNUSED_KHR;
      groups.push_back(g);
    }
    if (rmissIdx >= 0) {
      VkRayTracingShaderGroupCreateInfoKHR g{};
      g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      g.generalShader = static_cast<uint32_t>(rmissIdx);
      g.closestHitShader = VK_SHADER_UNUSED_KHR;
      g.anyHitShader = VK_SHADER_UNUSED_KHR;
      g.intersectionShader = VK_SHADER_UNUSED_KHR;
      groups.push_back(g);
    }
    if (rchitIdx >= 0 || rahitIdx >= 0 || rintIdx >= 0) {
      VkRayTracingShaderGroupCreateInfoKHR g{};
      g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      g.type = (rintIdx >= 0) ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                              : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      g.generalShader = VK_SHADER_UNUSED_KHR;
      g.closestHitShader = (rchitIdx >= 0) ? static_cast<uint32_t>(rchitIdx) : VK_SHADER_UNUSED_KHR;
      g.anyHitShader = (rahitIdx >= 0) ? static_cast<uint32_t>(rahitIdx) : VK_SHADER_UNUSED_KHR;
      g.intersectionShader = (rintIdx >= 0) ? static_cast<uint32_t>(rintIdx) : VK_SHADER_UNUSED_KHR;
      groups.push_back(g);
    }

    VkShaderStageFlags rtStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                  VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

    std::vector<VkDescriptorSetLayoutBinding> bindings(12);
    for (uint32_t i = 0; i < 12; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = rtStages;
    }

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    dslInfo.pBindings = bindings.data();

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl);

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = rtStages;
    pcRange.offset = 0;
    pcRange.size = 128;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dsl;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout);

    VkRayTracingPipelineCreateInfoKHR rtPipeInfo{};
    rtPipeInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtPipeInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                       VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    rtPipeInfo.stageCount = static_cast<uint32_t>(stages.size());
    rtPipeInfo.pStages = stages.data();
    rtPipeInfo.groupCount = static_cast<uint32_t>(groups.size());
    rtPipeInfo.pGroups = groups.data();
    rtPipeInfo.maxPipelineRayRecursionDepth = 1;
    rtPipeInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult res = pfnCreateRTPipelines(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtPipeInfo,
                                        nullptr, &pipeline);
    if (res == VK_SUCCESS && pipeline != VK_NULL_HANDLE) {
      printPipelineExecutables(pipeline, art, fullIsa);
      vkDestroyPipeline(device, pipeline, nullptr);
    } else {
      std::cerr << "  [Error] vkCreateRayTracingPipelinesKHR failed for " << art.filePath
                << " (VkResult=" << res << ")\n";
    }

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    for (auto m : modules) {
      vkDestroyShaderModule(device, m, nullptr);
    }
    return res == VK_SUCCESS;
  }

  bool inspectSpv(const KernelArtifact &art, bool fullIsa) {
    std::string fname = art.filePath.filename().string();
    if (fname.find(".rgen.spv") != std::string::npos ||
        fname.find(".rmiss.spv") != std::string::npos ||
        fname.find(".rchit.spv") != std::string::npos ||
        fname.find(".rahit.spv") != std::string::npos ||
        fname.find(".rint.spv") != std::string::npos) {
      return inspectRayTracingSpv(art, fullIsa);
    }
    if (fname.find(".vert.spv") != std::string::npos ||
        fname.find(".frag.spv") != std::string::npos) {
      return inspectRasterSpv(art, fullIsa);
    }

    // Pre-flight capability checks to avoid compiling shaders whose required
    // Vulkan sub-features are not supported by the active device/driver
    // (e.g. bf16.comp requires shaderBFloat16DotProduct, which is false on RDNA 3.5 / gfx1151
    // where RADV only exposes shaderBFloat16CooperativeMatrix, causing ACO to crash on scalar bfmul/bffma).
    if (fname == "bf16.comp.spv" && !bf16DotProductSupported) {
      std::cout << "━━━ [vulkan] " << art.configName << " (" << fname << ") ━━━\n"
                << "  [Skipped] BF16 vector dot product (VK_KHR_shader_bfloat16::shaderBFloat16DotProduct / v_dot2_bf16_bf16) "
                   "is not supported on " << deviceName << " under Vulkan (" << driverInfo << ").\n"
                << "            Use '-k rocm' to inspect the ROCm HIP BF16 vector kernel (v_dot2_f32_bf16 in bf16.co).\n\n";
      return true;
    }
    if (fname == "coop_matrix_bf16.comp.spv" && !bf16CoopMatrixSupported) {
      std::cout << "━━━ [vulkan] " << art.configName << " (" << fname << ") ━━━\n"
                << "  [Skipped] BF16 cooperative matrix (VK_KHR_shader_bfloat16::shaderBFloat16CooperativeMatrix) "
                   "is not supported on " << deviceName << " (" << driverInfo << ").\n\n";
      return true;
    }
    if (fname == "coop_matrix_fp8.comp.spv" && !fp8CoopMatrixSupported) {
      std::cout << "━━━ [vulkan] " << art.configName << " (" << fname << ") ━━━\n"
                << "  [Skipped] FP8 cooperative matrix (VK_EXT_shader_float8::shaderFloat8CooperativeMatrix) "
                   "is not supported on " << deviceName << " (" << driverInfo << ").\n\n";
      return true;
    }
    if ((fname == "coop_matrix_fp16.comp.spv" || fname == "coop_matrix_int8.comp.spv") &&
        !coopMatrixSupported) {
      std::cout << "━━━ [vulkan] " << art.configName << " (" << fname << ") ━━━\n"
                << "  [Skipped] Cooperative matrix (VK_KHR_cooperative_matrix) "
                   "is not supported on " << deviceName << " (" << driverInfo << ").\n\n";
      return true;
    }
    if (fname == "fp64.comp.spv" && !fp64Supported) {
      std::cout << "━━━ [vulkan] " << art.configName << " (" << fname << ") ━━━\n"
                << "  [Skipped] FP64 (shaderFloat64) is not supported on " << deviceName << ".\n\n";
      return true;
    }

    std::vector<uint32_t> spvCode;
    VkShaderModule shaderModule = loadSpvModule(art.filePath, &spvCode);
    if (!shaderModule) {
      std::cerr << "  [Error] Failed to create VkShaderModule for " << art.filePath << "\n";
      return false;
    }

    bool hasAccelStruct = spirvHasAccelerationStructure(spvCode);
    if (hasAccelStruct && !rayQuerySupported) {
      std::cout << "━━━ [vulkan] " << art.configName << " (" << fname << ") ━━━\n"
                << "  [Skipped] Ray query / acceleration structure (VK_KHR_ray_query) "
                   "is not supported on " << deviceName << " (" << driverInfo << ").\n\n";
      vkDestroyShaderModule(device, shaderModule, nullptr);
      return true;
    }

    // Create a generous descriptor set layout (up to 12 bindings; binding 0 is TLAS if ray_query)
    std::vector<VkDescriptorSetLayoutBinding> bindings(12);
    for (uint32_t i = 0; i < 12; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = (hasAccelStruct && i == 0)
                                       ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    dslInfo.pBindings = bindings.data();

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl);

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = 128;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dsl;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout);

    // Match VulkanContext::createKernel exactly: request Wave32 (requiredSubgroupSize = 32)
    // so RDNA 3 / RDNA 4 VOPD dual-issue instructions (v_dual_fmac_f32, etc.) are enabled!
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSubgroupInfo{};
    reqSubgroupInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    reqSubgroupInfo.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                     VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = shaderModule;
    pipeInfo.stage.pName = "main";
    if (subgroupSize32Supported) {
      pipeInfo.stage.pNext = &reqSubgroupInfo;
      pipeInfo.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    pipeInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline);
    if (res != VK_SUCCESS || pipeline == VK_NULL_HANDLE) {
      std::cerr << "  [Error] vkCreateComputePipelines failed for " << art.filePath
                << " (VkResult=" << res << ")\n";
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device, dsl, nullptr);
      vkDestroyShaderModule(device, shaderModule, nullptr);
      return false;
    }

    printPipelineExecutables(pipeline, art, fullIsa);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    return true;
  }
};
#endif

static int runCommandSilent(const std::string &cmd, std::string &output) {
  output.clear();
  std::string fullCmd = cmd + " 2>&1";
#ifdef _WIN32
  FILE *pipe = _popen(fullCmd.c_str(), "r");
#else
  FILE *pipe = popen(fullCmd.c_str(), "r");
#endif
  if (!pipe) {
    return -1;
  }
  char buffer[512];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }
#ifdef _WIN32
  return _pclose(pipe);
#else
  return pclose(pipe);
#endif
}

static std::string extractGfxToken(const std::string &s) {
  size_t pos = s.find("gfx");
  if (pos == std::string::npos) {
    return "";
  }
  size_t end = pos + 3;
  while (end < s.size() && std::isalnum(static_cast<unsigned char>(s[end]))) {
    ++end;
  }
  if (end > pos + 3) {
    return s.substr(pos, end - pos);
  }
  return "";
}

static bool inspectRocmCodeObject(const KernelArtifact &art, const std::string &asic, bool fullIsa) {
  // Locate clang-offload-bundler and llvm-objdump from standard ROCm LLVM paths
  std::string bundler = "clang-offload-bundler";
  std::string objdump = "llvm-objdump";

  std::error_code ec;
  if (fs::is_directory("/opt/rocm", ec)) {
    for (const auto &entry : fs::directory_iterator("/opt/rocm", ec)) {
      fs::path b = entry.path() / "lib" / "llvm" / "bin" / "clang-offload-bundler";
      fs::path d = entry.path() / "lib" / "llvm" / "bin" / "llvm-objdump";
      if (fs::exists(b, ec) && fs::exists(d, ec)) {
        bundler = b.string();
        objdump = d.string();
        break;
      }
    }
  }

  fs::path tmpObj = fs::temp_directory_path() / (art.filePath.stem().string() + "_" + asic + ".o");
  std::string out;
  std::string unbundleCmd = "\"" + bundler + "\" --type=o --bundle-align=4096 "
                            "--targets=hipv4-amdgcn-amd-amdhsa--" + asic +
                            " --input=\"" + art.filePath.string() + "\""
                            " --output=\"" + tmpObj.string() + "\" --unbundle";
  runCommandSilent(unbundleCmd, out);

  fs::path targetPath = fs::exists(tmpObj, ec) ? tmpObj : art.filePath;
  std::string symArg = !art.rocmSymbol.empty() ? (" --disassemble-symbols=" + art.rocmSymbol) : "";
  std::string disCmd = "\"" + objdump + "\" -d --mcpu=" + asic + symArg + " \"" + targetPath.string() + "\"";
  int rc = runCommandSilent(disCmd, out);
  if (fs::exists(tmpObj, ec)) {
    fs::remove(tmpObj, ec);
  }
  if (rc != 0 || out.empty()) {
    std::cerr << "  [Error] Failed to disassemble ROCm code object: " << art.filePath << "\n";
    return false;
  }

  std::cout << "━━━ [rocm] " << art.configName
            << " (" << art.filePath.filename().string()
            << (!art.rocmSymbol.empty() ? (" :: " + art.rocmSymbol) : "")
            << " — " << asic << ") ━━━\n\n";
  printIsaTextSummary(out, fullIsa);
  return true;
}

} // namespace

int IsaDumper::dump(const IsaDumperOptions &opts) {
  fs::path kernelDir = KernelPath::find();

  // Resolve requested backends (default to Vulkan native pipeline executable inspection)
  std::set<std::string> wantedBackends;
  for (const auto &b : opts.backends) {
    std::string low = normalizeKey(b);
    if (low == "all") {
      wantedBackends.insert("vulkan");
      wantedBackends.insert("rocm");
    } else if (low.empty() || low == "auto" || low == "vulkan") {
      wantedBackends.insert("vulkan");
    } else {
      wantedBackends.insert(low);
    }
  }
  if (wantedBackends.empty()) {
    wantedBackends = {"vulkan"};
  }

  std::vector<KernelArtifact> artifacts;
  auto mappings = getBenchmarkMappings();

  for (const auto &target : opts.targets) {
    fs::path directPath(target);
    if (fs::exists(directPath) && fs::is_regular_file(directPath)) {
      std::string ext = directPath.extension().string();
      std::string backend = (ext == ".co" || ext == ".hsaco" || ext == ".o") ? "rocm" : "vulkan";
      artifacts.push_back(
          {directPath.filename().string(), directPath.filename().string(), 0, backend, directPath, "", ""});
      continue;
    }

    std::string normTarget = normalizeKey(target);
    if (normTarget == "all") {
      for (const auto &m : mappings) {
        if (!opts.configs.empty() &&
            std::find(opts.configs.begin(), opts.configs.end(), m.configIndex) ==
                opts.configs.end()) {
          continue;
        }
        if (wantedBackends.count("vulkan") && !m.vulkanSpv.empty()) {
          fs::path p = kernelDir / "vulkan" / m.vulkanSpv;
          if (fs::exists(p)) {
            artifacts.push_back({m.benchName, m.configLabel, m.configIndex, "vulkan", p, "", ""});
          }
        }
        if (wantedBackends.count("rocm") && !m.rocmCo.empty()) {
          fs::path p = kernelDir / "rocm" / m.rocmCo;
          if (fs::exists(p)) {
            artifacts.push_back({m.benchName, m.configLabel, m.configIndex, "rocm", p, m.rocmSymbol, ""});
          }
        }
      }
      continue;
    }

    bool matchedBench = false;
    for (const auto &m : mappings) {
      bool nameMatch = (normalizeKey(m.benchName) == normTarget) ||
                       (normalizeKey(m.configLabel) == normTarget);
      if (!nameMatch) {
        for (const auto &al : m.aliases) {
          if (normalizeKey(al) == normTarget) {
            nameMatch = true;
            break;
          }
        }
      }
      if (!nameMatch)
        continue;

      if (!opts.configs.empty() &&
          std::find(opts.configs.begin(), opts.configs.end(), m.configIndex) ==
              opts.configs.end()) {
        continue;
      }

      matchedBench = true;
      if (wantedBackends.count("vulkan")) {
        if (!m.vulkanSpv.empty()) {
          fs::path p = kernelDir / "vulkan" / m.vulkanSpv;
          if (fs::exists(p)) {
            artifacts.push_back({m.benchName, m.configLabel, m.configIndex, "vulkan", p, "", ""});
          }
        } else if (wantedBackends.size() == 1) {
          std::string reason = !m.rocmCo.empty()
              ? (m.configLabel + " has no Vulkan shader artifact (supported on ROCm via " + m.rocmCo + "; pass '-k rocm').")
              : (m.configLabel + " is a capability-probed benchmark with no compiled shader artifact on this toolchain.");
          artifacts.push_back({m.benchName, m.configLabel, m.configIndex, "vulkan", {}, "", reason});
        }
      }
      if (wantedBackends.count("rocm")) {
        if (!m.rocmCo.empty()) {
          fs::path p = kernelDir / "rocm" / m.rocmCo;
          if (fs::exists(p)) {
            artifacts.push_back({m.benchName, m.configLabel, m.configIndex, "rocm", p, m.rocmSymbol, ""});
          }
        } else if (wantedBackends.size() == 1) {
          std::string reason = !m.vulkanSpv.empty()
              ? (m.configLabel + " has no ROCm kernel artifact (supported on Vulkan via " + m.vulkanSpv + "; pass '-k vulkan').")
              : (m.configLabel + " is a capability-probed benchmark with no compiled kernel artifact on this toolchain.");
          artifacts.push_back({m.benchName, m.configLabel, m.configIndex, "rocm", {}, "", reason});
        }
      }
    }

    if (!matchedBench) {
      if (wantedBackends.count("vulkan") && fs::is_directory(kernelDir / "vulkan")) {
        for (const auto &entry : fs::directory_iterator(kernelDir / "vulkan")) {
          if (entry.path().extension() == ".spv") {
            std::string fname = entry.path().filename().string();
            if (normalizeKey(fname).find(normTarget) != std::string::npos) {
              artifacts.push_back({fname, fname, 0, "vulkan", entry.path(), "", ""});
            }
          }
        }
      }
      if (wantedBackends.count("rocm") && fs::is_directory(kernelDir / "rocm")) {
        for (const auto &entry : fs::directory_iterator(kernelDir / "rocm")) {
          if (entry.path().extension() == ".co") {
            std::string fname = entry.path().filename().string();
            if (normalizeKey(fname).find(normTarget) != std::string::npos) {
              artifacts.push_back({fname, fname, 0, "rocm", entry.path(), "", ""});
            }
          }
        }
      }
    }
  }

  if (artifacts.empty()) {
    std::cerr << "Error: No compiled shaders or kernels matched for --dump-isa (searched in "
              << kernelDir.string() << ").\n";
    return EXIT_FAILURE;
  }

  std::string resolvedAsic = opts.asic;

#ifdef HAVE_VULKAN
  std::unique_ptr<NativeVulkanIsaInspector> vkInspector;
  bool needVulkan = false;
  bool needRocm = false;
  for (const auto &a : artifacts) {
    if (!a.skipReason.empty())
      continue;
    if (a.backend == "vulkan") {
      needVulkan = true;
    } else if (a.backend == "rocm") {
      needRocm = true;
    }
  }
  if (needVulkan || (needRocm && resolvedAsic.empty())) {
    vkInspector = std::make_unique<NativeVulkanIsaInspector>();
    if (vkInspector->init(opts.deviceIndex)) {
      if (resolvedAsic.empty()) {
        const auto &prof = DeviceDatabase::lookup(vkInspector->vendorId, vkInspector->deviceId,
                                                  vkInspector->deviceName);
        resolvedAsic = extractGfxToken(prof.archName);
      }
      if (needVulkan) {
        std::cout << "╭─ GPUBench Native Shader ISA Inspector (VK_KHR_pipeline_executable_properties) ─╮\n";
        std::cout << "│ Target Device : " << std::left << std::setw(63) << vkInspector->deviceName << "│\n";
        std::cout << "│ Active Driver : " << std::left << std::setw(63) << vkInspector->driverInfo << "│\n";
        std::cout << "╰────────────────────────────────────────────────────────────────────────────────╯\n\n";
      }
    } else if (needVulkan) {
      return EXIT_FAILURE;
    }
  }
#endif
  if (resolvedAsic.empty()) {
    resolvedAsic = "gfx1201";
  }

  std::set<std::string> visitedFiles;
  for (const auto &art : artifacts) {
    std::string visitKey = art.backend + ":" + art.filePath.string() + ":" + art.rocmSymbol + ":" + art.configName;
    if (!visitedFiles.insert(visitKey).second) {
      continue;
    }

    if (!art.skipReason.empty()) {
      std::cout << "━━━ [" << art.backend << "] " << art.configName << " ━━━\n"
                << "  [Skipped] " << art.skipReason << "\n\n";
      continue;
    }

    if (art.backend == "vulkan") {
#ifdef HAVE_VULKAN
      if (vkInspector) {
        vkInspector->inspectSpv(art, opts.fullIsa);
      }
#endif
    } else if (art.backend == "rocm") {
      inspectRocmCodeObject(art, resolvedAsic, opts.fullIsa);
    }
  }

  return EXIT_SUCCESS;
}
