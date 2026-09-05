#pragma once

#include "IBenchmark.h"
#include "scene/GltfScene.h"
#include <string>
#include <vector>
#ifdef HAVE_VULKAN
#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#include <vulkan/vulkan.h>
#include "core/VulkanContext.h"
#endif

class RaySchedulingBench : public IBenchmark {
public:
  enum class SceneType : uint32_t {
    Showroom = 0,
    IndoorAtrium = 1,
    OutdoorLandscape = 2
  };

  RaySchedulingBench(SceneType scene = SceneType::IndoorAtrium) : sceneType(scene) {}

  const char *GetName() const override {
    if (sceneType == SceneType::OutdoorLandscape) {
      return "RayScheduling (Outdoor Landscape)";
    } else if (sceneType == SceneType::IndoorAtrium) {
      return "RayScheduling (Indoor Atrium)";
    } else {
      return "RayScheduling (Showroom Studio)";
    }
  }

  void SetSceneType(SceneType type) { sceneType = type; }
  SceneType GetSceneType() const { return sceneType; }

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kApi;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    if (context && (context->getBackend() == ComputeBackend::OpenCL ||
                    context->getBackend() == ComputeBackend::ROCm)) {
      return SupportLimitation::kApi;
    }
    return SupportLimitation::kHardware;
  }
  std::string GetSupportNote() const override {
    return "Ray Scheduling benchmark requires Vulkan ray query acceleration structures";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for ray tracing acceleration structures in OpenCL API";
    }
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "No support for ray tracing acceleration structures in ROCm/HIP API";
    }
    if (!info.rayTracingSupport) {
      return "extension VK_KHR_acceleration_structure or VK_KHR_ray_query missing";
    }
    return "Ray Scheduling benchmark requires Vulkan ray query acceleration structures";
  }

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void RebuildAccelerationStructures() override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  int GetSortWeight(uint32_t config_idx = 0) const override;

  uint32_t GetNumConfigs() const override { return 28; }
  std::vector<std::string> GetAliases() const override {
    if (sceneType == SceneType::OutdoorLandscape) {
      return {"rayscheduling", "rtscheduling", "outdoor", "landscape", "rayscheduling_outdoor", "worklists", "dgc", "scene_render", "total_scene_render", "total_frame", "primary", "primary_rays", "shadow", "shadows", "rts", "ray_shadows", "ray_shadow"};
    } else if (sceneType == SceneType::IndoorAtrium) {
      return {"rayscheduling", "rtscheduling", "indoor", "atrium", "rayscheduling_indoor", "worklists", "dgc", "scene_render", "total_scene_render", "total_frame", "primary", "primary_rays", "shadow", "shadows", "rts", "ray_shadows", "ray_shadow"};
    } else {
      return {"rayscheduling", "rtscheduling", "showroom", "studio", "rayscheduling_showroom", "worklists", "dgc", "scene_render", "total_scene_render", "total_frame", "primary", "primary_rays", "shadow", "shadows", "rts", "ray_shadows", "ray_shadow"};
    }
  }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetMetric(uint32_t config_idx = 0) const override {
    if (config_idx < 4) return "MHits/s";
    if (config_idx == 17) return "MRecords/s";
    return "MRays/s";
  }
  bool IsConfigSupported(uint32_t config_idx) const override {
    return !unsupportedConfig[config_idx];
  }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return !unsupportedConfig[config_idx];
  }
  std::string GetConfigSupportNote(uint32_t config_idx) const override {
    return unsupportedReason[config_idx];
  }
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    if (!unsupportedReason[config_idx].empty()) {
      return unsupportedReason[config_idx];
    }
    if (config_idx == 1 || config_idx == 5 || config_idx == 9 || config_idx == 13 || config_idx == 24) {
      if (!unsupportedReason[config_idx].empty()) {
        return unsupportedReason[config_idx];
      }
      return "VK_EXT_ray_tracing_invocation_reorder requires Ray Tracing Pipeline (not supported in compute shaders)";
    }
    if (config_idx == 3 || config_idx == 7 || config_idx == 11 || config_idx == 15 || config_idx == 26) {
      return "extension VK_AMDX_shader_enqueue missing";
    }
    return GetSupportNote(info, context);
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx) const override {
    if (config_idx == 1 || config_idx == 5 || config_idx == 9 || config_idx == 13 || config_idx == 24) {
      return SupportLimitation::kHardware;
    }
    if (config_idx == 3 || config_idx == 7 || config_idx == 11 || config_idx == 15 || config_idx == 26) {
      return SupportLimitation::kApi;
    }
    return SupportLimitation::kNone;
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return GetConfigSupportLimitation(config_idx);
  }
  const char *GetComponent(uint32_t config_idx = 0) const override { return "Ray Tracing"; }
  const char *GetSubCategory(uint32_t config_idx = 0) const override;

  void SetDumpRenders(bool dump) { dumpRenders = dump; }
  bool GetDumpRenders() const { return dumpRenders; }

  void SetResolution(uint32_t w, uint32_t h) override {
    renderWidth = w;
    renderHeight = h;
    rayCount = renderWidth * renderHeight;
    queueCapacity = 65536;
    while (queueCapacity < rayCount / 6) {
      queueCapacity *= 2;
    }
  }
  void SetBounceDepth(uint32_t bounces);
  uint32_t GetBounceDepth() const { return bounceDepth; }

  uint32_t GetRenderWidth() const { return renderWidth; }
  uint32_t GetRenderHeight() const { return renderHeight; }
  uint32_t GetQueueCapacity() const { return queueCapacity; }

  void RecordRunResult(uint32_t config_idx, uint64_t total_invocations, double total_time_ms) override {
    if (config_idx < 28) {
      recordedInvocations[config_idx] = total_invocations;
      recordedTimeMs[config_idx] = total_time_ms;
    }
  }

private:
  uint64_t recordedInvocations[28] = {0};
  double recordedTimeMs[28] = {0.0};
  IComputeContext *context = nullptr;
  std::string findScriptPath(const std::string &scriptName) const;
  bool dumpRenders = true;
  ComputeBuffer fbTraditional = nullptr;
  ComputeBuffer fbWorkList = nullptr;
  void performVisualVerification();

  // Compute Kernels
  ComputeKernel kernelTraditional = nullptr;
  ComputeKernel kernelClassify = nullptr;
  ComputeKernel kernelMaterial = nullptr;
  ComputeKernel kernelMaterialSpecialized[8] = {nullptr};
  ComputeKernel kernelBounce = nullptr;
  ComputeKernel kernelBounceTerminal = nullptr;
  ComputeKernel kernelBounceOctant = nullptr;
  ComputeKernel kernelShadow = nullptr;
  ComputeKernel kernelWorkGraph = nullptr;
  ComputeKernel kernelReset = nullptr;
  ComputeKernel kernelResolve = nullptr;

  // Storage Buffers
  ComputeBuffer resultBuffer = nullptr;
  ComputeBuffer workListBuffer = nullptr;
  ComputeBuffer indirectBuffer = nullptr;

  // glTF Scene & PBR Storage Buffers
  GltfScene gltfScene;
  bool isGltf = false;
  ComputeBuffer materialBuffer = nullptr;
  ComputeBuffer triangleMaterialBuffer = nullptr;
  ComputeBuffer texHeaderBuffer = nullptr;
  ComputeBuffer texPixelBuffer = nullptr;
  std::string findModelPath(const std::string &modelName) const;

#ifdef HAVE_VULKAN
  VkAccelerationStructureKHR triangleBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR sceneTlas = VK_NULL_HANDLE;
  ComputeBuffer vertexBuffer = nullptr;
  ComputeBuffer instanceBuffer = nullptr;
  ComputeBuffer triangleBlasBuffer = nullptr;
  ComputeBuffer tlasBuffer = nullptr;
  ComputeBuffer scratchBuffer = nullptr;

  PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR_ptr = nullptr;
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ptr = nullptr;
  PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR_ptr = nullptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR_ptr = nullptr;
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ptr = nullptr;

  void loadRTProcs(VkDevice device);
  void buildAS();

  std::vector<VulkanContext::IndirectBatchEntry> materialBatches;
  std::vector<VulkanContext::IndirectBatchEntry> materialBatchesBreakdown;
  std::vector<VulkanContext::IndirectBatchEntry> bounceBatches;
  std::vector<VulkanContext::IndirectBatchEntry> octantBatches;
  std::vector<VulkanContext::IndirectBatchEntry> shadowBatches;
  std::vector<VulkanContext::IndirectBatchEntry> shadowBinBatches;
  void rebuildBounceBatches();
#endif

  uint32_t bounceDepth = 2;
  uint32_t renderWidth = 1920;
  uint32_t renderHeight = 1080;
  uint32_t rayCount = 1920 * 1080;
  uint32_t queueCapacity = 850000;
  uint32_t materialCapacity = 850000;
  uint32_t bounceCapacity = 1500000;
  uint32_t octantCapacity = 262144;
  uint32_t numPrimitives = 4096;
  SceneType sceneType = SceneType::IndoorAtrium;
  mutable double results[28] = {0.0};
  mutable bool unsupportedConfig[28] = {false};
  mutable std::string unsupportedReason[28];
};
