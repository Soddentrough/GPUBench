#pragma once

#include "IBenchmark.h"
#include "ReactorBayScene.h"
#include "scene/GltfScene.h"
#include <algorithm>
#include <random>
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
    ReactorBay = 0,
    Showroom = ReactorBay,
    IndoorAtrium = ReactorBay,
    OutdoorLandscape = ReactorBay,
    AAAOutdoorForest = ReactorBay
  };

  RaySchedulingBench(SceneType scene = SceneType::ReactorBay) : sceneType(scene) {}

  const char *GetName() const override {
    return "RayScheduling (Zero-Day Reactor Bay)";
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
  uint32_t GetSamplesPerFrame(uint32_t config_idx = 0) const override {
    if (config_idx >= 3 && config_idx <= 5) {
      return (samplesPerPixel > 1)
          ? ((lastCompletedPasses[config_idx] > 0) ? lastCompletedPasses[config_idx] : samplesPerPixel)
          : 1u;
    }
    return 1u;
  }
  bool ValidateResults(uint32_t config_idx = 0) const override;
  int GetSortWeight(uint32_t config_idx = 0) const override;
  uint32_t GetExpectedKernelCount() const override { return 29; }

  uint32_t GetNumConfigs() const override { return 24; }
  std::vector<std::string> GetAliases() const override {
    return {"rayscheduling", "rtscheduling", "reactor", "reactor_bay", "zeroday", "zero_day", "hangar", "rayscheduling_reactor",
            "showroom", "studio", "indoor", "atrium", "outdoor", "landscape", "forest", "aaa_forest",
            "wavefront_queues", "dgc", "scene_render", "total_scene_render", "total_frame",
            "primary", "primary_rays", "shadow", "shadows", "rts", "ray_shadows", "ray_shadow",
            "multilight", "multi_light", "lightbinning", "light_binning", "pathtracing", "path_tracing", "path", "pathtrace"};
  }
  std::string GetConfigName(uint32_t config_idx) const override;
  int32_t GetBaselineConfigIndex(uint32_t config_idx) const override;
  const char *GetMetric(uint32_t config_idx = 0) const override {
    if (config_idx < 3) return "MHits/s";
    if (config_idx == 10 || config_idx == 17 || config_idx == 19) return "MRecords/s";
    return "MRays/s";
  }
  void SetVerifyParity(bool verify) override { verifyParity = verify; }
  bool GetVerifyParity() const { return verifyParity; }
  bool HasParityFailure() const override { return parityFailure; }
  static constexpr bool isSERConfigIdx(uint32_t c) {
    return c == 1 || c == 4 || c == 7 || c == 16 || c == 21;
  }
  static constexpr bool isStandardDGCConfigIdx(uint32_t c) {
    return c == 5 || c == 8 || c == 23;
  }
  bool IsConfigSupported(uint32_t config_idx) const override {
    return config_idx < 24 && !unsupportedConfig[config_idx];
  }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override {
    if (!IsSupported(info, context)) {
      return false;
    }
    if (config_idx >= 24 || unsupportedConfig[config_idx]) {
      return false;
    }
    if (isSERConfigIdx(config_idx)) {
      if (!info.serSupported) return false;
#ifdef HAVE_VULKAN
      if (context && context->getBackend() == ComputeBackend::Vulkan) {
        const VulkanContext *vc = static_cast<const VulkanContext *>(context);
        if (!vc->isSERSupported()) return false;
      }
#endif
    }
    if (config_idx == 2 || isStandardDGCConfigIdx(config_idx)) {
      if (!info.dgcSupported) return false;
#ifdef HAVE_VULKAN
      if (context && context->getBackend() == ComputeBackend::Vulkan) {
        const VulkanContext *vc = static_cast<const VulkanContext *>(context);
        if (!vc->isDGCSupported()) return false;
      }
#endif
    }
    return true;
  }
  std::string GetConfigSupportNote(uint32_t config_idx) const override {
    return (config_idx < 24) ? unsupportedReason[config_idx] : std::string();
  }
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override {
    if (config_idx < 24 && !unsupportedReason[config_idx].empty()) {
      return unsupportedReason[config_idx];
    }
    if (isSERConfigIdx(config_idx)) {
      return "VK_EXT_ray_tracing_invocation_reorder not supported by device/driver";
    }
    if (config_idx == 20) {
      return "Vulkan Ray Tracing Pipeline unsupported or pipeline creation failed";
    }
    if (config_idx == 2 || isStandardDGCConfigIdx(config_idx)) {
      if (!info.dgcSupported) {
        return "VK_EXT_device_generated_commands unsupported by device/driver";
      }
#ifdef HAVE_VULKAN
      if (context && context->getBackend() == ComputeBackend::Vulkan) {
        const VulkanContext *vc = static_cast<const VulkanContext *>(context);
        if (!vc->isDGCSupported()) {
          return "VK_EXT_device_generated_commands unsupported by device/driver";
        }
      }
#endif
    }
    if (!IsSupported(info, context)) {
      return GetSupportNote(info, context);
    }
    return "";
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx) const override {
    if (isSERConfigIdx(config_idx) || config_idx == 20 ||
        config_idx == 2 || isStandardDGCConfigIdx(config_idx)) {
      return SupportLimitation::kHardware;
    }
    return SupportLimitation::kNone;
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context = nullptr) const override {
    if (context && (context->getBackend() == ComputeBackend::OpenCL ||
                    context->getBackend() == ComputeBackend::ROCm)) {
      return SupportLimitation::kApi;
    }
    if (config_idx == 2) {
      bool basicDgc = info.dgcSupported;
#ifdef HAVE_VULKAN
      if (context && context->getBackend() == ComputeBackend::Vulkan) {
        const VulkanContext *vc = static_cast<const VulkanContext *>(context);
        basicDgc = vc->isDGCSupported();
      }
#endif
      return basicDgc ? SupportLimitation::kNone : SupportLimitation::kHardware;
    }
    if (isSERConfigIdx(config_idx) && info.serSupported) {
      return SupportLimitation::kApi;
    }
    return GetConfigSupportLimitation(config_idx);
  }
  SupportDiscrepancy GetConfigSupportDiscrepancy(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override {
    SupportDiscrepancy d{};
    if (!context || context->getBackend() != ComputeBackend::Vulkan) {
      return d;
    }
#ifdef HAVE_VULKAN
    const VulkanContext *vc = static_cast<const VulkanContext *>(context);
    if (isSERConfigIdx(config_idx)) {
      bool driverSer = vc ? vc->isSERSupported() : info.serSupported;
      bool hwSer = vc ? vc->isSERHardwareSupported() : info.serHardwareSupported;
      if (driverSer && !hwSer) {
        d.differs = true;
        d.hardwareStatus = "Unsupported natively (reorderingHint = NONE; no dedicated hardware SER unit)";
        d.driverStatus = "Supported (VK_EXT_ray_tracing_invocation_reorder = true; allowed to run)";
        d.summary = "Driver exposes SER API (pass-through/no-op); GPU lacks dedicated hardware SER";
      }
    } else if (config_idx == 2) {
      bool basicDgc = vc ? vc->isDGCSupported() : info.dgcSupported;
      bool execSet = vc ? vc->isDGCExecutionSetSupported() : info.dgcExecutionSetSupported;
      if (basicDgc && !execSet) {
        d.differs = true;
        d.hardwareStatus = "Partial DGC active (VK_EXT_device_generated_commands compute stage supported)";
        d.driverStatus = "Unsupported for IndirectExecutionSet (supportedIndirectCommandsShaderStagesPipelineBinding omits COMPUTE_BIT)";
        d.summary = "Driver supports compute DGC, but omits compute pipeline binding for IndirectExecutionSet";
      }
    }
#endif
    return d;
  }
  bool IsEmulated(uint32_t config_idx = 0) const override {
    return isSERConfigIdx(config_idx) && serEmulated;
  }
  bool IsDGCAvailable() const { return isDGCAvailable; }
  bool IsDGCExecutionSetAvailable() const { return isDGCExecutionSetAvailable; }
  std::string GetConfigCaveat(uint32_t config_idx = 0) const override;
  std::string GetConfigCaveat(uint32_t config_idx,
                              const DeviceInfo &info,
                              IComputeContext *context = nullptr) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override { return "Ray Tracing"; }
  const char *GetSubCategory(uint32_t config_idx = 0) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  bool RequiresInterIterationBarrier(uint32_t /*config_idx*/ = 0) const override {
    return true;
  }

  void SetDumpRenders(bool dump) { wantDumpRenders = dump; }
  bool GetDumpRenders() const { return wantDumpRenders; }
  void SetCaptureGpuPreviews(bool capture) { captureGpuPreviews = capture; }
  bool GetCaptureGpuPreviews() const { return captureGpuPreviews; }

  void SetResolution(uint32_t w, uint32_t h) override {
    renderWidth = w;
    renderHeight = h;
    rayCount = renderWidth * renderHeight;
    queueCapacity = 65536;
    while (queueCapacity < rayCount / 6) {
      queueCapacity *= 2;
    }
  }
  bool IsResolutionDependent(uint32_t = 0) const override { return true; }
  void SetBounceDepth(uint32_t bounces);
  uint32_t GetBounceDepth() const { return bounceDepth; }

  void SetSamplesPerPixel(uint32_t spp) { samplesPerPixel = std::clamp(spp, 1u, 256u); }
  uint32_t GetSamplesPerPixel() const { return samplesPerPixel; }

  uint32_t GetRenderWidth() const { return renderWidth; }
  uint32_t GetRenderHeight() const { return renderHeight; }
  uint32_t GetQueueCapacity() const { return queueCapacity; }

  void RecordRunResult(uint32_t config_idx, uint64_t total_invocations, double total_time_ms) override {
    if (config_idx < 24) {
      recordedInvocations[config_idx] = total_invocations;
      recordedTimeMs[config_idx] = total_time_ms;
    }
  }

  bool HasVisualVerification() const override {
    return wantDumpRenders || verifyParity || captureGpuPreviews;
  }
  void RunVisualVerification(bool isInteractive = false) override;

private:
  uint64_t recordedInvocations[24] = {0};
  double recordedTimeMs[24] = {0.0};
  IComputeContext *context = nullptr;
  bool dumpRenders = false;
  bool captureFramebuffer = false;
  bool wantDumpRenders = false;
  bool captureGpuPreviews = false;
  bool verifyParity = false;
  bool parityFailure = false;
  std::string cachedKernelDir;
  std::vector<float> compactBvhPositions;
  ComputeBuffer fbTraditional = nullptr;
  ComputeBuffer fbWorkList = nullptr;
  void performVisualVerification(bool isInteractive = false);
  void dumpPipelineBreakdown(const std::string &tag, bool isInteractive = false);
  double bvhBuildTimeMs = 0.0;

  // Compute & RT Kernels
  ComputeKernel kernelTraditional = nullptr;
  ComputeKernel kernelTraditionalSpecialized[13] = {nullptr};
  ComputeKernel kernelRTP = nullptr;
  ComputeKernel kernelRTPSER = nullptr;
  ComputeKernel kernelClassify = nullptr;
  ComputeKernel kernelClassifySpecialized[10] = {nullptr};
  ComputeKernel kernelMaterial = nullptr;
  ComputeKernel kernelMaterialGltf = nullptr;
  ComputeKernel kernelMaterialSpecialized[8] = {nullptr};
  ComputeKernel kernelBounce = nullptr;
  ComputeKernel kernelBounceTerminal = nullptr;
  ComputeKernel kernelMultiLight = nullptr;
  ComputeKernel kernelReset = nullptr;
  ComputeKernel kernelResolve = nullptr;

  // Storage Buffers
  ComputeBuffer resultBuffer = nullptr;
  ComputeBuffer workListBuffer = nullptr;
  ComputeBuffer indirectBuffer = nullptr;
  ComputeBuffer indirectMaterialBuffer = nullptr;

  // PBR Scene & Material Storage Buffers
  bool isGltf = true;
  ComputeBuffer materialBuffer = nullptr;
  ComputeBuffer triangleMaterialBuffer = nullptr;
  ComputeBuffer texHeaderBuffer = nullptr;
  ComputeBuffer texPixelBuffer = nullptr;

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
  std::vector<VulkanContext::IndirectBatchEntry> multiLightBatches128;
  void rebuildBounceBatches();

#endif
  bool isDGCAvailable = false;
  bool isDGCExecutionSetAvailable = false;
  bool serEmulated = false;
#ifdef HAVE_VULKAN
  VkIndirectCommandsLayoutEXT dgcLayoutStandard = VK_NULL_HANDLE;
  VkIndirectCommandsLayoutEXT dgcLayoutSpecialized = VK_NULL_HANDLE;
  VkIndirectExecutionSetEXT dgcExecutionSetSpecialized = VK_NULL_HANDLE;
  ComputeBuffer dgcSequenceBuffer = nullptr;
  ComputeBuffer dgcSequenceCountBuffer = nullptr;
  ComputeBuffer dgcSpecializedSequenceBuffer = nullptr;
  ComputeBuffer dgcPreprocessBuffer = nullptr;
  VkDeviceSize dgcPreprocessBufferSize = 0;

  VulkanContext::DGCExecutionInfo dgcInfoStandard;
  VulkanContext::DGCExecutionInfo dgcInfoFullFrame;
  VulkanContext::DGCExecutionInfo dgcInfoSpecialized;
  std::vector<VulkanContext::DGCExecutionInfo> dgcBounceBatches;
  void rebuildDGCBounceBatches();
#endif

  uint32_t bounceDepth = 2;
  uint32_t samplesPerPixel = 1;
  uint32_t renderWidth = 1920;
  uint32_t renderHeight = 1080;
  uint32_t rayCount = 1920 * 1080;
  uint32_t queueCapacity = 850000;
  uint32_t materialCapacity = 850000;
  uint32_t bounceCapacity = 1500000;
  uint32_t octantCapacity = 262144;
  uint32_t numPrimitives = 4096;
  SceneType sceneType = SceneType::ReactorBay;
  mutable double results[24] = {0.0};
  mutable uint32_t lastCompletedPasses[24] = {0};
  mutable bool unsupportedConfig[24] = {false};
  mutable std::string unsupportedReason[24];
  uint32_t lastRunConfig = 0xFFFFFFFFu;
  std::mt19937 rng{1337u};

  template <typename Tag> friend struct ::gpubench::test::TestBody;
};
