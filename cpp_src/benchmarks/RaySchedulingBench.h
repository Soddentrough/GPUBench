#pragma once

#include "IBenchmark.h"
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
  const char *GetName() const override { return "RayScheduling"; }

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kApi;
  }
  std::string GetSupportNote() const override {
    return "Ray Scheduling benchmark requires Vulkan ray query / ray tracing acceleration structures";
  }

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;

  uint32_t GetNumConfigs() const override { return 16; }
  std::vector<std::string> GetAliases() const override {
    return {"rayscheduling", "rtscheduling", "rayexecutionparadigm", "rayparadigm", "rtparadigm", "workgraphs", "worklists", "dgc", "ser"};
  }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetMetric(uint32_t config_idx = 0) const override {
    if (config_idx == 13 || config_idx == 14) return "MHits/s";
    if (config_idx == 15) return "MRecords/s";
    return "MRays/s";
  }
  bool IsConfigSupported(uint32_t config_idx) const override {
    return !unsupportedConfig[config_idx];
  }
  std::string GetConfigSupportNote(uint32_t config_idx) const override {
    return unsupportedReason[config_idx];
  }
  SupportLimitation GetConfigSupportLimitation(uint32_t config_idx) const override {
    if (config_idx == 1 || config_idx == 5 || config_idx == 9) {
      return SupportLimitation::kHardware;
    }
    if (config_idx == 3 || config_idx == 7 || config_idx == 11) {
      return SupportLimitation::kApi;
    }
    return SupportLimitation::kNone;
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
  uint32_t GetRenderWidth() const { return renderWidth; }
  uint32_t GetRenderHeight() const { return renderHeight; }
  uint32_t GetQueueCapacity() const { return queueCapacity; }

  void RecordRunResult(uint32_t config_idx, uint64_t total_invocations, double total_time_ms) override {
    if (config_idx < 16) {
      recordedInvocations[config_idx] = total_invocations;
      recordedTimeMs[config_idx] = total_time_ms;
    }
  }

private:
  uint64_t recordedInvocations[16] = {0};
  double recordedTimeMs[16] = {0.0};
  IComputeContext *context = nullptr;
  bool dumpRenders = false;
  ComputeBuffer fbTraditional = nullptr;
  ComputeBuffer fbWorkList = nullptr;
  void performVisualVerification();

  // Compute Kernels
  ComputeKernel kernelTraditional = nullptr;
  ComputeKernel kernelClassify = nullptr;
  ComputeKernel kernelMaterial = nullptr;
  ComputeKernel kernelMaterialSpecialized[6] = {nullptr};
  ComputeKernel kernelBounce = nullptr;
  ComputeKernel kernelWorkGraph = nullptr;

  // Storage Buffers
  ComputeBuffer resultBuffer = nullptr;
  ComputeBuffer workListBuffer = nullptr;
  ComputeBuffer indirectBuffer = nullptr;

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
#endif

  uint32_t renderWidth = 1920;
  uint32_t renderHeight = 1080;
  uint32_t rayCount = 1920 * 1080;
  uint32_t queueCapacity = 262144;
  uint32_t numPrimitives = 4096;
  mutable double results[16] = {0.0};
  mutable bool unsupportedConfig[16] = {false};
  mutable std::string unsupportedReason[16];
};
