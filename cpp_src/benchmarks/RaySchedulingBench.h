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
  const char *GetMetric(uint32_t config_idx = 0) const override { return "MRays/s"; }

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

  uint32_t GetNumConfigs() const override { return 12; }
  std::vector<std::string> GetAliases() const override {
    return {"rayscheduling", "rtscheduling", "rayexecutionparadigm", "rayparadigm", "rtparadigm", "workgraphs", "worklists", "dgc", "ser"};
  }
  std::string GetConfigName(uint32_t config_idx) const override;
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

private:
  IComputeContext *context = nullptr;

  // Compute Kernels
  ComputeKernel kernelTraditional = nullptr;
  ComputeKernel kernelClassify = nullptr;
  ComputeKernel kernelMaterial = nullptr;
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
  std::vector<VulkanContext::IndirectBatchEntry> bounceBatches;
  std::vector<VulkanContext::IndirectBatchEntry> octantBatches;
#endif

  uint32_t rayCount = 1000000;
  uint32_t numPrimitives = 4096;
  mutable double results[12] = {0.0};
  mutable bool unsupportedConfig[12] = {false};
  mutable std::string unsupportedReason[12];
};
