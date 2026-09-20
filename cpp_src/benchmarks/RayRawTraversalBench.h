#pragma once

#include "IBenchmark.h"
#include <cstdint>
#include <string>
#include <vector>
#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

class RayRawTraversalBench : public IBenchmark {
public:
  const char *GetName() const override;
  std::vector<std::string> GetAliases() const override;
  const char *GetMetric() const override;
  const char *GetMetric(uint32_t config_idx) const override;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kApi;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && (context->getBackend() == ComputeBackend::OpenCL ||
                    context->getBackend() == ComputeBackend::ROCm)) {
      return SupportLimitation::kApi;
    }
    return SupportLimitation::kHardware;
  }
  std::string GetSupportNote() const override {
    return "Raw BVH traversal requires Vulkan ray query acceleration structures";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for ray query acceleration structures in OpenCL";
    }
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "No support for ray query acceleration structures in ROCm";
    }
    if (!info.rayTracingSupport) {
      return "Extension VK_KHR_acceleration_structure or VK_KHR_ray_query missing";
    }
    return "Raw BVH traversal requires Vulkan ray query acceleration structures";
  }

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  void RecordRunResult(uint32_t config_idx, uint64_t total_invocations, double total_time_ms) override;
  std::string GetConfigCaveat(uint32_t config_idx = 0) const override;
  std::string GetConfigCaveat(uint32_t config_idx, const DeviceInfo &info,
                              IComputeContext *context = nullptr) const override;
  void DumpGeometry() const override;

  uint32_t GetNumConfigs() const override { return 2; }
  int GetSortWeight(uint32_t = 0) const override { return 669; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override;
  const char *GetSubCategory(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

  // Un-fictitious telemetry accessors
  uint64_t GetTraversedRays(uint32_t config_idx = 0) const;
  uint64_t GetSustainedOperations(uint32_t config_idx = 0) const;
  double GetThroughputGIS(uint32_t config_idx = 0) const;
  double GetThroughputMRays(uint32_t config_idx = 0) const;

private:
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer resultBuffer = nullptr;

#ifdef HAVE_VULKAN
  // Acceleration Structures
  VkAccelerationStructureKHR coherentBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR coherentTlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR deepBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR deepTlas = VK_NULL_HANDLE;

  ComputeBuffer vertexBufferCoherent = nullptr;
  ComputeBuffer vertexBufferDeep = nullptr;
  ComputeBuffer instanceBufferCoherent = nullptr;
  ComputeBuffer instanceBufferDeep = nullptr;
  ComputeBuffer coherentBlasBuffer = nullptr;
  ComputeBuffer coherentTlasBuffer = nullptr;
  ComputeBuffer deepBlasBuffer = nullptr;
  ComputeBuffer deepTlasBuffer = nullptr;
  ComputeBuffer scratchBuffer = nullptr;

  // RT function pointers
  PFN_vkGetAccelerationStructureBuildSizesKHR
      vkGetAccelerationStructureBuildSizesKHR_ptr = nullptr;
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ptr =
      nullptr;
  PFN_vkCmdBuildAccelerationStructuresKHR
      vkCmdBuildAccelerationStructuresKHR_ptr = nullptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR
      vkGetAccelerationStructureDeviceAddressKHR_ptr = nullptr;
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ptr =
      nullptr;

  void loadRTProcs(VkDevice device);
  void buildAS();
#endif // HAVE_VULKAN

  uint32_t rayCount = 64000000;
  uint32_t coherentPrimitives = 0;
  uint32_t deepPrimitives = 0;
  uint64_t recordedInvocations[2] = {0, 0};
  double recordedTimeMs[2] = {0.0, 0.0};
};
