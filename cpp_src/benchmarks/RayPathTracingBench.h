#pragma once

#include "IBenchmark.h"
#include <string>
#include <vector>
#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

class RayPathTracingBench : public IBenchmark {
public:
  const char *GetName() const override;
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
    return "Ray tracing requires Vulkan ray query acceleration structures";
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
    return "Ray tracing requires Vulkan ray query acceleration structures";
  }

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void RebuildAccelerationStructures() override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;

  uint32_t GetNumConfigs() const override { return 3; }
  int GetSortWeight(uint32_t = 0) const override { return 655; }
  std::vector<std::string> GetAliases() const override {
    return {"raypathtracing", "pathtracing", "pt"};
  }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override;
  const char *GetSubCategory(uint32_t config_idx = 0) const override;

private:
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer resultBuffer = nullptr;

#ifdef HAVE_VULKAN
  // Acceleration Structures
  VkAccelerationStructureKHR triangleBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR boxBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR sceneTlas = VK_NULL_HANDLE;

  ComputeBuffer vertexBuffer = nullptr;
  ComputeBuffer aabbBuffer = nullptr;
  ComputeBuffer instanceBuffer = nullptr;
  ComputeBuffer triangleBlasBuffer = nullptr;
  ComputeBuffer boxBlasBuffer = nullptr;
  ComputeBuffer tlasBuffer = nullptr;
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
#endif // HAVE_VULKAN

  uint32_t rayCount = 4000000;
  uint32_t numPrimitives = 4096;
  double results[3] = {0.0, 0.0, 0.0};

#ifdef HAVE_VULKAN
  void loadRTProcs(VkDevice device);
  void buildAS();
#endif // HAVE_VULKAN
};
