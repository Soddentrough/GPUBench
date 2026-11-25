#pragma once

#include "IBenchmark.h"
#include <map>
#include <string>
#include <vulkan/vulkan.h>

class VulkanContext;

class RayASBuildBench : public IBenchmark {
public:
  RayASBuildBench() = default;
  ~RayASBuildBench() override = default;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context) const override;
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
    return "Ray tracing AS build requires Vulkan VK_KHR_acceleration_structure";
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
      return "extension VK_KHR_acceleration_structure missing";
    }
    return "Ray tracing AS build requires Vulkan VK_KHR_acceleration_structure";
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx) override;
  void Teardown() override;

  BenchmarkResult GetResult(uint32_t config_idx) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;
  const char *GetName() const override;
  const char *GetComponent(uint32_t config_idx) const override;
  const char *GetMetric(uint32_t config_idx) const override;
  const char *GetSubCategory(uint32_t config_idx) const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  uint32_t GetNumConfigs() const override { return 4; } // 1 batched multi-mesh BLAS build + 3 Real-World Branched TLAS scenes
  const char *GetDescription() const override {
    return "Hardware BVH batched multi-mesh Bottom-Level (BLAS) and Top-Level (TLAS) scene acceleration structure construction throughput.";
  }
  bool RequiresInterIterationBarrier(uint32_t /*config_idx*/ = 0) const override {
    return true;
  }
  int GetSortWeight(uint32_t config_idx = 0) const override {
    return 600 + static_cast<int>(config_idx);
  }

  static constexpr uint32_t kBlasBatchMeshes = 256;
  static constexpr uint32_t kBlasBatchTotalTris = 5000000;

  struct BlasInfo {
    uint32_t numPrimitives = 0;
    ComputeBuffer buffer = nullptr;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
  };

  struct TlasInfo {
    std::string name;
    uint32_t numInstances = 0;
    ComputeBuffer instanceBuffer = nullptr;
    ComputeBuffer buffer = nullptr;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
  };

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  void loadRTProcs(VkDevice device);
  void recordBuildCommand(VkCommandBuffer cmd, uint32_t config_idx,
                          VulkanContext *vContext);

  IComputeContext *context = nullptr;

  // Batched multi-mesh BLAS benchmark resources (256 meshes, 5M tris total)
  ComputeBuffer vertexBuffer = nullptr;
  ComputeBuffer scratchBuffer = nullptr;
  std::vector<BlasInfo> blases;
  std::vector<VkAccelerationStructureKHR> blasBatchHandles;
  std::vector<VkDeviceSize> blasBatchOffsets;
  std::vector<VkAccelerationStructureGeometryKHR> blasBatchGeometries;
  std::vector<VkAccelerationStructureBuildGeometryInfoKHR> blasBatchBuildInfos;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR> blasBatchRanges;
  std::vector<const VkAccelerationStructureBuildRangeInfoKHR *> blasBatchRangePtrs;

  // Multi-BLAS scene library (5,000 distinct geometries)
  ComputeBuffer blasLibVertexBuffer = nullptr;
  ComputeBuffer blasLibPooledBuffer = nullptr;
  std::vector<ComputeBuffer> blasLibBuffers;
  std::vector<VkAccelerationStructureKHR> blasLibHandles;
  std::vector<VkDeviceAddress> blasLibAddrs;

  // Real-world branched TLAS scenes
  std::vector<TlasInfo> tlases;

  // Persistent command resources for benchmark execution
  static constexpr uint32_t kNumFrames = 4;
  struct BuildFrame {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool inFlight = false;
  };
  VkCommandPool buildCmdPool = VK_NULL_HANDLE;
  BuildFrame buildFrames[kNumFrames];
  uint32_t frameIndex = 0;

  // Function pointers
  PFN_vkGetAccelerationStructureBuildSizesKHR
      vkGetAccelerationStructureBuildSizesKHR_ptr;
  PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR_ptr;
  PFN_vkCmdBuildAccelerationStructuresKHR
      vkCmdBuildAccelerationStructuresKHR_ptr;
  PFN_vkGetAccelerationStructureDeviceAddressKHR
      vkGetAccelerationStructureDeviceAddressKHR_ptr;
  PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR_ptr;
};
