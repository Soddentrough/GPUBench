#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif
#include <array>
#include <cstdint>
#include <string>
#include <vector>

enum class IndirectBenchMode {
  CPU_Direct,
  GPU_Indirect,
  GPU_MultiBucket_Legacy,
  GPU_MultiBucket_DGC
};

struct IndirectBenchConfig {
  std::string name;
  IndirectBenchMode mode;
  uint32_t selectivity; // 0..100%
  std::string hint;
};

class InShaderIndirectBench : public IBenchmark {
public:
  InShaderIndirectBench();
  ~InShaderIndirectBench() override;

  const char *GetName() const override { return "In-Shader Indirect Synthesis"; }
  std::vector<std::string> GetAliases() const override {
    return {"indirect", "indirect-dispatch", "inshaderindirect", "in-shader-indirect", "indirect_synth", "command_synth"};
  }
  const char *GetMetric() const override { return "us"; }
  const char *GetMetric(uint32_t /*config_idx*/) const override { return "us"; }
  int32_t GetBaselineConfigIndex(uint32_t config_idx) const override {
    switch (config_idx) {
    case 1:
    case 2:
      return 0;
    case 3:
      return 1;
    case 4:
      return 2;
    case 6:
      return 5;
    default:
      return -1;
    }
  }
  bool IsSupported(const DeviceInfo & /*info*/, IComputeContext *context = nullptr) const override {
    if (context && context->getBackend() != ComputeBackend::Vulkan &&
        context->getBackend() != ComputeBackend::ROCm) {
      return false;
    }
    return true;
  }
  bool IsConfigSupported(uint32_t config_idx) const override {
    if (config_idx >= configs.size()) return false;
    return !mbSetupComplete || (config_idx != 6u) || mbDgcAvailable;
  }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override;
  SupportLimitation GetConfigSupportLimitation(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  std::string GetSupportNote() const override {
    return "Requires Vulkan 1.3+ indirect dispatch or ROCm persistent GPU work queue";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override;
  std::string GetConfigSupportNote(uint32_t config_idx) const override;
  std::string GetConfigSupportNote(uint32_t config_idx,
                                   const DeviceInfo &info,
                                   IComputeContext *context = nullptr) const override;
  std::string GetConfigCaveat(uint32_t config_idx,
                              const DeviceInfo &info,
                              IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

  const char *GetComponent(uint32_t config_idx = 0) const override { (void)config_idx; return "Scheduling"; }
  const char *GetSubCategory(uint32_t config_idx = 0) const override { (void)config_idx; return "Indirect Command Synthesis"; }
  const char *GetDescription() const override {
    return "GPU-side indirect dispatch argument synthesis (Vulkan vkCmdDispatchIndirect / DGC ExecutionSet) or persistent work-queue compaction (ROCm HIP) vs CPU direct dispatch.";
  }
  const char *GetDescription(uint32_t config_idx) const override {
    if (config_idx < configs.size()) {
      return configs[config_idx].hint.c_str();
    }
    return GetDescription();
  }
  const char *GetDescription(uint32_t config_idx, const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override;
  bool RequiresInterIterationBarrier(uint32_t /*config_idx*/ = 0) const override {
    return true;
  }
  int GetSortWeight() const override { return 28; }

  uint32_t GetNumConfigs() const override { return static_cast<uint32_t>(configs.size()); }
  uint32_t GetExpectedKernelCount() const override { return 14; }
  std::string GetConfigName(uint32_t config_idx) const override;
  std::string GetConfigName(uint32_t config_idx, const DeviceInfo &info,
                            IComputeContext *context = nullptr) const override;

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  std::vector<IndirectBenchConfig> configs;

  ComputeKernel kernelReset = nullptr;
  ComputeKernel kernelClassify = nullptr;
  ComputeKernel kernelIndirectWorker = nullptr;
  ComputeKernel kernelDirectWorker = nullptr;

  ComputeBuffer inputBuffer = nullptr;
  ComputeBuffer workListBuffer = nullptr;
  ComputeBuffer indirectBuffer = nullptr;
  ComputeBuffer outputBuffer = nullptr;

  // 128-Bucket Sparse Multi-Pipeline Showcase (Legacy Indirect vs. DGC ExecutionSet)
  ComputeKernel kernelMbClassify = nullptr;
  ComputeKernel kernelMbResolve = nullptr;
  std::array<ComputeKernel, 8> kernelMbWorker{};
  ComputeBuffer mbWorkListBuffer = nullptr;
  ComputeBuffer mbBucketCountsBuffer = nullptr;
  ComputeBuffer mbIndirectBuffer = nullptr;
  ComputeBuffer mbDgcSeqBuffer = nullptr;
  ComputeBuffer mbDgcCountBuffer = nullptr;
  ComputeBuffer mbDgcPreprocessBuffer = nullptr;

  bool mbSetupComplete = false;
  bool mbDgcAvailable = false;
  std::string mbDgcUnsupportedReason;
  uint32_t m_rocmPersistentWorkgroups = 1024u;

#ifdef HAVE_VULKAN
  std::vector<VulkanContext::IndirectBatchEntry> mbLegacyEntries;
  VkIndirectCommandsLayoutEXT mbDgcLayout = VK_NULL_HANDLE;
  VkIndirectExecutionSetEXT mbDgcExecSet = VK_NULL_HANDLE;
  VulkanContext::DGCExecutionInfo mbDgcInfo{};
#endif

  static constexpr uint32_t kTotalItems = 4194304; // 4M items
  static constexpr uint32_t kWorkgroupSize = 64;
  static constexpr uint32_t kTotalWorkgroups = kTotalItems / kWorkgroupSize;
  static constexpr uint32_t kMbTotalItems = 131072u; // 128 * 1024 items -> 1024 items/bucket when active
  static constexpr uint32_t kMbNumBuckets = 128u;
  static constexpr uint32_t kMbBucketCapacity = 2048u;
};
