#pragma once

#include "IBenchmark.h"
#include "core/VulkanContext.h"
#include <cstdint>
#include <string>
#include <vector>

class WorkGraphsBench : public IBenchmark {
public:
  const char *GetName() const override {
    return "GPU Work Graphs (Producer-Consumer)";
  }
  std::vector<std::string> GetAliases() const override {
    return {"workgraphs", "shaderenqueue", "wg"};
  }
  const char *GetMetric() const override { return "MItems/s"; }
  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Scheduling";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "GPU Work Graphs";
  }
  int GetSortWeight() const override { return 415; }
  uint32_t GetExpectedKernelCount() const override { return 2; }

  uint32_t GetNumConfigs() const override { return 3; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  int32_t GetBaselineConfigIndex(uint32_t config_idx) const override {
    return (config_idx >= 1u && config_idx <= 2u) ? 0 : -1;
  }
  bool RequiresInterIterationBarrier(uint32_t /*config_idx*/ = 0) const override {
    return true;
  }

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation(
      const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  SupportLimitation GetConfigSupportLimitation(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override;
  std::string GetConfigSupportNote(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;

  bool UsesCustomTiming(uint32_t config_idx = 0) const override {
    return config_idx == 2u;
  }
  double GetLastRunDurationMs(uint32_t config_idx = 0) const override {
    return (config_idx == 2u) ? m_lastGraphDurationMs : 0.0;
  }

  void Setup(IComputeContext &context,
             const std::string &kernel_dir) override;
  void PrepareConfig(uint32_t config_idx) override;
  void ReleaseConfigResources(uint32_t config_idx) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

private:
  static constexpr uint32_t kTotalItems = 262144u; // 1024 * 256

  VulkanContext *m_vk = nullptr;
  std::string m_kernelDir;
  ComputeKernel m_producerKernel = nullptr;
  ComputeKernel m_consumerKernel = nullptr;

  ComputeBuffer m_indirectBuf = nullptr;
  ComputeBuffer m_payloadQueueBuf = nullptr;
  ComputeBuffer m_outBuf = nullptr;

  // DGC resources (Config 1)
  VkIndirectCommandsLayoutEXT m_dgcLayout = VK_NULL_HANDLE;
  ComputeBuffer m_dgcPreprocessBuf = nullptr;
  VkDeviceSize m_dgcPreprocessSize = 4096;
  ComputeBuffer m_dgcCountBuf = nullptr;

  // VK_AMDX_shader_enqueue resources (Config 2)
  VkDescriptorSetLayout m_wgSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout m_wgPipeLayout = VK_NULL_HANDLE;
  VkPipeline m_wgPipeline = VK_NULL_HANDLE;
  uint32_t m_wgProducerNodeIndex = 0;
  VkDescriptorPool m_wgDescPool = VK_NULL_HANDLE;
  VkDescriptorSet m_wgDescSet = VK_NULL_HANDLE;
  VkBuffer m_wgScratchBuf = VK_NULL_HANDLE;
  VmaAllocation m_wgScratchAlloc = VK_NULL_HANDLE;
  VkDeviceAddress m_wgScratchAddr = 0;
  VkDeviceSize m_wgScratchSize = 0;
  VkCommandPool m_wgCmdPool = VK_NULL_HANDLE;
  VkCommandBuffer m_wgCmd = VK_NULL_HANDLE;
  VkFence m_wgFence = VK_NULL_HANDLE;
  VkQueryPool m_wgTsPool = VK_NULL_HANDLE;
  double m_lastGraphDurationMs = 0.0;

  PFN_vkCreateExecutionGraphPipelinesAMDX m_pfnCreateGraphPipelines = nullptr;
  PFN_vkGetExecutionGraphPipelineScratchSizeAMDX m_pfnGetGraphScratchSize = nullptr;
  PFN_vkGetExecutionGraphPipelineNodeIndexAMDX m_pfnGetGraphNodeIndex = nullptr;
  PFN_vkCmdInitializeGraphScratchMemoryAMDX m_pfnCmdInitGraphScratch = nullptr;
  PFN_vkCmdDispatchGraphAMDX m_pfnCmdDispatchGraph = nullptr;
};
