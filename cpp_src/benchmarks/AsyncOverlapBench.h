#pragma once

#include "IBenchmark.h"
#include "core/VulkanContext.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

class AsyncOverlapBench : public IBenchmark {
public:
  const char *GetName() const override { return "Async Compute & Copy Overlap"; }
  std::vector<std::string> GetAliases() const override {
    return {"asyncoverlap", "async", "ace", "sdma"};
  }
  const char *GetMetric() const override { return "us"; }
  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Scheduling";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "Multi-Queue Concurrency";
  }
  int GetSortWeight() const override { return 420; }
  uint32_t GetExpectedKernelCount() const override { return 2; }

  uint32_t GetNumConfigs() const override { return 4; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  int32_t GetBaselineConfigIndex(uint32_t config_idx) const override {
    if (config_idx == 1u) return 0;
    if (config_idx == 3u) return 2;
    return -1;
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
  std::string GetConfigCaveat(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;

  bool UsesCustomTiming(uint32_t /*config_idx*/ = 0) const override {
    return true;
  }
  double GetLastRunDurationMs(uint32_t config_idx = 0) const override {
    return (config_idx < m_lastDurationMs.size()) ? m_lastDurationMs[config_idx]
                                                  : 0.0;
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
  static constexpr uint32_t kGfxWidth = 3840u;
  static constexpr uint32_t kGfxHeight = 2160u;
  static constexpr uint32_t kGfxDepthLayers = 2048u;
  static constexpr uint32_t kAceComputeElements = 65536u;
  static constexpr uint32_t kComputeElements = 262144u;
  static constexpr size_t kCopyBytes = 128ULL * 1024ULL * 1024ULL;

  VulkanContext *m_vk = nullptr;
  std::string m_kernelDir;

  // Graphics Dynamic Rendering pipeline
  VkPipelineLayout m_gfxPipeLayout = VK_NULL_HANDLE;
  VkPipeline m_gfxPipeline = VK_NULL_HANDLE;
  VkImage m_colorImage = VK_NULL_HANDLE;
  VmaAllocation m_colorImageAlloc = VK_NULL_HANDLE;
  VkImageView m_colorView = VK_NULL_HANDLE;
  VkImage m_depthImage = VK_NULL_HANDLE;
  VmaAllocation m_depthImageAlloc = VK_NULL_HANDLE;
  VkImageView m_depthView = VK_NULL_HANDLE;
  bool m_colorImageInitialized = false;

  // Async compute pipeline
  VkDescriptorSetLayout m_compSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout m_compPipeLayout = VK_NULL_HANDLE;
  VkPipeline m_compPipeline = VK_NULL_HANDLE;
  VkDescriptorPool m_compDescPool = VK_NULL_HANDLE;
  VkDescriptorSet m_compDescSet = VK_NULL_HANDLE;

  VkBuffer m_compOutBuf = VK_NULL_HANDLE;
  VmaAllocation m_compOutAlloc = VK_NULL_HANDLE;

  // DMA copy buffers for Config 2 & 3
  VkBuffer m_dmaSrcBuf = VK_NULL_HANDLE;
  VmaAllocation m_dmaSrcAlloc = VK_NULL_HANDLE;
  VkBuffer m_dmaDstBuf = VK_NULL_HANDLE;
  VmaAllocation m_dmaDstAlloc = VK_NULL_HANDLE;
  void *m_dmaDstMapped = nullptr;

  // Command pools, buffers, fences
  VkCommandPool m_gfxPool = VK_NULL_HANDLE;
  VkCommandBuffer m_gfxCmd = VK_NULL_HANDLE;
  VkFence m_gfxFence = VK_NULL_HANDLE;

  VkCommandPool m_asyncPool = VK_NULL_HANDLE;
  VkCommandBuffer m_asyncCmd = VK_NULL_HANDLE;
  VkFence m_asyncFence = VK_NULL_HANDLE;

  VkCommandPool m_xferPool = VK_NULL_HANDLE;
  VkCommandBuffer m_xferCmd = VK_NULL_HANDLE;
  VkFence m_xferFence = VK_NULL_HANDLE;

  std::array<double, 4> m_lastDurationMs{};
  float m_runSeed = 0.25f;

  VkShaderModule loadSpirvModule(const std::string &shaderStem) const;
};
