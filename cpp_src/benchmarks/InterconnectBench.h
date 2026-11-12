#pragma once

#include "IBenchmark.h"
#include "core/VulkanContext.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

class InterconnectBench : public IBenchmark {
public:
  const char *GetName() const override {
    return "PCIe & Host-Device Interconnect";
  }
  std::vector<std::string> GetAliases() const override {
    return {"interconnect", "pcie", "p2p", "rebar"};
  }
  const char *GetMetric() const override { return "GB/s"; }
  const char *GetMetric(uint32_t config_idx) const override {
    return (config_idx == 4u) ? "ns" : "GB/s";
  }
  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Memory";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "Interconnect";
  }
  int GetSortWeight() const override { return 205; }
  uint32_t GetExpectedKernelCount() const override { return 1; }

  uint32_t GetNumConfigs() const override { return 6; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;

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

  bool UsesCustomTiming(uint32_t config_idx = 0) const override;
  double GetLastRunDurationMs(uint32_t config_idx = 0) const override;

  void Setup(IComputeContext &context,
             const std::string &kernel_dir) override;
  void PrepareConfig(uint32_t config_idx) override;
  void ReleaseConfigResources(uint32_t config_idx) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

private:
  static constexpr size_t kDmaBytes = 128ULL * 1024ULL * 1024ULL;
  static constexpr size_t kRebarBytes = 64ULL * 1024ULL * 1024ULL;
  static constexpr uint32_t kPingPongSteps = 2048u;

  IComputeContext *m_context = nullptr;
  VulkanContext *m_vk = nullptr;
  std::string m_kernelDir;

  ComputeKernel m_helperKernel = nullptr;
  VkCommandPool m_primaryPool = VK_NULL_HANDLE;
  VkCommandBuffer m_primaryCmd = VK_NULL_HANDLE;
  VkFence m_primaryFence = VK_NULL_HANDLE;
  VkQueryPool m_tsPool = VK_NULL_HANDLE;

  VkCommandPool m_transferPool = VK_NULL_HANDLE;
  VkCommandBuffer m_transferCmd = VK_NULL_HANDLE;
  VkFence m_transferFence = VK_NULL_HANDLE;
  VkCommandPool m_secondaryTransferPool = VK_NULL_HANDLE;
  VkCommandBuffer m_secondaryTransferCmd = VK_NULL_HANDLE;
  VkFence m_secondaryTransferFence = VK_NULL_HANDLE;

  // Per-config lazily allocated buffers
  ComputeBuffer m_pinnedUpload = nullptr;
  void *m_mappedUpload = nullptr;
  ComputeBuffer m_devDst = nullptr;

  ComputeBuffer m_devSrc = nullptr;
  ComputeBuffer m_pinnedDownload = nullptr;
  void *m_mappedDownload = nullptr;

  // Concurrent-sharing buffers when dedicated transfer family differs
  VkBuffer m_xferHostSrcBuf = VK_NULL_HANDLE;
  VmaAllocation m_xferHostSrcAlloc = VK_NULL_HANDLE;
  void *m_xferHostSrcMapped = nullptr;
  VkBuffer m_xferDevDstBuf = VK_NULL_HANDLE;
  VmaAllocation m_xferDevDstAlloc = VK_NULL_HANDLE;
  VkBuffer m_xferDevSrcBuf = VK_NULL_HANDLE;
  VmaAllocation m_xferDevSrcAlloc = VK_NULL_HANDLE;
  VkBuffer m_xferHostDstBuf = VK_NULL_HANDLE;
  VmaAllocation m_xferHostDstAlloc = VK_NULL_HANDLE;
  void *m_xferHostDstMapped = nullptr;

  ComputeBuffer m_rebarBuf = nullptr;
  void *m_mappedRebar = nullptr;
  ComputeBuffer m_verifyOutBuf = nullptr;

  // Vulkan DeviceGroup P2P resources (Config 5)
  VkDevice m_p2pDevice = VK_NULL_HANDLE;
  VkQueue m_p2pQueue = VK_NULL_HANDLE;
  VkCommandPool m_p2pPool = VK_NULL_HANDLE;
  VkCommandBuffer m_p2pCmd = VK_NULL_HANDLE;
  VkFence m_p2pFence = VK_NULL_HANDLE;
  VkBuffer m_p2pSrcBuf = VK_NULL_HANDLE;
  VkDeviceMemory m_p2pSrcMem = VK_NULL_HANDLE;
  VkBuffer m_p2pDstBuf = VK_NULL_HANDLE;
  VkDeviceMemory m_p2pDstMem = VK_NULL_HANDLE;
  VkBuffer m_p2pStageBuf = VK_NULL_HANDLE;
  VkDeviceMemory m_p2pStageMem = VK_NULL_HANDLE;
  void *m_p2pStageMapped = nullptr;

  std::array<double, 6> m_lastDurationMs{};
  uint32_t m_rebarSeed = 0xC33C5A5Au;

  bool probeVulkanDeviceGroupP2P(const VulkanContext *vc,
                                 std::string &outReason) const;
};
