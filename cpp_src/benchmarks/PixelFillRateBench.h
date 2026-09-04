#pragma once

#include "benchmarks/IBenchmark.h"
#include <array>
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

class PixelFillRateBench : public IBenchmark {
public:
  PixelFillRateBench();
  ~PixelFillRateBench() override;

  const char *GetName() const override { return "Pixel Fill Rate"; }
  std::vector<std::string> GetAliases() const override {
    return {"pixelfill", "fillrate", "rop"};
  }
  const char *GetMetric() const override { return "GPixels/s"; }
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kApi;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return SupportLimitation::kApi;
  }
  std::string GetSupportNote() const override {
    return "Pixel Fill Rate benchmark requires Vulkan graphics rasterization pipeline (ROPs)";
  }
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)info;
    if (context && context->getBackend() == ComputeBackend::OpenCL) {
      return "No support for graphics rasterization pipeline (ROPs) in OpenCL API";
    }
    if (context && context->getBackend() == ComputeBackend::ROCm) {
      return "No support for graphics rasterization pipeline (ROPs) in ROCm API";
    }
    return "Pixel Fill Rate benchmark requires Vulkan graphics rasterization pipeline (ROPs)";
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;

  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Graphics";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    return "ROP Throughput";
  }
  int GetSortWeight() const override { return 450; }

  uint32_t GetNumConfigs() const override { return static_cast<uint32_t>(configs.size()); }
  std::string GetConfigName(uint32_t config_idx) const override;

private:
  struct Config {
    std::string name;
    VkFormat format;
    bool enableBlend;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
  };

  IComputeContext *context = nullptr;
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamilyIndex = 0;

  uint32_t width = 8192;
  uint32_t height = 8192;
  uint32_t passesPerDispatch = 8;

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkShaderModule vertShaderModule = VK_NULL_HANDLE;
  VkShaderModule fragShaderModule = VK_NULL_HANDLE;

  static constexpr size_t kMaxInFlight = 16;
  struct FrameSync {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool inFlight = false;
  };
  std::array<FrameSync, kMaxInFlight> frames{};
  size_t frameIndex = 0;
  VkCommandPool commandPool = VK_NULL_HANDLE;

  std::vector<Config> configs;
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
  void createPipelineForConfig(Config &cfg);
  VkShaderModule loadShaderModule(const std::string &path);
};
