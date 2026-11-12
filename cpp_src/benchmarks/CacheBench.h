#pragma once

#include "IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

class CacheBench : public IBenchmark {
public:
  CacheBench(std::string name, std::string metric, uint64_t bufferSize,
             std::string kernelFile, std::vector<uint32_t> initData = {},
             std::vector<std::string> aliases = {}, int targetCacheLevel = -1);
  ~CacheBench();

  const char *GetName() const override;
  std::vector<std::string> GetAliases() const override;
  const char *GetMetric() const override;
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)context;
    if (targetCacheLevel == 0) {
      return (metric == "GB/s")
                 ? "Measures nearest vector cache bandwidth: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1; Intel L1"
                 : "Measures nearest vector cache latency: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1 / Texture; Intel L1";
    }
    if (targetCacheLevel == 3 && info.l3CacheSize == 0) {
      return "Device does not have an L3 / Infinity Cache (l3CacheSize = 0)";
    }
    if (targetCacheLevel == 2 && info.l2CacheSize == 0) {
      return "Device does not have an L2 cache reported";
    }
    return "";
  }
  std::string GetSupportNote() const override {
    if (targetCacheLevel == 0) {
      return (metric == "GB/s")
                 ? "Measures nearest vector cache bandwidth: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1; Intel L1"
                 : "Measures nearest vector cache latency: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1 / Texture; Intel L1";
    }
    if (targetCacheLevel == 3) {
      return "Device does not have an L3 / Infinity Cache (l3CacheSize = 0)";
    }
    return "";
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kHardware;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return SupportLimitation::kHardware;
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override;
  const char *GetSubCategory(uint32_t config_idx = 0) const override;
  const char *GetDescription() const override {
    return (metric == "GB/s")
               ? "Multi-CU streaming GPU cache bandwidth across L0, L1, L2, and L3/MALL cache tiers."
               : "Pointer-chasing dependent load latency across GPU L0, L1, L2, L3/MALL, and VRAM tiers.";
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    (void)config_idx;
    // cache_bw_robust.comp executes 8 vec4 FP32 additions (32 FLOPs) per 8 vec4 loads (128 bytes) = 0.25 OP/Byte
    return (metric == "GB/s") ? 0.25 : 0.0;
  }
  int GetSortWeight() const override;

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  std::string name;
  std::vector<std::string> aliases;
  std::string metric;
  uint64_t bufferSize;
  std::string kernelFile;
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer outBuffer = nullptr;
  ComputeBuffer pcBuffer = nullptr;
  std::vector<uint32_t> initData;
  void *hostMem = nullptr;
  int targetCacheLevel = -1;
  uint32_t numWorkgroups = 1;
  bool debug = false;

public:
  void setDebug(bool d) { debug = d; }
};
