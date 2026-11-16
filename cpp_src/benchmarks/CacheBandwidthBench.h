#pragma once

#include "IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

class CacheBandwidthBench : public IBenchmark {
public:
  CacheBandwidthBench();
  ~CacheBandwidthBench() override;

  const char *GetName() const override { return "Cache Hierarchy Bandwidth"; }
  std::vector<std::string> GetAliases() const override {
    return {"cachebw", "cache_bw", "cache_bandwidth", "cachebandwidth",
            "l0b", "l1b", "l2b", "l3b", "l0bw", "l1bw", "l2bw", "l3bw",
            "l0_cache_bandwidth", "l1_cache_bandwidth",
            "l2_cache_bandwidth", "l3_cache_bandwidth"};
  }
  const char *GetMetric() const override { return "GB/s"; }
  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Memory";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "Cache Bandwidth";
  }
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  double GetOperationalIntensity(uint32_t /*config_idx*/ = 0) const override {
    // cache_bw_robust executes 8 vec4 FP32 additions (32 FLOPs) per 8 vec4 loads (128 bytes)
    return 0.25;
  }
  int GetSortWeight() const override { return 150; }
  uint32_t GetNumConfigs() const override { return 4; }
  uint32_t GetExpectedKernelCount() const override { return 1; }
  std::string GetConfigName(uint32_t config_idx) const override;

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return true;
  }
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kHardware;
  }
  SupportLimitation GetConfigSupportLimitation(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  std::string GetConfigSupportNote(uint32_t config_idx) const override;
  std::string GetConfigSupportNote(
      uint32_t config_idx, const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void PrepareConfig(uint32_t config_idx) override;
  void ReleaseConfigResources(uint32_t config_idx) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

  void setDebug(bool d) { debug = d; }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  uint64_t computeWorkingSetBytes(uint32_t config_idx, const DeviceInfo &info) const;

  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer outBuffer = nullptr;
  ComputeBuffer pcBuffer = nullptr;
  int32_t activeConfigIdx = -1;
  static constexpr uint32_t kNumWorkgroups = 2048u;
  bool debug = false;
};
