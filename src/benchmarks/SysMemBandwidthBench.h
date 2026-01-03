#pragma once

#include "benchmarks/IBenchmark.h"
#include <string>
#include <vector>

enum class SysMemTestMode { Read, Write, ReadWrite };

struct SysMemConfig {
  std::string name;
  SysMemTestMode mode;
  uint32_t numThreads = 0; // 0 = Auto/Max
};

class SysMemBandwidthBench : public IBenchmark {
public:
  SysMemBandwidthBench();
  virtual ~SysMemBandwidthBench();

  const char *GetName() const override;
  std::vector<std::string> GetAliases() const override {
    return {"sysmem", "ram", "bw"};
  }
  const char *GetMetric() const override;
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  uint32_t GetNumConfigs() const override;
  std::string GetConfigName(uint32_t config_idx) const override;

  // System benchmark is not tied to a specific GPU
  bool IsDeviceDependent() const override { return false; }

  // Not an emulated benchmark, it's a real system benchmark, but returns false
  // for GPU emulation
  bool IsEmulated() const override { return false; }

private:
  std::vector<SysMemConfig> configs;
  void *buffer = nullptr;
  void *destBuffer = nullptr; // For ReadWrite/Copy
  size_t bufferSize = 0;

  // Results
  double lastRunTimeMs = 0.0;
  uint64_t lastRunBytes = 0;
};
