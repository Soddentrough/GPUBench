#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

enum class TestMode { Read, Write, ReadWrite };

struct BandwidthConfig {
  std::string name;
  std::string kernelFile;
  uint32_t workgroupSize;
  uint32_t numWorkgroups;
  TestMode mode;
  ComputeKernel kernel;
};

class MemBandwidthBench : public IBenchmark {
public:
  MemBandwidthBench();
  const char *GetName() const override { return "Device Memory Bandwidth"; }
  std::vector<std::string> GetAliases() const override {
    return {"membw", "bandwidth", "devicemem"};
  }
  const char *GetMetric() const override;
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override;
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override {
    return "Memory";
  }
  const char *GetSubCategory(uint32_t config_idx = 0) const override {
    (void)config_idx;
    return "Bandwidth";
  }
  const char *GetDescription() const override {
    return "Streaming device VRAM read, write, and read-write bandwidth across coalesced workgroup sizes.";
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    // Streaming read/write does ~0.0625..0.125 ops per byte transferred
    return (config_idx % 3 == 2) ? 0.125 : 0.0625;
  }
  bool RequiresInterIterationBarrier(uint32_t config_idx = 0) const override {
    (void)config_idx;
    // Prevent Dispatch N+1 from overlapping with Dispatch N on high-CU GPUs
    // (e.g., 96 CUs / 96 MB Infinity Cache), ensuring each 512 MB sweep fully
    // evicts L3 before the next iteration reads chunk 0.
    return true;
  }
  int GetSortWeight() const override { return 300; }
  uint32_t GetNumConfigs() const override;
  std::string GetConfigName(uint32_t config_idx) const override;
  virtual uint32_t GetExpectedKernelCount() const override {
    return 9; // Read/Write/RW for 128, 256, and 1024 threads
  }

  void setDebug(bool debug) { this->debug = debug; }

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  IComputeContext *context = nullptr;
  std::vector<BandwidthConfig> configs;
  ComputeBuffer inputBuffer = nullptr;
  ComputeBuffer outputBuffer = nullptr;
  size_t bufferSize = 0;
  bool debug = false;

  void initDefaultConfigs(uint32_t maxWorkgroupSize = 1024);
  void createKernel(BandwidthConfig &config, const std::string &kernel_dir);
};
