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
                   IComputeContext *context) const override;
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override;
  const char *GetSubCategory(uint32_t config_idx = 0) const override;
  int GetSortWeight() const override;

private:
  std::string name;
  std::vector<std::string> aliases;
  std::string metric;
  uint64_t bufferSize;
  std::string kernelFile;
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer pcBuffer = nullptr;
  std::vector<uint32_t> initData;
  void *hostMem = nullptr;
  int targetCacheLevel = -1;
  uint32_t numWorkgroups = 1;
  bool debug = false;

public:
  void setDebug(bool d) { debug = d; }
};
