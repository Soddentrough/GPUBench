#pragma once

#include "IBenchmark.h"
#include <cstdint>
#include <string>
#include <vector>

class SparseMatrixBench : public IBenchmark {
public:
  const char *GetName() const override {
    return "2:4 Structured Sparse Matrix";
  }
  std::vector<std::string> GetAliases() const override {
    return {"sparsematrix", "sparse", "swmma", "smfma"};
  }
  const char *GetMetric() const override { return "TFLOPS"; }
  const char *GetMetric(uint32_t config_idx) const override {
    return (config_idx == 2u) ? "TOPS" : "TFLOPS";
  }
  const char *GetComponent(uint32_t /*config_idx*/ = 0) const override {
    return "Compute";
  }
  const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const override {
    return "Matrix";
  }
  int GetSortWeight() const override { return 108; }
  uint32_t GetExpectedKernelCount() const override { return 3; }

  uint32_t GetNumConfigs() const override { return 3; }
  std::string GetConfigName(uint32_t config_idx) const override;
  const char *GetDescription() const override { return GetDescription(0); }
  const char *GetDescription(uint32_t config_idx) const override;
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    return (config_idx == 0u) ? 4096.0 : 8192.0;
  }

  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                         IComputeContext *context = nullptr) const override;
  SupportLimitation GetSupportLimitation(
      const DeviceInfo &info,
      IComputeContext *context = nullptr) const override;
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override;

  void Setup(IComputeContext &context,
             const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;

private:
  IComputeContext *m_context = nullptr;
  ComputeBuffer m_inBuf = nullptr;
  ComputeBuffer m_outBuf = nullptr;
  ComputeKernel m_fp16Kernel = nullptr;
  ComputeKernel m_fp8Kernel = nullptr;
  ComputeKernel m_int8Kernel = nullptr;
  uint32_t m_blockSize = 32;
  uint32_t m_gridX = 4096;
};
