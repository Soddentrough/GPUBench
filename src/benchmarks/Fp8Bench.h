#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"

class Fp8Bench : public IBenchmark {
public:
    const char* GetName() const override;
    std::vector<std::string> GetAliases() const override { return {"f8"}; }
    bool IsSupported(const DeviceInfo& info, IComputeContext* context = nullptr) const override;
    void Setup(IComputeContext& context, const std::string& kernel_dir) override;
    void Run(uint32_t config_idx = 0) override;
    void Teardown() override;
    BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
    
    uint32_t GetNumConfigs() const override;
    std::string GetConfigName(uint32_t config_idx) const override;
    const char* GetMetric() const override { return "TFLOPs"; }
    
    bool IsEmulated() const override { return is_emulated; }

private:
    IComputeContext* context = nullptr;
    ComputeKernel vectorKernel = nullptr;
    ComputeKernel matrixKernel = nullptr;
    ComputeBuffer buffer = nullptr;
    bool is_emulated = false;
    bool is_native_vector = false;
    bool is_native_matrix = false;
    mutable std::string name = "FP8";
};
