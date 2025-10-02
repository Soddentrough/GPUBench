#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"

class Fp4Bench : public IBenchmark {
public:
    const char* GetName() const override;
    bool IsSupported(const DeviceInfo& info, IComputeContext* context = nullptr) const override;
    void Setup(IComputeContext& context, const std::string& kernel_dir) override;
    void Run() override;
    void Teardown() override;
    BenchmarkResult GetResult() const override;
    bool IsEmulated() const override { return is_emulated; }

private:
    IComputeContext* context = nullptr;
    ComputeKernel kernel = nullptr;
    ComputeBuffer buffer = nullptr;
    bool is_emulated = true;
    mutable std::string name = "FP4";
};
