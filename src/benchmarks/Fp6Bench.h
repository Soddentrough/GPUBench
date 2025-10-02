#pragma once

#include "benchmarks/IBenchmark.h"

class Fp6Bench : public IBenchmark {
public:
    Fp6Bench();
    ~Fp6Bench() override;

    const char* GetName() const override;
    bool IsSupported(const DeviceInfo& device, IComputeContext* context = nullptr) const override;
    void Setup(IComputeContext& context, const std::string& build_dir) override;
    void Run() override;
    BenchmarkResult GetResult() const override;
    void Teardown() override;
    bool IsEmulated() const override { return is_emulated; }

private:
    IComputeContext* context = nullptr;
    ComputeKernel kernel = nullptr;
    ComputeBuffer buffer = nullptr;
    uint64_t operations = 0;
    bool is_emulated = true;
    mutable std::string name = "FP6";
};
