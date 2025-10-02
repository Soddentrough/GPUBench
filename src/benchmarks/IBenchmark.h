#pragma once

#include "core/IComputeContext.h"
#include <string>

struct BenchmarkResult {
    uint64_t operations;
    double elapsedTime; // in milliseconds
};

class IBenchmark {
public:
    virtual ~IBenchmark() = default;
    virtual const char* GetName() const = 0;
    virtual const char* GetMetric() const { return "TFLOPS"; }
    virtual bool IsSupported(const DeviceInfo& info, IComputeContext* context = nullptr) const = 0;
    virtual void Setup(IComputeContext& context, const std::string& kernel_dir) = 0;
    virtual void Run() = 0;
    virtual void Teardown() = 0;
    virtual BenchmarkResult GetResult() const = 0;
    virtual bool IsEmulated() const { return false; }
};
