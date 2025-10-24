#pragma once

#include "core/IComputeContext.h"
#include <string>
#include <vector>

struct BenchmarkResult {
    uint64_t operations;
    double elapsedTime; // in milliseconds
};

class IBenchmark {
public:
    virtual ~IBenchmark() = default;
    virtual const char* GetName() const = 0;
    virtual std::vector<std::string> GetAliases() const { return {}; }
    virtual const char* GetMetric() const { return "TFLOPS"; }
    virtual bool IsSupported(const DeviceInfo& info, IComputeContext* context = nullptr) const = 0;
    virtual void Setup(IComputeContext& context, const std::string& kernel_dir) = 0;
    virtual void Run(uint32_t config_idx = 0) = 0;
    virtual void Teardown() = 0;
    virtual BenchmarkResult GetResult(uint32_t config_idx = 0) const = 0;
    virtual bool IsEmulated() const { return false; }
    virtual uint32_t GetNumConfigs() const { return 1; }
    virtual std::string GetConfigName(uint32_t config_idx) const { return ""; }
};
