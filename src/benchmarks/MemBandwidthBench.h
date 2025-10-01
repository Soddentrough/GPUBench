#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <vector>
#include <string>

struct BandwidthConfig {
    std::string name;
    std::string kernelFile;
    uint32_t workgroupSize;
    uint32_t numWorkgroups;
    ComputeKernel kernel;
};

class MemBandwidthBench : public IBenchmark {
public:
    const char* GetName() const override;
    const char* GetMetric() const override;
    bool IsSupported(const DeviceInfo& info, IComputeContext* context = nullptr) const override;
    void Setup(IComputeContext& context, const std::string& kernel_dir) override;
    void Run() override;
    void Teardown() override;
    BenchmarkResult GetResult() const override;

private:
    IComputeContext* context = nullptr;
    std::vector<BandwidthConfig> configs;
    ComputeBuffer inputBuffer = nullptr;
    ComputeBuffer outputBuffer = nullptr;
    
    size_t currentConfigIndex = 0;
    
    void createKernel(BandwidthConfig& config, const std::string& kernel_dir);
};
