#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <vector>
#include <string>

enum class TestMode {
    Read,
    Write,
    ReadWrite
};

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
    const char* GetName() const override;
    std::vector<std::string> GetAliases() const override { return {"membw"}; }
    const char* GetMetric() const override;
    bool IsSupported(const DeviceInfo& info, IComputeContext* context = nullptr) const override;
    void Setup(IComputeContext& context, const std::string& kernel_dir) override;
    void Run(uint32_t config_idx = 0) override;
    void Teardown() override;
    BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
    uint32_t GetNumConfigs() const override;
    std::string GetConfigName(uint32_t config_idx) const override;

private:
    IComputeContext* context = nullptr;
    std::vector<BandwidthConfig> configs;
    ComputeBuffer inputBuffer = nullptr;
    ComputeBuffer outputBuffer = nullptr;
    size_t bufferSize = 0;
    
    void createKernel(BandwidthConfig& config, const std::string& kernel_dir);
};
