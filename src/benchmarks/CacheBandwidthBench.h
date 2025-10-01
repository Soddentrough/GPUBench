#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <vector>
#include <string>

class CacheBandwidthBench : public IBenchmark {
public:
    const char* GetName() const override { return "Cache Bandwidth"; }
    bool IsSupported(const DeviceInfo& info, IComputeContext* context = nullptr) const override;
    void Setup(IComputeContext& context, const std::string& kernel_dir) override;
    void Run() override;
    void Teardown() override;
    BenchmarkResult GetResult() const override;

private:
    struct CacheConfig {
        std::string name;
        std::string kernelFile;
        uint32_t workgroupSize;
        uint32_t numWorkgroups;
        uint32_t bufferSizeBytes;  // Working set size for cache level
        uint32_t iterations;        // Number of times to access the data
        ComputeKernel kernel;
    };

    IComputeContext* context = nullptr;
    ComputeBuffer buffer = nullptr;
    
    std::vector<CacheConfig> configs;
    size_t currentConfigIndex = 0;
    
    void createKernel(CacheConfig& config, const std::string& kernel_dir);
};
