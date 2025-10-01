#pragma once

#include "IBenchmark.h"
#include "core/IComputeContext.h"
#include <vector>
#include <string>

class CacheBench : public IBenchmark {
public:
    CacheBench(std::string name, std::string metric, uint64_t bufferSize, std::string kernelFile, std::vector<uint32_t> initData = {});
    ~CacheBench();

    bool IsSupported(const DeviceInfo& info, IComputeContext* context) const override;
    void Setup(IComputeContext& context, const std::string& kernel_dir) override;
    void Run() override;
    void Teardown() override;
    const char* GetName() const override;
    const char* GetMetric() const override;
    BenchmarkResult GetResult() const override;

private:
    std::string name;
    std::string metric;
    uint64_t bufferSize;
    std::string kernelFile;
    IComputeContext* context = nullptr;
    ComputeKernel kernel = nullptr;
    ComputeBuffer buffer = nullptr;
    std::vector<uint32_t> initData;
};
