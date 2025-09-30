#pragma once

#include "core/IComputeContext.h"
#include "benchmarks/IBenchmark.h"
#include <vector>
#include <memory>

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

class BenchmarkRunner {
public:
    BenchmarkRunner(IComputeContext& context);
    ~BenchmarkRunner();

    void run(const std::vector<std::string>& benchmarks_to_run);

private:
    void discoverBenchmarks();

    IComputeContext& context;
    std::vector<std::unique_ptr<IBenchmark>> benchmarks;
    
#ifdef HAVE_VULKAN
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;
#endif
};
