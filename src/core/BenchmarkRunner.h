#pragma once

#include "core/VulkanContext.h"
#include "benchmarks/IBenchmark.h"
#include <vector>
#include <memory>

class BenchmarkRunner {
public:
    BenchmarkRunner(VulkanContext& context);
    ~BenchmarkRunner();

    void run(const std::vector<std::string>& benchmarks_to_run);

private:
    void discoverBenchmarks();

    VulkanContext& context;
    std::vector<std::unique_ptr<IBenchmark>> benchmarks;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;
};
