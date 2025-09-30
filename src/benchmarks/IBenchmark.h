#pragma once

#include "core/VulkanContext.h"
#include <vulkan/vulkan.h>
#include <string>

struct BenchmarkResult {
    uint64_t operations;
    double elapsedTime; // in milliseconds
};

class IBenchmark {
public:
    virtual ~IBenchmark() = default;
    virtual const char* GetName() const = 0;
    virtual bool IsSupported(VkPhysicalDevice device) const = 0;
    virtual void Setup(VulkanContext& context, const std::string& shader_dir) = 0;
    virtual void Run(VkCommandBuffer cmdBuffer) = 0;
    virtual void Teardown() = 0;
    virtual BenchmarkResult GetResult() const = 0;
};
