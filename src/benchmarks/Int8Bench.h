#pragma once

#include "IBenchmark.h"

class Int8Bench : public IBenchmark {
public:
    const char* GetName() const override { return "INT8 Benchmark"; }
    bool IsSupported(VkPhysicalDevice device) const override;
    void Setup(VulkanContext& context, const std::string& shader_dir) override;
    void Run(VkCommandBuffer cmdBuffer) override;
    void Teardown() override;
    BenchmarkResult GetResult() const override;

private:
    VulkanContext* context = nullptr;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory bufferMemory = VK_NULL_HANDLE;
};
