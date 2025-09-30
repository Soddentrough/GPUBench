#include "core/BenchmarkRunner.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Fp8Bench.h"
#include "benchmarks/Int8Bench.h"
#include "benchmarks/Int4Bench.h"
#include <iostream>
#include <stdexcept>

BenchmarkRunner::BenchmarkRunner(VulkanContext& context) : context(context) {
    discoverBenchmarks();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = context.getComputeQueueFamilyIndex();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(context.getDevice(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }

    VkQueryPoolCreateInfo queryPoolInfo{};
    queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = 2;

    if (vkCreateQueryPool(context.getDevice(), &queryPoolInfo, nullptr, &queryPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create query pool!");
    }
}

BenchmarkRunner::~BenchmarkRunner() {
    vkDestroyQueryPool(context.getDevice(), queryPool, nullptr);
    vkDestroyCommandPool(context.getDevice(), commandPool, nullptr);
}

void BenchmarkRunner::discoverBenchmarks() {
    benchmarks.push_back(std::make_unique<Fp64Bench>());
    benchmarks.push_back(std::make_unique<Fp32Bench>());
    benchmarks.push_back(std::make_unique<Fp16Bench>());
    benchmarks.push_back(std::make_unique<Fp8Bench>());
    benchmarks.push_back(std::make_unique<Int8Bench>());
    benchmarks.push_back(std::make_unique<Int4Bench>());
}

void BenchmarkRunner::run(const std::vector<std::string>& benchmarks_to_run) {
    for (auto& bench : benchmarks) {
        bool should_run = benchmarks_to_run.empty();
        for (const auto& name : benchmarks_to_run) {
            if (name == bench->GetName()) {
                should_run = true;
                break;
            }
        }

        if (should_run && bench->IsSupported(context.getPhysicalDevice())) {
            std::cout << "Running " << bench->GetName() << std::endl;
            bench->Setup(context, ".");

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            VkCommandBuffer commandBuffer;
            vkAllocateCommandBuffers(context.getDevice(), &allocInfo, &commandBuffer);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            vkBeginCommandBuffer(commandBuffer, &beginInfo);
            vkCmdResetQueryPool(commandBuffer, queryPool, 0, 2);
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 0);
            bench->Run(commandBuffer);
            vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1);
            vkEndCommandBuffer(commandBuffer);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            uint64_t total_invocations = 0;
            double total_elapsed_time = 0;
            
            while (total_elapsed_time < 5000) {
                vkQueueSubmit(context.getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
                vkQueueWaitIdle(context.getComputeQueue());

                uint64_t timestamps[2];
                vkGetQueryPoolResults(context.getDevice(), queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

                float timestampPeriod = context.getPhysicalDeviceProperties().limits.timestampPeriod;
                double elapsedTime = (timestamps[1] - timestamps[0]) * timestampPeriod / 1e6; // in milliseconds
                
                total_elapsed_time += elapsedTime;
                total_invocations++;
            }
            
            double elapsedTime = total_elapsed_time / total_invocations;

            vkFreeCommandBuffers(context.getDevice(), commandPool, 1, &commandBuffer);

            bench->Teardown();

            BenchmarkResult result = bench->GetResult();
            result.elapsedTime = elapsedTime;

            double total_ops = (double)result.operations * total_invocations;
            double tflops = total_ops / (total_elapsed_time / 1000.0) / 1e12;

            std::cout.precision(3);
            std::cout << "  Operations per invocation: " << result.operations << std::endl;
            std::cout << "  Invocations: " << total_invocations << std::endl;
            std::cout << "  Total runtime: " << std::fixed << total_elapsed_time / 1000.0 << " seconds" << std::endl;
            std::cout << "  Avg time per invocation: " << result.elapsedTime << " ms" << std::endl;
            std::cout << "  Performance: " << std::fixed << tflops << " TFLOPS" << std::endl;
        } else {
            std::cout << "Skipping " << bench->GetName() << " (not supported)" << std::endl;
        }
    }
}
