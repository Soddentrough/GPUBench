#include "core/BenchmarkRunner.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Fp8Bench.h"
#include "benchmarks/Int8Bench.h"
#include "benchmarks/Int4Bench.h"
#include "benchmarks/MemBandwidthBench.h"
#include "benchmarks/CacheBandwidthBench.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sstream>
#include <locale>

#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif

BenchmarkRunner::BenchmarkRunner(IComputeContext& context) : context(context) {
    discoverBenchmarks();

#ifdef HAVE_VULKAN
    if (context.getBackend() == ComputeBackend::Vulkan) {
        VulkanContext* vkContext = static_cast<VulkanContext*>(context.getVulkanContext());
        
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = vkContext->getComputeQueueFamilyIndex();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(vkContext->getDevice(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }

        VkQueryPoolCreateInfo queryPoolInfo{};
        queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolInfo.queryCount = 2;

        if (vkCreateQueryPool(vkContext->getDevice(), &queryPoolInfo, nullptr, &queryPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create query pool!");
        }
    }
#endif
}

BenchmarkRunner::~BenchmarkRunner() {
#ifdef HAVE_VULKAN
    if (context.getBackend() == ComputeBackend::Vulkan) {
        VulkanContext* vkContext = static_cast<VulkanContext*>(context.getVulkanContext());
        vkDestroyQueryPool(vkContext->getDevice(), queryPool, nullptr);
        vkDestroyCommandPool(vkContext->getDevice(), commandPool, nullptr);
    }
#endif
}

void BenchmarkRunner::discoverBenchmarks() {
    benchmarks.push_back(std::make_unique<Fp64Bench>());
    benchmarks.push_back(std::make_unique<Fp32Bench>());
    benchmarks.push_back(std::make_unique<Fp16Bench>());
    benchmarks.push_back(std::make_unique<Fp8Bench>());
    benchmarks.push_back(std::make_unique<Int8Bench>());
    benchmarks.push_back(std::make_unique<Int4Bench>());
    benchmarks.push_back(std::make_unique<MemBandwidthBench>());
    benchmarks.push_back(std::make_unique<CacheBandwidthBench>());
}

struct BenchmarkResultRow {
    std::string testName;
    double performance;
    std::string unit;
};

void BenchmarkRunner::run(const std::vector<std::string>& benchmarks_to_run) {
#ifndef HAVE_VULKAN
    std::cerr << "Error: Vulkan backend not available, cannot run benchmarks." << std::endl;
    return;
#else
    if (context.getBackend() != ComputeBackend::Vulkan) {
        std::cerr << "Note: Benchmarks are currently only implemented for Vulkan backend." << std::endl;
        std::cerr << "OpenCL benchmark implementations are planned for future releases." << std::endl;
        return;
    }
    
    VulkanContext* vkContext = static_cast<VulkanContext*>(context.getVulkanContext());
    std::vector<BenchmarkResultRow> results;

    for (auto& bench : benchmarks) {
        bool should_run = benchmarks_to_run.empty();
        for (const auto& name : benchmarks_to_run) {
            if (name == bench->GetName()) {
                should_run = true;
                break;
            }
        }

        if (should_run) {
            if (!bench->IsSupported(vkContext->getPhysicalDevice())) {
                std::cout << "Skipping " << bench->GetName() << " (not supported)" << std::endl;
                continue;
            }
            
            std::cout << "Running " << bench->GetName() << "..." << std::flush;
            bench->Setup(*vkContext, ".");

            // Check if this is a bandwidth benchmark (runs multiple configs)
            std::string bench_name = bench->GetName();
            bool is_mem_bandwidth_bench = bench_name.find("Memory Bandwidth") != std::string::npos;
            bool is_cache_bandwidth_bench = bench_name.find("Cache Bandwidth") != std::string::npos;
            bool is_bandwidth_bench = is_mem_bandwidth_bench || is_cache_bandwidth_bench;
            int num_configs = is_mem_bandwidth_bench ? 4 : (is_cache_bandwidth_bench ? 3 : 1);
            
            for (int config_idx = 0; config_idx < num_configs; config_idx++) {
                if (is_bandwidth_bench && config_idx > 0) {
                    std::cout << std::endl;
                }
                
                VkCommandBufferAllocateInfo allocInfo{};
                allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                allocInfo.commandPool = commandPool;
                allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocInfo.commandBufferCount = 1;

                VkCommandBuffer commandBuffer;
                vkAllocateCommandBuffers(vkContext->getDevice(), &allocInfo, &commandBuffer);

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
                
                // Run for ~2 seconds per config for bandwidth tests, 5 seconds for others
                double target_time = is_bandwidth_bench ? 2000 : 5000;
                
                while (total_elapsed_time < target_time) {
                    vkQueueSubmit(vkContext->getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
                    vkQueueWaitIdle(vkContext->getComputeQueue());

                    uint64_t timestamps[2];
                    vkGetQueryPoolResults(vkContext->getDevice(), queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

                    float timestampPeriod = vkContext->getPhysicalDeviceProperties().limits.timestampPeriod;
                    double elapsedTime = (timestamps[1] - timestamps[0]) * timestampPeriod / 1e6; // in milliseconds
                    
                    total_elapsed_time += elapsedTime;
                    total_invocations++;
                }
                
                double elapsedTime = total_elapsed_time / total_invocations;

                vkFreeCommandBuffers(vkContext->getDevice(), commandPool, 1, &commandBuffer);

                BenchmarkResult result = bench->GetResult();
                result.elapsedTime = elapsedTime;

                double total_ops = (double)result.operations * total_invocations;
                
                if (is_bandwidth_bench) {
                    // For bandwidth benchmarks, operations represent bytes transferred
                    double bandwidth_gbps = total_ops / (total_elapsed_time / 1000.0) / 1e9;
                    std::string testName;
                    if (is_mem_bandwidth_bench) {
                        const char* config_names[] = {"128 threads/group", "256 threads/group", "512 threads/group", "1024 threads/group"};
                        testName = std::string(bench->GetName()) + " (" + config_names[config_idx] + ")";
                    } else if (is_cache_bandwidth_bench) {
                        // Cache bandwidth configs have detailed names built-in
                        const char* config_names[] = {
                            "L1: 16 KB working set (16KB/CU × 80 CUs = 1.28 MB total)",
                            "L2: 2 MB working set (4 MB total shared)",
                            "Infinity Cache: 64 MB working set (128 MB total)"
                        };
                        testName = std::string(bench->GetName()) + " (" + config_names[config_idx] + ")";
                    }
                    results.push_back({testName, bandwidth_gbps, "GB/s"});
                } else {
                    // For compute benchmarks, operations represent FLOPS
                    double tflops = total_ops / (total_elapsed_time / 1000.0) / 1e12;
                    results.push_back({bench->GetName(), tflops, "TFLOPS"});
                }
            }
            
            std::cout << " Done" << std::endl;

            bench->Teardown();
        }
    }
    
    // Print results table
    if (!results.empty()) {
        std::cout << "\n";
        std::cout << "========================================================================================================\n";
        std::cout << "BENCHMARK RESULTS\n";
        std::cout << "========================================================================================================\n";
        std::cout << std::left << std::setw(70) << "Test" << " | " << std::right << std::setw(15) << "Performance" << "\n";
        std::cout << "--------------------------------------------------------------------------------------------------------\n";
        
        // Helper lambda to format numbers with commas
        auto formatWithCommas = [](double value) -> std::string {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << value;
            std::string result = ss.str();
            
            // Find the decimal point
            size_t decimal_pos = result.find('.');
            if (decimal_pos == std::string::npos) {
                decimal_pos = result.length();
            }
            
            // Insert commas in the integer part
            int insert_pos = decimal_pos - 3;
            while (insert_pos > 0) {
                result.insert(insert_pos, ",");
                insert_pos -= 3;
            }
            
            return result;
        };
        
        for (const auto& row : results) {
            std::string formatted_perf = formatWithCommas(row.performance);
            std::cout << std::left << std::setw(70) << row.testName << " | " 
                      << std::right << std::setw(15) << formatted_perf 
                      << " " << std::left << row.unit << "\n";
        }
        std::cout << "========================================================================================================\n";
    }
#endif
}
