#include "core/BenchmarkRunner.h"
#include "core/ComputeBackendFactory.h"
#include "core/ResultFormatter.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Fp8Bench.h"
#include "benchmarks/Fp4Bench.h"
#include "benchmarks/Int8Bench.h"
#include "benchmarks/Int4Bench.h"
#include "benchmarks/MemBandwidthBench.h"
#include "benchmarks/CacheBench.h"
#include "benchmarks/Fp6Bench.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <algorithm>
#include <locale>
#include <numeric>
#include <random>

// Helper function to create a shuffled index array for pointer chasing
std::vector<uint32_t> create_shuffled_indices(size_t size) {
    std::vector<uint32_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(1337); // Use a fixed seed for reproducibility
    std::shuffle(indices.begin(), indices.end(), g);
    return indices;
}

BenchmarkRunner::BenchmarkRunner(const std::vector<IComputeContext*>& contexts) : contexts(contexts) {
    discoverBenchmarks();
    formatter = std::make_unique<ResultFormatter>();
}

BenchmarkRunner::~BenchmarkRunner() {}

std::vector<std::string> BenchmarkRunner::getAvailableBenchmarks() const {
    std::vector<std::string> names;
    for (const auto& bench : benchmarks) {
        names.push_back(bench->GetName());
    }
    return names;
}

void BenchmarkRunner::discoverBenchmarks() {
    benchmarks.push_back(std::make_unique<Fp64Bench>());
    benchmarks.push_back(std::make_unique<Fp32Bench>());
    benchmarks.push_back(std::make_unique<Fp16Bench>());
    benchmarks.push_back(std::make_unique<Fp8Bench>());
    benchmarks.push_back(std::make_unique<Fp6Bench>());
    benchmarks.push_back(std::make_unique<Fp4Bench>());
    benchmarks.push_back(std::make_unique<Int8Bench>());
    benchmarks.push_back(std::make_unique<Int4Bench>());
    benchmarks.push_back(std::make_unique<MemBandwidthBench>());

    // Cache Bandwidth
    benchmarks.push_back(std::make_unique<CacheBench>("L0 Cache Bandwidth", "GB/s", 4, "l0_cache_bandwidth"));
    benchmarks.push_back(std::make_unique<CacheBench>("L1 Cache Bandwidth", "GB/s", 24 * 1024, "cache_bandwidth"));
    benchmarks.push_back(std::make_unique<CacheBench>("L2 Cache Bandwidth", "GB/s", 1 * 1024 * 1024, "cache_bandwidth"));
    benchmarks.push_back(std::make_unique<CacheBench>("L3 Cache Bandwidth", "GB/s", 16 * 1024 * 1024, "cache_bandwidth"));

    // Cache Latency
    const size_t l1_size = 24 * 1024;
    const size_t l2_size = 1 * 1024 * 1024;
    const size_t l3_size = 16 * 1024 * 1024;

    benchmarks.push_back(std::make_unique<CacheBench>("L0 Cache Latency", "ns", 4, "l0_cache_latency"));
    benchmarks.push_back(std::make_unique<CacheBench>("L1 Cache Latency", "ns", l1_size, "cache_latency", create_shuffled_indices(l1_size / sizeof(uint32_t))));
    benchmarks.push_back(std::make_unique<CacheBench>("L2 Cache Latency", "ns", l2_size, "cache_latency", create_shuffled_indices(l2_size / sizeof(uint32_t))));
    benchmarks.push_back(std::make_unique<CacheBench>("L3 Cache Latency", "ns", l3_size, "cache_latency", create_shuffled_indices(l3_size / sizeof(uint32_t))));
}

struct BenchmarkResultRow {
    std::string testName;
    double performance;
    std::string unit;
};

// Helper to lowercase a string
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

void BenchmarkRunner::run(const std::vector<std::string>& benchmarks_to_run) {
    std::vector<std::string> lower_benchmarks_to_run;
    for (const auto& b : benchmarks_to_run) {
        lower_benchmarks_to_run.push_back(to_lower(b));
    }

    for (auto* context : contexts) {
        try {
            DeviceInfo info = context->getCurrentDeviceInfo();
            std::cout << "================================================================" << std::endl;
            std::cout << "Running on device: " << info.name 
                      << " (" << ComputeBackendFactory::getBackendName(context->getBackend()) << ")" << std::endl;
            std::cout << "================================================================" << std::endl;
            
            // Display device statistics
            std::cout << "Device Statistics:" << std::endl;
            std::cout << "  Compute Units: " << info.computeUnits << std::endl;
            std::cout << "  VRAM: " << (info.memorySize / (1024 * 1024 * 1024)) << " GB" << std::endl;
            std::cout << "  Max Work Group Size: " << info.maxWorkGroupSize << std::endl;
            std::cout << "  Subgroup Size: " << info.subgroupSize << std::endl;
            std::cout << "  Max Shared Memory: " << (info.maxComputeSharedMemorySize / 1024) << " KB" << std::endl;
            std::cout << "================================================================" << std::endl;
            std::cout << std::endl;

            for (auto& bench : benchmarks) {
                bool should_run = false;
                if (benchmarks_to_run.empty()) {
                    should_run = true;
                } else {
                    std::string bench_name_lower = to_lower(bench->GetName());
                    for (const auto& run_name : lower_benchmarks_to_run) {
                        if (bench_name_lower.find(run_name) != std::string::npos) {
                            should_run = true;
                            break;
                        }
                    }
                }

                if (should_run && bench->IsSupported(info, context)) {
                    try {
                        std::cout << "Running " << bench->GetName() << "..." << std::endl;
                        bench->Setup(*context, ".");

                        // Timed run
                        double total_time_ms = 0;
                        uint64_t total_invocations = 0;
                        auto bench_start = std::chrono::high_resolution_clock::now();
                        while (total_time_ms < 5000) {
                            bench->Run();
                            context->waitIdle();
                            total_invocations++;
                            auto now = std::chrono::high_resolution_clock::now();
                            total_time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(now - bench_start).count() / 1e6;
                        }

                        BenchmarkResult bench_result = bench->GetResult();
                        
                        ResultData result_data;
                        result_data.backendName = ComputeBackendFactory::getBackendName(context->getBackend());
                        result_data.deviceName = info.name;
                        result_data.benchmarkName = bench->GetName();
                        result_data.metric = bench->GetMetric();
                        result_data.operations = bench_result.operations * total_invocations;
                        result_data.time_ms = total_time_ms;
                        result_data.isEmulated = false; // This will be updated later

                        formatter->addResult(result_data);

                        bench->Teardown();
                    } catch (const std::exception& e) {
                        std::cerr << "Error running " << bench->GetName() << ": " << e.what() << std::endl;
                        // Make sure to clean up
                        try {
                            bench->Teardown();
                        } catch (...) {
                            // Ignore errors during cleanup
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing device: " << e.what() << std::endl;
            continue;
        }
    }
    formatter->print();
}
