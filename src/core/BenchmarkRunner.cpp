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
// #include "benchmarks/Fp6Bench.h" // Temporarily disabled
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
    // benchmarks.push_back(std::make_unique<Fp6Bench>()); // Temporarily disabled
    benchmarks.push_back(std::make_unique<Fp4Bench>());
    benchmarks.push_back(std::make_unique<Int8Bench>());
    benchmarks.push_back(std::make_unique<Int4Bench>());
    benchmarks.push_back(std::make_unique<MemBandwidthBench>());

    // Cache Bandwidth
    std::vector<uint32_t> l0_init = {42};  // Initialize with a single value
    benchmarks.push_back(std::make_unique<CacheBench>("L0 Cache Bandwidth", "GB/s", 4, "l0_cache_bandwidth", l0_init));
    
    const size_t l1_size = 24 * 1024;
    const size_t l2_size = 1 * 1024 * 1024;
    const size_t l3_size = 16 * 1024 * 1024;
    
    // Cache bandwidth kernels use float4 arrays and access large index ranges
    // We need to allocate enough space based on the dispatch pattern (65536 workgroups * 256 threads)
    // cachebw_l1: max index = 65536 * 2 + 1 = ~131K float4 elements = ~2MB
    // cachebw_l2: max index = 65536 * 256 + 255 = ~16.7M float4 elements = ~268MB
    // cachebw_l3: max index = 65536 * 8192 + 255*32+31 = ~537M float4 elements = ~8.6GB (too large!)
    
    // For cachebw_l1 (L2 cache), allocate 2MB (enough for the access pattern)
    size_t cachebw_l1_size = 2 * 1024 * 1024;
    std::vector<uint32_t> l1_bw_init(cachebw_l1_size / sizeof(uint32_t), 1);
    
    // For cachebw_l2 (L1 cache), allocate 268MB (enough for the access pattern)
    size_t cachebw_l2_size = 268 * 1024 * 1024;
    std::vector<uint32_t> l2_bw_init(cachebw_l2_size / sizeof(uint32_t), 1);
    
    // For cachebw_l3 (L3 cache), the kernel would access ~8.6GB which is too much
    // We'll keep the 16MB allocation but need to fix the kernel or dispatch
    std::vector<uint32_t> l3_bw_init(l3_size / sizeof(uint32_t), 1);
    
    benchmarks.push_back(std::make_unique<CacheBench>("L1 Cache Bandwidth", "GB/s", cachebw_l2_size, "cachebw_l2", l2_bw_init));
    benchmarks.push_back(std::make_unique<CacheBench>("L2 Cache Bandwidth", "GB/s", cachebw_l1_size, "cachebw_l1", l1_bw_init));
    benchmarks.push_back(std::make_unique<CacheBench>("L3 Cache Bandwidth", "GB/s", l3_size, "cachebw_l3", l3_bw_init));

    // Cache Latency
    benchmarks.push_back(std::make_unique<CacheBench>("L0 Cache Latency", "ns", 4, "l0_cache_latency", l0_init));
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
            std::cout << "  VRAM: " << static_cast<int>(std::round(info.memorySize / (1024.0 * 1024.0 * 1024.0))) << " GB" << std::endl;
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
                        std::cout << "Running " << bench->GetName() <<   "..." << std::endl;
                        bench->Setup(*context, "kernels");

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
                    result_data.isEmulated = bench->IsEmulated();

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
