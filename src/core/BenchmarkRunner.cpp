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
#include "benchmarks/CacheBandwidthBench.h"
#include "benchmarks/CacheLatencyBench.h"
#include "benchmarks/Fp6Bench.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <algorithm>
#include <locale>

BenchmarkRunner::BenchmarkRunner(const std::vector<IComputeContext*>& contexts) : contexts(contexts) {
    discoverBenchmarks();
    formatter = std::make_unique<ResultFormatter>();
}

BenchmarkRunner::~BenchmarkRunner() {}

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
    benchmarks.push_back(std::make_unique<CacheBandwidthBench>());
    benchmarks.push_back(std::make_unique<CacheLatencyBench>());
}

struct BenchmarkResultRow {
    std::string testName;
    double performance;
    std::string unit;
};

void BenchmarkRunner::run(const std::vector<std::string>& benchmarks_to_run) {
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
            if (bench->IsSupported(info, context)) {
                try {
                    std::cout << "Running " << bench->GetName() << "..." << std::endl;
                    bench->Setup(*context, "build");

                    // Timed run
                    double total_time_ms = 0;
                    uint64_t total_invocations = 0;
                    auto bench_start = std::chrono::high_resolution_clock::now();
                    while (total_time_ms < 8000) {
                        bench->Run();
                        context->waitIdle();
                        total_invocations++;
                        auto now = std::chrono::high_resolution_clock::now();
                        total_time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(now - bench_start).count() / 1e6;
                    }
                    double time_ms = total_time_ms / total_invocations;

                    BenchmarkResult bench_result = bench->GetResult();
                    
                    ResultData result_data;
                    result_data.backendName = ComputeBackendFactory::getBackendName(context->getBackend());
                    result_data.deviceName = info.name;
                    result_data.benchmarkName = bench->GetName();
                    result_data.operations = bench_result.operations;
                    result_data.time_ms = time_ms;
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
