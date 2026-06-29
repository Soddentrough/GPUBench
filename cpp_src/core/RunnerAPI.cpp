#include "core/RunnerAPI.h"
#include "core/BenchmarkRunner.h"
#include "core/ComputeBackendFactory.h"
#include <iostream>
#include <memory>

std::vector<ResultData> RunBenchmarksAPI(
    const std::vector<std::string>& benchmarks_to_run,
    const std::vector<uint32_t>& device_indices,
    const std::vector<std::string>& backend_strs,
    bool verbose, bool debug, bool dump_geometry,
    std::function<void(const ResultData&)> callback) 
{
    std::vector<std::unique_ptr<IComputeContext>> contexts;
    if (backend_strs.empty() || (backend_strs.size() == 1 && backend_strs[0] == "auto")) {
        if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
            contexts.push_back(ComputeBackendFactory::create(ComputeBackend::Vulkan, verbose, debug));
        } else if (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
            contexts.push_back(ComputeBackendFactory::create(ComputeBackend::OpenCL, verbose, debug));
        } else if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
            contexts.push_back(ComputeBackendFactory::create(ComputeBackend::ROCm, verbose, debug));
        }
    } else {
        for (const auto& backend_str : backend_strs) {
            if (backend_str == "vulkan" && ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
                contexts.push_back(ComputeBackendFactory::create(ComputeBackend::Vulkan, verbose, debug));
            } else if (backend_str == "opencl" && ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
                contexts.push_back(ComputeBackendFactory::create(ComputeBackend::OpenCL, verbose, debug));
            } else if (backend_str == "rocm" && ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
                contexts.push_back(ComputeBackendFactory::create(ComputeBackend::ROCm, verbose, debug));
            }
        }
    }

    std::vector<IComputeContext*> context_ptrs;
    std::vector<std::unique_ptr<IComputeContext>> execution_contexts;

    for (auto& proto_context : contexts) {
        ComputeBackend backend = proto_context->getBackend();
        const auto& devices = proto_context->getDevices();

        std::vector<uint32_t> target_indices = device_indices;
        if (target_indices.empty()) {
            target_indices.push_back(0);
        }

        for (uint32_t device_idx : target_indices) {
            if (device_idx < devices.size()) {
                std::unique_ptr<IComputeContext> new_context = ComputeBackendFactory::create(backend, verbose, debug);
                if (new_context) {
                    new_context->pickDevice(device_idx);
                    execution_contexts.push_back(std::move(new_context));
                }
            }
        }
    }

    for (const auto& ctx : execution_contexts) {
        context_ptrs.push_back(ctx.get());
    }

    BenchmarkRunner runner(context_ptrs, verbose, debug, dump_geometry);
    if (callback) {
        runner.onResult = callback;
    }
    runner.run(benchmarks_to_run);
    return runner.getResults();
}

std::vector<std::string> GetAvailableHardwareAPI() {
    std::vector<std::string> results;
    
    // System
    results.push_back("System|0|System Memory / Host CPU");

    if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
        auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
        if (ctx) {
            uint32_t i = 0;
            for (const auto& dev : ctx->getDevices()) {
                results.push_back("vulkan|" + std::to_string(i) + "|" + dev.name);
                i++;
            }
        }
    }
    if (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
        auto ctx = ComputeBackendFactory::create(ComputeBackend::OpenCL, false, false);
        if (ctx) {
            uint32_t i = 0;
            for (const auto& dev : ctx->getDevices()) {
                results.push_back("opencl|" + std::to_string(i) + "|" + dev.name);
                i++;
            }
        }
    }
    if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
        auto ctx = ComputeBackendFactory::create(ComputeBackend::ROCm, false, false);
        if (ctx) {
            uint32_t i = 0;
            for (const auto& dev : ctx->getDevices()) {
                results.push_back("rocm|" + std::to_string(i) + "|" + dev.name);
                i++;
            }
        }
    }
    return results;
}

std::vector<std::string> GetAvailableBenchmarksAPI() {
    std::vector<IComputeContext*> dummy;
    BenchmarkRunner runner(dummy, false, false, false);
    return runner.getAvailableBenchmarks();
}
