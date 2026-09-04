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
    bool dump_renders,
    uint32_t renderWidth,
    uint32_t renderHeight,
    std::function<void(const ResultData&)> callback)
{
    // Never let C++ exceptions cross the cxx FFI boundary into Rust (that
    // would call std::terminate). On error, return an empty result list.
    try {
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

    BenchmarkRunner runner(context_ptrs, verbose, debug, dump_geometry, dump_renders);
    runner.setResolution(renderWidth, renderHeight);
    if (callback) {
        runner.onResult = callback;
    }
    runner.run(benchmarks_to_run);
    return runner.getResults();
    } catch (const std::exception& e) {
        std::cerr << "RunBenchmarksAPI failed: " << e.what() << std::endl;
        return {};
    } catch (...) {
        std::cerr << "RunBenchmarksAPI failed: unknown error" << std::endl;
        return {};
    }
}

std::vector<std::string> GetAvailableHardwareAPI() {
    // Never let C++ exceptions cross the cxx FFI boundary into Rust. On
    // error, return whatever was gathered (possibly just the System entry).
    std::vector<std::string> results;

    // System
    results.push_back("System|0|System Memory / Host CPU");

    try {
    if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
        try {
            auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
            if (ctx) {
                uint32_t i = 0;
                for (const auto& dev : ctx->getDevices()) {
                    results.push_back("vulkan|" + std::to_string(i) + "|" + dev.name);
                    i++;
                }
            }
        } catch (...) {
            // Vulkan compiled in but not usable at runtime; skip it
        }
    }
    if (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
        try {
            auto ctx = ComputeBackendFactory::create(ComputeBackend::OpenCL, false, false);
            if (ctx) {
                uint32_t i = 0;
                for (const auto& dev : ctx->getDevices()) {
                    results.push_back("opencl|" + std::to_string(i) + "|" + dev.name);
                    i++;
                }
            }
        } catch (...) {
            // OpenCL compiled in but not usable at runtime; skip it
        }
    }
    if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
        try {
            auto ctx = ComputeBackendFactory::create(ComputeBackend::ROCm, false, false);
            if (ctx) {
                uint32_t i = 0;
                for (const auto& dev : ctx->getDevices()) {
                    results.push_back("rocm|" + std::to_string(i) + "|" + dev.name);
                    i++;
                }
            }
        } catch (...) {
            // ROCm compiled in but not usable at runtime; skip it
        }
    }
    } catch (const std::exception& e) {
        std::cerr << "GetAvailableHardwareAPI failed: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "GetAvailableHardwareAPI failed: unknown error" << std::endl;
    }
    return results;
}

std::vector<std::string> GetAvailableBenchmarksAPI() {
    // Never let C++ exceptions cross the cxx FFI boundary into Rust. On
    // error, return an empty list.
    try {
        std::vector<IComputeContext*> dummy;
        BenchmarkRunner runner(dummy, false, false, false);
        return runner.getAvailableBenchmarks();
    } catch (const std::exception& e) {
        std::cerr << "GetAvailableBenchmarksAPI failed: " << e.what() << std::endl;
        return {};
    } catch (...) {
        std::cerr << "GetAvailableBenchmarksAPI failed: unknown error" << std::endl;
        return {};
    }
}

std::vector<DeviceProfile> GetDeviceProfilesAPI() {
    std::vector<DeviceProfile> profiles;
    try {
        if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
            try {
                auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
                if (ctx) {
                    uint32_t i = 0;
                    for (const auto& dev : ctx->getDevices()) {
                        DeviceProfile p;
                        p.backend = "Vulkan";
                        p.deviceIndex = i;
                        p.deviceName = dev.name;
                        p.vendorID = dev.vendorID;
                        p.deviceID = dev.deviceID;
                        p.driverName = dev.driverName;
                        p.driverInfo = dev.driverInfo;
                        p.driverVersion = dev.driverVersionStr;
                        uint32_t apiMajor = dev.apiVersion >> 22;
                        uint32_t apiMinor = (dev.apiVersion >> 12) & 0x3FF;
                        uint32_t apiPatch = dev.apiVersion & 0xFFF;
                        p.apiVersion = std::to_string(apiMajor) + "." + std::to_string(apiMinor) + "." + std::to_string(apiPatch);
                        p.vramTotalMb = dev.memorySize / (1024 * 1024);
                        p.subgroupSize = dev.subgroupSize;
                        p.maxWorkGroupSize = dev.maxWorkGroupSize;
                        p.rayTracingSupported = dev.rayTracingSupport;
                        p.serSupported = dev.serSupported;
                        p.workGraphsSupported = dev.workGraphsSupported;
                        p.cooperativeMatrixSupported = dev.cooperativeMatrixSupport;
                        p.float16Supported = dev.fp16Support;
                        p.int8Supported = dev.int8Support;
                        profiles.push_back(p);
                        i++;
                    }
                }
            } catch (...) {}
        }
    } catch (const std::exception& e) {
        std::cerr << "GetDeviceProfilesAPI failed: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "GetDeviceProfilesAPI failed: unknown error" << std::endl;
    }
    return profiles;
}

