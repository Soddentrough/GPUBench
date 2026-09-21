#include "core/RunnerAPI.h"
#include "core/BenchmarkRunner.h"
#include "core/ComputeBackendFactory.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <map>

std::vector<ResultData> RunBenchmarksAPI(
    const std::vector<std::string>& benchmarks_to_run,
    const std::vector<uint32_t>& device_indices,
    const std::vector<std::string>& backend_strs,
    bool verbose, bool debug, bool dump_geometry,
    bool dump_renders,
    uint32_t renderWidth,
    uint32_t renderHeight,
    std::function<void(const ResultData&)> callback,
    const std::string& scene,
    uint32_t samples_per_pixel,
    std::atomic<bool>* cancel_token)
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

    BenchmarkRunner runner({}, verbose, debug, dump_geometry, dump_renders, scene.empty() ? "all" : scene);
    runner.setResolution(renderWidth, renderHeight);
    runner.setSamplesPerPixel(samples_per_pixel);
    if (cancel_token) {
        runner.setCancelToken(cancel_token);
    }
    if (callback) {
        runner.onResult = callback;
    }

    std::vector<uint32_t> target_indices = device_indices;
    // Only default to GPU 0 if no device was specified AND there are non-host workloads to run
    if (target_indices.empty()) {
        bool hasGpuWorkloads = false;
        for (const auto& b : benchmarks_to_run) {
            if (b.find("System Memory") == std::string::npos && b.find("SysMem") == std::string::npos) {
                hasGpuWorkloads = true;
                break;
            }
        }
        if (hasGpuWorkloads) {
            target_indices.push_back(0);
        }
    }

    std::vector<ComputeBackend> target_backends;
    for (const auto &proto_context : contexts) {
        target_backends.push_back(proto_context->getBackend());
    }
    contexts.clear();

    for (ComputeBackend backend : target_backends) {
        if (cancel_token && cancel_token->load()) break;
        for (uint32_t device_idx : target_indices) {
            if (cancel_token && cancel_token->load()) break;
            std::unique_ptr<IComputeContext> new_context =
                ComputeBackendFactory::create(backend, verbose, debug);
            if (new_context) {
                if (device_idx < new_context->getDevices().size()) {
                    new_context->pickDevice(device_idx);
                    runner.runForContext(new_context.get(), benchmarks_to_run);
                }
            }
        }
    }

    if (!cancel_token || !cancel_token->load()) {
        runner.runHostBenchmarks(benchmarks_to_run);
    }
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
        BenchmarkRunner runner(dummy, false, false, false, false, "all");
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
        if (profiles.empty() && ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
            try {
                auto ctx = ComputeBackendFactory::create(ComputeBackend::OpenCL, false, false);
                if (ctx) {
                    uint32_t i = 0;
                    for (const auto& dev : ctx->getDevices()) {
                        DeviceProfile p;
                        p.backend = "OpenCL";
                        p.deviceIndex = i;
                        p.deviceName = dev.name;
                        p.vendorID = dev.vendorID;
                        p.deviceID = dev.deviceID;
                        p.driverName = dev.driverName.empty() ? "OpenCL Driver" : dev.driverName;
                        p.driverInfo = dev.driverInfo.empty() ? "OpenCL" : dev.driverInfo;
                        p.driverVersion = dev.driverVersionStr.empty() ? std::to_string(dev.driverVersion) : dev.driverVersionStr;
                        if (dev.apiVersion > 0) {
                            uint32_t apiMajor = dev.apiVersion >> 22;
                            uint32_t apiMinor = (dev.apiVersion >> 12) & 0x3FF;
                            uint32_t apiPatch = dev.apiVersion & 0xFFF;
                            p.apiVersion = std::to_string(apiMajor) + "." + std::to_string(apiMinor) + "." + std::to_string(apiPatch);
                        } else {
                            p.apiVersion = "OpenCL";
                        }
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
        if (profiles.empty() && ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
            try {
                auto ctx = ComputeBackendFactory::create(ComputeBackend::ROCm, false, false);
                if (ctx) {
                    uint32_t i = 0;
                    for (const auto& dev : ctx->getDevices()) {
                        DeviceProfile p;
                        p.backend = "ROCm";
                        p.deviceIndex = i;
                        p.deviceName = dev.name;
                        p.vendorID = dev.vendorID;
                        p.deviceID = dev.deviceID;
                        p.driverName = dev.driverName.empty() ? "AMD ROCm / HIP" : dev.driverName;
                        p.driverInfo = dev.driverInfo.empty() ? (dev.archName.empty() ? "HIP Runtime" : dev.archName) : dev.driverInfo;
                        p.driverVersion = dev.driverVersionStr.empty() ? std::to_string(dev.driverVersion) : dev.driverVersionStr;
                        if (dev.driverVersion > 0) {
                            p.apiVersion = "HIP " + std::to_string(dev.driverVersion / 10000000) + "." + std::to_string((dev.driverVersion / 100000) % 100);
                        } else {
                            p.apiVersion = "ROCm";
                        }
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

std::vector<BenchmarkSupportInfo> ProbeBenchmarkSupportAPI(
    const std::string& backend_name,
    uint32_t device_idx)
{
    static std::map<std::pair<std::string, uint32_t>, std::vector<BenchmarkSupportInfo>> s_supportCache;
    static std::mutex s_supportCacheMutex;

    {
        std::lock_guard<std::mutex> lock(s_supportCacheMutex);
        auto it = s_supportCache.find({backend_name, device_idx});
        if (it != s_supportCache.end()) {
            return it->second;
        }
    }

    std::vector<BenchmarkSupportInfo> results;
    try {
        ComputeBackend backend = ComputeBackend::Vulkan;
        if (backend_name == "rocm") backend = ComputeBackend::ROCm;
        else if (backend_name == "opencl") backend = ComputeBackend::OpenCL;

        if (ComputeBackendFactory::isAvailable(backend)) {
            auto ctx = ComputeBackendFactory::create(backend, false, false);
            if (ctx) {
                const auto& devices = ctx->getDevices();
                if (device_idx < devices.size()) {
                    ctx->pickDevice(device_idx);
                    DeviceInfo info = ctx->getCurrentDeviceInfo();
                    BenchmarkRunner runner({}, false, false, false, false, "all");
                    for (const auto& bench : runner.getBenchmarkList()) {
                        BenchmarkSupportInfo item;
                        item.id = bench->GetName();
                        item.isSupported = bench->IsSupported(info, ctx.get());
                        if (!item.isSupported) {
                            item.reason = bench->GetSupportNote(info, ctx.get());
                            switch (bench->GetSupportLimitation(info, ctx.get())) {
                            case IBenchmark::SupportLimitation::kHardware:
                                item.limitationCategory = "Hardware Limitation";
                                break;
                            case IBenchmark::SupportLimitation::kApi:
                                item.limitationCategory = "API Limitation";
                                break;
                            case IBenchmark::SupportLimitation::kToolchain:
                                item.limitationCategory = "Toolchain Limitation";
                                break;
                            default:
                                item.limitationCategory = "Unsupported";
                                break;
                            }
                        }
                        results.push_back(item);
                    }
                }
            }
        }

        if (!results.empty()) {
            std::lock_guard<std::mutex> lock(s_supportCacheMutex);
            s_supportCache[{backend_name, device_idx}] = results;
        }
    } catch (const std::exception& e) {
        std::cerr << "ProbeBenchmarkSupportAPI failed: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "ProbeBenchmarkSupportAPI failed: unknown error" << std::endl;
    }
    return results;
}

