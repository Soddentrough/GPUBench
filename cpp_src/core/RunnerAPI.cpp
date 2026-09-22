#include "core/RunnerAPI.h"
#include "core/BenchmarkRunner.h"
#include "core/ComputeBackendFactory.h"
#include <iostream>
#include <memory>
#include <mutex>
#include <map>
#include <algorithm>
#include <cctype>

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

ComputeApiSupportInfo ProbeComputeApiSupportAPI(const std::string& backend_name) {
    static std::map<std::string, ComputeApiSupportInfo> s_apiSupportCache;
    static std::mutex s_apiSupportCacheMutex;

    std::string lower = backend_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    {
        std::lock_guard<std::mutex> lock(s_apiSupportCacheMutex);
        auto it = s_apiSupportCache.find(lower);
        if (it != s_apiSupportCache.end()) {
            return it->second;
        }
    }

    ComputeApiSupportInfo info;
    info.name = lower;

    if (lower == "vulkan") {
        info.label = "Vulkan";
#ifdef HAVE_VULKAN
        try {
            auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
            if (ctx) {
                const auto& devices = ctx->getDevices();
                if (devices.empty()) {
                    info.isSupported = false;
                    info.reason = "Vulkan runtime initialized, but no compatible Vulkan compute devices were found.";
                    info.missingRequirement = "Compatible Vulkan-capable graphics hardware or vendor ICD driver.";
                } else {
                    info.isSupported = true;
                    info.reason = "Vulkan compute backend is operational with " + std::to_string(devices.size()) + " detected device(s).";
                    info.missingRequirement = "";
                }
            } else {
                info.isSupported = false;
                info.reason = "Vulkan context could not be instantiated.";
                info.missingRequirement = "Vulkan loader runtime (vulkan-1.dll / libvulkan.so.1).";
            }
        } catch (const std::exception& e) {
            info.isSupported = false;
            info.reason = std::string("Vulkan runtime initialization failed: ") + e.what();
            info.missingRequirement = "Vulkan ICD driver or loader runtime (vulkan-1.dll).";
        } catch (...) {
            info.isSupported = false;
            info.reason = "Vulkan runtime initialization failed with an unknown error.";
            info.missingRequirement = "Vulkan ICD driver or loader runtime.";
        }
#else
        info.isSupported = false;
        info.reason = "Vulkan compute backend was not compiled into this GPUBench binary.";
        info.missingRequirement = "Vulkan SDK headers and libraries (vulkan/vulkan.h, Vulkan-1) at build time.";
#endif
    } else if (lower == "rocm") {
        info.label = "ROCm";
#ifdef HAVE_ROCM
        try {
            auto ctx = ComputeBackendFactory::create(ComputeBackend::ROCm, false, false);
            if (ctx) {
                const auto& devices = ctx->getDevices();
                if (devices.empty()) {
                    info.isSupported = false;
                    info.reason = "ROCm/HIP runtime initialized, but no compatible AMD ROCm devices were detected.";
                    info.missingRequirement = "Compatible AMD GPU with ROCm/KFD kernel driver support.";
                } else {
                    info.isSupported = true;
                    info.reason = "AMD ROCm (HIP) backend is operational with " + std::to_string(devices.size()) + " detected device(s).";
                    info.missingRequirement = "";
                }
            } else {
                info.isSupported = false;
                info.reason = "ROCm context could not be created.";
                info.missingRequirement = "ROCm / HIP runtime libraries (amdhip64).";
            }
        } catch (const std::exception& e) {
            info.isSupported = false;
            info.reason = std::string("ROCm initialization failed: ") + e.what();
            info.missingRequirement = "AMD ROCm driver and HIP runtime stack.";
        } catch (...) {
            info.isSupported = false;
            info.reason = "ROCm initialization failed with an unknown error.";
            info.missingRequirement = "AMD ROCm driver and HIP runtime stack.";
        }
#else
        info.isSupported = false;
#ifdef _WIN32
        info.reason = "AMD ROCm (HIP) compute backend is not supported on Windows.";
        info.missingRequirement = "Linux operating system with AMD ROCm kernel driver (amdgpu/kfd) and HIP runtime.";
#else
        info.reason = "ROCm compute backend was not compiled into this GPUBench binary.";
        info.missingRequirement = "AMD ROCm development packages and HIP runtime headers at build time.";
#endif
#endif
    } else if (lower == "opencl") {
        info.label = "OpenCL";
#ifdef HAVE_OPENCL
        try {
            auto ctx = ComputeBackendFactory::create(ComputeBackend::OpenCL, false, false);
            if (ctx) {
                const auto& devices = ctx->getDevices();
                if (devices.empty()) {
                    info.isSupported = false;
                    info.reason = "OpenCL runtime initialized, but no OpenCL compute devices were enumerated.";
                    info.missingRequirement = "OpenCL ICD vendor driver (AMD, NVIDIA, or Intel OpenCL runtime).";
                } else {
                    info.isSupported = true;
                    info.reason = "OpenCL compute backend is operational with " + std::to_string(devices.size()) + " detected device(s).";
                    info.missingRequirement = "";
                }
            } else {
                info.isSupported = false;
                info.reason = "OpenCL context could not be created.";
                info.missingRequirement = "OpenCL ICD loader (OpenCL.dll / libOpenCL.so).";
            }
        } catch (const std::exception& e) {
            info.isSupported = false;
            info.reason = std::string("OpenCL initialization failed: ") + e.what();
            info.missingRequirement = "OpenCL ICD loader (OpenCL.dll / libOpenCL.so) or vendor runtime.";
        } catch (...) {
            info.isSupported = false;
            info.reason = "OpenCL initialization failed with an unknown error.";
            info.missingRequirement = "OpenCL ICD loader or vendor runtime.";
        }
#else
        info.isSupported = false;
        info.reason = "OpenCL compute backend was not compiled into this GPUBench binary.";
        info.missingRequirement = "OpenCL SDK development headers (CL/cl.h) and ICD loader library at build time.";
#endif
    } else if (lower == "auto") {
        info.label = "Auto";
        auto vInfo = ProbeComputeApiSupportAPI("vulkan");
        auto rInfo = ProbeComputeApiSupportAPI("rocm");
        auto oInfo = ProbeComputeApiSupportAPI("opencl");

        if (vInfo.isSupported) {
            info.isSupported = true;
            info.reason = "Auto selects Vulkan (highest priority available backend on this workstation).";
            info.missingRequirement = "";
        } else if (rInfo.isSupported) {
            info.isSupported = true;
            info.reason = "Auto selects ROCm (HIP).";
            info.missingRequirement = "";
        } else if (oInfo.isSupported) {
            info.isSupported = true;
            info.reason = "Auto selects OpenCL.";
            info.missingRequirement = "";
        } else {
            info.isSupported = false;
            info.reason = "No compatible compute backend runtime is available on this system.";
            info.missingRequirement = "A functional graphics driver and runtime supporting Vulkan, ROCm, or OpenCL.";
        }
    } else {
        info.label = backend_name;
        info.isSupported = false;
        info.reason = "Unknown compute API: " + backend_name;
        info.missingRequirement = "Valid compute backend identifier (Vulkan, ROCm, OpenCL, Auto).";
    }

    {
        std::lock_guard<std::mutex> lock(s_apiSupportCacheMutex);
        s_apiSupportCache[lower] = info;
    }

    return info;
}

std::vector<ComputeApiSupportInfo> GetAllComputeApiSupportAPI() {
    return {
        ProbeComputeApiSupportAPI("vulkan"),
        ProbeComputeApiSupportAPI("rocm"),
        ProbeComputeApiSupportAPI("opencl"),
        ProbeComputeApiSupportAPI("auto")
    };
}

