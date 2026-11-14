#include "core/RunnerAPI.h"
#include "core/BenchmarkRunner.h"
#include "core/ComputeBackendFactory.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif
#include <iostream>
#include <memory>
#include <mutex>
#include <map>
#include <algorithm>
#include <cctype>

namespace {
std::mutex g_sharedVkMutex;
std::shared_ptr<VulkanContext> g_sharedVulkanContext;

void populateDeviceProfileFromInfo(DeviceProfile &p, const DeviceInfo &dev) {
    p.deviceName = dev.name;
    p.vendorID = dev.vendorID;
    p.deviceID = dev.deviceID;
    p.vramTotalMb = dev.memorySize / (1024 * 1024);
    p.computeUnits = dev.computeUnits;
    p.subgroupSize = dev.subgroupSize;
    p.maxWorkGroupSize = dev.maxWorkGroupSize;
    p.isApu = dev.isApu;
    p.rebarSupported = dev.rebarSupported;
    p.rebarSizeBytes = dev.rebarSizeBytes;
    p.rayTracingSupported = dev.rayTracingSupport;
    p.serSupported = dev.serSupported;
    p.opacityMicromapSupported = dev.opacityMicromapSupport;
    p.workGraphsSupported = dev.workGraphsSupported;
    p.cooperativeMatrixSupported = dev.cooperativeMatrixSupport;
    p.structuredSparsitySupported = dev.structuredSparsitySupport;
    p.cooperativeVectorSupported = dev.cooperativeVectorSupport;
    p.float16Supported = dev.fp16Support;
    p.int8Supported = dev.int8Support;
    p.ocpFp6Supported = dev.fp6Support;
    p.ocpFp4Supported = dev.fp4Support;
    p.performanceQuerySupported = dev.performanceQuerySupported;
}
} // namespace

void SetSharedVulkanContextAPI(std::shared_ptr<VulkanContext> sharedCtx) {
    std::lock_guard<std::mutex> lock(g_sharedVkMutex);
    g_sharedVulkanContext = std::move(sharedCtx);
}

std::shared_ptr<VulkanContext> GetSharedVulkanContextAPI() {
    std::lock_guard<std::mutex> lock(g_sharedVkMutex);
    return g_sharedVulkanContext;
}

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
    std::atomic<bool>* cancel_token,
    bool quiet,
    const std::vector<std::string>& selected_workloads,
    uint32_t bounce_depth)
{
    try {
    auto sharedVk = GetSharedVulkanContextAPI();

    std::vector<ComputeBackend> target_backends;
    if (backend_strs.empty() || (backend_strs.size() == 1 && backend_strs[0] == "auto")) {
#ifdef HAVE_VULKAN
        if (sharedVk && sharedVk->isAvailable() && !sharedVk->getDevices().empty()) {
            target_backends.push_back(ComputeBackend::Vulkan);
        }
#endif
        if (target_backends.empty()) {
            const ComputeBackend auto_order[] = {
                ComputeBackend::Vulkan, ComputeBackend::ROCm, ComputeBackend::OpenCL};
            for (ComputeBackend backend : auto_order) {
                if (!ComputeBackendFactory::isAvailable(backend)) {
                    continue;
                }
                try {
                    auto ctx = ComputeBackendFactory::create(backend, verbose, debug);
                    if (ctx && ctx->isAvailable() && !ctx->getDevices().empty()) {
                        target_backends.push_back(backend);
                        break;
                    }
                } catch (...) {
                }
            }
        }
    } else {
        for (const auto& backend_str : backend_strs) {
            if (backend_str == "vulkan" && ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
                target_backends.push_back(ComputeBackend::Vulkan);
            } else if (backend_str == "rocm" && ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
                target_backends.push_back(ComputeBackend::ROCm);
            } else if (backend_str == "opencl" && ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
                target_backends.push_back(ComputeBackend::OpenCL);
            }
        }
    }

    BenchmarkRunner runner({}, verbose, debug, dump_geometry, dump_renders, scene.empty() ? "all" : scene);
    runner.setResolution(renderWidth, renderHeight);
    runner.setBounceDepth(bounce_depth);
    runner.setSamplesPerPixel(samples_per_pixel);
    runner.setQuiet(quiet || (callback != nullptr));
    if (sharedVk != nullptr) {
        runner.setCaptureGpuPreviews(true);
    }
    if (!selected_workloads.empty()) {
        runner.setWorkloadFilter(selected_workloads);
    }
    if (cancel_token) {
        runner.setCancelToken(cancel_token);
    }
    if (callback) {
        runner.onResult = callback;
    }

    std::vector<uint32_t> target_indices = device_indices;
    if (target_indices.empty()) {
        bool hasGpuWorkloads = false;
        for (const auto& b : benchmarks_to_run) {
            if (b.find("System Memory") == std::string::npos && b.find("SysMem") == std::string::npos) {
                hasGpuWorkloads = true;
                break;
            }
        }
        if (hasGpuWorkloads) {
#ifdef HAVE_VULKAN
            if (sharedVk) {
                target_indices.push_back(sharedVk->getSelectedDeviceIndex());
            } else
#endif
            {
                target_indices.push_back(0);
            }
        }
    }

    for (ComputeBackend backend : target_backends) {
        if (cancel_token && cancel_token->load()) break;
        for (uint32_t device_idx : target_indices) {
            if (cancel_token && cancel_token->load()) break;
#ifdef HAVE_VULKAN
            if (backend == ComputeBackend::Vulkan && sharedVk &&
                sharedVk->isAvailable() &&
                device_idx == sharedVk->getSelectedDeviceIndex()) {
                runner.runForContext(sharedVk.get(), benchmarks_to_run);
                continue;
            }
#endif
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
    std::vector<std::string> results;
    results.push_back("System|0|System Memory / Host CPU");

    try {
    if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
        try {
#ifdef HAVE_VULKAN
            auto sharedVk = GetSharedVulkanContextAPI();
            if (sharedVk && sharedVk->isAvailable()) {
                uint32_t i = 0;
                for (const auto& dev : sharedVk->getDevices()) {
                    results.push_back("vulkan|" + std::to_string(i) + "|" + dev.name);
                    i++;
                }
            } else
#endif
            {
                auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
                if (ctx) {
                    uint32_t i = 0;
                    for (const auto& dev : ctx->getDevices()) {
                        results.push_back("vulkan|" + std::to_string(i) + "|" + dev.name);
                        i++;
                    }
                }
            }
        } catch (...) {
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

std::vector<BenchmarkConfigMetadata> GetBenchmarkMetadataCatalogAPI() {
    std::vector<BenchmarkConfigMetadata> catalog;
    try {
        std::vector<IComputeContext*> dummy;
        BenchmarkRunner runner(dummy, false, false, false, false, "all");
        for (const auto& bench : runner.getBenchmarkList()) {
            std::string bName = bench->GetName();
            uint32_t numConfigs = bench->GetNumConfigs();
            auto aliases = bench->GetAliases();
            for (uint32_t ci = 0; ci < numConfigs; ++ci) {
                BenchmarkConfigMetadata meta;
                meta.benchmarkName = bName;
                meta.configIndex = ci;
                meta.configName = bench->GetConfigName(ci);
                if (meta.configName.rfind(bName, 0) == 0) {
                    meta.displayName = meta.configName;
                } else if (!meta.configName.empty()) {
                    meta.displayName = bName + " (" + meta.configName + ")";
                } else {
                    meta.displayName = bName;
                }
                meta.component = bench->GetComponent(ci);
                meta.subcategory = bench->GetSubCategory(ci);
                meta.metric = bench->GetMetric(ci);
                const char* desc = bench->GetDescription(ci);
                meta.description = desc ? desc : "";
                meta.sortWeight = bench->GetSortWeight(ci);
                meta.operationalIntensity = bench->GetOperationalIntensity(ci);
                meta.isDeviceDependent = bench->IsDeviceDependent();
                meta.aliases = aliases;
                catalog.push_back(std::move(meta));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "GetBenchmarkMetadataCatalogAPI failed: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "GetBenchmarkMetadataCatalogAPI failed: unknown error" << std::endl;
    }
    return catalog;
}

std::vector<DeviceProfile> GetDeviceProfilesAPI() {
    std::vector<DeviceProfile> profiles;
    try {
        if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
            try {
                auto appendVulkanDevices = [&](const std::vector<DeviceInfo>& devices) {
                    uint32_t i = 0;
                    for (const auto& dev : devices) {
                        DeviceProfile p{};
                        p.backend = "Vulkan";
                        p.deviceIndex = i;
                        populateDeviceProfileFromInfo(p, dev);
                        p.driverName = dev.driverName;
                        p.driverInfo = dev.driverInfo;
                        p.driverVersion = dev.driverVersionStr;
                        uint32_t apiMajor = dev.apiVersion >> 22;
                        uint32_t apiMinor = (dev.apiVersion >> 12) & 0x3FF;
                        uint32_t apiPatch = dev.apiVersion & 0xFFF;
                        p.apiVersion = std::to_string(apiMajor) + "." + std::to_string(apiMinor) + "." + std::to_string(apiPatch);
                        profiles.push_back(std::move(p));
                        i++;
                    }
                };
#ifdef HAVE_VULKAN
                auto sharedVk = GetSharedVulkanContextAPI();
                if (sharedVk && sharedVk->isAvailable()) {
                    appendVulkanDevices(sharedVk->getDevices());
                } else
#endif
                {
                    auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
                    if (ctx) {
                        appendVulkanDevices(ctx->getDevices());
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
                        DeviceProfile p{};
                        p.backend = "ROCm";
                        p.deviceIndex = i;
                        populateDeviceProfileFromInfo(p, dev);
                        p.driverName = dev.driverName.empty() ? "AMD ROCm / HIP" : dev.driverName;
                        p.driverInfo = dev.driverInfo.empty() ? (dev.archName.empty() ? "HIP Runtime" : dev.archName) : dev.driverInfo;
                        p.driverVersion = dev.driverVersionStr.empty() ? std::to_string(dev.driverVersion) : dev.driverVersionStr;
                        if (dev.driverVersion > 0) {
                            p.apiVersion = "HIP " + std::to_string(dev.driverVersion / 10000000) + "." + std::to_string((dev.driverVersion / 100000) % 100);
                        } else {
                            p.apiVersion = "ROCm";
                        }
                        profiles.push_back(std::move(p));
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
                        DeviceProfile p{};
                        p.backend = "OpenCL";
                        p.deviceIndex = i;
                        populateDeviceProfileFromInfo(p, dev);
                        p.driverName = dev.driverName.empty() ? "OpenCL Driver" : dev.driverName;
                        p.driverInfo = dev.driverInfo.empty() ? "OpenCL 3.0" : dev.driverInfo;
                        p.driverVersion = dev.driverVersionStr.empty() ? std::to_string(dev.driverVersion) : dev.driverVersionStr;
                        if (dev.apiVersion > 0) {
                            uint32_t apiMajor = dev.apiVersion >> 22;
                            uint32_t apiMinor = (dev.apiVersion >> 12) & 0x3FF;
                            uint32_t apiPatch = dev.apiVersion & 0xFFF;
                            p.apiVersion = std::to_string(apiMajor) + "." + std::to_string(apiMinor) + "." + std::to_string(apiPatch);
                        } else {
                            p.apiVersion = "OpenCL 3.0";
                        }
                        profiles.push_back(std::move(p));
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
            std::unique_ptr<IComputeContext> ownedCtx;
            IComputeContext* activeCtx = nullptr;
#ifdef HAVE_VULKAN
            auto sharedVk = GetSharedVulkanContextAPI();
            if (backend == ComputeBackend::Vulkan && sharedVk &&
                sharedVk->isAvailable() &&
                device_idx == sharedVk->getSelectedDeviceIndex()) {
                activeCtx = sharedVk.get();
            }
#endif
            if (!activeCtx) {
                ownedCtx = ComputeBackendFactory::create(backend, false, false);
                if (ownedCtx && device_idx < ownedCtx->getDevices().size()) {
                    ownedCtx->pickDevice(device_idx);
                    activeCtx = ownedCtx.get();
                }
            }

            if (activeCtx) {
                DeviceInfo info = activeCtx->getCurrentDeviceInfo();
                BenchmarkRunner runner({}, false, false, false, false, "all");
                for (const auto& bench : runner.getBenchmarkList()) {
                    BenchmarkSupportInfo item;
                    item.id = bench->GetName();
                    item.isSupported = bench->IsSupported(info, activeCtx);
                    if (!item.isSupported) {
                        item.reason = bench->GetSupportNote(info, activeCtx);
                        switch (bench->GetSupportLimitation(info, activeCtx)) {
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

                    if (std::string(bench->GetName()).rfind("RayScheduling", 0) == 0) {
                        BenchmarkSupportInfo baseItem = item;
                        baseItem.id = "RayScheduling";
                        results.push_back(baseItem);
                    }

                    uint32_t num_configs = bench->GetNumConfigs();
                    for (uint32_t ci = 0; ci < num_configs; ++ci) {
                        if (!bench->IsConfigSupported(ci, info, activeCtx)) {
                            BenchmarkSupportInfo cfgItem;
                            cfgItem.id = bench->GetConfigName(ci);
                            cfgItem.isSupported = false;
                            cfgItem.reason = bench->GetConfigSupportNote(ci, info, activeCtx);
                            if (cfgItem.reason.empty()) {
                                cfgItem.reason = bench->GetSupportNote(info, activeCtx);
                            }
                            switch (bench->GetConfigSupportLimitation(ci, info, activeCtx)) {
                            case IBenchmark::SupportLimitation::kHardware:
                                cfgItem.limitationCategory = "Hardware Limitation";
                                break;
                            case IBenchmark::SupportLimitation::kApi:
                                cfgItem.limitationCategory = "API Limitation";
                                break;
                            case IBenchmark::SupportLimitation::kToolchain:
                                cfgItem.limitationCategory = "Toolchain Limitation";
                                break;
                            default:
                                cfgItem.limitationCategory = "Unsupported";
                                break;
                            }
                            results.push_back(cfgItem);
                            BenchmarkSupportInfo indexedItem = cfgItem;
                            indexedItem.id = std::string(bench->GetName()) + "#" + std::to_string(ci);
                            results.push_back(indexedItem);
                            if (std::string(bench->GetName()).rfind("RayScheduling", 0) == 0) {
                                BenchmarkSupportInfo baseIndexed = cfgItem;
                                baseIndexed.id = "RayScheduling#" + std::to_string(ci);
                                results.push_back(baseIndexed);
                            }
                        }
                    }
                }

                if (!info.serSupported) {
                    BenchmarkSupportInfo serItem;
                    serItem.id = "SER";
                    serItem.isSupported = false;
                    serItem.reason = "VK_EXT_ray_tracing_invocation_reorder requires Ray Tracing Pipeline with hardware SER support";
                    serItem.limitationCategory = "Hardware Limitation";
                    results.push_back(serItem);
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
            auto sharedVk = GetSharedVulkanContextAPI();
            if (sharedVk && sharedVk->isAvailable() && !sharedVk->getDevices().empty()) {
                info.isSupported = true;
                info.reason = "Vulkan 1.4 compute backend is operational with " +
                              std::to_string(sharedVk->getDevices().size()) +
                              " detected device(s) (shared device active).";
                info.missingRequirement = "";
            } else {
                auto ctx = ComputeBackendFactory::create(ComputeBackend::Vulkan, false, false);
                if (ctx) {
                    const auto& devices = ctx->getDevices();
                    if (devices.empty()) {
                        info.isSupported = false;
                        info.reason = "Vulkan 1.4 runtime initialized, but no Vulkan 1.4 capable compute devices were found.";
                        info.missingRequirement = "Vulkan 1.4-capable graphics hardware and vendor ICD driver.";
                    } else {
                        info.isSupported = true;
                        info.reason = "Vulkan 1.4 compute backend is operational with " + std::to_string(devices.size()) + " detected device(s).";
                        info.missingRequirement = "";
                    }
                } else {
                    info.isSupported = false;
                    info.reason = "Vulkan 1.4 context could not be instantiated.";
                    info.missingRequirement = "Vulkan 1.4 loader runtime (vulkan-1.dll / libvulkan.so.1).";
                }
            }
        } catch (const std::exception& e) {
            info.isSupported = false;
            info.reason = std::string("Vulkan 1.4 runtime initialization failed: ") + e.what();
            info.missingRequirement = "Vulkan 1.4 ICD driver or loader runtime (vulkan-1.dll).";
        } catch (...) {
            info.isSupported = false;
            info.reason = "Vulkan 1.4 runtime initialization failed with an unknown error.";
            info.missingRequirement = "Vulkan 1.4 ICD driver or loader runtime.";
        }
#else
        info.isSupported = false;
        info.reason = "Vulkan compute backend was not compiled into this GPUBench binary.";
        info.missingRequirement = "Vulkan 1.4 SDK headers and libraries (vulkan/vulkan.h, Vulkan-1) at build time.";
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
                    info.missingRequirement = "Compatible AMD GPU with HIP runtime driver support.";
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
            info.missingRequirement = "AMD ROCm / HIP runtime stack (amdhip64).";
        } catch (...) {
            info.isSupported = false;
            info.reason = "ROCm initialization failed with an unknown error.";
            info.missingRequirement = "AMD ROCm / HIP runtime stack.";
        }
#else
        info.isSupported = false;
#ifdef _WIN32
        info.reason = "ROCm (HIP) backend was not compiled into this Windows binary (HIP SDK headers absent at build time).";
        info.missingRequirement = "AMD HIP SDK headers at build time + amdhip64_7.dll / amdhip64_6.dll at runtime.";
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
                    info.reason = "OpenCL 3.0 compute backend is operational with " + std::to_string(devices.size()) + " detected device(s).";
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
            info.reason = "Auto selects Vulkan 1.4 (highest priority available backend on this workstation).";
            info.missingRequirement = "";
        } else if (rInfo.isSupported) {
            info.isSupported = true;
            info.reason = "Auto selects ROCm (HIP).";
            info.missingRequirement = "";
        } else if (oInfo.isSupported) {
            info.isSupported = true;
            info.reason = "Auto selects OpenCL 3.0.";
            info.missingRequirement = "";
        } else {
            info.isSupported = false;
            info.reason = "No compatible compute backend runtime is available on this system.";
            info.missingRequirement = "A functional graphics driver and runtime supporting Vulkan 1.4, ROCm, or OpenCL.";
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

