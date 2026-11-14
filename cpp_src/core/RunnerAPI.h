#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "core/ResultFormatter.h"

#include <functional>
#include <atomic>

#include <memory>

class VulkanContext;

void SetSharedVulkanContextAPI(std::shared_ptr<VulkanContext> sharedCtx);
std::shared_ptr<VulkanContext> GetSharedVulkanContextAPI();

std::vector<ResultData> RunBenchmarksAPI(
    const std::vector<std::string>& benchmarks_to_run,
    const std::vector<uint32_t>& device_indices,
    const std::vector<std::string>& backend_strs,
    bool verbose, bool debug, bool dump_geometry,
    bool dump_renders = true,
    uint32_t renderWidth = 3840,
    uint32_t renderHeight = 2160,
    std::function<void(const ResultData&)> callback = nullptr,
    const std::string& scene = "all",
    uint32_t samples_per_pixel = 1,
    std::atomic<bool>* cancel_token = nullptr,
    bool quiet = false,
    const std::vector<std::string>& selected_workloads = {},
    uint32_t bounce_depth = 2);

struct DeviceProfile {
    std::string backend;
    uint32_t deviceIndex = 0;
    std::string deviceName;
    uint32_t vendorID = 0;
    uint32_t deviceID = 0;
    std::string driverName;
    std::string driverInfo;
    std::string driverVersion;
    std::string apiVersion;
    uint64_t vramTotalMb = 0;
    uint32_t computeUnits = 0;
    uint32_t subgroupSize = 0;
    uint32_t maxWorkGroupSize = 0;
    bool isApu = false;
    bool rebarSupported = false;
    uint64_t rebarSizeBytes = 0;
    bool rayTracingSupported = false;
    bool serSupported = false;
    bool opacityMicromapSupported = false;
    bool workGraphsSupported = false;
    bool cooperativeMatrixSupported = false;
    bool structuredSparsitySupported = false;
    bool cooperativeVectorSupported = false;
    bool float16Supported = false;
    bool int8Supported = false;
    bool ocpFp6Supported = false;
    bool ocpFp4Supported = false;
    bool performanceQuerySupported = false;
};

struct BenchmarkConfigMetadata {
    std::string benchmarkName;
    uint32_t configIndex = 0;
    std::string configName;
    std::string displayName;
    std::string component;
    std::string subcategory;
    std::string metric;
    std::string description;
    int sortWeight = 0;
    double operationalIntensity = 0.0;
    bool isDeviceDependent = true;
    std::vector<std::string> aliases;
};

std::vector<std::string> GetAvailableHardwareAPI();
std::vector<std::string> GetAvailableBenchmarksAPI();
std::vector<DeviceProfile> GetDeviceProfilesAPI();
std::vector<BenchmarkConfigMetadata> GetBenchmarkMetadataCatalogAPI();

struct BenchmarkSupportInfo {
    std::string id;
    bool isSupported{true};
    std::string reason;
    std::string limitationCategory;
};

std::vector<BenchmarkSupportInfo> ProbeBenchmarkSupportAPI(
    const std::string& backend_name,
    uint32_t device_idx);

struct ComputeApiSupportInfo {
    std::string name;               // "vulkan", "rocm", "opencl", "auto"
    std::string label;              // "Vulkan", "ROCm", "OpenCL", "Auto"
    bool isSupported{false};
    std::string reason;             // Why it is supported or unsupported
    std::string missingRequirement; // Specific requirement missing (SDK, driver, OS)
};

ComputeApiSupportInfo ProbeComputeApiSupportAPI(const std::string& backend_name);
std::vector<ComputeApiSupportInfo> GetAllComputeApiSupportAPI();
