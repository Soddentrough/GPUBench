#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "core/ResultFormatter.h"

#include <functional>
#include <atomic>

std::vector<ResultData> RunBenchmarksAPI(
    const std::vector<std::string>& benchmarks_to_run,
    const std::vector<uint32_t>& device_indices,
    const std::vector<std::string>& backend_strs,
    bool verbose, bool debug, bool dump_geometry,
    bool dump_renders = true,
    uint32_t renderWidth = 0,
    uint32_t renderHeight = 0,
    std::function<void(const ResultData&)> callback = nullptr,
    const std::string& scene = "all",
    uint32_t samples_per_pixel = 1,
    std::atomic<bool>* cancel_token = nullptr);

struct DeviceProfile {
    std::string backend;
    uint32_t deviceIndex;
    std::string deviceName;
    uint32_t vendorID;
    uint32_t deviceID;
    std::string driverName;
    std::string driverInfo;
    std::string driverVersion;
    std::string apiVersion;
    uint64_t vramTotalMb;
    uint32_t subgroupSize;
    uint32_t maxWorkGroupSize;
    bool rayTracingSupported;
    bool serSupported;
    bool workGraphsSupported;
    bool cooperativeMatrixSupported;
    bool float16Supported;
    bool int8Supported;
};

std::vector<std::string> GetAvailableHardwareAPI();
std::vector<std::string> GetAvailableBenchmarksAPI();
std::vector<DeviceProfile> GetDeviceProfilesAPI();

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
