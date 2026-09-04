#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "core/ResultFormatter.h"

#include <functional>

std::vector<ResultData> RunBenchmarksAPI(
    const std::vector<std::string>& benchmarks_to_run,
    const std::vector<uint32_t>& device_indices,
    const std::vector<std::string>& backend_strs,
    bool verbose, bool debug, bool dump_geometry,
    bool dump_renders = false,
    uint32_t renderWidth = 0,
    uint32_t renderHeight = 0,
    std::function<void(const ResultData&)> callback = nullptr);

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
