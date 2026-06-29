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
    std::function<void(const ResultData&)> callback = nullptr);

std::vector<std::string> GetAvailableHardwareAPI();
std::vector<std::string> GetAvailableBenchmarksAPI();
