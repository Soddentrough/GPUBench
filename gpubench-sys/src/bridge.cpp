#include "gpubench-sys/src/lib.rs.h"
#include "bridge.h"
#include <iostream>
#include <vector>
#include <string>
#include "core/RunnerAPI.h"

void gpubench_init() {
    std::cout << "GPUBench FFI initialized!" << std::endl;
}

rust::Vec<FfiResultData> gpubench_run_benchmarks(const rust::Vec<rust::String>& benchmarks_to_run_rust,
                                                 const rust::Vec<uint32_t>& device_indices_rust,
                                                 const rust::Vec<rust::String>& backend_strs_rust,
                                                 bool verbose, bool debug, bool dump_geometry,
                                                 bool dump_renders,
                                                 uint32_t render_width,
                                                 uint32_t render_height,
                                                 rust::Fn<void(const FfiResultData&)> callback,
                                                 rust::String scene_rust,
                                                 uint32_t samples_per_pixel) {
    
    std::vector<std::string> benchmarks_to_run;
    for (const auto& b : benchmarks_to_run_rust) {
        benchmarks_to_run.push_back(std::string(b));
    }

    std::vector<uint32_t> device_indices;
    for (const auto& d : device_indices_rust) {
        device_indices.push_back(d);
    }

    std::vector<std::string> backend_strs;
    for (const auto& b : backend_strs_rust) {
        backend_strs.push_back(std::string(b));
    }

    std::string scene_str = std::string(scene_rust);
    if (scene_str.empty()) scene_str = "all";

    auto raw_results = RunBenchmarksAPI(benchmarks_to_run, device_indices, backend_strs, verbose, debug, dump_geometry, dump_renders, render_width, render_height,
        [&callback](const ResultData& res) {
            FfiResultData r;
            r.backendName = res.backendName;
            r.deviceName = res.deviceName;
            r.benchmarkName = res.benchmarkName;
            r.component = res.component;
            r.subcategory = res.subcategory;
            r.metric = res.metric;
            r.operations = res.operations;
            r.time_ms = res.time_ms;
            r.isEmulated = res.isEmulated;
            r.isUnsupported = res.isUnsupported;
            r.supportNote = res.supportNote;
            r.supportCategory = res.supportCategory;
            r.maxWorkGroupSize = res.maxWorkGroupSize;
            r.deviceIndex = res.deviceIndex;
            r.configIndex = res.configIndex;
            r.sortWeight = res.sortWeight;
            r.width = res.width;
            r.height = res.height;
            callback(r);
        },
        scene_str,
        samples_per_pixel);

    rust::Vec<FfiResultData> ffi_results;
    for (const auto& res : raw_results) {
        FfiResultData r;
        r.backendName = res.backendName;
        r.deviceName = res.deviceName;
        r.benchmarkName = res.benchmarkName;
        r.component = res.component;
        r.subcategory = res.subcategory;
        r.metric = res.metric;
        r.operations = res.operations;
        r.time_ms = res.time_ms;
        r.isEmulated = res.isEmulated;
        r.isUnsupported = res.isUnsupported;
        r.supportNote = res.supportNote;
        r.supportCategory = res.supportCategory;
        r.maxWorkGroupSize = res.maxWorkGroupSize;
        r.deviceIndex = res.deviceIndex;
        r.configIndex = res.configIndex;
        r.sortWeight = res.sortWeight;
        r.width = res.width;
        r.height = res.height;
        ffi_results.push_back(r);
    }

    return ffi_results;
}

rust::Vec<rust::String> gpubench_get_available_hardware() {
    auto cpp_results = GetAvailableHardwareAPI();
    rust::Vec<rust::String> rust_results;
    for (const auto& r : cpp_results) {
        rust_results.push_back(r);
    }
    return rust_results;
}

rust::Vec<rust::String> gpubench_get_available_benchmarks() {
    auto cpp_results = GetAvailableBenchmarksAPI();
    rust::Vec<rust::String> rust_results;
    for (const auto& r : cpp_results) {
        rust_results.push_back(r);
    }
    return rust_results;
}

rust::Vec<FfiDeviceProfile> gpubench_get_device_profiles() {
    auto profiles = GetDeviceProfilesAPI();
    rust::Vec<FfiDeviceProfile> rust_profiles;
    for (const auto& p : profiles) {
        FfiDeviceProfile f;
        f.backend = p.backend;
        f.deviceIndex = p.deviceIndex;
        f.deviceName = p.deviceName;
        f.vendorId = p.vendorID;
        f.deviceId = p.deviceID;
        f.driverName = p.driverName;
        f.driverInfo = p.driverInfo;
        f.driverVersion = p.driverVersion;
        f.apiVersion = p.apiVersion;
        f.vramTotalMb = p.vramTotalMb;
        f.subgroupSize = p.subgroupSize;
        f.maxWorkGroupSize = p.maxWorkGroupSize;
        f.rayTracingSupported = p.rayTracingSupported;
        f.serSupported = p.serSupported;
        f.workGraphsSupported = p.workGraphsSupported;
        f.cooperativeMatrixSupported = p.cooperativeMatrixSupported;
        f.float16Supported = p.float16Supported;
        f.int8Supported = p.int8Supported;
        rust_profiles.push_back(f);
    }
    return rust_profiles;
}

