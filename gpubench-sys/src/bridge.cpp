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
                                                 rust::Fn<void(const FfiResultData&)> callback) {
    
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

    auto raw_results = RunBenchmarksAPI(benchmarks_to_run, device_indices, backend_strs, verbose, debug, dump_geometry, 
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
            r.maxWorkGroupSize = res.maxWorkGroupSize;
            r.deviceIndex = res.deviceIndex;
            r.configIndex = res.configIndex;
            r.sortWeight = res.sortWeight;
            callback(r);
        });

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
        r.maxWorkGroupSize = res.maxWorkGroupSize;
        r.deviceIndex = res.deviceIndex;
        r.configIndex = res.configIndex;
        r.sortWeight = res.sortWeight;
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
