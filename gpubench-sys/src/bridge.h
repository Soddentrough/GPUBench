#pragma once
#include "rust/cxx.h"
#include <cstdint>

struct FfiResultData;



rust::Vec<FfiResultData> gpubench_run_benchmarks(const rust::Vec<rust::String>& benchmarks_to_run_rust,
                                                 const rust::Vec<uint32_t>& device_indices_rust,
                                                 const rust::Vec<rust::String>& backend_strs_rust,
                                                 bool verbose, bool debug, bool dump_geometry,
                                                 rust::Fn<void(const FfiResultData&)> callback);

rust::Vec<rust::String> gpubench_get_available_hardware();
rust::Vec<rust::String> gpubench_get_available_benchmarks();

void gpubench_init();
