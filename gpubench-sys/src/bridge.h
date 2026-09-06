#pragma once
#include "rust/cxx.h"
#include <cstdint>

struct FfiResultData;
struct FfiDeviceProfile;

rust::Vec<FfiResultData> gpubench_run_benchmarks(const rust::Vec<rust::String>& benchmarks_to_run_rust,
                                                 const rust::Vec<uint32_t>& device_indices_rust,
                                                 const rust::Vec<rust::String>& backend_strs_rust,
                                                 bool verbose, bool debug, bool dump_geometry,
                                                 bool dump_renders,
                                                 uint32_t render_width,
                                                 uint32_t render_height,
                                                 rust::Fn<void(const FfiResultData&)> callback,
                                                 rust::String scene_rust,
                                                 uint32_t samples_per_pixel);

rust::Vec<rust::String> gpubench_get_available_hardware();
rust::Vec<rust::String> gpubench_get_available_benchmarks();
rust::Vec<FfiDeviceProfile> gpubench_get_device_profiles();

void gpubench_init();
