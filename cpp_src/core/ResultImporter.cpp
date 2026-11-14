#include "ResultImporter.h"
#include "third_party/yyjson.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

struct YyjsonDocGuard {
  yyjson_doc *doc = nullptr;
  ~YyjsonDocGuard() {
    if (doc != nullptr) {
      yyjson_doc_free(doc);
    }
  }
};

struct JsonView {
  yyjson_val *v = nullptr;

  bool is_null() const { return v == nullptr || yyjson_is_null(v); }
  bool is_bool() const { return v != nullptr && yyjson_is_bool(v); }
  bool is_number() const { return v != nullptr && yyjson_is_num(v); }
  bool is_string() const { return v != nullptr && yyjson_is_str(v); }
  bool is_array() const { return v != nullptr && yyjson_is_arr(v); }
  bool is_object() const { return v != nullptr && yyjson_is_obj(v); }

  size_t size() const {
    if (is_array()) return yyjson_arr_size(v);
    if (is_object()) return yyjson_obj_size(v);
    return 0;
  }

  bool empty() const { return size() == 0; }

  JsonView operator[](const char *key) const {
    if (!is_object() || key == nullptr) return {nullptr};
    return {yyjson_obj_get(v, key)};
  }

  JsonView operator[](const std::string &key) const {
    return (*this)[key.c_str()];
  }

  JsonView operator[](size_t idx) const {
    if (!is_array()) return {nullptr};
    return {yyjson_arr_get(v, idx)};
  }

  bool has(const char *key) const {
    return is_object() && key != nullptr && yyjson_obj_get(v, key) != nullptr;
  }

  std::string as_string(const std::string &fallback = "") const {
    if (!is_string()) return fallback;
    const char *s = yyjson_get_str(v);
    const size_t n = yyjson_get_len(v);
    return s != nullptr ? std::string(s, n) : fallback;
  }

  double as_double(double fallback = 0.0) const {
    if (!is_number()) return fallback;
    return yyjson_get_num(v);
  }

  uint64_t as_uint64(uint64_t fallback = 0) const {
    if (v == nullptr) return fallback;
    if (yyjson_is_uint(v)) return yyjson_get_uint(v);
    if (yyjson_is_sint(v)) {
      const int64_t s = yyjson_get_sint(v);
      return s > 0 ? static_cast<uint64_t>(s) : 0ULL;
    }
    if (yyjson_is_real(v)) {
      const double d = yyjson_get_real(v);
      return d > 0.0 ? static_cast<uint64_t>(d) : 0ULL;
    }
    return fallback;
  }

  uint32_t as_uint32(uint32_t fallback = 0) const {
    return static_cast<uint32_t>(as_uint64(fallback));
  }

  int32_t as_int32(int32_t fallback = 0) const {
    if (v == nullptr) return fallback;
    if (yyjson_is_sint(v)) return static_cast<int32_t>(yyjson_get_sint(v));
    if (yyjson_is_uint(v)) return static_cast<int32_t>(yyjson_get_uint(v));
    if (yyjson_is_real(v)) return static_cast<int32_t>(yyjson_get_real(v));
    return fallback;
  }

  bool as_bool(bool fallback = false) const {
    return is_bool() ? yyjson_get_bool(v) : fallback;
  }
};

uint32_t parseHexUint32(const std::string &hexStr) {
  if (hexStr.empty()) return 0;
  char *endPtr = nullptr;
  unsigned long parsed = std::strtoul(hexStr.c_str(), &endPtr, 0);
  return static_cast<uint32_t>(parsed);
}

void parseResolution(const std::string &resStr, uint32_t &width, uint32_t &height) {
  width = 0;
  height = 0;
  // Look for "(WxH)" or "WxH"
  size_t openParen = resStr.find('(');
  size_t closeParen = resStr.find(')', openParen != std::string::npos ? openParen : 0);
  std::string target = resStr;
  if (openParen != std::string::npos && closeParen != std::string::npos && closeParen > openParen) {
    target = resStr.substr(openParen + 1, closeParen - openParen - 1);
  }
  size_t xPos = target.find('x');
  if (xPos == std::string::npos) {
    xPos = target.find('X');
  }
  if (xPos != std::string::npos) {
    try {
      width = std::stoul(target.substr(0, xPos));
      height = std::stoul(target.substr(xPos + 1));
    } catch (...) {}
  }
}

} // namespace

bool ResultImporter::loadFromFile(const std::string &filepath, ImportedRun &outRun,
                                  std::string &errorMessage) {
  std::ifstream file(filepath, std::ios::in | std::ios::binary);
  if (!file) {
    errorMessage = "Could not open file: " + filepath;
    return false;
  }

  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  file.close();

  if (content.empty()) {
    errorMessage = "File is empty: " + filepath;
    return false;
  }

  return loadFromString(content, outRun, errorMessage);
}

bool ResultImporter::loadFromString(const std::string &content, ImportedRun &outRun,
                                    std::string &errorMessage) {
  if (content.empty()) {
    errorMessage = "JSON content is empty";
    return false;
  }

  yyjson_read_err readErr{};
  YyjsonDocGuard docGuard{yyjson_read_opts(const_cast<char *>(content.data()),
                                           content.size(), 0, nullptr, &readErr)};
  if (docGuard.doc == nullptr) {
    errorMessage = std::string("JSON parse error: ") +
                   (readErr.msg != nullptr ? readErr.msg : "invalid syntax") +
                   " at byte " + std::to_string(readErr.pos);
    return false;
  }

  const JsonView root{yyjson_doc_get_root(docGuard.doc)};
  if (root.is_null() || (!root.is_object() && !root.is_array())) {
    errorMessage = "Invalid JSON root: expected object or array";
    return false;
  }

  outRun = ImportedRun{};
  outRun.schemaVersion = root["schema_version"].as_string("1.0");
  outRun.appVersion = root["app_version"].as_string(root["version"].as_string("1.0.0"));
  outRun.timestamp = root["timestamp"].as_uint64(0);
  outRun.backend = root["backend"].as_string("Vulkan");
  outRun.resolution = root["resolution"].as_string("4K (UHD) (3840x2160)");

  uint32_t resWidth = 0, resHeight = 0;
  parseResolution(outRun.resolution, resWidth, resHeight);

  if (root.has("system_info")) {
    const JsonView sys = root["system_info"];
    outRun.osName = sys["os_name"].as_string();
    outRun.cpuModel = sys["cpu_model"].as_string();
    outRun.cpuCores = sys["cpu_logical_cores"].as_uint32(0);
    outRun.totalRamGb = sys["total_ram_gb"].as_double(0.0);
  }

  if (root.has("gui_session") && root["gui_session"].is_object()) {
    const JsonView gs = root["gui_session"];
    outRun.hasGuiSession = true;
    outRun.sessionScene = gs["scene"].as_string("all");
    outRun.sessionSamplesPerPixel = gs["samples_per_pixel"].as_uint32(1);
    outRun.sessionDumpRenders = gs["dump_renders"].as_bool(true);
    outRun.sessionUiScale = static_cast<float>(gs["ui_scale"].as_double(1.0));
  }

  uint32_t profileVendorId = 0;
  uint32_t profileDeviceId = 0;
  if (root["device_profiles"].is_array() && !root["device_profiles"].empty()) {
    const JsonView p = root["device_profiles"][static_cast<size_t>(0)];
    outRun.deviceProfile.backend = p["backend"].as_string();
    outRun.deviceProfile.deviceIndex = p["device_index"].as_uint32(0);
    outRun.deviceProfile.deviceName = p["device_name"].as_string();
    outRun.deviceProfile.vendorId = p["vendor_id"].as_string();
    outRun.deviceProfile.deviceId = p["device_id"].as_string(p["device_id_hex"].as_string());
    outRun.deviceProfile.driverName = p["driver_name"].as_string();
    outRun.deviceProfile.driverInfo = p["driver_info"].as_string();
    outRun.deviceProfile.driverVersion = p["driver_version"].as_string();
    outRun.deviceProfile.apiVersion = p["api_version"].as_string();
    outRun.deviceProfile.vramTotalMb = p["vram_total_mb"].as_uint64(0);
    outRun.deviceProfile.computeUnits = p["compute_units"].as_uint32(0);
    outRun.deviceProfile.subgroupSize = p["subgroup_size"].as_uint32(0);
    outRun.deviceProfile.maxWorkgroupSize = p["max_workgroup_size"].as_uint32(1024);
    outRun.deviceProfile.isApu = p["is_apu"].as_bool(false);
    outRun.deviceProfile.rebarSupported = p["rebar_supported"].as_bool(false);
    outRun.deviceProfile.rebarSizeMb = p["rebar_size_mb"].as_uint64(0);
    outRun.deviceProfile.rayTracingSupported = p["ray_tracing_supported"].as_bool(false);
    outRun.deviceProfile.serSupported = p["ser_supported"].as_bool(false);
    outRun.deviceProfile.opacityMicromapSupported = p["opacity_micromap_supported"].as_bool(false);
    outRun.deviceProfile.workGraphsSupported = p["work_graphs_supported"].as_bool(false);
    outRun.deviceProfile.cooperativeMatrixSupported = p["cooperative_matrix_supported"].as_bool(false);
    outRun.deviceProfile.structuredSparsitySupported = p["structured_sparsity_supported"].as_bool(false);
    outRun.deviceProfile.cooperativeVectorSupported = p["cooperative_vector_supported"].as_bool(false);
    profileVendorId = parseHexUint32(outRun.deviceProfile.vendorId);
    profileDeviceId = parseHexUint32(outRun.deviceProfile.deviceId);
  }

  if (outRun.deviceProfile.deviceName.empty()) {
    if (root["devices"].is_array() && !root["devices"].empty()) {
      std::string dn = root["devices"][static_cast<size_t>(0)].as_string();
      size_t colonPos = dn.find(": ");
      if (colonPos != std::string::npos && colonPos <= 3) dn = dn.substr(colonPos + 2);
      outRun.deviceProfile.deviceName = dn;
    }
  }

  JsonView resultsJson;
  bool isGuiSchema = false;

  if (root.is_array()) {
    resultsJson = root;
    if (!root.empty()) {
      const JsonView first = root[static_cast<size_t>(0)];
      outRun.backend = first["backend"].as_string("Vulkan");
      outRun.deviceProfile.deviceName = first["device"].as_string();
      outRun.deviceProfile.deviceIndex = first["device_index"].as_uint32(0);
      outRun.deviceProfile.backend = outRun.backend;
      outRun.deviceProfile.maxWorkgroupSize = first["max_workgroup_size"].as_uint32(1024);
      if (first.has("resolution")) {
        outRun.resolution = first["resolution"].as_string();
        parseResolution(outRun.resolution, resWidth, resHeight);
      }
    }
  } else {
    const JsonView rj = root["results"];
    if (!rj.is_array()) {
      errorMessage = "Invalid JSON: missing 'results' array";
      return false;
    }
    resultsJson = rj;
    if (!resultsJson.empty() && resultsJson[static_cast<size_t>(0)].has("benchmarks")) {
      isGuiSchema = true;
    }
  }

  if (isGuiSchema) {
    size_t devIdxIter = 0, devMaxIter = 0;
    yyjson_val *devRaw = nullptr;
    yyjson_arr_foreach(resultsJson.v, devIdxIter, devMaxIter, devRaw) {
      const JsonView devEntry{devRaw};
      uint32_t devId = devEntry["device_id"].as_uint32(0);
      std::string devName = devEntry["device_name"].as_string();

      // Clean prefix like "0: " from device name if present
      size_t colonPos = devName.find(": ");
      if (colonPos != std::string::npos && colonPos <= 3) {
        devName = devName.substr(colonPos + 2);
      }

      bool isSystem = (devId == 999 || devName.find("Host CPU") != std::string::npos);
      if (outRun.deviceProfile.deviceName.empty() && !isSystem) {
        outRun.deviceProfile.deviceName = devName;
      }

      const JsonView benchmarksArr = devEntry["benchmarks"];
      if (!benchmarksArr.is_array()) continue;

      size_t bmIdxIter = 0, bmMaxIter = 0;
      yyjson_val *bmRaw = nullptr;
      yyjson_arr_foreach(benchmarksArr.v, bmIdxIter, bmMaxIter, bmRaw) {
        const JsonView bm{bmRaw};
        std::string id = bm["id"].as_string();
        std::string label = bm["label"].as_string();
        std::string category = bm["category"].as_string();
        std::string approach = bm["approach"].as_string();
        std::string unit = bm["unit"].as_string();
        std::string status = bm["status"].as_string();
        std::string note = bm["support_note"].as_string(bm["unsupported_reason"].as_string());
        std::string catLimitation = bm["support_category"].as_string();
        uint64_t ops = bm["raw_operations"].as_uint64(0);
        double timeMs = bm["raw_time_ms"].as_double(0.0);
        double numericVal = bm["numeric"].as_double(0.0);

        if (ops == 0 && timeMs <= 0.0 && numericVal > 0.0) {
          timeMs = 1000.0;
          if (unit == "TFLOPS" || unit == "TOPS") {
            ops = static_cast<uint64_t>(numericVal * 1e12);
          } else if (unit == "GB/s" || unit == "GIS/s" || unit == "GPixels/s") {
            ops = static_cast<uint64_t>(numericVal * 1e9);
          } else if (unit == "MRays/s" || unit == "MTris/s" || unit == "MInst/s" || unit == "MPixels/s") {
            ops = static_cast<uint64_t>(numericVal * 1e6);
          } else if (unit == "ns") {
            timeMs = numericVal;
            ops = 1000000;
          }
        }

        ResultData rd;
        rd.backendName = isSystem ? "System" : outRun.backend;
        rd.deviceName = devName;
        rd.deviceIndex = isSystem ? 0xFFFFFFFF : devId;
        rd.metric = unit;
        rd.operations = ops;
        rd.time_ms = timeMs;
        rd.isUnsupported = (status == "unsupported");
        rd.supportNote = note;
        rd.supportCategory = catLimitation;
        rd.maxWorkGroupSize = outRun.deviceProfile.maxWorkgroupSize ? outRun.deviceProfile.maxWorkgroupSize : 1024;
        rd.width = resWidth;
        rd.height = resHeight;

        // Map GUI workload definition ID to hierarchical benchmark properties
        if (id == "fp64") {
          rd.component = "Compute";
          rd.subcategory = "Double Precision";
          rd.benchmarkName = "FP64";
          rd.sortWeight = 10;
          rd.configIndex = 0;
        } else if (id == "fp32") {
          rd.component = "Compute";
          rd.subcategory = "Single Precision";
          rd.benchmarkName = "FP32";
          rd.sortWeight = 20;
          rd.configIndex = 0;
        } else if (id == "fp16_vec") {
          rd.component = "Compute";
          rd.subcategory = "Half Precision (FP16)";
          rd.benchmarkName = "Vector";
          rd.sortWeight = 30;
          rd.configIndex = 0;
        } else if (id == "fp16_mat") {
          rd.component = "Compute";
          rd.subcategory = "Half Precision (FP16)";
          rd.benchmarkName = "Matrix";
          rd.sortWeight = 30;
          rd.configIndex = 1;
        } else if (id == "bf16_vec") {
          rd.component = "Compute";
          rd.subcategory = "Bfloat16 (BF16)";
          rd.benchmarkName = "Vector";
          rd.sortWeight = 40;
          rd.configIndex = 0;
        } else if (id == "bf16_mat") {
          rd.component = "Compute";
          rd.subcategory = "Bfloat16 (BF16)";
          rd.benchmarkName = "Matrix";
          rd.sortWeight = 40;
          rd.configIndex = 1;
        } else if (id == "fp8_vec") {
          rd.component = "Compute";
          rd.subcategory = "Quarter Precision (FP8)";
          rd.benchmarkName = "Vector";
          rd.sortWeight = 50;
          rd.configIndex = 0;
        } else if (id == "fp8_mat") {
          rd.component = "Compute";
          rd.subcategory = "Quarter Precision (FP8)";
          rd.benchmarkName = "Matrix";
          rd.sortWeight = 50;
          rd.configIndex = 1;
        } else if (id == "int8_vec") {
          rd.component = "Compute";
          rd.subcategory = "8-bit Integer (INT8)";
          rd.benchmarkName = "Vector";
          rd.sortWeight = 60;
          rd.configIndex = 0;
        } else if (id == "int8_mat") {
          rd.component = "Compute";
          rd.subcategory = "8-bit Integer (INT8)";
          rd.benchmarkName = "Matrix";
          rd.sortWeight = 60;
          rd.configIndex = 1;
        } else if (id == "int4_vec") {
          rd.component = "Compute";
          rd.subcategory = "4-bit Integer (INT4)";
          rd.benchmarkName = "Vector";
          rd.sortWeight = 70;
          rd.configIndex = 0;
        } else if (id == "int4_mat") {
          rd.component = "Compute";
          rd.subcategory = "4-bit Integer (INT4)";
          rd.benchmarkName = "Matrix";
          rd.sortWeight = 70;
          rd.configIndex = 1;
        } else if (id == "gpu_vram_bw") {
          rd.component = "Memory";
          rd.subcategory = "Bandwidth";
          rd.benchmarkName = "GPU VRAM Bandwidth";
          rd.sortWeight = 100;
          rd.configIndex = 0;
        } else if (id == "cache_l0") {
          rd.component = "Memory";
          rd.subcategory = "Latency";
          rd.benchmarkName = "L0 Cache Latency";
          rd.sortWeight = 110;
          rd.configIndex = 0;
        } else if (id == "cache_l1") {
          rd.component = "Memory";
          rd.subcategory = "Latency";
          rd.benchmarkName = "L1 Cache Latency";
          rd.sortWeight = 120;
          rd.configIndex = 1;
        } else if (id == "cache_l2") {
          rd.component = "Memory";
          rd.subcategory = "Latency";
          rd.benchmarkName = "L2 Cache Latency";
          rd.sortWeight = 130;
          rd.configIndex = 2;
        } else if (id == "cache_l3") {
          rd.component = "Memory";
          rd.subcategory = "Latency";
          rd.benchmarkName = "L3 Cache Latency";
          rd.sortWeight = 140;
          rd.configIndex = 3;
        } else if (id == "rop_rgba8") {
          rd.component = "Rasterization & ROP";
          rd.subcategory = "Pixel Fill Rate";
          rd.benchmarkName = "RGBA8 Color Fill";
          rd.sortWeight = 200;
          rd.configIndex = 0;
        } else if (id == "rop_rgba16f") {
          rd.component = "Rasterization & ROP";
          rd.subcategory = "Pixel Fill Rate";
          rd.benchmarkName = "RGBA16F HDR Fill";
          rd.sortWeight = 210;
          rd.configIndex = 1;
        } else if (id == "rop_blend") {
          rd.component = "Rasterization & ROP";
          rd.subcategory = "Pixel Fill Rate";
          rd.benchmarkName = "Alpha Blending Fill";
          rd.sortWeight = 220;
          rd.configIndex = 2;
        } else if (id == "rt_blas_build_1m") {
          rd.component = "Ray Tracing";
          rd.subcategory = "BLAS Construction";
          rd.benchmarkName = "RayASBuild (BLAS Build (1M Tris))";
          rd.sortWeight = 300;
          rd.configIndex = 0;
        } else if (id == "rt_blas_update_1m") {
          rd.component = "Ray Tracing";
          rd.subcategory = "BLAS Construction";
          rd.benchmarkName = "RayASBuild (BLAS Update (1M Tris))";
          rd.sortWeight = 305;
          rd.configIndex = 1;
        } else if (id == "rt_blas_build_5m") {
          rd.component = "Ray Tracing";
          rd.subcategory = "BLAS Construction";
          rd.benchmarkName = "RayASBuild (BLAS Build (5M Tris))";
          rd.sortWeight = 310;
          rd.configIndex = 2;
        } else if (id == "rt_blas_update_5m") {
          rd.component = "Ray Tracing";
          rd.subcategory = "BLAS Construction";
          rd.benchmarkName = "RayASBuild (BLAS Update (5M Tris))";
          rd.sortWeight = 315;
          rd.configIndex = 3;
        } else if (id == "rt_blas_build_10m") {
          rd.component = "Ray Tracing";
          rd.subcategory = "BLAS Construction";
          rd.benchmarkName = "RayASBuild (BLAS Build (10M Tris))";
          rd.sortWeight = 320;
          rd.configIndex = 4;
        } else if (id == "rt_tlas_indoor") {
          rd.component = "Ray Tracing";
          rd.subcategory = "TLAS Construction";
          rd.benchmarkName = "TLAS: Indoor Corridor (20k Inst)";
          rd.sortWeight = 330;
          rd.configIndex = 0;
        } else if (id == "rt_tlas_jungle") {
          rd.component = "Ray Tracing";
          rd.subcategory = "TLAS Construction";
          rd.benchmarkName = "TLAS: Dense Jungle (50k Inst)";
          rd.sortWeight = 335;
          rd.configIndex = 1;
        } else if (id == "rt_tlas_openworld") {
          rd.component = "Ray Tracing";
          rd.subcategory = "TLAS Construction";
          rd.benchmarkName = "TLAS: Open World (200k Inst)";
          rd.sortWeight = 340;
          rd.configIndex = 2;
        } else if (id == "rt_triangle") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Ray Intersection";
          rd.benchmarkName = "RayIntersect (Ray-Triangle)";
          rd.sortWeight = 350;
          rd.configIndex = 0;
        } else if (id == "rt_anyhit") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Ray Intersection";
          rd.benchmarkName = "RayAnyHit (Alpha-Tested)";
          rd.sortWeight = 355;
          rd.configIndex = 0;
        } else if (id == "rt_procedural") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Procedural Geometry";
          rd.benchmarkName = "RayProcedural (Procedural Geometry)";
          rd.sortWeight = 360;
          rd.configIndex = 0;
        } else if (id == "rt_sched_shadow_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Directional Shadows";
          rd.benchmarkName = "RayScheduling (Shadows - Single Light (Megakernel))";
          rd.sortWeight = 400;
          rd.configIndex = 0;
        } else if (id == "rt_sched_shadow_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Directional Shadows";
          rd.benchmarkName = "RayScheduling (Shadows - Single Light (DGC))";
          rd.sortWeight = 401;
          rd.configIndex = 1;
        } else if (id == "rt_sched_shadow_bin") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Directional Shadows";
          rd.benchmarkName = "RayScheduling (Shadows - Multi-Light (3 Lights, DGC))";
          rd.sortWeight = 402;
          rd.configIndex = 2;
        } else if (id == "rt_sched_multilight_single_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Multi-Light Evaluation";
          rd.benchmarkName = "RayScheduling (Multi-Light - Single Light (Megakernel))";
          rd.sortWeight = 405;
          rd.configIndex = 0;
        } else if (id == "rt_sched_multilight_single_dgc") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Multi-Light Evaluation";
          rd.benchmarkName = "RayScheduling (Multi-Light - Single Light (DGC))";
          rd.sortWeight = 406;
          rd.configIndex = 1;
        } else if (id == "rt_sched_multilight_128_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Multi-Light Evaluation";
          rd.benchmarkName = "RayScheduling (Multi-Light - 128 Lights (Megakernel))";
          rd.sortWeight = 407;
          rd.configIndex = 2;
        } else if (id == "rt_sched_multilight_128_dgc") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Multi-Light Evaluation";
          rd.benchmarkName = "RayScheduling (Multi-Light - 128 Lights (DGC Light Binning))";
          rd.sortWeight = 408;
          rd.configIndex = 3;
        } else if (id == "rt_sched_mat_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Material Shading";
          rd.benchmarkName = "RayScheduling (Material Shading (Megakernel))";
          rd.sortWeight = 410;
          rd.configIndex = 0;
        } else if (id == "rt_sched_mat_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Material Shading";
          rd.benchmarkName = "RayScheduling (Material Shading (DGC))";
          rd.sortWeight = 411;
          rd.configIndex = 1;
        } else if (id == "rt_sched_incoh_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Secondary Rays";
          rd.benchmarkName = "RayScheduling (Incoherent Rays (Megakernel))";
          rd.sortWeight = 420;
          rd.configIndex = 0;
        } else if (id == "rt_sched_incoh_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Secondary Rays";
          rd.benchmarkName = "RayScheduling (Incoherent Rays (DGC))";
          rd.sortWeight = 421;
          rd.configIndex = 1;
        } else if (id == "rt_incoherent") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Ray Traversal";
          rd.benchmarkName = "RayIncoherent (Incoherent Bounces)";
          rd.sortWeight = 425;
          rd.configIndex = 0;
        } else if (id == "rt_divergence") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Ray Traversal";
          rd.benchmarkName = "RayDivergence (Divergence Traversal)";
          rd.sortWeight = 427;
          rd.configIndex = 0;
        } else if (id == "rt_sched_pt_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Multi-Bounce Path Tracing";
          rd.benchmarkName = "RayScheduling (Path Tracing (16 SPP) (Megakernel))";
          rd.sortWeight = 430;
          rd.configIndex = 0;
        } else if (id == "rt_sched_pt_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Multi-Bounce Path Tracing";
          rd.benchmarkName = "RayScheduling (Path Tracing (16 SPP) (DGC))";
          rd.sortWeight = 431;
          rd.configIndex = 1;
        } else if (id == "rt_payload") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Ray Traversal";
          rd.benchmarkName = "RayPayload (Payload Pressure)";
          rd.sortWeight = 435;
          rd.configIndex = 0;
        } else if (id == "rt_sched_ser") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Hardware Scheduling";
          rd.benchmarkName = "RayScheduling (Hardware Reordering (SER))";
          rd.sortWeight = 440;
          rd.configIndex = 0;
        } else if (id == "rt_sched_workgraph") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Hardware Scheduling";
          rd.benchmarkName = "RayScheduling (GPU Work Graphs)";
          rd.sortWeight = 445;
          rd.configIndex = 0;
        } else if (id == "rt_sched_stage_bvh_linear") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Traversal Ordering & Coherence";
          rd.benchmarkName = "BVH Traversal - Linear 1D Scanline (Baseline)";
          rd.sortWeight = 500;
          rd.configIndex = 0;
        } else if (id == "rt_sched_stage_queue_compaction") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Wavefront Stream Compaction";
          rd.benchmarkName = "Queue Compaction - Wave Stream Sort";
          rd.sortWeight = 505;
          rd.configIndex = 1;
        } else if (id == "rt_sched_stage_bvh_tiled") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Traversal Ordering & Coherence";
          rd.benchmarkName = "BVH Traversal - 2D Screen Tiled (8x4)";
          rd.sortWeight = 510;
          rd.configIndex = 2;
        } else if (id == "rt_sched_stage_bvh_morton8x4") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Traversal Ordering & Coherence";
          rd.benchmarkName = "BVH Traversal - 2D Morton Z-Curve (8x4)";
          rd.sortWeight = 515;
          rd.configIndex = 3;
        } else if (id == "rt_sched_stage_bvh_morton4x8") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Traversal Ordering & Coherence";
          rd.benchmarkName = "BVH Traversal - 2D Morton Z-Curve (4x8)";
          rd.sortWeight = 520;
          rd.configIndex = 4;
        } else if (id == "rt_sched_full_showroom_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Showroom)";
          rd.benchmarkName = "RayScheduling (Showroom (Megakernel))";
          rd.sortWeight = 600;
          rd.configIndex = 0;
        } else if (id == "rt_sched_full_showroom_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Showroom)";
          rd.benchmarkName = "RayScheduling (Showroom (DGC))";
          rd.sortWeight = 601;
          rd.configIndex = 1;
        } else if (id == "rt_sched_full_showroom_rtp") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Showroom)";
          rd.benchmarkName = "RayScheduling (Showroom (RTP))";
          rd.sortWeight = 602;
          rd.configIndex = 2;
        } else if (id == "rt_sched_full_showroom_ser") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Showroom)";
          rd.benchmarkName = "RayScheduling (Showroom (RTP + SER))";
          rd.sortWeight = 603;
          rd.configIndex = 3;
        } else if (id == "rt_sched_full_indoor_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Indoor)";
          rd.benchmarkName = "RayScheduling (Indoor (Megakernel))";
          rd.sortWeight = 610;
          rd.configIndex = 0;
        } else if (id == "rt_sched_full_indoor_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Indoor)";
          rd.benchmarkName = "RayScheduling (Indoor (DGC))";
          rd.sortWeight = 611;
          rd.configIndex = 1;
        } else if (id == "rt_sched_full_indoor_rtp") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Indoor)";
          rd.benchmarkName = "RayScheduling (Indoor (RTP))";
          rd.sortWeight = 612;
          rd.configIndex = 2;
        } else if (id == "rt_sched_full_indoor_ser") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Indoor)";
          rd.benchmarkName = "RayScheduling (Indoor (RTP + SER))";
          rd.sortWeight = 613;
          rd.configIndex = 3;
        } else if (id == "rt_sched_full_outdoor_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Outdoor)";
          rd.benchmarkName = "RayScheduling (Outdoor (Megakernel))";
          rd.sortWeight = 620;
          rd.configIndex = 0;
        } else if (id == "rt_sched_full_outdoor_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Outdoor)";
          rd.benchmarkName = "RayScheduling (Outdoor (DGC))";
          rd.sortWeight = 621;
          rd.configIndex = 1;
        } else if (id == "rt_sched_full_outdoor_rtp") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Outdoor)";
          rd.benchmarkName = "RayScheduling (Outdoor (RTP))";
          rd.sortWeight = 622;
          rd.configIndex = 2;
        } else if (id == "rt_sched_full_outdoor_ser") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Outdoor)";
          rd.benchmarkName = "RayScheduling (Outdoor (RTP + SER))";
          rd.sortWeight = 623;
          rd.configIndex = 3;
        } else if (id == "rt_sched_full_forest_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Forest)";
          rd.benchmarkName = "RayScheduling (Forest (Megakernel))";
          rd.sortWeight = 630;
          rd.configIndex = 0;
        } else if (id == "rt_sched_full_forest_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Forest)";
          rd.benchmarkName = "RayScheduling (Forest (DGC))";
          rd.sortWeight = 631;
          rd.configIndex = 1;
        } else if (id == "rt_sched_full_forest_rtp") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Forest)";
          rd.benchmarkName = "RayScheduling (Forest (RTP))";
          rd.sortWeight = 632;
          rd.configIndex = 2;
        } else if (id == "rt_sched_full_forest_ser") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Scene Ray Tracing (Forest)";
          rd.benchmarkName = "RayScheduling (Forest (RTP + SER))";
          rd.sortWeight = 633;
          rd.configIndex = 3;
        } else if (id == "sys_mem_bw_multi") {
          rd.component = "Memory";
          rd.subcategory = "Host System Memory";
          rd.benchmarkName = "System Memory Bandwidth (Multi-Threaded)";
          rd.sortWeight = 700;
          rd.configIndex = 0;
        } else if (id == "sys_mem_bw_single") {
          rd.component = "Memory";
          rd.subcategory = "Host System Memory";
          rd.benchmarkName = "System Memory Bandwidth (Single-Threaded)";
          rd.sortWeight = 701;
          rd.configIndex = 1;
        } else if (id == "sys_mem_lat") {
          rd.component = "Memory";
          rd.subcategory = "Host System Memory";
          rd.benchmarkName = "System Memory Latency";
          rd.sortWeight = 702;
          rd.configIndex = 2;
        } else {
          // Fallback for unmapped or custom IDs
          rd.component = category.empty() ? "Compute" : category;
          rd.subcategory = approach.empty() ? label : approach;
          rd.benchmarkName = label.empty() ? id : label;
          rd.sortWeight = 900;
          rd.configIndex = 0;
        }

        outRun.results.push_back(rd);
      }
    }
  } else {
    // Unified v2.0 / CLI schema: flat results array
    size_t entryIdxIter = 0, entryMaxIter = 0;
    yyjson_val *entryRaw = nullptr;
    yyjson_arr_foreach(resultsJson.v, entryIdxIter, entryMaxIter, entryRaw) {
      const JsonView entry{entryRaw};
      ResultData rd;
      rd.backendName = entry["backend"].as_string("Vulkan");
      rd.deviceName = entry["device"].as_string("Target GPU");
      rd.deviceIndex = entry["device_index"].is_null() ? 0xFFFFFFFF : entry["device_index"].as_uint32(0);
      rd.vendorId = profileVendorId;
      rd.deviceId = profileDeviceId;
      rd.benchmarkName = entry["benchmark"].as_string();
      rd.component = entry["component"].as_string("Compute");
      rd.subcategory = entry["subcategory"].as_string();
      rd.metric = entry["metric"].as_string();
      rd.operations = entry["operations"].as_uint64(0);
      rd.time_ms = entry["time_ms"].as_double(0.0);
      rd.operationalIntensity = entry["operational_intensity"].as_double(0.0);
      rd.isValid = entry["is_valid"].as_bool(true);
      rd.errorString = entry["error"].as_string();
      std::string status = entry["status"].as_string();
      if (status == "INVALID") {
        rd.isValid = false;
      } else if (status == "ABORTED" && rd.time_ms >= 0.0) {
        rd.time_ms = -3.0;
      } else if (status == "FAILED" && rd.time_ms >= 0.0) {
        rd.time_ms = -2.0;
      }
      rd.isEmulated = entry["is_emulated"].as_bool(entry["emulated"].as_bool(false));
      rd.isUnsupported = entry["unsupported"].as_bool(false) || (status == "UNSUPPORTED");
      rd.supportCategory = entry["unsupported_category"].as_string();
      rd.supportNote = entry["unsupported_reason"].as_string(
          entry["support_note"].as_string(
              entry["caveat"].as_string(
                  entry["notes"].as_string())));
      rd.maxWorkGroupSize = entry["max_workgroup_size"].as_uint32(1024);
      bool hasExplicitConfigIndex = entry.has("config_index");
      rd.configIndex = entry["config_index"].as_uint32(0);
      rd.baselineConfigIndex = entry["baseline_config_index"].as_int32(-1);

      uint32_t entryW = resWidth, entryH = resHeight;
      if (entry.has("resolution")) {
        parseResolution(entry["resolution"].as_string(), entryW, entryH);
      }
      rd.width = entry["width"].as_uint32(entryW);
      rd.height = entry["height"].as_uint32(entryH);

      if (entry.has("statistics")) {
        const JsonView st = entry["statistics"];
        rd.sample_count = st["samples"].as_uint32(st["sample_count"].as_uint32(0));
        rd.min_time_ms = st["min_time_ms"].as_double(0.0);
        rd.median_time_ms = st["median_time_ms"].as_double(0.0);
        rd.mean_time_ms = st["mean_time_ms"].as_double(0.0);
        rd.p95_time_ms = st["p95_time_ms"].as_double(0.0);
      }
      if (entry.has("compiler_telemetry")) {
        const JsonView ct = entry["compiler_telemetry"];
        rd.hasRegisterTelemetry = ct["available"].as_bool(true);
        rd.vgprCount = ct["vgpr_count"].as_uint32(0);
        rd.sgprCount = ct["sgpr_count"].as_uint32(0);
        rd.ldsSizeBytes = ct["lds_size_bytes"].as_uint32(0);
        rd.scratchSizeBytes = ct["scratch_spill_bytes"].as_uint32(0);
        rd.codeSizeBytes = ct["code_size_bytes"].as_uint32(0);
        rd.maxWavesPerSimd = ct["max_waves_per_simd"].as_uint32(0);
        rd.compilerTarget = ct["compiler"].as_string();
      }
      if (entry.has("power_telemetry")) {
        const JsonView pt = entry["power_telemetry"];
        rd.hasPowerTelemetry = pt["available"].as_bool(true);
        rd.powerWatts = static_cast<float>(pt["power_watts"].as_double(0.0));
        rd.energyJoules = pt["energy_joules"].as_double(0.0);
        rd.joulesPerUnit = pt["joules_per_unit"].as_double(0.0);
        rd.unitPerWatt = pt["unit_per_watt"].as_double(0.0);
        rd.efficiencyUnit = pt["efficiency_unit"].as_string();
      }
      if (entry.has("performance_query")) {
        const JsonView pq = entry["performance_query"];
        rd.hasPerfQueryTelemetry = pq["available"].as_bool(true);
        rd.perfCounters.available = rd.hasPerfQueryTelemetry;
        rd.perfCounters.gpuActiveCycles = pq["gpu_active_cycles"].as_uint64(0);
        rd.perfCounters.waves = pq["waves"].as_uint64(0);
        rd.perfCounters.valuInstructions = pq["valu_instructions"].as_uint64(0);
        rd.perfCounters.saluInstructions = pq["salu_instructions"].as_uint64(0);
        rd.perfCounters.vmemLoadInstructions = pq["vmem_load_instructions"].as_uint64(0);
        rd.perfCounters.smemLoadInstructions = pq["smem_load_instructions"].as_uint64(0);
        rd.perfCounters.vmemStoreInstructions = pq["vmem_store_instructions"].as_uint64(0);
        rd.perfCounters.ldsInstructions = pq["lds_instructions"].as_uint64(0);
        rd.perfCounters.valuBusyPct = static_cast<float>(pq["valu_busy_pct"].as_double(0.0));
        rd.perfCounters.saluBusyPct = static_cast<float>(pq["salu_busy_pct"].as_double(0.0));
        rd.perfCounters.vramReadBytes = pq["vram_read_bytes"].as_uint64(0);
        rd.perfCounters.l0CacheHitRatio = static_cast<float>(pq["l0_cache_hit_ratio"].as_double(0.0));
        rd.perfCounters.l1CacheHitRatio = static_cast<float>(pq["l1_cache_hit_ratio"].as_double(0.0));
        if (pq.has("raw_counters") && pq["raw_counters"].is_object()) {
          yyjson_val *rawObj = pq["raw_counters"].v;
          size_t rcIdx = 0, rcMax = 0;
          yyjson_val *rcKey = nullptr, *rcVal = nullptr;
          yyjson_obj_foreach(rawObj, rcIdx, rcMax, rcKey, rcVal) {
            const char *kStr = yyjson_get_str(rcKey);
            if (kStr != nullptr) {
              rd.perfCounters.rawCounters[kStr] = JsonView{rcVal}.as_double(0.0);
            }
          }
        }
      }

      // Deduce sort weight and normalize legacy CLI schema while preserving 2026 subcategories
      if (rd.component == "Compute") {
        rd.sortWeight = 10;
        if (rd.subcategory == "Sparse Matrix (2:4)") {
          rd.sortWeight = 75;
        } else if (rd.subcategory == "Cooperative Vector") {
          rd.sortWeight = 78;
        } else if (rd.subcategory == "Dual-Issue" || rd.subcategory == "ILP & Concurrency" ||
                   rd.benchmarkName.rfind("Dual-Issue", 0) == 0 || rd.benchmarkName.rfind("ILP & Dual-Issue", 0) == 0) {
          if (rd.subcategory.empty()) rd.subcategory = "ILP & Concurrency";
          rd.sortWeight = 25;
        } else if (rd.subcategory == "LDS Bank Conflicts" || rd.benchmarkName.find("LDS") != std::string::npos) {
          rd.subcategory = "LDS Bank Conflicts";
          rd.sortWeight = 26;
        } else if (rd.subcategory == "Indirect Command Synthesis" || rd.benchmarkName.find("Indirect") != std::string::npos) {
          rd.subcategory = "Indirect Command Synthesis";
          rd.sortWeight = 28;
        } else if (rd.subcategory == "FP64" || rd.subcategory == "Double Precision" || rd.benchmarkName == "FP64") {
          rd.subcategory = "Double Precision";
          rd.benchmarkName = "FP64";
          rd.sortWeight = 10;
          if (!hasExplicitConfigIndex) rd.configIndex = 0;
        } else if (rd.subcategory == "FP32" || rd.subcategory == "Single Precision" || rd.benchmarkName == "FP32") {
          rd.subcategory = "Single Precision";
          rd.benchmarkName = "FP32";
          rd.sortWeight = 20;
          if (!hasExplicitConfigIndex) rd.configIndex = 0;
        } else if (rd.subcategory == "FP16" || rd.subcategory == "Half Precision (FP16)" || rd.benchmarkName.rfind("FP16", 0) == 0) {
          rd.subcategory = "Half Precision (FP16)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 30;
        } else if (rd.subcategory == "BF16" || rd.subcategory == "Bfloat16 (BF16)" || rd.benchmarkName.rfind("BF16", 0) == 0) {
          rd.subcategory = "Bfloat16 (BF16)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 40;
        } else if (rd.subcategory == "FP8" || rd.subcategory == "Quarter Precision (FP8)" || rd.benchmarkName.rfind("FP8", 0) == 0) {
          rd.subcategory = "Quarter Precision (FP8)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 50;
        } else if (rd.subcategory == "INT8" || rd.subcategory == "8-bit Integer (INT8)" || rd.benchmarkName.rfind("INT8", 0) == 0) {
          rd.subcategory = "8-bit Integer (INT8)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 60;
        } else if (rd.subcategory == "INT4" || rd.subcategory == "4-bit Integer (INT4)" || rd.benchmarkName.rfind("INT4", 0) == 0) {
          rd.subcategory = "4-bit Integer (INT4)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 70;
        } else if (rd.subcategory == "FP6" || rd.benchmarkName == "FP6" || rd.benchmarkName.rfind("FP6", 0) == 0) {
          if (rd.subcategory.empty()) rd.subcategory = "FP6";
          rd.sortWeight = 45;
        } else if (rd.subcategory == "FP4" || rd.subcategory.find("FP4") != std::string::npos || rd.benchmarkName.rfind("FP4", 0) == 0) {
          if (rd.subcategory.empty()) rd.subcategory = "Sub-Byte Precision (FP4)";
          rd.sortWeight = 55;
        }
      } else if (rd.component == "Memory") {
        rd.sortWeight = 100;
        if (rd.subcategory == "Host-Device Interconnect") {
          rd.sortWeight = 160;
        } else if (rd.subcategory == "Host System Memory") {
          rd.sortWeight = 700;
        } else if (rd.subcategory == "Performance" || rd.benchmarkName.find("Performance") != std::string::npos) {
          if (rd.benchmarkName.find("Read 256") != std::string::npos || rd.benchmarkName.find("Read 1024") != std::string::npos) {
            rd.subcategory = "Bandwidth";
            rd.benchmarkName = "GPU VRAM Bandwidth";
            rd.sortWeight = 100;
            rd.configIndex = 0;
          } else {
            rd.subcategory = "Device Memory Bandwidth";
            rd.sortWeight = 105;
            bool isWrite = (rd.benchmarkName.find("Write") != std::string::npos);
            bool isRW = (rd.benchmarkName.find("R/W") != std::string::npos);
            if (rd.benchmarkName.find("128") != std::string::npos) {
              rd.benchmarkName = "VRAM Bandwidth (128 threads/group)";
              if (!hasExplicitConfigIndex) rd.configIndex = isWrite ? 1 : (isRW ? 2 : 0);
            } else if (rd.benchmarkName.find("256") != std::string::npos) {
              rd.benchmarkName = "VRAM Bandwidth (256 threads/group)";
              if (!hasExplicitConfigIndex) rd.configIndex = isWrite ? 4 : (isRW ? 5 : 3);
            } else if (rd.benchmarkName.find("1024") != std::string::npos) {
              rd.benchmarkName = "VRAM Bandwidth (1024 threads/group)";
              if (!hasExplicitConfigIndex) rd.configIndex = isWrite ? 7 : (isRW ? 8 : 6);
            }
          }
        } else if (rd.subcategory == "Cache Latency Curve" || rd.benchmarkName.find("Cache Latency Curve") != std::string::npos) {
          rd.subcategory = "Cache Latency Curve";
          rd.sortWeight = 115;
        } else if (rd.subcategory == "Latency" ||
                   (rd.subcategory.empty() && rd.benchmarkName.find("Latency") != std::string::npos)) {
          rd.subcategory = "Latency";
          rd.sortWeight = 110;
        }
      } else if (rd.component == "Scheduling") {
        rd.sortWeight = (rd.subcategory == "GPU Work Graphs") ? 185 : 180;
      } else if (rd.component == "Rasterization & ROP") {
        rd.sortWeight = 200;
      } else if (rd.component == "Ray Tracing") {
        rd.sortWeight = 500;
        if (rd.subcategory == "Opacity Micromap (OMM)") {
          rd.sortWeight = 525;
        } else if (rd.benchmarkName.find("Ray-Triangle") != std::string::npos) {
          rd.subcategory = "Hardware Traversal & Intersection";
          rd.benchmarkName = "Ray-Triangle Intersection";
          rd.sortWeight = 500;
          if (!hasExplicitConfigIndex) rd.configIndex = 0;
        } else if (rd.benchmarkName.find("Ray-Box") != std::string::npos) {
          rd.subcategory = "Hardware Traversal & Intersection";
          rd.benchmarkName = "Ray-Box Intersection";
          rd.sortWeight = 500;
          if (!hasExplicitConfigIndex) rd.configIndex = 1;
        } else if (rd.benchmarkName.find("RayDivergence") != std::string::npos) {
          rd.subcategory = "Ray Divergence";
          rd.sortWeight = 510;
        } else if (rd.benchmarkName.find("RayAnyHit") != std::string::npos) {
          rd.subcategory = "AnyHit Traversal (Alpha Testing)";
          rd.sortWeight = 520;
        } else if (rd.benchmarkName.find("RayIncoherent") != std::string::npos) {
          rd.subcategory = "Incoherent Traversal";
          rd.sortWeight = 530;
        } else if (rd.benchmarkName.find("RayPayload") != std::string::npos) {
          rd.subcategory = "Payload Overhead";
          rd.sortWeight = 540;
        } else if (rd.benchmarkName.find("BLAS Build") != std::string::npos) {
          rd.subcategory = "BLAS Build & Update";
          rd.sortWeight = 550;
          if (!hasExplicitConfigIndex) rd.configIndex = 0;
        } else if (rd.benchmarkName.find("BLAS Update") != std::string::npos) {
          rd.subcategory = "BLAS Build & Update";
          rd.sortWeight = 550;
          if (!hasExplicitConfigIndex) rd.configIndex = 1;
        } else if (rd.benchmarkName.find("TLAS Build") != std::string::npos) {
          rd.subcategory = "TLAS Build";
          rd.sortWeight = 560;
        } else if (rd.benchmarkName.find("Material Shading") != std::string::npos) {
          rd.subcategory = "Material Shading Wavefront";
          bool isDgc = (rd.benchmarkName.find("DGC") != std::string::npos || rd.benchmarkName.find("Work Lists") != std::string::npos);
          rd.sortWeight = isDgc ? 581 : 580;
          if (!hasExplicitConfigIndex) rd.configIndex = isDgc ? 1 : 0;
        } else if (rd.benchmarkName.find("Path Tracing (1 SPP)") != std::string::npos) {
          rd.subcategory = "Path Tracing (1 SPP)";
          bool isDgc = (rd.benchmarkName.find("DGC") != std::string::npos || rd.benchmarkName.find("Work Lists") != std::string::npos);
          rd.sortWeight = isDgc ? 591 : 590;
          if (!hasExplicitConfigIndex) rd.configIndex = isDgc ? 1 : 0;
        } else if (rd.benchmarkName.find("Path Tracing (16 SPP") != std::string::npos) {
          rd.subcategory = "Path Tracing (16 SPP Stress)";
          bool isDgc = (rd.benchmarkName.find("DGC") != std::string::npos || rd.benchmarkName.find("Work Lists") != std::string::npos);
          rd.sortWeight = isDgc ? 596 : 595;
          if (!hasExplicitConfigIndex) rd.configIndex = isDgc ? 1 : 0;
        }
      } else {
        rd.sortWeight = 300;
      }

      if (entry.has("sort_weight")) {
        int explicitSortWeight = entry["sort_weight"].as_int32(0);
        if (explicitSortWeight > 0) {
          rd.sortWeight = explicitSortWeight;
        }
      }

      if (outRun.backend == "Vulkan" && !root.has("backend") && !rd.backendName.empty() && rd.backendName != "System") {
        outRun.backend = rd.backendName;
      }

      outRun.results.push_back(rd);
    }
  }

  return true;
}

bool ResultImporter::loadFromFiles(const std::vector<std::string> &filepaths,
                                   std::vector<ImportedRun> &outRuns,
                                   std::string &errorMessage) {
  outRuns.clear();
  outRuns.reserve(filepaths.size());
  for (size_t i = 0; i < filepaths.size(); ++i) {
    ImportedRun run;
    if (!loadFromFile(filepaths[i], run, errorMessage)) {
      char letter = static_cast<char>('A' + i);
      errorMessage = "Failed to load Run " + std::string(1, letter) + " ('" + filepaths[i] + "'): " + errorMessage;
      return false;
    }
    outRuns.push_back(std::move(run));
  }
  return true;
}

bool ResultImporter::loadFromFiles(const std::string &fileA, const std::string &fileB,
                                   ImportedRun &runA, ImportedRun &runB,
                                   std::string &errorMessage) {
  std::vector<ImportedRun> runs;
  if (!loadFromFiles({fileA, fileB}, runs, errorMessage)) {
    return false;
  }
  runA = std::move(runs[0]);
  runB = std::move(runs[1]);
  return true;
}
