#include "ResultImporter.h"
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

// Lightweight self-contained JSON parser
enum class JsonType { Null, Bool, Number, String, Array, Object };

struct JsonValue {
  JsonType type = JsonType::Null;
  bool boolVal = false;
  double numVal = 0.0;
  std::string strVal;
  std::vector<JsonValue> arrVal;
  std::map<std::string, JsonValue> objVal;

  bool is_null() const { return type == JsonType::Null; }
  bool is_bool() const { return type == JsonType::Bool; }
  bool is_number() const { return type == JsonType::Number; }
  bool is_string() const { return type == JsonType::String; }
  bool is_array() const { return type == JsonType::Array; }
  bool is_object() const { return type == JsonType::Object; }

  const JsonValue &operator[](const std::string &key) const {
    static const JsonValue nullVal;
    if (type != JsonType::Object) return nullVal;
    auto it = objVal.find(key);
    return (it != objVal.end()) ? it->second : nullVal;
  }

  const JsonValue &operator[](size_t idx) const {
    static const JsonValue nullVal;
    if (type != JsonType::Array || idx >= arrVal.size()) return nullVal;
    return arrVal[idx];
  }

  bool has(const std::string &key) const {
    return type == JsonType::Object && objVal.find(key) != objVal.end();
  }

  std::string as_string(const std::string &fallback = "") const {
    return is_string() ? strVal : fallback;
  }

  double as_double(double fallback = 0.0) const {
    return is_number() ? numVal : fallback;
  }

  uint64_t as_uint64(uint64_t fallback = 0) const {
    return is_number() ? static_cast<uint64_t>(std::max(0.0, numVal)) : fallback;
  }

  uint32_t as_uint32(uint32_t fallback = 0) const {
    return is_number() ? static_cast<uint32_t>(std::max(0.0, numVal)) : fallback;
  }

  bool as_bool(bool fallback = false) const {
    return is_bool() ? boolVal : fallback;
  }
};

class SimpleJsonReader {
public:
  static bool parse(const std::string &content, JsonValue &outVal, std::string &err) {
    size_t pos = 0;
    skipWhitespace(content, pos);
    if (!parseValue(content, pos, outVal, err)) {
      return false;
    }
    return true;
  }

private:
  static void skipWhitespace(const std::string &s, size_t &pos) {
    while (pos < s.size() && (std::isspace(static_cast<unsigned char>(s[pos])) || s[pos] == '\0')) {
      pos++;
    }
  }

  static bool parseValue(const std::string &s, size_t &pos, JsonValue &val, std::string &err) {
    skipWhitespace(s, pos);
    if (pos >= s.size()) {
      err = "Unexpected end of input";
      return false;
    }

    char c = s[pos];
    if (c == '{') return parseObject(s, pos, val, err);
    if (c == '[') return parseArray(s, pos, val, err);
    if (c == '"') return parseString(s, pos, val, err);
    if (c == 't' || c == 'f') return parseBool(s, pos, val, err);
    if (c == 'n') return parseNull(s, pos, val, err);
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parseNumber(s, pos, val, err);

    err = std::string("Unexpected character: '") + c + "'";
    return false;
  }

  static bool parseObject(const std::string &s, size_t &pos, JsonValue &val, std::string &err) {
    pos++; // skip '{'
    val.type = JsonType::Object;
    val.objVal.clear();

    while (pos < s.size()) {
      skipWhitespace(s, pos);
      if (pos >= s.size()) {
        err = "Unterminated object";
        return false;
      }
      if (s[pos] == '}') {
        pos++;
        return true;
      }

      if (s[pos] != '"') {
        err = "Expected string key in object";
        return false;
      }

      JsonValue keyVal;
      if (!parseString(s, pos, keyVal, err)) return false;

      skipWhitespace(s, pos);
      if (pos >= s.size() || s[pos] != ':') {
        err = "Expected ':' after key";
        return false;
      }
      pos++; // skip ':'

      JsonValue memberVal;
      if (!parseValue(s, pos, memberVal, err)) return false;

      val.objVal[keyVal.strVal] = std::move(memberVal);

      skipWhitespace(s, pos);
      if (pos < s.size() && s[pos] == ',') {
        pos++;
      } else if (pos < s.size() && s[pos] == '}') {
        pos++;
        return true;
      } else {
        err = "Expected ',' or '}' in object";
        return false;
      }
    }
    err = "Unterminated object at EOF";
    return false;
  }

  static bool parseArray(const std::string &s, size_t &pos, JsonValue &val, std::string &err) {
    pos++; // skip '['
    val.type = JsonType::Array;
    val.arrVal.clear();

    while (pos < s.size()) {
      skipWhitespace(s, pos);
      if (pos >= s.size()) {
        err = "Unterminated array";
        return false;
      }
      if (s[pos] == ']') {
        pos++;
        return true;
      }

      JsonValue item;
      if (!parseValue(s, pos, item, err)) return false;
      val.arrVal.push_back(std::move(item));

      skipWhitespace(s, pos);
      if (pos < s.size() && s[pos] == ',') {
        pos++;
      } else if (pos < s.size() && s[pos] == ']') {
        pos++;
        return true;
      } else {
        err = "Expected ',' or ']' in array";
        return false;
      }
    }
    err = "Unterminated array at EOF";
    return false;
  }

  static bool parseString(const std::string &s, size_t &pos, JsonValue &val, std::string &err) {
    pos++; // skip '"'
    val.type = JsonType::String;
    val.strVal.clear();

    while (pos < s.size()) {
      char c = s[pos++];
      if (c == '"') {
        return true;
      }
      if (c == '\\') {
        if (pos >= s.size()) {
          err = "Unterminated string escape";
          return false;
        }
        char esc = s[pos++];
        switch (esc) {
        case '"': val.strVal += '"'; break;
        case '\\': val.strVal += '\\'; break;
        case '/': val.strVal += '/'; break;
        case 'b': val.strVal += '\b'; break;
        case 'f': val.strVal += '\f'; break;
        case 'n': val.strVal += '\n'; break;
        case 'r': val.strVal += '\r'; break;
        case 't': val.strVal += '\t'; break;
        case 'u': {
          // Skip 4 hex characters
          if (pos + 4 <= s.size()) {
            pos += 4;
            val.strVal += '?';
          }
          break;
        }
        default: val.strVal += esc; break;
        }
      } else {
        val.strVal += c;
      }
    }
    err = "Unterminated string literal";
    return false;
  }

  static bool parseNumber(const std::string &s, size_t &pos, JsonValue &val, std::string &/*err*/) {
    size_t start = pos;
    if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) pos++;
    while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) ||
                              s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' ||
                              s[pos] == '+' || s[pos] == '-')) {
      pos++;
    }
    std::string numStr = s.substr(start, pos - start);
    val.type = JsonType::Number;
    try {
      val.numVal = std::stod(numStr);
    } catch (...) {
      val.numVal = 0.0;
    }
    return true;
  }

  static bool parseBool(const std::string &s, size_t &pos, JsonValue &val, std::string &err) {
    if (s.compare(pos, 4, "true") == 0) {
      pos += 4;
      val.type = JsonType::Bool;
      val.boolVal = true;
      return true;
    }
    if (s.compare(pos, 5, "false") == 0) {
      pos += 5;
      val.type = JsonType::Bool;
      val.boolVal = false;
      return true;
    }
    err = "Invalid boolean literal";
    return false;
  }

  static bool parseNull(const std::string &s, size_t &pos, JsonValue &val, std::string &err) {
    if (s.compare(pos, 4, "null") == 0) {
      pos += 4;
      val.type = JsonType::Null;
      return true;
    }
    err = "Invalid null literal";
    return false;
  }
};

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

  JsonValue root;
  if (!SimpleJsonReader::parse(content, root, errorMessage)) {
    return false;
  }

  outRun.results.clear();
  outRun.appVersion = root["app_version"].as_string(root["version"].as_string("1.0.0"));
  outRun.timestamp = root["timestamp"].as_uint64(0);
  outRun.backend = root["backend"].as_string("Vulkan");
  outRun.resolution = root["resolution"].as_string("4K (UHD) (3840x2160)");

  uint32_t resWidth = 0, resHeight = 0;
  parseResolution(outRun.resolution, resWidth, resHeight);

  if (root.has("system_info")) {
    const auto &sys = root["system_info"];
    outRun.osName = sys["os_name"].as_string();
    outRun.cpuModel = sys["cpu_model"].as_string();
    outRun.cpuCores = sys["cpu_logical_cores"].as_uint32(0);
    outRun.totalRamGb = sys["total_ram_gb"].as_double(0.0);
  }

  if (root["device_profiles"].is_array() && !root["device_profiles"].arrVal.empty()) {
    const auto &p = root["device_profiles"][0];
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
    outRun.deviceProfile.subgroupSize = p["subgroup_size"].as_uint32(0);
    outRun.deviceProfile.maxWorkgroupSize = p["max_workgroup_size"].as_uint32(1024);
  }

  if (outRun.deviceProfile.deviceName.empty()) {
    if (root["devices"].is_array() && !root["devices"].arrVal.empty()) {
      std::string dn = root["devices"][0].as_string();
      size_t colonPos = dn.find(": ");
      if (colonPos != std::string::npos && colonPos <= 3) dn = dn.substr(colonPos + 2);
      outRun.deviceProfile.deviceName = dn;
    }
  }

  JsonValue resultsJson;
  bool isGuiSchema = false;

  if (root.is_array()) {
    resultsJson = root;
    if (!root.arrVal.empty()) {
      outRun.backend = root[0]["backend"].as_string("Vulkan");
      outRun.deviceProfile.deviceName = root[0]["device"].as_string();
      outRun.deviceProfile.deviceIndex = root[0]["device_index"].as_uint32(0);
      outRun.deviceProfile.backend = outRun.backend;
      outRun.deviceProfile.maxWorkgroupSize = root[0]["max_workgroup_size"].as_uint32(1024);
      if (root[0].has("resolution")) {
        outRun.resolution = root[0]["resolution"].as_string();
        parseResolution(outRun.resolution, resWidth, resHeight);
      }
    }
  } else {
    const auto &rj = root["results"];
    if (!rj.is_array()) {
      errorMessage = "Invalid JSON: missing 'results' array";
      return false;
    }
    resultsJson = rj;
    if (!resultsJson.arrVal.empty() && resultsJson[0].has("benchmarks")) {
      isGuiSchema = true;
    }
  }

  if (isGuiSchema) {
    for (const auto &devEntry : resultsJson.arrVal) {
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

      const auto &benchmarksArr = devEntry["benchmarks"];
      if (!benchmarksArr.is_array()) continue;

      for (const auto &bm : benchmarksArr.arrVal) {
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
          rd.subcategory = "Ray Intersection";
          rd.benchmarkName = "RayProcedural (Procedural Geometry)";
          rd.sortWeight = 360;
          rd.configIndex = 0;
        } else if (id == "rt_sched_shadow_trad") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Directional Shadows";
          rd.benchmarkName = "RayScheduling (Shadows - Traditional Megakernel)";
          rd.sortWeight = 400;
          rd.configIndex = 0;
        } else if (id == "rt_sched_shadow_wl") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Directional Shadows";
          rd.benchmarkName = "RayScheduling (Shadows (DGC))";
          rd.sortWeight = 401;
          rd.configIndex = 1;
        } else if (id == "rt_sched_shadow_bin") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Directional Shadows";
          rd.benchmarkName = "RayScheduling (Shadows - Directional Binning)";
          rd.sortWeight = 402;
          rd.configIndex = 2;
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
          rd.subcategory = "Pipeline Breakdown";
          rd.benchmarkName = "BVH Traversal - Linear 1D Scanline (Baseline)";
          rd.sortWeight = 500;
          rd.configIndex = 0;
        } else if (id == "rt_sched_stage_queue_compaction") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Pipeline Breakdown";
          rd.benchmarkName = "Queue Compaction - Wave Stream Sort";
          rd.sortWeight = 505;
          rd.configIndex = 1;
        } else if (id == "rt_sched_stage_bvh_tiled") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Pipeline Breakdown";
          rd.benchmarkName = "BVH Traversal - 2D Screen Tiled (8x4)";
          rd.sortWeight = 510;
          rd.configIndex = 2;
        } else if (id == "rt_sched_stage_bvh_morton8x4") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Pipeline Breakdown";
          rd.benchmarkName = "BVH Traversal - 2D Morton Z-Curve (8x4)";
          rd.sortWeight = 515;
          rd.configIndex = 3;
        } else if (id == "rt_sched_stage_bvh_morton4x8") {
          rd.component = "Ray Tracing";
          rd.subcategory = "Pipeline Breakdown";
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
          rd.benchmarkName = "RayScheduling (Showroom (Dedicated RTP))";
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
          rd.benchmarkName = "RayScheduling (Indoor (Dedicated RTP))";
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
          rd.benchmarkName = "RayScheduling (Outdoor (Dedicated RTP))";
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
          rd.benchmarkName = "RayScheduling (Forest (Dedicated RTP))";
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
    // CLI schema: flat results array
    for (const auto &entry : resultsJson.arrVal) {
      ResultData rd;
      rd.backendName = entry["backend"].as_string("Vulkan");
      rd.deviceName = entry["device"].as_string("Target GPU");
      rd.deviceIndex = entry["device_index"].is_null() ? 0xFFFFFFFF : entry["device_index"].as_uint32(0);
      rd.benchmarkName = entry["benchmark"].as_string();
      rd.component = entry["component"].as_string("Compute");
      rd.subcategory = entry["subcategory"].as_string();
      rd.metric = entry["metric"].as_string();
      rd.operations = entry["operations"].as_uint64(0);
      rd.time_ms = entry["time_ms"].as_double(0.0);
      rd.isEmulated = entry["is_emulated"].as_bool(false);
      rd.isUnsupported = entry["unsupported"].as_bool(false);
      rd.supportCategory = entry["unsupported_category"].as_string();
      rd.supportNote = entry["unsupported_reason"].as_string(entry["support_note"].as_string());
      rd.maxWorkGroupSize = entry["max_workgroup_size"].as_uint32(1024);
      rd.configIndex = entry["config_index"].as_uint32(0);
      rd.width = resWidth;
      rd.height = resHeight;

      // Deduce sort weight and normalize to standard hierarchical schema
      if (rd.component == "Compute") {
        rd.sortWeight = 10;
        if (rd.subcategory == "FP64" || rd.benchmarkName.find("FP64") != std::string::npos) {
          rd.subcategory = "Double Precision";
          rd.benchmarkName = "FP64";
          rd.sortWeight = 10;
          rd.configIndex = 0;
        } else if (rd.subcategory == "FP32" || rd.benchmarkName.find("FP32") != std::string::npos) {
          rd.subcategory = "Single Precision";
          rd.benchmarkName = "FP32";
          rd.sortWeight = 20;
          rd.configIndex = 0;
        } else if (rd.subcategory == "FP16" || rd.benchmarkName.find("FP16") != std::string::npos) {
          rd.subcategory = "Half Precision (FP16)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 30;
        } else if (rd.subcategory == "BF16" || rd.benchmarkName.find("BF16") != std::string::npos) {
          rd.subcategory = "Bfloat16 (BF16)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 40;
        } else if (rd.subcategory == "FP8" || rd.benchmarkName.find("FP8") != std::string::npos) {
          rd.subcategory = "Quarter Precision (FP8)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 50;
        } else if (rd.subcategory == "INT8" || rd.benchmarkName.find("INT8") != std::string::npos) {
          rd.subcategory = "8-bit Integer (INT8)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 60;
        } else if (rd.subcategory == "INT4" || rd.benchmarkName.find("INT4") != std::string::npos) {
          rd.subcategory = "4-bit Integer (INT4)";
          rd.benchmarkName = (rd.configIndex == 1 || rd.benchmarkName.find("Matrix") != std::string::npos) ? "Matrix" : "Vector";
          rd.configIndex = (rd.benchmarkName == "Matrix") ? 1 : 0;
          rd.sortWeight = 70;
        }
      } else if (rd.component == "Memory") {
        rd.sortWeight = 100;
        if (rd.subcategory == "Performance" || rd.benchmarkName.find("Performance") != std::string::npos) {
          if (rd.benchmarkName.find("Read 256") != std::string::npos || rd.benchmarkName.find("Read 1024") != std::string::npos) {
            rd.subcategory = "Bandwidth";
            rd.benchmarkName = "GPU VRAM Bandwidth";
            rd.sortWeight = 100;
            rd.configIndex = 0;
          } else {
            rd.subcategory = "Device Memory Bandwidth";
            rd.sortWeight = 105;
            if (rd.benchmarkName.find("128") != std::string::npos) {
              rd.benchmarkName = "VRAM Bandwidth (128 threads/group)";
              rd.configIndex = (rd.benchmarkName.find("Write") != std::string::npos) ? 1 : (rd.benchmarkName.find("R/W") != std::string::npos ? 2 : 0);
            } else if (rd.benchmarkName.find("256") != std::string::npos) {
              rd.benchmarkName = "VRAM Bandwidth (256 threads/group)";
              rd.configIndex = (rd.benchmarkName.find("Write") != std::string::npos) ? 4 : (rd.benchmarkName.find("R/W") != std::string::npos ? 5 : 3);
            } else if (rd.benchmarkName.find("1024") != std::string::npos) {
              rd.benchmarkName = "VRAM Bandwidth (1024 threads/group)";
              rd.configIndex = (rd.benchmarkName.find("Write") != std::string::npos) ? 7 : (rd.benchmarkName.find("R/W") != std::string::npos ? 8 : 6);
            }
          }
        } else if (rd.subcategory == "Latency" || rd.benchmarkName.find("Latency") != std::string::npos) {
          rd.subcategory = "Latency";
          rd.sortWeight = 110;
        }
      } else if (rd.component == "Rasterization & ROP") {
        rd.sortWeight = 200;
      } else if (rd.component == "Ray Tracing") {
        rd.sortWeight = 500;
        if (rd.benchmarkName.find("Ray-Triangle") != std::string::npos) {
          rd.subcategory = "Hardware Traversal & Intersection";
          rd.benchmarkName = "Ray-Triangle Intersection";
          rd.sortWeight = 500;
          rd.configIndex = 0;
        } else if (rd.benchmarkName.find("Ray-Box") != std::string::npos) {
          rd.subcategory = "Hardware Traversal & Intersection";
          rd.benchmarkName = "Ray-Box Intersection";
          rd.sortWeight = 500;
          rd.configIndex = 1;
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
          rd.configIndex = 0;
        } else if (rd.benchmarkName.find("BLAS Update") != std::string::npos) {
          rd.subcategory = "BLAS Build & Update";
          rd.sortWeight = 550;
          rd.configIndex = 1;
        } else if (rd.benchmarkName.find("TLAS Build") != std::string::npos) {
          rd.subcategory = "TLAS Build";
          rd.sortWeight = 560;
        } else if (rd.benchmarkName.find("Material Shading") != std::string::npos) {
          rd.subcategory = "Material Shading Wavefront";
          bool isDgc = (rd.benchmarkName.find("DGC") != std::string::npos || rd.benchmarkName.find("Work Lists") != std::string::npos);
          rd.sortWeight = isDgc ? 581 : 580;
          rd.configIndex = isDgc ? 1 : 0;
        } else if (rd.benchmarkName.find("Path Tracing (1 SPP)") != std::string::npos) {
          rd.subcategory = "Path Tracing (1 SPP)";
          bool isDgc = (rd.benchmarkName.find("DGC") != std::string::npos || rd.benchmarkName.find("Work Lists") != std::string::npos);
          rd.sortWeight = isDgc ? 591 : 590;
          rd.configIndex = isDgc ? 1 : 0;
        } else if (rd.benchmarkName.find("Path Tracing (16 SPP") != std::string::npos) {
          rd.subcategory = "Path Tracing (16 SPP Stress)";
          bool isDgc = (rd.benchmarkName.find("DGC") != std::string::npos || rd.benchmarkName.find("Work Lists") != std::string::npos);
          rd.sortWeight = isDgc ? 596 : 595;
          rd.configIndex = isDgc ? 1 : 0;
        }
      } else {
        rd.sortWeight = 300;
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
