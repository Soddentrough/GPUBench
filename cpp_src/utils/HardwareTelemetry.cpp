#include "HardwareTelemetry.h"
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

static std::string readSysfsFile(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) return "";
  std::string line;
  std::getline(f, line);
  return line;
}

GpuTelemetryData HardwareTelemetry::queryGpu(uint32_t deviceIndex) {
  GpuTelemetryData data;
  data.deviceIndex = deviceIndex;
  data.isAvailable = false;

  // Check AMD sysfs paths: /sys/class/drm/card0/device/, /sys/class/drm/card1/device/
  std::string cardPath = "/sys/class/drm/card" + std::to_string(deviceIndex) + "/device/";
  if (std::filesystem::exists(cardPath)) {
    data.isAvailable = true;

    // GPU load / activity
    std::string busyStr = readSysfsFile(cardPath + "gpu_busy_percent");
    if (!busyStr.empty()) {
      try { data.gpuActivityPct = std::stoul(busyStr); } catch (...) {}
    }

    // Temperature (hwmon)
    for (int h = 0; h < 5; ++h) {
      std::string hwmonPath = cardPath + "hwmon/hwmon" + std::to_string(h) + "/";
      if (std::filesystem::exists(hwmonPath)) {
        std::string tempStr = readSysfsFile(hwmonPath + "temp1_input");
        if (!tempStr.empty()) {
          try { data.temperatureC = std::stof(tempStr) / 1000.0f; } catch (...) {}
        }
        std::string powerStr = readSysfsFile(hwmonPath + "power1_average");
        if (powerStr.empty()) {
          powerStr = readSysfsFile(hwmonPath + "power1_input");
        }
        if (!powerStr.empty()) {
          try { data.powerWatts = std::stof(powerStr) / 1000000.0f; } catch (...) {}
        }
        std::string fanStr = readSysfsFile(hwmonPath + "pwm1");
        if (!fanStr.empty()) {
          try { data.fanSpeedPct = static_cast<uint32_t>(std::stoul(fanStr) * 100 / 255); } catch (...) {}
        }
        break;
      }
    }

    // VRAM usage
    std::string vramUsedStr = readSysfsFile(cardPath + "mem_info_vram_used");
    std::string vramTotalStr = readSysfsFile(cardPath + "mem_info_vram_total");
    if (!vramUsedStr.empty()) {
      try { data.vramUsedMb = std::stoull(vramUsedStr) / (1024 * 1024); } catch (...) {}
    }
    if (!vramTotalStr.empty()) {
      try { data.vramTotalMb = std::stoull(vramTotalStr) / (1024 * 1024); } catch (...) {}
    }

    // Clocks
    std::string sclkStr = readSysfsFile(cardPath + "current_gfxclk");
    if (!sclkStr.empty()) {
      try { data.coreClockMhz = std::stoul(sclkStr); } catch (...) {}
    }
    std::string mclkStr = readSysfsFile(cardPath + "current_uclk");
    if (!mclkStr.empty()) {
      try { data.memoryClockMhz = std::stoul(mclkStr); } catch (...) {}
    }
  }

  return data;
}

std::vector<GpuTelemetryData> HardwareTelemetry::queryAllGpus() {
  std::vector<GpuTelemetryData> result;
  for (uint32_t i = 0; i < 8; ++i) {
    GpuTelemetryData data = queryGpu(i);
    if (data.isAvailable) {
      result.push_back(data);
    }
  }
  return result;
}
