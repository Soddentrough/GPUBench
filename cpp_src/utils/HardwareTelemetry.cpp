#include "HardwareTelemetry.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>

static std::string readSysfsFile(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) return "";
  std::string line;
  std::getline(f, line);
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
    line.pop_back();
  }
  return line;
}

static std::vector<std::string> getSortedDrmDeviceDirs() {
  std::vector<std::pair<std::string, std::string>> cards; // {canonicalPciPath, deviceDir}
  const std::string drmRoot = "/sys/class/drm";
  if (std::filesystem::exists(drmRoot)) {
    for (const auto &entry : std::filesystem::directory_iterator(drmRoot)) {
      std::string name = entry.path().filename().string();
      // Match only card0, card1, etc. - skip connectors card0-DP-1, etc.
      if (name.rfind("card", 0) == 0 && name.find('-') == std::string::npos) {
        std::string devDir = entry.path().string() + "/device";
        if (std::filesystem::exists(devDir)) {
          std::string pciPath = devDir;
          try {
            pciPath = std::filesystem::canonical(devDir).string();
          } catch (...) {}
          cards.push_back({pciPath, devDir});
        }
      }
    }
  }
  std::sort(cards.begin(), cards.end());
  std::vector<std::string> result;
  for (const auto &c : cards) {
    result.push_back(c.second);
  }
  return result;
}

GpuTelemetryData HardwareTelemetry::queryGpu(uint32_t deviceIndex) {
  GpuTelemetryData data;
  data.deviceIndex = deviceIndex;
  data.isAvailable = false;

  std::vector<std::string> sortedDirs = getSortedDrmDeviceDirs();
  std::string cardPath;
  if (deviceIndex < sortedDirs.size()) {
    cardPath = sortedDirs[deviceIndex] + "/";
  } else {
    cardPath = "/sys/class/drm/card" + std::to_string(deviceIndex) + "/device/";
  }

  if (std::filesystem::exists(cardPath)) {
    data.isAvailable = true;

    // Product name
    std::string prodName = readSysfsFile(cardPath + "product_name");
    if (!prodName.empty()) {
      data.deviceName = prodName;
    }

    // GPU load / activity
    std::string busyStr = readSysfsFile(cardPath + "gpu_busy_percent");
    if (!busyStr.empty()) {
      try { data.gpuActivityPct = std::stoul(busyStr); } catch (...) {}
    }

    // Temperature & Power (hwmon)
    for (int h = 0; h < 8; ++h) {
      std::string hwmonPath = cardPath + "hwmon/hwmon" + std::to_string(h) + "/";
      if (std::filesystem::exists(hwmonPath)) {
        if (data.deviceName.empty()) {
          std::string hwmonName = readSysfsFile(hwmonPath + "name");
          if (!hwmonName.empty()) data.deviceName = hwmonName;
        }
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

    // Unified memory / GTT usage
    std::string gttUsedStr = readSysfsFile(cardPath + "mem_info_gtt_used");
    std::string gttTotalStr = readSysfsFile(cardPath + "mem_info_gtt_total");
    if (!gttUsedStr.empty()) {
      try { data.gttUsedMb = std::stoull(gttUsedStr) / (1024 * 1024); } catch (...) {}
    }
    if (!gttTotalStr.empty()) {
      try { data.gttTotalMb = std::stoull(gttTotalStr) / (1024 * 1024); } catch (...) {}
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
  std::vector<std::string> sortedDirs = getSortedDrmDeviceDirs();
  uint32_t count = sortedDirs.empty() ? 4u : static_cast<uint32_t>(sortedDirs.size());
  for (uint32_t i = 0; i < count; ++i) {
    GpuTelemetryData data = queryGpu(i);
    if (data.isAvailable) {
      result.push_back(data);
    }
  }
  return result;
}

std::string GpuContentionInfo::getFormattedSummary() const {
  std::ostringstream oss;
  for (size_t i = 0; i < reasons.size(); ++i) {
    oss << "  • " << reasons[i];
    if (i + 1 < reasons.size()) oss << "\n";
  }
  return oss.str();
}

GpuContentionInfo HardwareTelemetry::checkContention(uint32_t deviceIndex) {
  GpuContentionInfo info;
  info.cpuCores = std::thread::hardware_concurrency();

  std::string loadStr = readSysfsFile("/proc/loadavg");
  if (!loadStr.empty()) {
    std::istringstream iss(loadStr);
    iss >> info.cpuLoad1Min;
  }

  GpuTelemetryData data = queryGpu(deviceIndex);
  if (data.isAvailable) {
    info.gpuBusyPct = data.gpuActivityPct;
    info.vramUsedMb = data.vramUsedMb;
    info.vramTotalMb = data.vramTotalMb;
    info.gttUsedMb = data.gttUsedMb;
    info.gttTotalMb = data.gttTotalMb;

    // 1. GPU Activity Threshold:
    // If GPU busy is >= 15%, background compute or 3D is active.
    // If GPU busy is >= 25%, mark as critical contention.
    if (data.gpuActivityPct >= 15) {
      info.hasContention = true;
      if (data.gpuActivityPct >= 25) {
        info.isCritical = true;
      }
      info.reasons.push_back("Active GPU load: " + std::to_string(data.gpuActivityPct) +
                             "% busy (background LLM or compute workload active)");
    }

    // 2. High Unified Memory (GTT) Allocation (APU / Strix Halo / Unified LPDDR5X)
    // If >16 GB of GTT is allocated, an LLM (e.g. llama-server) is resident in memory.
    if (data.gttUsedMb >= 16384) {
      info.hasContention = true;
      info.reasons.push_back("High resident memory usage: " +
                             std::to_string(data.gttUsedMb / 1024) + " GB GTT allocated by background processes");
    }

    // 3. High Dedicated VRAM Allocation
    if (data.vramTotalMb > 0) {
      uint32_t vramPct = static_cast<uint32_t>((data.vramUsedMb * 100) / data.vramTotalMb);
      if (vramPct >= 80) {
        info.hasContention = true;
        if (vramPct >= 90) {
          info.isCritical = true;
        }
        info.reasons.push_back("High VRAM allocation: " + std::to_string(data.vramUsedMb) + " MB / " +
                               std::to_string(data.vramTotalMb) + " MB (" + std::to_string(vramPct) + "% in-use)");
      }
    }
  }

  // 4. CPU Saturation Check
  if (info.cpuCores > 0 && info.cpuLoad1Min > static_cast<float>(info.cpuCores) * 1.5f) {
    info.hasContention = true;
    info.reasons.push_back("High CPU load average: " + std::to_string(info.cpuLoad1Min) +
                           " on " + std::to_string(info.cpuCores) + " hardware threads");
  }

  return info;
}
