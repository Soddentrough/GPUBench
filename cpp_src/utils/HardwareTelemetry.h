#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GpuTelemetryData {
  uint32_t deviceIndex = 0;
  std::string deviceName;
  uint32_t coreClockMhz = 0;
  uint32_t memoryClockMhz = 0;
  float temperatureC = 0.0f;
  float powerWatts = 0.0f;
  uint64_t vramUsedMb = 0;
  uint64_t vramTotalMb = 0;
  uint64_t gttUsedMb = 0;
  uint64_t gttTotalMb = 0;
  uint32_t gpuActivityPct = 0;
  uint32_t fanSpeedPct = 0;
  bool isAvailable = false;
};

struct GpuContentionInfo {
  bool hasContention = false;
  bool isCritical = false;
  uint32_t gpuBusyPct = 0;
  uint64_t vramUsedMb = 0;
  uint64_t vramTotalMb = 0;
  uint64_t gttUsedMb = 0;
  uint64_t gttTotalMb = 0;
  float cpuLoad1Min = 0.0f;
  uint32_t cpuCores = 0;
  std::vector<std::string> reasons;
  std::string getFormattedSummary() const;
};

class HardwareTelemetry {
public:
  static GpuTelemetryData queryGpu(uint32_t deviceIndex);
  static std::vector<GpuTelemetryData> queryAllGpus();
  static GpuContentionInfo checkContention(uint32_t deviceIndex);
};
