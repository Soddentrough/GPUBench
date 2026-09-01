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
  uint32_t gpuActivityPct = 0;
  uint32_t fanSpeedPct = 0;
  bool isAvailable = false;
};

class HardwareTelemetry {
public:
  static GpuTelemetryData queryGpu(uint32_t deviceIndex);
  static std::vector<GpuTelemetryData> queryAllGpus();
};
