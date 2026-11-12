#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GpuTelemetryData {
  uint32_t deviceIndex = 0;
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  std::string deviceName;
  std::string pciBusId;

  uint32_t coreClockMhz = 0;
  uint32_t memoryClockMhz = 0;
  float temperatureC = 0.0f;       // Edge / primary die temperature
  float tempJunctionC = 0.0f;      // Hotspot / junction temperature
  float tempMemoryC = 0.0f;        // VRAM temperature
  float powerWatts = 0.0f;
  uint64_t energyMicroJoules = 0;  // Monotonic hardware energy accumulator (Linux energy1_input / NVML)

  uint64_t vramUsedBytes = 0;
  uint64_t vramTotalBytes = 0;
  uint64_t gttUsedBytes = 0;
  uint64_t gttTotalBytes = 0;
  uint64_t vramUsedMb = 0;
  uint64_t vramTotalMb = 0;
  uint64_t gttUsedMb = 0;
  uint64_t gttTotalMb = 0;

  uint32_t gpuActivityPct = 0;
  uint32_t memActivityPct = 0;
  uint32_t fanSpeedPct = 0;
  uint32_t fanRpm = 0;

  // Explicit per-sensor availability flags (never fake 0.0 W / 0 C when absent)
  bool isAvailable = false;
  bool hasCoreClock = false;
  bool hasMemoryClock = false;
  bool hasTempEdge = false;
  bool hasTempJct = false;
  bool hasTempMem = false;
  bool hasPower = false;
  bool hasEnergyCounter = false;
  bool hasVram = false;
  bool hasGtt = false;
  bool hasGpuActivity = false;
  bool hasMemActivity = false;
  bool hasFanSpeed = false;
  bool hasFanRpm = false;
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

  // Computes average package/board power across a timed window using monotonic
  // hardware microjoule energy accumulators (AMDGPU/Intel energy1_input, NVML)
  // when available, falling back to instantaneous sensor samples otherwise.
  static float computeWindowPowerWatts(const GpuTelemetryData &startSample,
                                       const GpuTelemetryData &endSample,
                                       double wallElapsedMs);
  static bool containsAsciiCaseInsensitive(const char *haystack,
                                           const char *needle);
};
