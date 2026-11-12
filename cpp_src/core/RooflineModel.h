#pragma once

#include "core/ResultFormatter.h"
#include <string>
#include <vector>

namespace gpubench {

struct RooflineBandwidthCeiling {
  std::string level;       // e.g., "VRAM", "L3", "L2", "L1", "L0", "PCIe"
  std::string sourceLabel; // e.g., "VRAM Bandwidth (1024-bit)"
  double bandwidthGBps = 0.0;
};

struct RooflineComputeCeiling {
  std::string name;        // e.g., "FP32 Vector", "FP16 Vector", "FP16/BF16 Matrix", "FP8/INT8 Matrix"
  std::string metric;      // "TFLOPS" or "TOPS"
  std::string sourceLabel; // benchmarkName that set this peak
  double peakTops = 0.0;   // TFLOPS or TOPS
  double peakGops = 0.0;   // peakTops * 1000.0 (GFLOPS or GOPS for log-log Y axis)
};

struct RooflineRidgePoint {
  std::string memoryLevel;
  std::string computeCeiling;
  double bandwidthGBps = 0.0;
  double peakTops = 0.0;
  double ridgeOpsPerByte = 0.0; // (peakTops * 1000.0) / bandwidthGBps
};

struct RooflineWorkloadPoint {
  std::string label;
  std::string subcategory;
  std::string metric;
  std::string targetCeilingName;
  double operationalIntensity = 0.0; // FLOP/Byte or OP/Byte
  double achievedGops = 0.0;         // GFLOPS or GOPS
  double rooflineBoundGops = 0.0;    // min(ceilingGops, OI * vramGBps)
  double attainmentPct = 0.0;        // achievedGops / rooflineBoundGops * 100
  bool isMemoryBound = false;
  bool isMatrix = false;
};

struct DeviceRoofline {
  std::string deviceName;
  std::string backendName;
  uint32_t deviceIndex = 0;
  std::vector<RooflineBandwidthCeiling> bandwidthCeilings;
  std::vector<RooflineComputeCeiling> computeCeilings;
  std::vector<RooflineRidgePoint> ridgePoints;
  std::vector<RooflineWorkloadPoint> workloadPoints;

  double primaryVramGBps() const;
  double primaryFp32Gops() const;
  double highestComputeGops() const;
};

class RooflineModel {
public:
  static std::vector<DeviceRoofline>
  analyze(const std::vector<ResultData> &results);

  static double inferOperationalIntensity(const ResultData &r);

  static void printReport(const std::vector<DeviceRoofline> &rooflines);
};

} // namespace gpubench
