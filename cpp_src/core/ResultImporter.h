#pragma once

#include "ResultFormatter.h"
#include <string>
#include <vector>

struct ImportedDeviceProfile {
  std::string backend;
  uint32_t deviceIndex = 0;
  std::string deviceName;
  std::string vendorId;
  std::string deviceId;
  std::string driverName;
  std::string driverInfo;
  std::string driverVersion;
  std::string apiVersion;
  uint64_t vramTotalMb = 0;
  uint32_t subgroupSize = 0;
  uint32_t maxWorkgroupSize = 0;
};

struct ImportedRun {
  std::string appVersion;
  uint64_t timestamp = 0;
  std::string backend;
  std::string resolution;
  std::string osName;
  std::string cpuModel;
  uint32_t cpuCores = 0;
  double totalRamGb = 0.0;
  ImportedDeviceProfile deviceProfile;
  std::vector<ResultData> results;
};

class ResultImporter {
public:
  // Load a single run from a JSON file (GUI or CLI schema)
  static bool loadFromFile(const std::string &filepath, ImportedRun &outRun,
                           std::string &errorMessage);

  // Load two runs from JSON files for comparison
  static bool loadFromFiles(const std::string &fileA, const std::string &fileB,
                            ImportedRun &runA, ImportedRun &runB,
                            std::string &errorMessage);

  // Load N runs from JSON files for multi-way comparison
  static bool loadFromFiles(const std::vector<std::string> &filepaths,
                            std::vector<ImportedRun> &outRuns,
                            std::string &errorMessage);
};
