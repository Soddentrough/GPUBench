#pragma once

#include "IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

namespace gpubench {

struct HardwareProfile {
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  std::string marketingName;
  std::string archName;
  std::string archFamily;
  bool isApu = false;
  uint32_t l1CacheBytes = 0;
  uint32_t l2CacheBytes = 0;
  uint32_t l3CacheBytes = 0;
  double theoreticalFp32Tflops = 0.0;
  double theoreticalTriangleGis = 0.0;
  double theoreticalBoxGis = 0.0;
  double theoreticalBandwidthGBps = 0.0;
  std::string memoryType;
};

class DeviceDatabase {
public:
  // Look up a device profile by PCI vendor/device ID, with fallback to name matching
  static const HardwareProfile &lookup(uint32_t vendorId, uint32_t deviceId,
                                       const std::string &name = "");

  // Enrich a DeviceInfo struct with database values for any unset/zero fields
  static void enrichDeviceInfo(DeviceInfo &info);

  // Helper to obtain a clean architecture display string
  static std::string getArchitectureName(uint32_t vendorId, uint32_t deviceId,
                                         const std::string &name = "");

  // Helper to check if a device is an APU / integrated graphics
  static bool isApuDevice(uint32_t vendorId, uint32_t deviceId,
                          const std::string &name = "");
};

} // namespace gpubench

using gpubench::DeviceDatabase;
using gpubench::HardwareProfile;
