#include "DeviceDatabase.h"
#include <algorithm>
#include <cctype>
#include <mutex>
#include <unordered_map>

namespace gpubench {

namespace {

// Case-insensitive string contains helper
bool containsCi(const std::string &haystack, const std::string &needle) {
  if (needle.empty()) return true;
  if (haystack.size() < needle.size()) return false;
  auto it = std::search(
      haystack.begin(), haystack.end(), needle.begin(), needle.end(),
      [](char ch1, char ch2) { return std::tolower(static_cast<unsigned char>(ch1)) ==
                                      std::tolower(static_cast<unsigned char>(ch2)); });
  return (it != haystack.end());
}

static const std::vector<HardwareProfile> s_knownProfiles = {
    // -------------------------------------------------------------------------
    // AMD RDNA 4 (GFX1201 / Navi 48)
    // -------------------------------------------------------------------------
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x7448,
        /* marketingName */ "AMD Radeon AI PRO R9700",
        /* archName */ "gfx1201 (RDNA 4)",
        /* archFamily */ "RDNA 4",
        /* isApu */ false,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 8 * 1024 * 1024,
        /* l3CacheBytes */ 64 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 48.66,
        /* theoreticalTriangleGis */ 300.8,
        /* theoreticalBoxGis */ 1203.2,
        /* theoreticalBandwidthGBps */ 640.0,
        /* memoryType */ "GDDR6",
    },
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x7449,
        /* marketingName */ "AMD Radeon RX 9070 XT",
        /* archName */ "gfx1201 (RDNA 4)",
        /* archFamily */ "RDNA 4",
        /* isApu */ false,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 8 * 1024 * 1024,
        /* l3CacheBytes */ 64 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 48.66,
        /* theoreticalTriangleGis */ 300.8,
        /* theoreticalBoxGis */ 1203.2,
        /* theoreticalBandwidthGBps */ 640.0,
        /* memoryType */ "GDDR6",
    },
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x744A,
        /* marketingName */ "AMD Radeon RX 9070",
        /* archName */ "gfx1201 (RDNA 4)",
        /* archFamily */ "RDNA 4",
        /* isApu */ false,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 8 * 1024 * 1024,
        /* l3CacheBytes */ 64 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 40.55,
        /* theoreticalTriangleGis */ 250.6,
        /* theoreticalBoxGis */ 1002.4,
        /* theoreticalBandwidthGBps */ 640.0,
        /* memoryType */ "GDDR6",
    },

    // -------------------------------------------------------------------------
    // AMD RDNA 3.5 (Strix Halo & Strix Point)
    // -------------------------------------------------------------------------
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x17F0,
        /* marketingName */ "AMD Radeon 8060S Graphics",
        /* archName */ "gfx1151 (RDNA 3.5)",
        /* archFamily */ "RDNA 3.5",
        /* isApu */ true,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 2 * 1024 * 1024,
        /* l3CacheBytes */ 32 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 23.56,
        /* theoreticalTriangleGis */ 145.0,
        /* theoreticalBoxGis */ 580.0,
        /* theoreticalBandwidthGBps */ 273.0,
        /* memoryType */ "Unified LPDDR5X",
    },
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x15BF,
        /* marketingName */ "AMD Radeon 890M",
        /* archName */ "gfx1150 (RDNA 3.5)",
        /* archFamily */ "RDNA 3.5",
        /* isApu */ true,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 2 * 1024 * 1024,
        /* l3CacheBytes */ 0,
        /* theoreticalFp32Tflops */ 11.2,
        /* theoreticalTriangleGis */ 70.0,
        /* theoreticalBoxGis */ 280.0,
        /* theoreticalBandwidthGBps */ 120.0,
        /* memoryType */ "Unified LPDDR5X",
    },
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x15C8,
        /* marketingName */ "AMD Radeon 880M",
        /* archName */ "gfx1150 (RDNA 3.5)",
        /* archFamily */ "RDNA 3.5",
        /* isApu */ true,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 2 * 1024 * 1024,
        /* l3CacheBytes */ 0,
        /* theoreticalFp32Tflops */ 8.9,
        /* theoreticalTriangleGis */ 55.0,
        /* theoreticalBoxGis */ 220.0,
        /* theoreticalBandwidthGBps */ 120.0,
        /* memoryType */ "Unified LPDDR5X",
    },

    // -------------------------------------------------------------------------
    // AMD RDNA 3 (Navi 31, 32, 33)
    // -------------------------------------------------------------------------
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x744C,
        /* marketingName */ "AMD Radeon RX 7900 XTX",
        /* archName */ "gfx1100 (RDNA 3)",
        /* archFamily */ "RDNA 3",
        /* isApu */ false,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 6 * 1024 * 1024,
        /* l3CacheBytes */ 96 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 61.4,
        /* theoreticalTriangleGis */ 240.0,
        /* theoreticalBoxGis */ 960.0,
        /* theoreticalBandwidthGBps */ 960.0,
        /* memoryType */ "GDDR6",
    },
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x7460,
        /* marketingName */ "AMD Radeon RX 7800 XT",
        /* archName */ "gfx1101 (RDNA 3)",
        /* archFamily */ "RDNA 3",
        /* isApu */ false,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 4 * 1024 * 1024,
        /* l3CacheBytes */ 64 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 37.3,
        /* theoreticalTriangleGis */ 160.0,
        /* theoreticalBoxGis */ 640.0,
        /* theoreticalBandwidthGBps */ 624.0,
        /* memoryType */ "GDDR6",
    },
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x7480,
        /* marketingName */ "AMD Radeon RX 7600",
        /* archName */ "gfx1102 (RDNA 3)",
        /* archFamily */ "RDNA 3",
        /* isApu */ false,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 2 * 1024 * 1024,
        /* l3CacheBytes */ 32 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 21.7,
        /* theoreticalTriangleGis */ 105.0,
        /* theoreticalBoxGis */ 420.0,
        /* theoreticalBandwidthGBps */ 288.0,
        /* memoryType */ "GDDR6",
    },

    // -------------------------------------------------------------------------
    // AMD RDNA 2 (Navi 21, 22, 23)
    // -------------------------------------------------------------------------
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x73BF,
        /* marketingName */ "AMD Radeon RX 6900 XT",
        /* archName */ "gfx1030 (RDNA 2)",
        /* archFamily */ "RDNA 2",
        /* isApu */ false,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 4 * 1024 * 1024,
        /* l3CacheBytes */ 128 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 23.0,
        /* theoreticalTriangleGis */ 110.0,
        /* theoreticalBoxGis */ 440.0,
        /* theoreticalBandwidthGBps */ 512.0,
        /* memoryType */ "GDDR6",
    },

    // -------------------------------------------------------------------------
    // AMD CDNA 3 (MI300 Series)
    // -------------------------------------------------------------------------
    {
        /* vendorId */ 0x1002,
        /* deviceId */ 0x740F,
        /* marketingName */ "AMD Instinct MI300X",
        /* archName */ "gfx942 (CDNA 3)",
        /* archFamily */ "CDNA 3",
        /* isApu */ false,
        /* l1CacheBytes */ 32 * 1024,
        /* l2CacheBytes */ 4 * 1024 * 1024,
        /* l3CacheBytes */ 256 * 1024 * 1024,
        /* theoreticalFp32Tflops */ 163.4,
        /* theoreticalTriangleGis */ 0.0,
        /* theoreticalBoxGis */ 0.0,
        /* theoreticalBandwidthGBps */ 5300.0,
        /* memoryType */ "HBM3",
    },

    // -------------------------------------------------------------------------
    // NVIDIA Ada Lovelace
    // -------------------------------------------------------------------------
    {
        /* vendorId */ 0x10DE,
        /* deviceId */ 0x2684,
        /* marketingName */ "NVIDIA GeForce RTX 4090",
        /* archName */ "Ada Lovelace (AD102)",
        /* archFamily */ "Ada Lovelace",
        /* isApu */ false,
        /* l1CacheBytes */ 128 * 1024,
        /* l2CacheBytes */ 72 * 1024 * 1024,
        /* l3CacheBytes */ 0,
        /* theoreticalFp32Tflops */ 82.6,
        /* theoreticalTriangleGis */ 380.0,
        /* theoreticalBoxGis */ 1520.0,
        /* theoreticalBandwidthGBps */ 1008.0,
        /* memoryType */ "GDDR6X",
    },
    {
        /* vendorId */ 0x10DE,
        /* deviceId */ 0x2704,
        /* marketingName */ "NVIDIA GeForce RTX 4080",
        /* archName */ "Ada Lovelace (AD103)",
        /* archFamily */ "Ada Lovelace",
        /* isApu */ false,
        /* l1CacheBytes */ 128 * 1024,
        /* l2CacheBytes */ 64 * 1024 * 1024,
        /* l3CacheBytes */ 0,
        /* theoreticalFp32Tflops */ 48.7,
        /* theoreticalTriangleGis */ 230.0,
        /* theoreticalBoxGis */ 920.0,
        /* theoreticalBandwidthGBps */ 716.8,
        /* memoryType */ "GDDR6X",
    },

    // -------------------------------------------------------------------------
    // NVIDIA Ampere
    // -------------------------------------------------------------------------
    {
        /* vendorId */ 0x10DE,
        /* deviceId */ 0x2204,
        /* marketingName */ "NVIDIA GeForce RTX 3090",
        /* archName */ "Ampere (GA102)",
        /* archFamily */ "Ampere",
        /* isApu */ false,
        /* l1CacheBytes */ 128 * 1024,
        /* l2CacheBytes */ 6 * 1024 * 1024,
        /* l3CacheBytes */ 0,
        /* theoreticalFp32Tflops */ 35.6,
        /* theoreticalTriangleGis */ 140.0,
        /* theoreticalBoxGis */ 560.0,
        /* theoreticalBandwidthGBps */ 936.0,
        /* memoryType */ "GDDR6X",
    },

    // -------------------------------------------------------------------------
    // Intel Arc (Battlemage & Alchemist)
    // -------------------------------------------------------------------------
    {
        /* vendorId */ 0x8086,
        /* deviceId */ 0xE20B,
        /* marketingName */ "Intel Arc B580",
        /* archName */ "Battlemage (Xe2)",
        /* archFamily */ "Battlemage",
        /* isApu */ false,
        /* l1CacheBytes */ 64 * 1024,
        /* l2CacheBytes */ 16 * 1024 * 1024,
        /* l3CacheBytes */ 0,
        /* theoreticalFp32Tflops */ 19.2,
        /* theoreticalTriangleGis */ 85.0,
        /* theoreticalBoxGis */ 340.0,
        /* theoreticalBandwidthGBps */ 456.0,
        /* memoryType */ "GDDR6",
    },
    {
        /* vendorId */ 0x8086,
        /* deviceId */ 0x56A0,
        /* marketingName */ "Intel Arc A770",
        /* archName */ "Alchemist (Xe-HPG)",
        /* archFamily */ "Alchemist",
        /* isApu */ false,
        /* l1CacheBytes */ 64 * 1024,
        /* l2CacheBytes */ 16 * 1024 * 1024,
        /* l3CacheBytes */ 0,
        /* theoreticalFp32Tflops */ 19.7,
        /* theoreticalTriangleGis */ 80.0,
        /* theoreticalBoxGis */ 320.0,
        /* theoreticalBandwidthGBps */ 560.0,
        /* memoryType */ "GDDR6",
    },
};

static HardwareProfile synthesizePatternProfile(uint32_t vendorId, uint32_t deviceId,
                                              const std::string &name) {
  HardwareProfile p;
  p.vendorId = vendorId;
  p.deviceId = deviceId;
  p.marketingName = name;

  if (vendorId == 0x1002 || containsCi(name, "AMD") || containsCi(name, "Radeon")) {
    if (containsCi(name, "gfx12") || containsCi(name, "r9700") || containsCi(name, "rdna 4") ||
        containsCi(name, "rdna4") || containsCi(name, "navi 48") || containsCi(name, "rx 9070")) {
      p.archName = "gfx1201 (RDNA 4)";
      p.archFamily = "RDNA 4";
      p.isApu = false;
      p.l2CacheBytes = 8 * 1024 * 1024;
      p.l3CacheBytes = 64 * 1024 * 1024;
      p.theoreticalTriangleGis = 300.8;
      p.theoreticalBoxGis = 1203.2;
      p.theoreticalFp32Tflops = 48.66;
      p.memoryType = "GDDR6";
      return p;
    }
    if (containsCi(name, "gfx115") || containsCi(name, "strix") || containsCi(name, "8060") ||
        containsCi(name, "8050") || containsCi(name, "890m") || containsCi(name, "880m")) {
      p.archName = "gfx1150 (RDNA 3.5)";
      p.archFamily = "RDNA 3.5";
      p.isApu = true;
      p.l2CacheBytes = 2 * 1024 * 1024;
      p.l3CacheBytes = containsCi(name, "8060") ? (32 * 1024 * 1024) : 0;
      p.memoryType = "Unified LPDDR5X";
      return p;
    }
    if (containsCi(name, "gfx11") || containsCi(name, "rdna 3") || containsCi(name, "rdna3") ||
        containsCi(name, "7900") || containsCi(name, "7800") || containsCi(name, "7700") ||
        containsCi(name, "7600")) {
      p.archName = "gfx1100 (RDNA 3)";
      p.archFamily = "RDNA 3";
      p.isApu = false;
      p.l2CacheBytes = containsCi(name, "7900") ? (6 * 1024 * 1024) : (4 * 1024 * 1024);
      p.l3CacheBytes = containsCi(name, "7900") ? (96 * 1024 * 1024) : (64 * 1024 * 1024);
      p.memoryType = "GDDR6";
      return p;
    }
    if (containsCi(name, "gfx103") || containsCi(name, "rdna 2") || containsCi(name, "rdna2") ||
        containsCi(name, "6900") || containsCi(name, "6800") || containsCi(name, "6700") ||
        containsCi(name, "6600")) {
      p.archName = "gfx1030 (RDNA 2)";
      p.archFamily = "RDNA 2";
      p.isApu = false;
      p.l2CacheBytes = 4 * 1024 * 1024;
      p.l3CacheBytes = 128 * 1024 * 1024;
      p.memoryType = "GDDR6";
      return p;
    }
    if (containsCi(name, "gfx942") || containsCi(name, "mi300")) {
      p.archName = "gfx942 (CDNA 3)";
      p.archFamily = "CDNA 3";
      p.isApu = false;
      p.l2CacheBytes = 4 * 1024 * 1024;
      p.l3CacheBytes = 256 * 1024 * 1024;
      p.memoryType = "HBM3";
      return p;
    }
    p.archName = "AMD Radeon";
    p.archFamily = "AMD GCN/RDNA";
    p.l2CacheBytes = 4 * 1024 * 1024;
    p.l3CacheBytes = 32 * 1024 * 1024;
    return p;
  }

  if (vendorId == 0x10DE || containsCi(name, "NVIDIA") || containsCi(name, "GeForce") ||
      containsCi(name, "RTX")) {
    if (containsCi(name, "5090") || containsCi(name, "5080") || containsCi(name, "blackwell")) {
      p.archName = "Blackwell";
      p.archFamily = "Blackwell";
      p.l2CacheBytes = 96 * 1024 * 1024;
      return p;
    }
    if (containsCi(name, "4090") || containsCi(name, "4080") || containsCi(name, "4070") ||
        containsCi(name, "ada")) {
      p.archName = "Ada Lovelace";
      p.archFamily = "Ada Lovelace";
      p.l2CacheBytes = containsCi(name, "4090") ? (72 * 1024 * 1024) : (64 * 1024 * 1024);
      return p;
    }
    if (containsCi(name, "3090") || containsCi(name, "3080") || containsCi(name, "3070") ||
        containsCi(name, "ampere")) {
      p.archName = "Ampere";
      p.archFamily = "Ampere";
      p.l2CacheBytes = 6 * 1024 * 1024;
      return p;
    }
    p.archName = "NVIDIA";
    p.archFamily = "NVIDIA";
    return p;
  }

  if (vendorId == 0x8086 || containsCi(name, "Intel") || containsCi(name, "Arc")) {
    if (containsCi(name, "b580") || containsCi(name, "battlemage") || containsCi(name, "xe2")) {
      p.archName = "Battlemage (Xe2)";
      p.archFamily = "Xe2";
      p.l2CacheBytes = 16 * 1024 * 1024;
      return p;
    }
    if (containsCi(name, "a770") || containsCi(name, "a750") || containsCi(name, "alchemist")) {
      p.archName = "Alchemist (Xe-HPG)";
      p.archFamily = "Xe-HPG";
      p.l2CacheBytes = 16 * 1024 * 1024;
      return p;
    }
    p.archName = "Intel Xe";
    p.archFamily = "Intel Xe";
    return p;
  }

  p.archName = "Discrete GPU";
  p.archFamily = "Generic";
  return p;
}

} // namespace

const HardwareProfile &DeviceDatabase::lookup(uint32_t vendorId, uint32_t deviceId,
                                            const std::string &name) {
  // First attempt exact vendorId + deviceId match
  if (vendorId != 0 && deviceId != 0) {
    for (const auto &p : s_knownProfiles) {
      if (p.vendorId == vendorId && p.deviceId == deviceId) {
        return p;
      }
    }
  }

  // Fallback: Pattern-based synthesis cached per unique query
  static std::mutex s_mutex;
  static std::unordered_map<std::string, HardwareProfile> s_synthCache;

  std::lock_guard<std::mutex> lock(s_mutex);
  std::string cacheKey = std::to_string(vendorId) + ":" + std::to_string(deviceId) + ":" + name;
  auto it = s_synthCache.find(cacheKey);
  if (it != s_synthCache.end()) {
    return it->second;
  }

  HardwareProfile synth = synthesizePatternProfile(vendorId, deviceId, name);
  auto res = s_synthCache.emplace(cacheKey, std::move(synth));
  return res.first->second;
}

void DeviceDatabase::enrichDeviceInfo(DeviceInfo &info) {
  const HardwareProfile &prof = lookup(info.vendorID, info.deviceID, info.name);

  if (info.archName.empty() || info.archName == "Discrete GPU" ||
      info.archName == "AMD Radeon (Vulkan)") {
    info.archName = prof.archName;
  }

  if (info.l1CacheSize == 0 && prof.l1CacheBytes > 0) {
    info.l1CacheSize = prof.l1CacheBytes;
  }
  if (info.l2CacheSize == 0 && prof.l2CacheBytes > 0) {
    info.l2CacheSize = prof.l2CacheBytes;
  }
  if (info.l3CacheSize == 0 && prof.l3CacheBytes > 0) {
    info.l3CacheSize = prof.l3CacheBytes;
  }

  if (!info.isApu && prof.isApu) {
    info.isApu = true;
  }
}

std::string DeviceDatabase::getArchitectureName(uint32_t vendorId, uint32_t deviceId,
                                                const std::string &name) {
  return lookup(vendorId, deviceId, name).archName;
}

bool DeviceDatabase::isApuDevice(uint32_t vendorId, uint32_t deviceId,
                                 const std::string &name) {
  return lookup(vendorId, deviceId, name).isApu;
}

} // namespace gpubench
