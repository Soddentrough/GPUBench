#include "HardwareTelemetry.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#else
#include <dlfcn.h>
#endif

namespace {

// ============================================================================
// Dynamic NVML Loader (Windows nvml.dll & Linux libnvidia-ml.so.1)
// ============================================================================
using nvmlReturn_t = int;
using nvmlDevice_t = void *;
static constexpr nvmlReturn_t kNvmlSuccess = 0;
static constexpr int kNvmlTempGpu = 0;
static constexpr int kNvmlClockGraphics = 0;
static constexpr int kNvmlClockMem = 2;

struct NvmlUtilization {
  unsigned int gpu;
  unsigned int memory;
};

struct NvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct NvmlPciInfo {
  char busIdLegacy[16];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  unsigned int pciSubSystemId;
  char busId[32];
};

class DynamicNvml {
public:
  static DynamicNvml &instance() {
    static DynamicNvml inst;
    return inst;
  }

  bool available() const { return m_ready; }

  uint32_t deviceCount() const {
    if (!m_ready || !pfnGetCount) return 0;
    unsigned int cnt = 0;
    if (pfnGetCount(&cnt) == kNvmlSuccess) return cnt;
    return 0;
  }

  bool queryDevice(uint32_t index, GpuTelemetryData &out) const {
    return queryDeviceByPciOrIndex("", index, out);
  }

  bool queryDeviceByPciOrIndex(const std::string &pciBusId,
                               uint32_t fallbackIndex,
                               GpuTelemetryData &out) const {
    if (!m_ready) return false;
    nvmlDevice_t dev = nullptr;
    if (!pciBusId.empty() && pfnGetHandleByPciBusId) {
      if (pfnGetHandleByPciBusId(pciBusId.c_str(), &dev) != kNvmlSuccess) {
        dev = nullptr;
      }
    }
    if (!dev) {
      if (!pfnGetHandle ||
          pfnGetHandle(fallbackIndex, &dev) != kNvmlSuccess || !dev) {
        return false;
      }
    }

    out.isAvailable = true;
    out.vendorId = 0x10DE;

    if (pfnGetName) {
      char nameBuf[96] = {};
      if (pfnGetName(dev, nameBuf, sizeof(nameBuf)) == kNvmlSuccess && nameBuf[0] != '\0') {
        out.deviceName = nameBuf;
      }
    }
    if (pfnGetPciInfo) {
      NvmlPciInfo pci{};
      if (pfnGetPciInfo(dev, &pci) == kNvmlSuccess) {
        out.pciBusId = pci.busId[0] ? pci.busId : pci.busIdLegacy;
        out.deviceId = (pci.pciDeviceId >> 16) & 0xFFFFu;
      }
    }
    if (pfnGetClock) {
      unsigned int clk = 0;
      if (pfnGetClock(dev, kNvmlClockGraphics, &clk) == kNvmlSuccess && clk > 0) {
        out.coreClockMhz = clk;
        out.hasCoreClock = true;
      }
      if (pfnGetClock(dev, kNvmlClockMem, &clk) == kNvmlSuccess && clk > 0) {
        out.memoryClockMhz = clk;
        out.hasMemoryClock = true;
      }
    }
    if (pfnGetTemp) {
      unsigned int temp = 0;
      if (pfnGetTemp(dev, kNvmlTempGpu, &temp) == kNvmlSuccess && temp > 0 && temp < 150) {
        out.temperatureC = static_cast<float>(temp);
        out.hasTempEdge = true;
      }
    }
    if (pfnGetPower) {
      unsigned int mw = 0;
      if (pfnGetPower(dev, &mw) == kNvmlSuccess && mw > 0) {
        out.powerWatts = static_cast<float>(mw) / 1000.0f;
        out.hasPower = true;
      }
    }
    if (pfnGetEnergy) {
      unsigned long long mj = 0;
      if (pfnGetEnergy(dev, &mj) == kNvmlSuccess && mj > 0) {
        out.energyMicroJoules = static_cast<uint64_t>(mj) * 1000ULL;
        out.hasEnergyCounter = true;
      }
    }
    if (pfnGetUtil) {
      NvmlUtilization util{};
      if (pfnGetUtil(dev, &util) == kNvmlSuccess) {
        out.gpuActivityPct = util.gpu;
        out.memActivityPct = util.memory;
        out.hasGpuActivity = true;
        out.hasMemActivity = true;
      }
    }
    if (pfnGetMem) {
      NvmlMemory mem{};
      if (pfnGetMem(dev, &mem) == kNvmlSuccess && mem.total > 0) {
        out.vramUsedBytes = mem.used;
        out.vramTotalBytes = mem.total;
        out.vramUsedMb = mem.used / (1024ULL * 1024ULL);
        out.vramTotalMb = mem.total / (1024ULL * 1024ULL);
        out.hasVram = true;
      }
    }
    if (pfnGetFan) {
      unsigned int fan = 0;
      if (pfnGetFan(dev, &fan) == kNvmlSuccess) {
        out.fanSpeedPct = fan;
        out.hasFanSpeed = true;
      }
    }
    return true;
  }

private:
  DynamicNvml() {
#ifdef _WIN32
    m_lib = LoadLibraryA("nvml.dll");
    if (!m_lib) {
      const char *progW64 = std::getenv("ProgramW6432");
      if (progW64) {
        std::string alt = std::string(progW64) + "\\NVIDIA Corporation\\NVSMI\\nvml.dll";
        m_lib = LoadLibraryA(alt.c_str());
      }
    }
    auto loadSym = [&](const char *sym) -> void * {
      return m_lib ? reinterpret_cast<void *>(GetProcAddress(m_lib, sym)) : nullptr;
    };
#else
    m_lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!m_lib) {
      m_lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
    }
    auto loadSym = [&](const char *sym) -> void * {
      return m_lib ? dlsym(m_lib, sym) : nullptr;
    };
#endif
    if (!m_lib) return;

    pfnInit = reinterpret_cast<nvmlReturn_t (*)()>(loadSym("nvmlInit_v2"));
    if (!pfnInit) pfnInit = reinterpret_cast<nvmlReturn_t (*)()>(loadSym("nvmlInit"));
    pfnShutdown = reinterpret_cast<nvmlReturn_t (*)()>(loadSym("nvmlShutdown"));
    pfnGetCount = reinterpret_cast<nvmlReturn_t (*)(unsigned int *)>(loadSym("nvmlDeviceGetCount_v2"));
    if (!pfnGetCount) pfnGetCount = reinterpret_cast<nvmlReturn_t (*)(unsigned int *)>(loadSym("nvmlDeviceGetCount"));
    pfnGetHandle = reinterpret_cast<nvmlReturn_t (*)(unsigned int, nvmlDevice_t *)>(loadSym("nvmlDeviceGetHandleByIndex_v2"));
    if (!pfnGetHandle) pfnGetHandle = reinterpret_cast<nvmlReturn_t (*)(unsigned int, nvmlDevice_t *)>(loadSym("nvmlDeviceGetHandleByIndex"));
    pfnGetHandleByPciBusId = reinterpret_cast<nvmlReturn_t (*)(const char *, nvmlDevice_t *)>(loadSym("nvmlDeviceGetHandleByPciBusId_v2"));
    if (!pfnGetHandleByPciBusId) pfnGetHandleByPciBusId = reinterpret_cast<nvmlReturn_t (*)(const char *, nvmlDevice_t *)>(loadSym("nvmlDeviceGetHandleByPciBusId"));
    pfnGetName = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int)>(loadSym("nvmlDeviceGetName"));
    pfnGetPciInfo = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, NvmlPciInfo *)>(loadSym("nvmlDeviceGetPciInfo_v3"));
    if (!pfnGetPciInfo) pfnGetPciInfo = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, NvmlPciInfo *)>(loadSym("nvmlDeviceGetPciInfo"));
    pfnGetClock = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, int, unsigned int *)>(loadSym("nvmlDeviceGetClockInfo"));
    pfnGetTemp = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, int, unsigned int *)>(loadSym("nvmlDeviceGetTemperature"));
    pfnGetPower = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, unsigned int *)>(loadSym("nvmlDeviceGetPowerUsage"));
    pfnGetEnergy = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, unsigned long long *)>(loadSym("nvmlDeviceGetTotalEnergyConsumption"));
    pfnGetUtil = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, NvmlUtilization *)>(loadSym("nvmlDeviceGetUtilizationRates"));
    pfnGetMem = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, NvmlMemory *)>(loadSym("nvmlDeviceGetMemoryInfo"));
    pfnGetFan = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, unsigned int *)>(loadSym("nvmlDeviceGetFanSpeed"));

    if (pfnInit && pfnGetCount && pfnGetHandle && pfnInit() == kNvmlSuccess) {
      m_ready = true;
    }
  }

  ~DynamicNvml() {
    if (m_ready && pfnShutdown) {
      pfnShutdown();
    }
#ifdef _WIN32
    if (m_lib) FreeLibrary(m_lib);
#else
    if (m_lib) dlclose(m_lib);
#endif
  }

#ifdef _WIN32
  HMODULE m_lib = nullptr;
#else
  void *m_lib = nullptr;
#endif
  bool m_ready = false;
  nvmlReturn_t (*pfnInit)() = nullptr;
  nvmlReturn_t (*pfnShutdown)() = nullptr;
  nvmlReturn_t (*pfnGetCount)(unsigned int *) = nullptr;
  nvmlReturn_t (*pfnGetHandle)(unsigned int, nvmlDevice_t *) = nullptr;
  nvmlReturn_t (*pfnGetHandleByPciBusId)(const char *, nvmlDevice_t *) = nullptr;
  nvmlReturn_t (*pfnGetName)(nvmlDevice_t, char *, unsigned int) = nullptr;
  nvmlReturn_t (*pfnGetPciInfo)(nvmlDevice_t, NvmlPciInfo *) = nullptr;
  nvmlReturn_t (*pfnGetClock)(nvmlDevice_t, int, unsigned int *) = nullptr;
  nvmlReturn_t (*pfnGetTemp)(nvmlDevice_t, int, unsigned int *) = nullptr;
  nvmlReturn_t (*pfnGetPower)(nvmlDevice_t, unsigned int *) = nullptr;
  nvmlReturn_t (*pfnGetEnergy)(nvmlDevice_t, unsigned long long *) = nullptr;
  nvmlReturn_t (*pfnGetUtil)(nvmlDevice_t, NvmlUtilization *) = nullptr;
  nvmlReturn_t (*pfnGetMem)(nvmlDevice_t, NvmlMemory *) = nullptr;
  nvmlReturn_t (*pfnGetFan)(nvmlDevice_t, unsigned int *) = nullptr;
};

#ifdef _WIN32
// ============================================================================
// Windows AMD Display Library (ADL2 via atiadlxx.dll) + DXGI 1.4 + D3DKMT
// ============================================================================
static constexpr int kAdlOk = 0;
static constexpr int kAdlMaxPath = 256;
static constexpr int kAdlPmlogMaxSensors = 256;

// Verified ADL_PMLOG_SENSORS indices from AMD ADL SDK adl_defines.h
static constexpr int kAdlSensorGfxClk = 1;
static constexpr int kAdlSensorMemClk = 2;
static constexpr int kAdlSensorTempEdge = 8;
static constexpr int kAdlSensorTempMem = 9;
static constexpr int kAdlSensorFanRpm = 14;
static constexpr int kAdlSensorFanPct = 15;
static constexpr int kAdlSensorActivityGfx = 19;
static constexpr int kAdlSensorActivityMem = 20;
static constexpr int kAdlSensorAsicPower = 23;
static constexpr int kAdlSensorTempHotspot = 27;
static constexpr int kAdlSensorGfxPower = 30;
static constexpr int kAdlSensorBoardPower = 73;

struct AdlAdapterInfo {
  int iSize;
  int iAdapterIndex;
  char strUDID[kAdlMaxPath];
  int iBusNumber;
  int iDeviceNumber;
  int iFunctionNumber;
  int iVendorID;
  char strAdapterName[kAdlMaxPath];
  char strDisplayName[kAdlMaxPath];
  int iPresent;
  int iExist;
  char strDriverPath[kAdlMaxPath];
  char strDriverPathExt[kAdlMaxPath];
  char strPNPString[kAdlMaxPath];
  int iOSDisplayIndex;
};

struct AdlSingleSensorData {
  int supported;
  int value;
};

struct AdlPmLogDataOutput {
  int size;
  AdlSingleSensorData sensors[kAdlPmlogMaxSensors];
};

static void *__stdcall adlMallocCallback(int size) {
  return std::malloc(static_cast<size_t>(size));
}

template <typename Fn>
static Fn loadWin32Proc(HMODULE lib, const char *sym) {
  return lib ? reinterpret_cast<Fn>(reinterpret_cast<void *>(GetProcAddress(lib, sym)))
             : nullptr;
}

class WindowsAmdAdl {
public:
  static WindowsAmdAdl &instance() {
    static WindowsAmdAdl inst;
    return inst;
  }

  bool available() const { return m_ready; }

  bool queryByOrdinal(size_t amdOrdinal, GpuTelemetryData &out) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_ready || amdOrdinal >= m_uniqueAdapters.size() || !pfnQueryPmLog) {
      return false;
    }
    const auto &adap = m_uniqueAdapters[amdOrdinal];
    AdlPmLogDataOutput pm{};
    pm.size = sizeof(AdlPmLogDataOutput);
    if (pfnQueryPmLog(m_context, adap.iAdapterIndex, &pm) != kAdlOk) {
      return false;
    }

    char pciBuf[32];
    std::snprintf(pciBuf, sizeof(pciBuf), "0000:%02x:%02x.%x",
                  adap.iBusNumber & 0xFF, adap.iDeviceNumber & 0xFF,
                  adap.iFunctionNumber & 0x7);
    if (out.pciBusId.empty()) out.pciBusId = pciBuf;
    if (out.deviceName.empty() && adap.strAdapterName[0] != '\0') {
      out.deviceName = adap.strAdapterName;
    }

    if (pm.sensors[kAdlSensorGfxClk].supported && pm.sensors[kAdlSensorGfxClk].value > 0) {
      out.coreClockMhz = static_cast<uint32_t>(pm.sensors[kAdlSensorGfxClk].value);
      out.hasCoreClock = true;
    }
    if (pm.sensors[kAdlSensorMemClk].supported && pm.sensors[kAdlSensorMemClk].value > 0) {
      out.memoryClockMhz = static_cast<uint32_t>(pm.sensors[kAdlSensorMemClk].value);
      out.hasMemoryClock = true;
    }
    if (pm.sensors[kAdlSensorTempEdge].supported && pm.sensors[kAdlSensorTempEdge].value > 0 &&
        pm.sensors[kAdlSensorTempEdge].value < 150) {
      out.temperatureC = static_cast<float>(pm.sensors[kAdlSensorTempEdge].value);
      out.hasTempEdge = true;
    }
    if (pm.sensors[kAdlSensorTempHotspot].supported && pm.sensors[kAdlSensorTempHotspot].value > 0 &&
        pm.sensors[kAdlSensorTempHotspot].value < 150) {
      out.tempJunctionC = static_cast<float>(pm.sensors[kAdlSensorTempHotspot].value);
      out.hasTempJct = true;
      if (!out.hasTempEdge) {
        out.temperatureC = out.tempJunctionC;
        out.hasTempEdge = true;
      }
    }
    if (pm.sensors[kAdlSensorTempMem].supported && pm.sensors[kAdlSensorTempMem].value > 0 &&
        pm.sensors[kAdlSensorTempMem].value < 150) {
      out.tempMemoryC = static_cast<float>(pm.sensors[kAdlSensorTempMem].value);
      out.hasTempMem = true;
    }
    if (pm.sensors[kAdlSensorBoardPower].supported && pm.sensors[kAdlSensorBoardPower].value > 0 &&
        pm.sensors[kAdlSensorBoardPower].value < 1500) {
      out.powerWatts = static_cast<float>(pm.sensors[kAdlSensorBoardPower].value);
      out.hasPower = true;
    } else if (pm.sensors[kAdlSensorAsicPower].supported && pm.sensors[kAdlSensorAsicPower].value > 0 &&
               pm.sensors[kAdlSensorAsicPower].value < 1500) {
      out.powerWatts = static_cast<float>(pm.sensors[kAdlSensorAsicPower].value);
      out.hasPower = true;
    } else if (pm.sensors[kAdlSensorGfxPower].supported && pm.sensors[kAdlSensorGfxPower].value > 0 &&
               pm.sensors[kAdlSensorGfxPower].value < 1500) {
      out.powerWatts = static_cast<float>(pm.sensors[kAdlSensorGfxPower].value);
      out.hasPower = true;
    }
    if (pm.sensors[kAdlSensorActivityGfx].supported && pm.sensors[kAdlSensorActivityGfx].value >= 0 &&
        pm.sensors[kAdlSensorActivityGfx].value <= 100) {
      out.gpuActivityPct = static_cast<uint32_t>(pm.sensors[kAdlSensorActivityGfx].value);
      out.hasGpuActivity = true;
    }
    if (pm.sensors[kAdlSensorActivityMem].supported && pm.sensors[kAdlSensorActivityMem].value >= 0 &&
        pm.sensors[kAdlSensorActivityMem].value <= 100) {
      out.memActivityPct = static_cast<uint32_t>(pm.sensors[kAdlSensorActivityMem].value);
      out.hasMemActivity = true;
    }
    if (pm.sensors[kAdlSensorFanRpm].supported && pm.sensors[kAdlSensorFanRpm].value >= 0) {
      out.fanRpm = static_cast<uint32_t>(pm.sensors[kAdlSensorFanRpm].value);
      out.hasFanRpm = true;
    }
    if (pm.sensors[kAdlSensorFanPct].supported && pm.sensors[kAdlSensorFanPct].value >= 0 &&
        pm.sensors[kAdlSensorFanPct].value <= 100) {
      out.fanSpeedPct = static_cast<uint32_t>(pm.sensors[kAdlSensorFanPct].value);
      out.hasFanSpeed = true;
    }
    return true;
  }

private:
  using AdlContext = void *;
  using PFN_ADL2_Main_Control_Create = int (*)(void *(__stdcall *)(int), int, AdlContext *);
  using PFN_ADL2_Main_Control_Destroy = int (*)(AdlContext);
  using PFN_ADL2_Adapter_NumberOfAdapters_Get = int (*)(AdlContext, int *);
  using PFN_ADL2_Adapter_AdapterInfo_Get = int (*)(AdlContext, AdlAdapterInfo *, int);
  using PFN_ADL2_New_QueryPMLogData_Get = int (*)(AdlContext, int, AdlPmLogDataOutput *);

  WindowsAmdAdl() {
    m_lib = LoadLibraryA("atiadlxx.dll");
    if (!m_lib) m_lib = LoadLibraryA("atiadlxy.dll");
    if (!m_lib) return;

    pfnCreate = loadWin32Proc<PFN_ADL2_Main_Control_Create>(m_lib, "ADL2_Main_Control_Create");
    pfnDestroy = loadWin32Proc<PFN_ADL2_Main_Control_Destroy>(m_lib, "ADL2_Main_Control_Destroy");
    pfnGetNum = loadWin32Proc<PFN_ADL2_Adapter_NumberOfAdapters_Get>(m_lib, "ADL2_Adapter_NumberOfAdapters_Get");
    pfnGetInfo = loadWin32Proc<PFN_ADL2_Adapter_AdapterInfo_Get>(m_lib, "ADL2_Adapter_AdapterInfo_Get");
    pfnQueryPmLog = loadWin32Proc<PFN_ADL2_New_QueryPMLogData_Get>(m_lib, "ADL2_New_QueryPMLogData_Get");

    if (!pfnCreate || !pfnDestroy || !pfnGetNum || !pfnGetInfo) return;
    if (pfnCreate(adlMallocCallback, 0, &m_context) != kAdlOk || !m_context) return;

    int numAdapters = 0;
    if (pfnGetNum(m_context, &numAdapters) == kAdlOk && numAdapters > 0) {
      std::vector<AdlAdapterInfo> infos(static_cast<size_t>(numAdapters));
      std::memset(infos.data(), 0, infos.size() * sizeof(AdlAdapterInfo));
      if (pfnGetInfo(m_context, infos.data(),
                     static_cast<int>(infos.size() * sizeof(AdlAdapterInfo))) == kAdlOk) {
        std::map<uint32_t, AdlAdapterInfo> byBdf;
        for (const auto &inf : infos) {
          if (inf.iBusNumber < 0) continue;
          uint32_t bdfKey = (static_cast<uint32_t>(inf.iBusNumber) << 16) |
                            (static_cast<uint32_t>(inf.iDeviceNumber) << 8) |
                            static_cast<uint32_t>(inf.iFunctionNumber);
          if (byBdf.find(bdfKey) == byBdf.end()) {
            // Verify PM log works on this adapter index if possible
            if (pfnQueryPmLog) {
              AdlPmLogDataOutput probe{};
              probe.size = sizeof(AdlPmLogDataOutput);
              if (pfnQueryPmLog(m_context, inf.iAdapterIndex, &probe) == kAdlOk) {
                byBdf[bdfKey] = inf;
              }
            } else {
              byBdf[bdfKey] = inf;
            }
          }
        }
        for (const auto &kv : byBdf) {
          m_uniqueAdapters.push_back(kv.second);
        }
      }
    }
    m_ready = !m_uniqueAdapters.empty();
  }

  ~WindowsAmdAdl() {
    if (m_context && pfnDestroy) {
      pfnDestroy(m_context);
    }
    if (m_lib) {
      FreeLibrary(m_lib);
    }
  }

  HMODULE m_lib = nullptr;
  AdlContext m_context = nullptr;
  bool m_ready = false;
  std::mutex m_mutex;
  std::vector<AdlAdapterInfo> m_uniqueAdapters;
  PFN_ADL2_Main_Control_Create pfnCreate = nullptr;
  PFN_ADL2_Main_Control_Destroy pfnDestroy = nullptr;
  PFN_ADL2_Adapter_NumberOfAdapters_Get pfnGetNum = nullptr;
  PFN_ADL2_Adapter_AdapterInfo_Get pfnGetInfo = nullptr;
  PFN_ADL2_New_QueryPMLogData_Get pfnQueryPmLog = nullptr;
};

static std::string wideToUtf8(const wchar_t *wstr) {
  if (!wstr || wstr[0] == L'\0') return "";
  int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
  if (len <= 1) return "";
  std::string out(static_cast<size_t>(len - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out.data(), len, nullptr, nullptr);
  return out;
}

class WindowsWddmGpuUtilization {
public:
  static WindowsWddmGpuUtilization &instance() {
    static WindowsWddmGpuUtilization inst;
    return inst;
  }

  bool queryAdapterBusyPct(LUID luid, uint32_t &outBusyPct) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_ready || !pfnCollect || !pfnGetArray) return false;
    if (pfnCollect(m_query) != 0) return false;

    DWORD bufSize = 0;
    DWORD itemCount = 0;
    constexpr DWORD kPdhFmtDouble = 0x00000200;
    pfnGetArray(m_counter, kPdhFmtDouble, &bufSize, &itemCount, nullptr);
    if (bufSize == 0 || itemCount == 0) return false;

    m_scratch.resize(bufSize);
    auto *items = reinterpret_cast<PdhFmtCounterValueItemA *>(m_scratch.data());
    if (pfnGetArray(m_counter, kPdhFmtDouble, &bufSize, &itemCount, items) != 0) {
      return false;
    }

    char luidTag[48];
    std::snprintf(luidTag, sizeof(luidTag), "luid_0x%08lx_0x%08lx",
                  static_cast<unsigned long>(luid.HighPart),
                  static_cast<unsigned long>(luid.LowPart));

    std::map<std::string, double> engineSums;
    bool matchedAny = false;
    for (DWORD i = 0; i < itemCount; ++i) {
      if (!items[i].szName || items[i].FmtValue.CStatus != 0) continue;
      const char *name = items[i].szName;
      if (!HardwareTelemetry::containsAsciiCaseInsensitive(name, luidTag)) continue;
      matchedAny = true;
      const char *engType = std::strstr(name, "engtype_");
      std::string key = engType ? std::string(engType) : std::string("total");
      engineSums[key] += items[i].FmtValue.doubleValue;
    }
    if (!matchedAny) return false;

    double maxEng = 0.0;
    for (const auto &kv : engineSums) {
      if (kv.second > maxEng) maxEng = kv.second;
    }
    outBusyPct = static_cast<uint32_t>(std::clamp(std::round(maxEng), 0.0, 100.0));
    return true;
  }

private:
  struct PdhFmtCounterValue {
    DWORD CStatus;
    double doubleValue;
  };
  struct PdhFmtCounterValueItemA {
    char *szName;
    PdhFmtCounterValue FmtValue;
  };
  using PdhQueryHandle = void *;
  using PdhCounterHandle = void *;
  using PFN_PdhOpenQueryA = long(WINAPI *)(const char *, DWORD_PTR, PdhQueryHandle *);
  using PFN_PdhAddEnglishCounterA = long(WINAPI *)(PdhQueryHandle, const char *, DWORD_PTR, PdhCounterHandle *);
  using PFN_PdhCollectQueryData = long(WINAPI *)(PdhQueryHandle);
  using PFN_PdhGetFormattedCounterArrayA = long(WINAPI *)(PdhCounterHandle, DWORD, DWORD *, DWORD *, PdhFmtCounterValueItemA *);
  using PFN_PdhCloseQuery = long(WINAPI *)(PdhQueryHandle);

  WindowsWddmGpuUtilization() {
    m_pdhLib = LoadLibraryA("pdh.dll");
    if (!m_pdhLib) return;
    pfnOpen = loadWin32Proc<PFN_PdhOpenQueryA>(m_pdhLib, "PdhOpenQueryA");
    pfnAdd = loadWin32Proc<PFN_PdhAddEnglishCounterA>(m_pdhLib, "PdhAddEnglishCounterA");
    pfnCollect = loadWin32Proc<PFN_PdhCollectQueryData>(m_pdhLib, "PdhCollectQueryData");
    pfnGetArray = loadWin32Proc<PFN_PdhGetFormattedCounterArrayA>(m_pdhLib, "PdhGetFormattedCounterArrayA");
    pfnClose = loadWin32Proc<PFN_PdhCloseQuery>(m_pdhLib, "PdhCloseQuery");
    if (!pfnOpen || !pfnAdd || !pfnCollect || !pfnGetArray || !pfnClose) return;

    if (pfnOpen(nullptr, 0, &m_query) == 0 && m_query) {
      if (pfnAdd(m_query, "\\GPU Engine(*)\\Utilization Percentage", 0, &m_counter) == 0 && m_counter) {
        pfnCollect(m_query); // Prime baseline sample for rate counters
        m_ready = true;
      }
    }
  }

  ~WindowsWddmGpuUtilization() {
    if (m_query && pfnClose) pfnClose(m_query);
    if (m_pdhLib) FreeLibrary(m_pdhLib);
  }

  HMODULE m_pdhLib = nullptr;
  PdhQueryHandle m_query = nullptr;
  PdhCounterHandle m_counter = nullptr;
  bool m_ready = false;
  std::mutex m_mutex;
  std::vector<uint8_t> m_scratch;
  PFN_PdhOpenQueryA pfnOpen = nullptr;
  PFN_PdhAddEnglishCounterA pfnAdd = nullptr;
  PFN_PdhCollectQueryData pfnCollect = nullptr;
  PFN_PdhGetFormattedCounterArrayA pfnGetArray = nullptr;
  PFN_PdhCloseQuery pfnClose = nullptr;
};

class WindowsDxgiTelemetry {
public:
  struct AdapterRecord {
    IDXGIAdapter3 *adapter3 = nullptr;
    DXGI_ADAPTER_DESC1 desc{};
    std::string nameUtf8;
    bool isDiscrete = false;
  };

  static WindowsDxgiTelemetry &instance() {
    static WindowsDxgiTelemetry inst;
    return inst;
  }

  std::vector<GpuTelemetryData> queryAll() {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<GpuTelemetryData> list;
    if (!m_ready) return list;

    size_t amdOrdinal = 0;
    size_t nvOrdinal = 0;

    for (size_t i = 0; i < m_adapters.size(); ++i) {
      const auto &rec = m_adapters[i];
      GpuTelemetryData d;
      d.deviceIndex = static_cast<uint32_t>(i);
      d.vendorId = rec.desc.VendorId;
      d.deviceId = rec.desc.DeviceId;
      d.deviceName = rec.nameUtf8;
      d.isAvailable = true;

      d.vramTotalBytes = static_cast<uint64_t>(rec.desc.DedicatedVideoMemory);
      d.gttTotalBytes = static_cast<uint64_t>(rec.desc.SharedSystemMemory);

      if (rec.adapter3) {
        DXGI_QUERY_VIDEO_MEMORY_INFO localInfo{};
        if (SUCCEEDED(rec.adapter3->QueryVideoMemoryInfo(
                0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &localInfo))) {
          d.vramUsedBytes = localInfo.CurrentUsage;
          if (d.vramTotalBytes == 0 && localInfo.Budget > 0) {
            d.vramTotalBytes = localInfo.Budget;
          }
          d.hasVram = (d.vramTotalBytes > 0);
        }
        DXGI_QUERY_VIDEO_MEMORY_INFO nonLocalInfo{};
        if (SUCCEEDED(rec.adapter3->QueryVideoMemoryInfo(
                0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonLocalInfo))) {
          d.gttUsedBytes = nonLocalInfo.CurrentUsage;
          if (d.gttTotalBytes == 0 && nonLocalInfo.Budget > 0) {
            d.gttTotalBytes = nonLocalInfo.Budget;
          }
          d.hasGtt = (d.gttTotalBytes > 0);
        }
      }

      d.vramUsedMb = d.vramUsedBytes / (1024ULL * 1024ULL);
      d.vramTotalMb = d.vramTotalBytes / (1024ULL * 1024ULL);
      d.gttUsedMb = d.gttUsedBytes / (1024ULL * 1024ULL);
      d.gttTotalMb = d.gttTotalBytes / (1024ULL * 1024ULL);

      if (d.vendorId == 0x1002) {
        WindowsAmdAdl::instance().queryByOrdinal(amdOrdinal++, d);
      } else if (d.vendorId == 0x10DE) {
        DynamicNvml::instance().queryDeviceByPciOrIndex(
            d.pciBusId, static_cast<uint32_t>(nvOrdinal++), d);
      }

      if (!d.hasGpuActivity) {
        uint32_t wddmBusy = 0;
        if (WindowsWddmGpuUtilization::instance().queryAdapterBusyPct(
                rec.desc.AdapterLuid, wddmBusy)) {
          d.gpuActivityPct = wddmBusy;
          d.hasGpuActivity = true;
        }
      }

      list.push_back(std::move(d));
    }
    return list;
  }

private:
  using PFN_CreateDXGIFactory1 = HRESULT(WINAPI *)(REFIID, void **);

  WindowsDxgiTelemetry() {
    m_dxgiLib = LoadLibraryA("dxgi.dll");
    if (!m_dxgiLib) return;
    auto pfnCreateFactory1 =
        loadWin32Proc<PFN_CreateDXGIFactory1>(m_dxgiLib, "CreateDXGIFactory1");
    if (!pfnCreateFactory1) return;

    // IID_IDXGIFactory1: {770aae78-f26f-4dba-a829-253c83d1b387}
    static const GUID kIidDxgiFactory1 = {
        0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
    // IID_IDXGIAdapter3: {645967A4-1392-4310-A798-8053CE3E93FD}
    static const GUID kIidDxgiAdapter3 = {
        0x645967a4, 0x1392, 0x4310, {0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd}};

    if (FAILED(pfnCreateFactory1(kIidDxgiFactory1, reinterpret_cast<void **>(&m_factory))) ||
        !m_factory) {
      return;
    }

    IDXGIAdapter1 *adap1 = nullptr;
    for (UINT idx = 0; m_factory->EnumAdapters1(idx, &adap1) != DXGI_ERROR_NOT_FOUND; ++idx) {
      DXGI_ADAPTER_DESC1 desc{};
      if (SUCCEEDED(adap1->GetDesc1(&desc))) {
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 && desc.VendorId != 0x1414) {
          AdapterRecord rec;
          rec.desc = desc;
          rec.nameUtf8 = wideToUtf8(desc.Description);
          // Match VulkanContext discrete GPU preference (dedicated VRAM > 2 GB treated as discrete)
          rec.isDiscrete = (desc.DedicatedVideoMemory > (2ULL * 1024ULL * 1024ULL * 1024ULL));
          adap1->QueryInterface(kIidDxgiAdapter3, reinterpret_cast<void **>(&rec.adapter3));
          m_adapters.push_back(rec);
        }
      }
      adap1->Release();
    }

    // Sort discrete GPUs ahead of integrated GPUs to match VulkanContext::initVulkan enumeration order
    std::stable_sort(m_adapters.begin(), m_adapters.end(),
                     [](const AdapterRecord &a, const AdapterRecord &b) {
                       return a.isDiscrete > b.isDiscrete;
                     });
    m_ready = !m_adapters.empty();
  }

  ~WindowsDxgiTelemetry() {
    for (auto &rec : m_adapters) {
      if (rec.adapter3) rec.adapter3->Release();
    }
    m_adapters.clear();
    if (m_factory) m_factory->Release();
    if (m_dxgiLib) FreeLibrary(m_dxgiLib);
  }

  HMODULE m_dxgiLib = nullptr;
  IDXGIFactory1 *m_factory = nullptr;
  bool m_ready = false;
  std::mutex m_mutex;
  std::vector<AdapterRecord> m_adapters;
};
#else // !_WIN32
// ============================================================================
// Linux Sysfs / Hwmon Telemetry (AMDGPU, Intel xe / i915) + Energy Delta Fallback
// ============================================================================
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

struct LinuxCardEntry {
  std::string cardDir;   // e.g. /sys/class/drm/card0
  std::string deviceDir; // e.g. /sys/class/drm/card0/device
  std::string pciBusId;
  bool isDiscrete = false;
};

static std::vector<LinuxCardEntry> getSortedLinuxDrmDevices() {
  std::vector<LinuxCardEntry> cards;
  const std::string drmRoot = "/sys/class/drm";
  std::error_code ec;
  if (!std::filesystem::exists(drmRoot, ec)) return cards;

  for (const auto &entry : std::filesystem::directory_iterator(drmRoot, ec)) {
    std::string name = entry.path().filename().string();
    if (name.rfind("card", 0) == 0 && name.find('-') == std::string::npos) {
      std::string cardDir = entry.path().string();
      std::string devDir = cardDir + "/device";
      if (std::filesystem::exists(devDir, ec)) {
        LinuxCardEntry ce;
        ce.cardDir = cardDir;
        ce.deviceDir = devDir;
        try {
          ce.pciBusId = std::filesystem::canonical(devDir).filename().string();
        } catch (...) {
          ce.pciBusId = name;
        }
        std::string vramTotalStr = readSysfsFile(devDir + "/mem_info_vram_total");
        uint64_t vramBytes = 0;
        if (!vramTotalStr.empty()) {
          try { vramBytes = std::stoull(vramTotalStr); } catch (...) {}
        }
        std::string gttTotalStr = readSysfsFile(devDir + "/mem_info_gtt_total");
        uint64_t gttBytes = 0;
        if (!gttTotalStr.empty()) {
          try { gttBytes = std::stoull(gttTotalStr); } catch (...) {}
        }
        ce.isDiscrete = (vramBytes > (4ULL * 1024ULL * 1024ULL * 1024ULL)) ||
                        (vramBytes > 0 && gttBytes > 0 && vramBytes >= gttBytes / 2);
        cards.push_back(std::move(ce));
      }
    }
  }

  std::sort(cards.begin(), cards.end(), [](const LinuxCardEntry &a, const LinuxCardEntry &b) {
    if (a.isDiscrete != b.isDiscrete) return a.isDiscrete > b.isDiscrete;
    return a.pciBusId < b.pciBusId;
  });
  return cards;
}

struct EnergyRateTracker {
  uint64_t lastEnergyUj = 0;
  std::chrono::steady_clock::time_point lastTime{};
  float lastDerivedWatts = 0.0f;
  bool valid = false;
};

static std::mutex g_energyMutex;
static std::map<uint32_t, EnergyRateTracker> g_energyTrackers;
#endif // _WIN32

} // namespace

GpuTelemetryData HardwareTelemetry::queryGpu(uint32_t deviceIndex) {
#ifdef _WIN32
  auto allWin = WindowsDxgiTelemetry::instance().queryAll();
  if (deviceIndex < allWin.size()) {
    return allWin[deviceIndex];
  }
  if (DynamicNvml::instance().available() && deviceIndex < DynamicNvml::instance().deviceCount()) {
    GpuTelemetryData nvData;
    nvData.deviceIndex = deviceIndex;
    DynamicNvml::instance().queryDevice(deviceIndex, nvData);
    return nvData;
  }
  GpuTelemetryData empty;
  empty.deviceIndex = deviceIndex;
  return empty;
#else
  GpuTelemetryData data;
  data.deviceIndex = deviceIndex;
  data.isAvailable = false;

  std::vector<LinuxCardEntry> sortedCards = getSortedLinuxDrmDevices();
  std::string cardRoot;
  std::string cardPath;
  if (deviceIndex < sortedCards.size()) {
    cardRoot = sortedCards[deviceIndex].cardDir + "/";
    cardPath = sortedCards[deviceIndex].deviceDir + "/";
    data.pciBusId = sortedCards[deviceIndex].pciBusId;
  } else {
    cardRoot = "/sys/class/drm/card" + std::to_string(deviceIndex) + "/";
    cardPath = cardRoot + "device/";
  }

  std::error_code ec;
  if (std::filesystem::exists(cardPath, ec)) {
    data.isAvailable = true;

    std::string vendorHex = readSysfsFile(cardPath + "vendor");
    std::string deviceHex = readSysfsFile(cardPath + "device");
    if (!vendorHex.empty()) {
      try { data.vendorId = static_cast<uint32_t>(std::stoul(vendorHex, nullptr, 0)); } catch (...) {}
    }
    if (!deviceHex.empty()) {
      try { data.deviceId = static_cast<uint32_t>(std::stoul(deviceHex, nullptr, 0)); } catch (...) {}
    }

    // If NVIDIA card on Linux, query NVML for rich sensors (preferring exact PCI Bus ID)
    if (data.vendorId == 0x10DE && DynamicNvml::instance().available()) {
      DynamicNvml::instance().queryDeviceByPciOrIndex(data.pciBusId, deviceIndex, data);
      return data;
    }

    std::string prodName = readSysfsFile(cardPath + "product_name");
    if (!prodName.empty()) {
      data.deviceName = prodName;
    }

    std::string busyStr = readSysfsFile(cardPath + "gpu_busy_percent");
    if (!busyStr.empty()) {
      try {
        data.gpuActivityPct = static_cast<uint32_t>(std::stoul(busyStr));
        data.hasGpuActivity = true;
      } catch (...) {}
    }
    std::string memBusyStr = readSysfsFile(cardPath + "mem_busy_percent");
    if (!memBusyStr.empty()) {
      try {
        data.memActivityPct = static_cast<uint32_t>(std::stoul(memBusyStr));
        data.hasMemActivity = true;
      } catch (...) {}
    }

    // Locate hwmon directory (AMDGPU, Intel xe, Intel i915)
    std::string foundHwmonPath;
    if (std::filesystem::exists(cardPath + "hwmon", ec)) {
      for (const auto &entry : std::filesystem::directory_iterator(cardPath + "hwmon", ec)) {
        if (entry.is_directory(ec)) {
          foundHwmonPath = entry.path().string() + "/";
          break;
        }
      }
    }

    if (!foundHwmonPath.empty()) {
      if (data.deviceName.empty()) {
        std::string hwmonName = readSysfsFile(foundHwmonPath + "name");
        if (!hwmonName.empty()) data.deviceName = hwmonName;
      }
      std::string temp1Str = readSysfsFile(foundHwmonPath + "temp1_input");
      if (!temp1Str.empty()) {
        try {
          data.temperatureC = std::stof(temp1Str) / 1000.0f;
          data.hasTempEdge = (data.temperatureC > 0.0f);
        } catch (...) {}
      }
      std::string temp2Str = readSysfsFile(foundHwmonPath + "temp2_input");
      if (!temp2Str.empty()) {
        try {
          data.tempJunctionC = std::stof(temp2Str) / 1000.0f;
          data.hasTempJct = (data.tempJunctionC > 0.0f);
        } catch (...) {}
      }
      std::string temp3Str = readSysfsFile(foundHwmonPath + "temp3_input");
      if (!temp3Str.empty()) {
        try {
          data.tempMemoryC = std::stof(temp3Str) / 1000.0f;
          data.hasTempMem = (data.tempMemoryC > 0.0f);
        } catch (...) {}
      }

      // Monotonic microjoule energy accumulator (AMDGPU & Intel xe/i915 energy1_input)
      std::string energyStr = readSysfsFile(foundHwmonPath + "energy1_input");
      if (!energyStr.empty()) {
        try {
          data.energyMicroJoules = std::stoull(energyStr);
          data.hasEnergyCounter = (data.energyMicroJoules > 0);
        } catch (...) {}
      }

      std::string powerStr = readSysfsFile(foundHwmonPath + "power1_average");
      if (powerStr.empty()) {
        powerStr = readSysfsFile(foundHwmonPath + "power1_input");
      }
      if (!powerStr.empty()) {
        try {
          data.powerWatts = std::stof(powerStr) / 1000000.0f;
          data.hasPower = (data.powerWatts > 0.0f);
        } catch (...) {}
      } else if (data.hasEnergyCounter) {
        // Derive live power from energy1_input delta (Intel xe/i915 & APUs lacking power1_average)
        std::lock_guard<std::mutex> lock(g_energyMutex);
        auto now = std::chrono::steady_clock::now();
        auto &tr = g_energyTrackers[deviceIndex];
        if (tr.valid && data.energyMicroJoules >= tr.lastEnergyUj) {
          double dt = std::chrono::duration<double>(now - tr.lastTime).count();
          if (dt >= 0.02) {
            double dJ = static_cast<double>(data.energyMicroJoules - tr.lastEnergyUj) * 1e-6;
            float w = static_cast<float>(dJ / dt);
            if (w > 0.0f && w < 1500.0f) {
              tr.lastDerivedWatts = w;
            }
            tr.lastEnergyUj = data.energyMicroJoules;
            tr.lastTime = now;
          }
          if (tr.lastDerivedWatts > 0.0f) {
            data.powerWatts = tr.lastDerivedWatts;
            data.hasPower = true;
          }
        } else {
          tr.lastEnergyUj = data.energyMicroJoules;
          tr.lastTime = now;
          tr.valid = true;
        }
      }

      std::string fanRpmStr = readSysfsFile(foundHwmonPath + "fan1_input");
      if (!fanRpmStr.empty()) {
        try {
          data.fanRpm = static_cast<uint32_t>(std::stoul(fanRpmStr));
          data.hasFanRpm = true;
        } catch (...) {}
      }
      std::string pwmStr = readSysfsFile(foundHwmonPath + "pwm1");
      if (!pwmStr.empty()) {
        try {
          data.fanSpeedPct = static_cast<uint32_t>(std::stoul(pwmStr) * 100 / 255);
          data.hasFanSpeed = true;
        } catch (...) {}
      }
    }

    // VRAM & GTT usage
    std::string vramUsedStr = readSysfsFile(cardPath + "mem_info_vram_used");
    std::string vramTotalStr = readSysfsFile(cardPath + "mem_info_vram_total");
    if (!vramUsedStr.empty() && !vramTotalStr.empty()) {
      try {
        data.vramUsedBytes = std::stoull(vramUsedStr);
        data.vramTotalBytes = std::stoull(vramTotalStr);
        data.vramUsedMb = data.vramUsedBytes / (1024ULL * 1024ULL);
        data.vramTotalMb = data.vramTotalBytes / (1024ULL * 1024ULL);
        data.hasVram = (data.vramTotalBytes > 0);
      } catch (...) {}
    }

    std::string gttUsedStr = readSysfsFile(cardPath + "mem_info_gtt_used");
    std::string gttTotalStr = readSysfsFile(cardPath + "mem_info_gtt_total");
    if (!gttUsedStr.empty() && !gttTotalStr.empty()) {
      try {
        data.gttUsedBytes = std::stoull(gttUsedStr);
        data.gttTotalBytes = std::stoull(gttTotalStr);
        data.gttUsedMb = data.gttUsedBytes / (1024ULL * 1024ULL);
        data.gttTotalMb = data.gttTotalBytes / (1024ULL * 1024ULL);
        data.hasGtt = (data.gttTotalBytes > 0);
      } catch (...) {}
    }

    // Core & Memory clocks (AMDGPU + Intel i915 gt_act_freq_mhz + Intel xe tile0/gt0/freq0/act_freq)
    std::string sclkStr = readSysfsFile(cardPath + "current_gfxclk");
    if (!sclkStr.empty()) {
      try { data.coreClockMhz = static_cast<uint32_t>(std::stoul(sclkStr)); } catch (...) {}
    }
    if (data.coreClockMhz == 0 && !foundHwmonPath.empty()) {
      std::string freq1Str = readSysfsFile(foundHwmonPath + "freq1_input");
      if (!freq1Str.empty()) {
        try { data.coreClockMhz = static_cast<uint32_t>(std::stoull(freq1Str) / 1000000ULL); } catch (...) {}
      }
    }
    if (data.coreClockMhz == 0) {
      std::string i915Clk = readSysfsFile(cardRoot + "gt_act_freq_mhz");
      if (i915Clk.empty()) {
        i915Clk = readSysfsFile(cardPath + "tile0/gt0/freq0/act_freq");
      }
      if (!i915Clk.empty()) {
        try { data.coreClockMhz = static_cast<uint32_t>(std::stoul(i915Clk)); } catch (...) {}
      }
    }
    data.hasCoreClock = (data.coreClockMhz > 0);

    std::string mclkStr = readSysfsFile(cardPath + "current_uclk");
    if (!mclkStr.empty()) {
      try { data.memoryClockMhz = static_cast<uint32_t>(std::stoul(mclkStr)); } catch (...) {}
    } else if (!foundHwmonPath.empty()) {
      std::string freq2Str = readSysfsFile(foundHwmonPath + "freq2_input");
      if (!freq2Str.empty()) {
        try { data.memoryClockMhz = static_cast<uint32_t>(std::stoull(freq2Str) / 1000000ULL); } catch (...) {}
      }
    }
    data.hasMemoryClock = (data.memoryClockMhz > 0);
  } else if (DynamicNvml::instance().available() &&
             deviceIndex < DynamicNvml::instance().deviceCount()) {
    DynamicNvml::instance().queryDevice(deviceIndex, data);
  }

  return data;
#endif
}

std::vector<GpuTelemetryData> HardwareTelemetry::queryAllGpus() {
#ifdef _WIN32
  auto allWin = WindowsDxgiTelemetry::instance().queryAll();
  if (!allWin.empty()) return allWin;
  std::vector<GpuTelemetryData> nvList;
  uint32_t nvCount = DynamicNvml::instance().deviceCount();
  for (uint32_t i = 0; i < nvCount; ++i) {
    GpuTelemetryData d;
    d.deviceIndex = i;
    if (DynamicNvml::instance().queryDevice(i, d)) {
      nvList.push_back(std::move(d));
    }
  }
  return nvList;
#else
  std::vector<GpuTelemetryData> result;
  std::vector<LinuxCardEntry> sortedCards = getSortedLinuxDrmDevices();
  uint32_t count = sortedCards.empty()
                       ? std::max(4u, DynamicNvml::instance().deviceCount())
                       : static_cast<uint32_t>(sortedCards.size());
  for (uint32_t i = 0; i < count; ++i) {
    GpuTelemetryData data = queryGpu(i);
    if (data.isAvailable) {
      result.push_back(std::move(data));
    }
  }
  return result;
#endif
}

float HardwareTelemetry::computeWindowPowerWatts(
    const GpuTelemetryData &startSample, const GpuTelemetryData &endSample,
    double wallElapsedMs) {
  if (startSample.hasEnergyCounter && endSample.hasEnergyCounter &&
      endSample.energyMicroJoules > startSample.energyMicroJoules &&
      wallElapsedMs > 1.0) {
    double deltaJoules =
        static_cast<double>(endSample.energyMicroJoules - startSample.energyMicroJoules) * 1e-6;
    double elapsedSec = wallElapsedMs * 1e-3;
    float counterWatts = static_cast<float>(deltaJoules / elapsedSec);
    if (std::isfinite(counterWatts) && counterWatts > 0.1f && counterWatts < 2000.0f) {
      return counterWatts;
    }
  }
  if (startSample.hasPower && endSample.hasPower &&
      startSample.powerWatts > 0.0f && endSample.powerWatts > 0.0f) {
    return (startSample.powerWatts + endSample.powerWatts) * 0.5f;
  }
  if (endSample.hasPower && endSample.powerWatts > 0.0f) {
    return endSample.powerWatts;
  }
  if (startSample.hasPower && startSample.powerWatts > 0.0f) {
    return startSample.powerWatts;
  }
  return 0.0f;
}

bool HardwareTelemetry::containsAsciiCaseInsensitive(const char *haystack,
                                                     const char *needle) {
  if (!haystack || !needle) return false;
  if (*needle == '\0') return true;
  auto toLowerAscii = [](unsigned char c) -> unsigned char {
    return (c >= 'A' && c <= 'Z') ? static_cast<unsigned char>(c + ('a' - 'A')) : c;
  };
  for (const char *h = haystack; *h != '\0'; ++h) {
    const char *p = h;
    const char *n = needle;
    while (*p != '\0' && *n != '\0' &&
           toLowerAscii(static_cast<unsigned char>(*p)) ==
               toLowerAscii(static_cast<unsigned char>(*n))) {
      ++p;
      ++n;
    }
    if (*n == '\0') return true;
  }
  return false;
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

#ifndef _WIN32
  std::string loadStr = readSysfsFile("/proc/loadavg");
  if (!loadStr.empty()) {
    std::istringstream iss(loadStr);
    iss >> info.cpuLoad1Min;
  }
#endif

  GpuTelemetryData data = queryGpu(deviceIndex);
  if (data.isAvailable) {
    info.gpuBusyPct = data.gpuActivityPct;
    info.vramUsedMb = data.vramUsedMb;
    info.vramTotalMb = data.vramTotalMb;
    info.gttUsedMb = data.gttUsedMb;
    info.gttTotalMb = data.gttTotalMb;

    if (data.hasGpuActivity && data.gpuActivityPct >= 15) {
      info.hasContention = true;
      if (data.gpuActivityPct >= 25) {
        info.isCritical = true;
      }
      info.reasons.push_back("Active GPU load: " + std::to_string(data.gpuActivityPct) +
                             "% busy (background LLM or compute workload active)");
    }

    if (data.hasGtt && data.gttUsedMb >= 16384) {
      info.hasContention = true;
      info.reasons.push_back("High resident memory usage: " +
                             std::to_string(data.gttUsedMb / 1024) +
                             " GB shared/GTT memory allocated by background processes");
    }

    if (data.hasVram && data.vramTotalMb > 0) {
      bool isSmallCarveoutApu =
          (data.vramTotalMb <= 4096 && data.gttTotalMb > data.vramTotalMb * 2);
      uint64_t poolUsedMb =
          isSmallCarveoutApu ? (data.vramUsedMb + data.gttUsedMb) : data.vramUsedMb;
      uint64_t poolTotalMb =
          isSmallCarveoutApu ? (data.vramTotalMb + data.gttTotalMb) : data.vramTotalMb;
      uint32_t vramPct = static_cast<uint32_t>((poolUsedMb * 100) / poolTotalMb);
      if (vramPct >= 80) {
        info.hasContention = true;
        if (vramPct >= 90) {
          info.isCritical = true;
        }
        info.reasons.push_back("High VRAM allocation: " + std::to_string(poolUsedMb) +
                               " MB / " + std::to_string(poolTotalMb) + " MB (" +
                               std::to_string(vramPct) + "% in-use)");
      }
    }
  }

  if (info.cpuCores > 0 && info.cpuLoad1Min > static_cast<float>(info.cpuCores) * 1.5f) {
    info.hasContention = true;
    info.reasons.push_back("High CPU load average: " + std::to_string(info.cpuLoad1Min) +
                           " on " + std::to_string(info.cpuCores) + " hardware threads");
  }

  return info;
}
