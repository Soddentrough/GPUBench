#include "ResultFormatter.h"
#include "DeviceDatabase.h"
#include "ResultImporter.h"
#include "RooflineModel.h"
#include "RunnerAPI.h"
#include "third_party/yyjson.h"
#include "utils/HardwareTelemetry.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <sys/ioctl.h>
#endif

static bool shouldUseColor() {
  if (std::getenv("NO_COLOR") != nullptr) {
    return false;
  }
#if defined(_WIN32)
  return true;
#else
  return isatty(fileno(stdout));
#endif
}

static size_t getTerminalWidth() {
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
    return std::max<size_t>(80, csbi.srWindow.Right - csbi.srWindow.Left + 1);
  }
  return 128;
#else
  struct winsize w;
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
    return std::max<size_t>(80, static_cast<size_t>(w.ws_col));
  }
  return 128;
#endif
}

std::string ResultFormatter::formatDouble(double value, int precision) {
  if (!std::isfinite(value)) {
    return "0.0";
  }
  std::stringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::fixed << std::setprecision(precision) << value;
  std::string str = stream.str();
  size_t dotPos = str.find('.');
  if (dotPos == std::string::npos) {
    return str;
  }
  std::string integerPart = str.substr(0, dotPos);
  std::string fractionalPart = str.substr(dotPos);
  int insertPosition = static_cast<int>(integerPart.length()) - 3;
  while (insertPosition > 0) {
    integerPart.insert(insertPosition, ",");
    insertPosition -= 3;
  }
  return integerPart + fractionalPart;
}

std::string ResultFormatter::formatNumber(uint64_t value) {
  std::string numWithCommas = std::to_string(value);
  int insertPosition = static_cast<int>(numWithCommas.length()) - 3;
  while (insertPosition > 0) {
    numWithCommas.insert(insertPosition, ",");
    insertPosition -= 3;
  }
  return numWithCommas;
}

ResultFormatter::ResultFormatter() {}
ResultFormatter::~ResultFormatter() {}

void ResultFormatter::addResult(const ResultData &result) {
  results.push_back(result);
}

static std::string repeatUtf8(const std::string &ch, size_t count) {
  std::string s;
  s.reserve(ch.size() * count);
  for (size_t i = 0; i < count; ++i) {
    s += ch;
  }
  return s;
}

static size_t visualLength(const std::string &s) {
  size_t len = 0;
  bool inEscape = false;
  for (size_t i = 0; i < s.length(); ++i) {
    if (s[i] == '\033') {
      inEscape = true;
    } else if (inEscape) {
      if (s[i] == 'm') inEscape = false;
    } else {
      if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
        len++;
      }
    }
  }
  return len;
}

static std::string truncateUtf8(const std::string &s, size_t maxVisualCols,
                                const std::string &suffix = "..") {
  if (visualLength(s) <= maxVisualCols) {
    return s;
  }
  size_t suffixLen = visualLength(suffix);
  if (maxVisualCols <= suffixLen) {
    return suffix.substr(0, maxVisualCols);
  }
  size_t targetCols = maxVisualCols - suffixLen;
  size_t cols = 0;
  bool inEscape = false;
  size_t cutByte = 0;
  for (size_t i = 0; i < s.length();) {
    if (s[i] == '\033') {
      inEscape = true;
      i++;
    } else if (inEscape) {
      if (s[i] == 'm') inEscape = false;
      i++;
    } else {
      if (cols >= targetCols) {
        break;
      }
      unsigned char c = static_cast<unsigned char>(s[i]);
      size_t charBytes = 1;
      if ((c & 0x80) == 0x00)
        charBytes = 1;
      else if ((c & 0xE0) == 0xC0)
        charBytes = 2;
      else if ((c & 0xF0) == 0xE0)
        charBytes = 3;
      else if ((c & 0xF8) == 0xF0)
        charBytes = 4;
      if (i + charBytes > s.length()) {
        break;
      }
      i += charBytes;
      cols++;
      cutByte = i;
    }
  }
  return s.substr(0, cutByte) + suffix;
}

static std::string cleanSupportNote(const std::string &rawNote) {
  std::string note = rawNote;
  if (note.empty()) return note;

  // Handle specific long reason strings
  if (note.find("requires Ray Tracing Pipeline") != std::string::npos) {
    return "Requires RT Pipeline (SER)";
  }

  // Strip leading "extension "
  if (note.rfind("extension ", 0) == 0) {
    note = note.substr(10);
  }

  // Strip trailing " missing"
  if (note.length() >= 8 && note.substr(note.length() - 8) == " missing") {
    note = note.substr(0, note.length() - 8);
  }

  return note;
}

std::string extractSceneName(const std::string &rawName) {
  if (rawName.rfind("RayScheduling (", 0) == 0) {
    size_t firstParen = 15; // length of "RayScheduling ("
    size_t closeParen = rawName.find(')', firstParen);
    if (closeParen != std::string::npos) {
      return rawName.substr(firstParen, closeParen - firstParen);
    }
  }
  return "";
}

static int getSceneOrder(const std::string &name) {
  if (name.find("[Showroom Studio]") != std::string::npos || name.find("Showroom") != std::string::npos) return 1;
  if (name.find("[Indoor Atrium]") != std::string::npos || name.find("Indoor") != std::string::npos || name.find("Atrium") != std::string::npos) return 2;
  if (name.find("[Outdoor Landscape]") != std::string::npos || name.find("Outdoor") != std::string::npos || name.find("Landscape") != std::string::npos) return 3;
  if (name.find("[Open-World Forest]") != std::string::npos || name.find("Forest") != std::string::npos) return 4;
  return 99;
}

std::string cleanWorkloadName(const std::string &rawName, const std::string &subcat) {
  std::string name = rawName;

  // Handle "RayScheduling (Indoor Atrium) (Config Name)"
  if (name.rfind("RayScheduling (", 0) == 0) {
    size_t secondOpen = name.find(") (");
    if (secondOpen != std::string::npos && name.back() == ')') {
      name = name.substr(secondOpen + 3, name.length() - (secondOpen + 4));
    }
  } else if (subcat == "Dual-Issue" || subcat == "ILP & Concurrency" ||
             name.rfind("Dual-Issue", 0) == 0 || name.rfind("ILP & Dual-Issue", 0) == 0) {
    // Preserve full precision qualification for Dual-Issue and ILP workloads
    const std::string prefix1 = "ILP & Dual-Issue (";
    const std::string prefix2 = "Dual-Issue (";
    if (name.rfind(prefix1, 0) == 0 && name.back() == ')') {
      name = name.substr(prefix1.length(), name.length() - prefix1.length() - 1);
    } else if (name.rfind(prefix2, 0) == 0 && name.back() == ')') {
      name = name.substr(prefix2.length(), name.length() - prefix2.length() - 1);
    } else if (name.rfind("Dual-Issue ", 0) == 0) {
      name = name.substr(11);
    }
  } else {
    static const std::vector<std::string> kExactPrefixes = {
        "PCIe & Host-Device Interconnect (",
        "Async Compute & Copy Overlap (",
        "2:4 Structured Sparsity (",
        "Cooperative Vector (Neural Shading) (",
        "GPU Work Graphs (Producer-Consumer) (",
    };
    bool strippedExact = false;
    for (const auto &pfx : kExactPrefixes) {
      if (name.rfind(pfx, 0) == 0 && name.back() == ')') {
        name = name.substr(pfx.length(), name.length() - pfx.length() - 1);
        strippedExact = true;
        break;
      }
    }
    if (!strippedExact &&
        (name.rfind("Ray", 0) == 0 ||
         name.rfind("LDS Bank Conflicts (", 0) == 0 ||
         name.rfind("In-Shader Indirect Synthesis (", 0) == 0 ||
         name.rfind("Cache Latency Curve (", 0) == 0 ||
         name.rfind("Device Memory Bandwidth (", 0) == 0 ||
         name.rfind("Pixel Fill Rate (", 0) == 0 ||
         name.rfind("System Memory Bandwidth (", 0) == 0)) {
      // Strip redundant outer benchmark prefix when the subcategory header already conveys it
      size_t firstOpen = name.find(" (");
      if (firstOpen != std::string::npos && name.back() == ')') {
        name = name.substr(firstOpen + 2, name.length() - (firstOpen + 3));
      }
    }
  }

  if (name == "Raw BVH - Coherent Triangle Traversal (Max Occupancy)" ||
      name == "Raw BVH Traversal - Coherent Triangle Traversal (Max Occupancy)") {
    name = "Raw BVH - Coherent Triangle Traversal";
  } else if (name == "Raw BVH Traversal - Deep BVH Multi-Layer Stress") {
    name = "Raw BVH - Deep Multi-Layer Box Stress";
  }

  // Strip subcategory prefix if redundantly present
  if (!subcat.empty()) {
    if (name.rfind(subcat + " - ", 0) == 0) {
      name = name.substr(subcat.length() + 3);
    } else if (name.rfind(subcat + ": ", 0) == 0) {
      name = name.substr(subcat.length() + 2);
    }
  }

  // Strip redundant TLAS prefix
  if (name.rfind("TLAS: ", 0) == 0) {
    name = name.substr(6);
  }

  // Strip repeated scene path tracing patterns
  if (subcat.find("Path Tracing") != std::string::npos && name.rfind("Primary Rays", 0) != 0) {
    if (name.rfind("Full Scene Path Tracing - ", 0) == 0) {
      name = name.substr(26);
    }
  }

  // Strip repeated scene ray tracing patterns
  if (subcat.find("Scene Ray Tracing") != std::string::npos && name.rfind("Primary Rays", 0) != 0) {
    if (name.rfind("Full Scene Ray Tracing (PBR) - ", 0) == 0) {
      name = name.substr(31);
    } else if (name.rfind("Full Scene Ray Tracing - ", 0) == 0) {
      name = name.substr(25);
    }
  }

  // Clean verbose divergence string
  static const std::string kDivPrefix1 = "Traversal: Isolated Microbenchmark - ";
  static const std::string kDivPrefix2 = "Isolated Microbenchmark - ";
  if (name.rfind(kDivPrefix1, 0) == 0) {
    name = name.substr(kDivPrefix1.length());
  } else if (name.find(kDivPrefix2) != std::string::npos) {
    size_t pos = name.find(kDivPrefix2);
    name = name.substr(pos + kDivPrefix2.length());
  } else if (name.find("0% Coherence (Diffuse)") != std::string::npos ||
      name.find("0% Diffuse") != std::string::npos) {
    name = "Secondary bounce rays - 0% Diffuse";
  }

  // Clean Stage prefix if present
  if (name.rfind("Stage: BVH Traversal - ", 0) == 0) {
    name = name.substr(23);
  } else if (name.rfind("Stage: Queue Compaction - ", 0) == 0) {
    name = name.substr(26);
  } else if (name.rfind("Stage: Queue Memory - ", 0) == 0) {
    name = name.substr(22);
  } else if (name.rfind("Stage: Traversal Divergence - ", 0) == 0) {
    name = name.substr(30);
  } else if (name.rfind("Stage: ", 0) == 0) {
    name = name.substr(7);
  }

  // Strip subcategory prefix again if revealed after Stage: removal
  if (!subcat.empty()) {
    if (name.rfind(subcat + " - ", 0) == 0) {
      name = name.substr(subcat.length() + 3);
    } else if (name.rfind(subcat + ": ", 0) == 0) {
      name = name.substr(subcat.length() + 2);
    }
  }

  // Clean long pipeline breakdown names
  if (name == "BVH Traversal - Linear 1D Scanline (Baseline)") {
    name = "BVH Traversal - Linear 1D Scanline";
  } else if (name == "Queue Compaction - Single-Pass Unified Stream") {
    name = "Queue Compaction - Single-Pass Stream";
  } else if (name.find("Queue Compaction - Queue Compaction") != std::string::npos ||
             name.find("Queue Memory - VRAM Round-Trip") != std::string::npos) {
    name = "Queue Compaction Throughput";
  }

  // Strip unmatched closing parenthesis
  int openCount = 0, closeCount = 0;
  for (char ch : name) {
    if (ch == '(') openCount++;
    else if (ch == ')') closeCount++;
  }
  while (closeCount > openCount && !name.empty() && name.back() == ')') {
    name.pop_back();
    closeCount--;
  }

  if (name == "Megakernel") name = "Megakernel";
  if (name == "Work Lists" || name == "Device-Generated Commands" || name == "Device-Generated Commands (DGC)") name = "DGC";

  return name;
}

void ResultFormatter::print() {
  if (results.empty()) {
    return;
  }

  // 1. Identify all unique devices (by index) and map to names
  std::set<uint32_t> deviceIndices;
  std::map<uint32_t, std::string> deviceNames;
  for (const auto &res : results) {
    deviceIndices.insert(res.deviceIndex);
    if (deviceNames.find(res.deviceIndex) == deviceNames.end()) {
      deviceNames[res.deviceIndex] = res.deviceName;
    }
  }

  // 2. Identify all unique backends
  std::set<std::string> backends;
  for (const auto &res : results) {
    backends.insert(res.backendName);
  }

  // 3. Structure subcategories cleanly:
  struct SubcategoryGroup {
    std::string name;
    int minSortWeight = 999999;
    // Map of benchmark key -> (backend -> ResultData)
    // Key: {sortWeight, configIndex, cleanName}
    std::map<std::tuple<int, uint32_t, std::string>, std::map<std::string, ResultData>> benchmarks;
  };

  // Device -> Component -> (SubcategoryName -> SubcategoryGroup)
  std::map<uint32_t, std::map<std::pair<int, std::string>, std::map<std::string, SubcategoryGroup>>> organizedData;

  // Track executive takeaways
  double maxRayRate = 0.0;
  std::string maxRayWorkload = "";
  double megakernelPBRRate = 0.0;
  double dgcPBRRate = 0.0;
  std::string pbrSceneName = "";
  bool hasPbrScene = false;
  double scanlineRate = 0.0;
  double tiledRate = 0.0;
  double maxBlasUpdateRate = 0.0;
  double maxTlasRate = 0.0;
  double rawBoxGis = 0.0;
  double rawTriangleGis = 0.0;
  size_t totalWorkloadsEvaluated = 0;

  for (const auto &res : results) {
    totalWorkloadsEvaluated++;
    int compWeight = 6;
    if (res.component == "Compute") compWeight = 1;
    else if (res.component == "Memory") compWeight = 2;
    else if (res.component == "Scheduling") compWeight = 3;
    else if (res.component == "Graphics" || res.component == "Rasterization & ROP") compWeight = 4;
    else if (res.component == "Ray Tracing") compWeight = 5;

    std::string cleanName = cleanWorkloadName(res.benchmarkName, res.subcategory);
    if (res.isEmulated) cleanName += " (Emulated)";

    std::string effectiveSubcat = res.subcategory;
    std::string sceneName = extractSceneName(res.benchmarkName);
    if (!sceneName.empty() &&
        res.subcategory != "Traversal Ordering & Coherence" &&
        res.subcategory != "Wavefront Stream Compaction") {
      effectiveSubcat = res.subcategory + " [" + sceneName + "]";
    }

    auto &subcatGroup = organizedData[res.deviceIndex][{compWeight, res.component}][effectiveSubcat];
    subcatGroup.name = effectiveSubcat;
    subcatGroup.minSortWeight = (std::min)(subcatGroup.minSortWeight, res.sortWeight);
    subcatGroup.benchmarks[{res.sortWeight, res.configIndex, cleanName}][res.backendName] = res;

    // Track metrics for executive summary
    if (!res.isUnsupported && res.isValid && res.time_ms > 0.0) {
      double rate = (static_cast<double>(res.operations) / (res.time_ms / 1000.0)) / 1e6;
      if (res.benchmarkName.find("RayRawTraversal") != std::string::npos ||
          res.subcategory == "Hardware BVH Traversal") {
        double time_s = res.time_ms / 1000.0;
        uint64_t ops = (res.configIndex == 0) ? res.operations : (res.operations * 64);
        double gis = (time_s > 0.0) ? ((static_cast<double>(ops) / time_s) / 1e9) : 0.0;
        if (res.configIndex == 0) {
          rawTriangleGis = (std::max)(rawTriangleGis, gis);
        } else {
          rawBoxGis = (std::max)(rawBoxGis, gis);
        }
      }
      if (res.metric == "MRays/s") {
        if (rate > maxRayRate) {
          maxRayRate = rate;
          maxRayWorkload = cleanName;
        }
        if (res.benchmarkName.find("Scene Render: Full Frame (Megakernel)") != std::string::npos ||
            res.benchmarkName.find("Full Scene Ray Tracing (PBR) - Megakernel") != std::string::npos) {
          if (!hasPbrScene || sceneName == pbrSceneName) {
            megakernelPBRRate = (std::max)(megakernelPBRRate, rate);
            pbrSceneName = sceneName;
            hasPbrScene = true;
          }
        } else if (res.benchmarkName.find("Scene Render: Full Frame (DGC)") != std::string::npos ||
                   res.benchmarkName.find("Full Scene Ray Tracing (PBR) - DGC") != std::string::npos ||
                   res.benchmarkName.find("Full Scene Ray Tracing (PBR) - Work Lists") != std::string::npos ||
                   res.benchmarkName.find("Full Scene Ray Tracing (PBR) - Device-Generated Commands") != std::string::npos) {
          if (!hasPbrScene || sceneName == pbrSceneName) {
            dgcPBRRate = (std::max)(dgcPBRRate, rate);
            pbrSceneName = sceneName;
            hasPbrScene = true;
          }
        } else if (res.benchmarkName.find("1D Scanline") != std::string::npos) {
          scanlineRate = (std::max)(scanlineRate, rate);
        } else if (res.benchmarkName.find("2D Screen Tiled") != std::string::npos) {
          tiledRate = (std::max)(tiledRate, rate);
        }
      } else if (res.metric == "MTris/s") {
        if (res.benchmarkName.find("BLAS Update") != std::string::npos && rate > maxBlasUpdateRate) {
          maxBlasUpdateRate = rate;
        }
      } else if (res.metric == "MInst/s" && rate > maxTlasRate) {
        maxTlasRate = rate;
      }
    }
  }

  const bool useColor = shouldUseColor();
  const std::string RESET = useColor ? "\033[0m" : "";
  const std::string BOLD = useColor ? "\033[1m" : "";
  const std::string DIM = useColor ? "\033[2m" : "";
  const std::string CYAN = useColor ? "\033[36m" : "";
  const std::string GREEN = useColor ? "\033[32m" : "";
  const std::string RED = useColor ? "\033[31m" : "";
  const std::string YELLOW = useColor ? "\033[33m" : "";
  const std::string MAGENTA = useColor ? "\033[35m" : "";
  const std::string BLUE = useColor ? "\033[34m" : "";

  const size_t cardBoxWidth = std::clamp<size_t>(getTerminalWidth(), 80, 128);
  const size_t cardInnerWidth = cardBoxWidth - 4; // 124

  const size_t w1 = 46;
  const size_t w2 = 8;
  const size_t w3 = 22;
  const size_t w4 = 39;

  std::cout << std::endl;
  for (uint32_t devIdx : deviceIndices) {
    std::string reportTitle = " GPUBench Hierarchical Benchmark Report ";
    size_t dashCount = (cardBoxWidth > reportTitle.length() + 3) ? (cardBoxWidth - 3 - reportTitle.length()) : 10;

    std::cout << BOLD << CYAN << "  ╭─" << RESET << BOLD << reportTitle << RESET
              << BOLD << CYAN << repeatUtf8("─", dashCount) << "╮" << RESET << "\n";

    std::string devLabel = "Target Device : ";
    std::string devNameStr;
    if (devIdx == 0xFFFFFFFF) {
      devNameStr = "System (Host CPU)";
    } else {
      devNameStr = deviceNames[devIdx] + " (ID: " + std::to_string(devIdx) + ")";
    }

    std::string fullLine = devLabel + devNameStr;
    size_t visLen = visualLength(fullLine);
    size_t pad = (cardInnerWidth > visLen) ? (cardInnerWidth - visLen) : 0;

    std::cout << BOLD << CYAN << "  │ " << RESET
              << BOLD << devLabel << RESET << MAGENTA << devNameStr << RESET
              << std::string(pad, ' ')
              << BOLD << CYAN << " │" << RESET << "\n";

    // Query canvas resolution if available
    uint32_t reportWidth = 0;
    uint32_t reportHeight = 0;
    for (const auto &compPair : organizedData[devIdx]) {
      for (const auto &subcatPair : compPair.second) {
        for (const auto &benchPair : subcatPair.second.benchmarks) {
          for (const auto &bPair : benchPair.second) {
            if (bPair.second.width > 0 && bPair.second.height > 0) {
              reportWidth = bPair.second.width;
              reportHeight = bPair.second.height;
              break;
            }
          }
          if (reportWidth > 0) break;
        }
        if (reportWidth > 0) break;
      }
      if (reportWidth > 0) break;
    }

    if (reportWidth > 0 && reportHeight > 0) {
      std::string resPreset = "";
      if (reportWidth == 3840 && reportHeight == 2160) resPreset = " (4K UHD)";
      else if (reportWidth == 2560 && reportHeight == 1440) resPreset = " (1440p QHD)";
      else if (reportWidth == 1920 && reportHeight == 1080) resPreset = " (1080p FHD)";
      else if (reportWidth == 1280 && reportHeight == 720) resPreset = " (720p HD)";
      else if (reportWidth == 1024 && reportHeight == 1024) resPreset = " (1024x1024 Square)";

      std::string resLabel = "Canvas / Res  : ";
      std::string resValStr = std::to_string(reportWidth) + "x" + std::to_string(reportHeight) + resPreset;
      std::string fullResLine = resLabel + resValStr;
      size_t visResLen = visualLength(fullResLine);
      size_t resPad = (cardInnerWidth > visResLen) ? (cardInnerWidth - visResLen) : 0;

      std::cout << BOLD << CYAN << "  │ " << RESET
                << BOLD << resLabel << RESET << CYAN << resValStr << RESET
                << std::string(resPad, ' ')
                << BOLD << CYAN << " │" << RESET << "\n";
    }

    std::cout << BOLD << CYAN << "  ╰─" << repeatUtf8("─", cardBoxWidth - 3) << "╯" << RESET << "\n";

    const auto &components = organizedData[devIdx];
    for (const auto &compPair : components) {
      const std::string &compName = compPair.first.second;
      std::cout << "\n  [" << BOLD << CYAN << compName << RESET << "]" << std::endl;

      // Sort subcategories by their minimum sortWeight
      std::vector<SubcategoryGroup> sortedSubcats;
      for (const auto &subcatPair : compPair.second) {
        sortedSubcats.push_back(subcatPair.second);
      }
      std::sort(sortedSubcats.begin(), sortedSubcats.end(),
                [](const SubcategoryGroup &a, const SubcategoryGroup &b) {
                  if (a.minSortWeight != b.minSortWeight)
                    return a.minSortWeight < b.minSortWeight;
                  int orderA = getSceneOrder(a.name);
                  int orderB = getSceneOrder(b.name);
                  if (orderA != orderB)
                    return orderA < orderB;
                  return a.name < b.name;
                });

      for (const auto &subcat : sortedSubcats) {
        std::string title = " " + subcat.name + " ";
        size_t subcatDashCount = (cardBoxWidth > title.length() + 3)
                                     ? (cardBoxWidth - 3 - title.length())
                                     : 10;

        // Top Border
        std::cout << BOLD << CYAN << "  ╭─" << RESET << BOLD << title << RESET
                  << BOLD << CYAN << repeatUtf8("─", subcatDashCount) << "╮" << RESET << "\n";

        // Header Row
        std::cout << BOLD << CYAN << "  │ " << RESET << BOLD << std::left << std::setw(w1) << "Workload" << RESET
                  << BOLD << CYAN << " │ " << RESET << BOLD << std::left << std::setw(w2) << "Backend" << RESET
                  << BOLD << CYAN << " │ " << RESET << BOLD << std::right << std::setw(w3) << "Throughput" << RESET
                  << BOLD << CYAN << " │ " << RESET << BOLD << std::left << std::setw(w4) << "Details / Speedup" << RESET
                  << BOLD << CYAN << " │" << RESET << "\n";

        // Divider Row
        std::cout << BOLD << CYAN << "  ├─" << repeatUtf8("─", w1)
                  << "─┼─" << repeatUtf8("─", w2)
                  << "─┼─" << repeatUtf8("─", w3)
                  << "─┼─" << repeatUtf8("─", w4)
                  << "─┤" << RESET << "\n";

        // Determine whether this subcategory has a genuine architectural baseline comparison.
        bool hasComparison = false;
        std::string baselineKeyName = "";
        double baselineVal = 0.0;
        std::string baselineMetric = "";

        if (subcat.benchmarks.size() > 1 && subcat.name != "Latency" && subcat.name != "Bandwidth") {
          bool hasMegakernel = false;
          bool hasVector = false;
          bool hasScanline = false;
          bool hasSinglePassCompaction = false;

          for (const auto &benchPair : subcat.benchmarks) {
            std::string cName = std::get<2>(benchPair.first);
            if (cName.find("Megakernel") != std::string::npos ||
                cName.find("Traditional") != std::string::npos) {
              hasMegakernel = true;
            }
            if (cName.find("Vector") != std::string::npos) {
              hasVector = true;
            }
            if (cName.find("Scanline") != std::string::npos ||
                cName.find("Baseline") != std::string::npos) {
              hasScanline = true;
            }
            if (cName.find("Single-Pass") != std::string::npos) {
              hasSinglePassCompaction = true;
            }
          }

          if (hasMegakernel || (hasVector && subcat.benchmarks.size() >= 2) || hasScanline || hasSinglePassCompaction) {
            hasComparison = true;
            for (const auto &benchPair : subcat.benchmarks) {
              std::string cName = std::get<2>(benchPair.first);
              bool isCandidate = false;
              if (hasMegakernel && ((cName.find("Megakernel") != std::string::npos && cName.find("RTP") == std::string::npos) || cName.find("Traditional") != std::string::npos)) {
                isCandidate = true;
              } else if (!hasMegakernel && hasVector && cName.find("Vector") != std::string::npos) {
                isCandidate = true;
              } else if (!hasMegakernel && !hasVector && hasScanline && (cName.find("Scanline") != std::string::npos || cName.find("Baseline") != std::string::npos)) {
                isCandidate = true;
              } else if (!hasMegakernel && !hasVector && !hasScanline && hasSinglePassCompaction && cName.find("Single-Pass") != std::string::npos) {
                isCandidate = true;
              }

              if (isCandidate) {
                for (const auto &backend : backends) {
                  if (benchPair.second.count(backend)) {
                    const auto &res = benchPair.second.at(backend);
                    if (!res.isUnsupported && res.isValid && res.time_ms > 0.0) {
                      double val = 0.0;
                      if (res.metric == "us") {
                        val = (res.time_ms * 1000.0) / static_cast<double>(res.operations);
                      } else if (res.metric == "ns") {
                        val = (res.time_ms * 1e6) / static_cast<double>(res.operations);
                      } else {
                        val = static_cast<double>(res.operations) / (res.time_ms / 1000.0);
                        if (res.metric == "TB/s" || res.metric == "TFLOPS" || res.metric == "TOPS") {
                          val /= 1e12;
                        } else if (res.metric == "GB/s" || res.metric == "GIS/s" || res.metric == "GRays/s" ||
                                   res.metric == "GPixels/s" || res.metric == "GOPS" || res.metric == "GFLOPS") {
                          val /= 1e9;
                        } else if (res.metric == "MItems/s" || res.metric == "MSamples/s" || res.metric == "MBVH/s" ||
                                   res.metric == "MRays/s" || res.metric == "MHits/s" || res.metric == "MTris/s" ||
                                   res.metric == "MInst/s" || res.metric == "MRecords/s") {
                          val /= 1e6;
                        } else if (res.component == "Compute") {
                          val /= 1e12;
                        } else if (res.component == "Memory") {
                          val = (res.subcategory == "Latency") ? ((res.time_ms * 1e6) / res.operations) : (val / 1e9);
                        } else {
                          val /= 1e6;
                        }
                      }

                      baselineVal = val;
                      baselineMetric = res.metric;
                      baselineKeyName = cName;
                      break;
                    }
                  }
                }
                if (baselineVal > 0.0) break;
              }
            }
          }
        }

        // Print benchmark rows
        for (const auto &benchPair : subcat.benchmarks) {
          std::string displayName = std::get<2>(benchPair.first);
          std::string fullName = displayName;
          displayName = truncateUtf8(displayName, w1);

          const auto &backendData = benchPair.second;
          bool firstBackend = true;
          for (const auto &backend : backends) {
            if (backendData.count(backend)) {
              const auto &res = backendData.at(backend);

              std::string valStr;
              std::string noteStr;
              std::string statusColor = GREEN;

              if (res.isUnsupported) {
                valStr = "UNSUPPORTED";
                statusColor = RED;
                std::string note = cleanSupportNote(!res.supportNote.empty() ? res.supportNote : res.supportCategory);
                if (!note.empty()) {
                  noteStr = "[" + note + "]";
                }
              } else if (!res.isValid) {
                valStr = "INVALID";
                statusColor = RED;
                noteStr = "[Validation Failed]";
              } else if (res.time_ms < 0.0) {
                if (res.time_ms == -3.0) {
                  valStr = "ABORTED";
                  statusColor = YELLOW;
                } else {
                  valStr = "FAILED";
                  statusColor = RED;
                }
                if (!res.errorString.empty()) {
                  noteStr = "[" + res.errorString + "]";
                }
              } else if (res.component == "Compute") {
                double value = 0.0;
                if (res.metric == "MItems/s" || res.metric == "MSamples/s" || res.metric == "MBVH/s") {
                  value = (static_cast<double>(res.operations) / (res.time_ms / 1000.0)) / 1e6;
                } else if (res.metric == "GB/s" || res.metric == "GOPS" || res.metric == "GFLOPS") {
                  value = (static_cast<double>(res.operations) / (res.time_ms / 1000.0)) / 1e9;
                } else if (res.metric == "us") {
                  value = (res.time_ms * 1000.0) / static_cast<double>(res.operations);
                } else {
                  value = (static_cast<double>(res.operations) / (res.time_ms / 1000.0)) / 1e12;
                }
                valStr = formatDouble(value, 2) + " " + res.metric;
                if (res.subcategory == "LDS Bank Conflicts") {
                  if (res.configIndex == 0) {
                    noteStr = "[Baseline]";
                  } else {
                    double baseVal = 0.0;
                    for (const auto &bp : subcat.benchmarks) {
                      if (bp.second.count(backend)) {
                        const auto &br = bp.second.at(backend);
                        if (br.subcategory == "LDS Bank Conflicts" && br.configIndex == 0 && br.time_ms > 0.0) {
                          baseVal = (static_cast<double>(br.operations) / (br.time_ms / 1000.0)) / 1e12;
                          break;
                        }
                      }
                    }
                    if (baseVal > 0.0) {
                      double ratio = value / baseVal;
                      double pct = (ratio - 1.0) * 100.0;
                      noteStr = "└──> " + formatDouble(ratio, 2) + "x (" + (pct >= 0 ? "+" : "") + formatDouble(pct, 1) + "%)";
                    }
                  }
                } else if (res.subcategory == "Indirect Command Synthesis") {
                  if (res.configIndex == 0) {
                    noteStr = "[Baseline]";
                  } else {
                    double baseVal = 0.0;
                    for (const auto &bp : subcat.benchmarks) {
                      if (bp.second.count(backend)) {
                        const auto &br = bp.second.at(backend);
                        if (br.subcategory == "Indirect Command Synthesis" && br.configIndex == 0 && br.time_ms > 0.0) {
                          if (br.metric == "us") {
                            baseVal = (br.time_ms * 1000.0) / static_cast<double>(br.operations);
                          } else {
                            baseVal = (static_cast<double>(br.operations) / (br.time_ms / 1000.0)) / 1e6;
                          }
                          break;
                        }
                      }
                    }
                    if (baseVal > 0.0 && value > 0.0) {
                      double ratio = (res.metric == "us") ? (baseVal / value) : (value / baseVal);
                      double pct = (ratio - 1.0) * 100.0;
                      noteStr = "└──> " + formatDouble(ratio, 2) + "x (" + (pct >= 0 ? "+" : "") + formatDouble(pct, 1) + "%)";
                    }
                  }
                } else if (res.subcategory == "Dual-Issue" || res.subcategory == "ILP & Concurrency") {
                  if (res.configIndex == 0 || res.configIndex == 3) {
                    noteStr = "[Baseline]";
                  } else if (res.configIndex == 6) {
                    // Config 6: Mixed (1:1 instruction mix: 16 FP32 FMA instructions = 32 FLOPs + 16 INT32 instructions = 16 IOPs).
                    // The expected single-issue serialized baseline throughput is the weighted harmonic mean of
                    // Config 0 (FP32 baseline) and Config 3 (INT32 baseline):
                    // B_mixed = 3 / (2 / B_FP32 + 1 / B_INT32)
                    double baseFP32 = 0.0;
                    double baseINT32 = 0.0;
                    for (const auto &bp : subcat.benchmarks) {
                      if (bp.second.count(backend)) {
                        const auto &br = bp.second.at(backend);
                        if ((br.subcategory == "Dual-Issue" || br.subcategory == "ILP & Concurrency") && br.time_ms > 0.0) {
                          if (br.configIndex == 0) {
                            baseFP32 = (static_cast<double>(br.operations) / (br.time_ms / 1000.0)) / 1e12;
                          } else if (br.configIndex == 3) {
                            baseINT32 = (static_cast<double>(br.operations) / (br.time_ms / 1000.0)) / 1e12;
                          }
                        }
                      }
                    }
                    if (baseFP32 > 0.0 && baseINT32 > 0.0) {
                      double baseVal = 3.0 / ((2.0 / baseFP32) + (1.0 / baseINT32));
                      double ratio = value / baseVal;
                      double pct = (ratio - 1.0) * 100.0;
                      noteStr = "└──> " + formatDouble(ratio, 2) + "x (" + (pct >= 0 ? "+" : "") + formatDouble(pct, 1) + "%)";
                    } else {
                      noteStr = "[Mixed FP32+INT32]";
                    }
                  } else {
                    uint32_t targetBaseConfig = (res.configIndex >= 3) ? 3 : 0;
                    for (const auto &bp : subcat.benchmarks) {
                      if (bp.second.count(backend)) {
                        const auto &br = bp.second.at(backend);
                        if ((br.subcategory == "Dual-Issue" || br.subcategory == "ILP & Concurrency") && br.configIndex == targetBaseConfig && br.time_ms > 0.0) {
                          double baseVal = (static_cast<double>(br.operations) / (br.time_ms / 1000.0)) / 1e12;
                          if (baseVal > 0.0) {
                            double ratio = value / baseVal;
                            double pct = (ratio - 1.0) * 100.0;
                            noteStr = "└──> " + formatDouble(ratio, 2) + "x (" + (pct >= 0 ? "+" : "") + formatDouble(pct, 1) + "%)";
                          }
                          break;
                        }
                      }
                    }
                  }
                } else if (hasComparison && baselineVal > 0.0 && value > 0.0) {
                  if (fullName == baselineKeyName || fullName.find("Vector") != std::string::npos) {
                    noteStr = "[Baseline]";
                  } else {
                    double ratio = (res.metric == "us" || res.metric == "ns") ? (baselineVal / value) : (value / baselineVal);
                    double pct = (ratio - 1.0) * 100.0;
                    noteStr = "└──> " + formatDouble(ratio, 2) + "x (" + (pct >= 0 ? "+" : "") + formatDouble(pct, 1) + "%)";
                  }
                }
              } else if (res.component == "Memory") {
                if (res.subcategory == "Latency" || res.subcategory == "Cache Latency Curve" || res.metric == "ns") {
                  double value = (res.time_ms * 1e6) / res.operations; // ns
                  valStr = formatDouble(value, 2) + " ns";
                  if (!res.supportNote.empty()) {
                    noteStr = "[" + res.supportNote + "]";
                  }
                } else {
                  if (res.metric == "TB/s") {
                    double value = (static_cast<double>(res.operations) /
                                   (res.time_ms / 1000.0)) / 1e12; // TB/s
                    valStr = formatDouble(value, 2) + " TB/s";
                  } else {
                    double value = (static_cast<double>(res.operations) /
                                   (res.time_ms / 1000.0)) / 1e9; // GB/s
                    valStr = formatDouble(value, 2) + " GB/s";
                  }
                  if (!res.supportNote.empty()) {
                    noteStr = "[" + cleanSupportNote(res.supportNote) + "]";
                  }
                }
              } else {
                double value = 0.0;
                int precision = 2;
                if (res.metric == "us") {
                  value = (res.operations > 0)
                              ? ((res.time_ms * 1000.0) / static_cast<double>(res.operations))
                              : (res.time_ms * 1000.0);
                } else if (res.metric == "ns") {
                  value = (res.operations > 0)
                              ? ((res.time_ms * 1e6) / static_cast<double>(res.operations))
                              : 0.0;
                } else if (res.metric == "ms") {
                  value = (res.operations > 0)
                              ? (res.time_ms / static_cast<double>(res.operations))
                              : res.time_ms;
                } else {
                  value = static_cast<double>(res.operations) / (res.time_ms / 1000.0);
                  if (res.metric == "TB/s" || res.metric == "TFLOPS" || res.metric == "TOPS") {
                    value /= 1e12;
                  } else if (res.metric == "GIS/s" || res.metric == "GRays/s" ||
                             res.metric == "GPixels/s" || res.metric == "GB/s" ||
                             res.metric == "GOPS" || res.metric == "GFLOPS") {
                    value /= 1e9;
                    if (res.metric == "GRays/s") precision = 3;
                  } else if (res.metric == "MTris/s" || res.metric == "MInst/s" ||
                             res.metric == "MRays/s" || res.metric == "MHits/s" ||
                             res.metric == "MRecords/s" || res.metric == "MSamples/s" ||
                             res.metric == "MBVH/s" || res.metric == "MItems/s" ||
                             res.metric == "MPixels/s") {
                    value /= 1e6;
                  }
                }
                valStr = formatDouble(value, precision) + " " + res.metric;

                std::string fpsStr = "";
                if (res.benchmarkName.find("RayScheduling") != std::string::npos && res.metric == "MRays/s") {
                  uint32_t w = res.width ? res.width : 1920;
                  uint32_t h = res.height ? res.height : 1080;
                  double fps = (value * 1e6) / static_cast<double>(w * h);
                  fpsStr = formatDouble(fps, 1) + " FPS";
                }

                if (res.benchmarkName.find("RayRawTraversal") != std::string::npos ||
                    res.subcategory == "Hardware BVH Traversal") {
                  double time_s = res.time_ms / 1000.0;
                  uint64_t ops = (res.configIndex == 0) ? res.operations : (res.operations * 64);
                  double gis = (time_s > 0.0) ? ((static_cast<double>(ops) / time_s) / 1e9) : 0.0;
                  if (res.configIndex == 0) {
                    noteStr = "[" + formatDouble(gis, 1) + " GIS/s Max Occupancy]";
                  } else {
                    noteStr = "[" + formatDouble(gis, 1) + " Box GIS/s (64-AABB Model)]";
                  }
                } else {
                  double localBaseVal = 0.0;
                  std::string localBaseMetric = "";
                  bool isLocalBase = false;

                  if (res.benchmarkName.find("Multi-Light Evaluation") != std::string::npos) {
                    if (res.configIndex == 31 || res.configIndex == 33) {
                      isLocalBase = true;
                    } else if (res.configIndex == 32) {
                      for (const auto &bp : subcat.benchmarks) {
                        if (bp.second.count(backend)) {
                          const auto &br = bp.second.at(backend);
                          if (br.configIndex == 31 && br.time_ms > 0.0) {
                            localBaseVal = static_cast<double>(br.operations) / (br.time_ms / 1000.0) / 1e6;
                            localBaseMetric = br.metric;
                            break;
                          }
                        }
                      }
                    } else if (res.configIndex == 34) {
                      for (const auto &bp : subcat.benchmarks) {
                        if (bp.second.count(backend)) {
                          const auto &br = bp.second.at(backend);
                          if (br.configIndex == 33 && br.time_ms > 0.0) {
                            localBaseVal = static_cast<double>(br.operations) / (br.time_ms / 1000.0) / 1e6;
                            localBaseMetric = br.metric;
                            break;
                          }
                        }
                      }
                    }
                  } else if (res.benchmarkName.find("RayDivergence") != std::string::npos) {
                    if (res.configIndex == 0) isLocalBase = true;
                    else {
                      for (const auto &bp : subcat.benchmarks) {
                        if (bp.second.count(backend)) {
                          const auto &br = bp.second.at(backend);
                          if (br.benchmarkName.find("RayDivergence") != std::string::npos && br.configIndex == 0 && br.time_ms > 0.0) {
                            localBaseVal = static_cast<double>(br.operations) / (br.time_ms / 1000.0) / 1e6;
                            localBaseMetric = br.metric;
                            break;
                          }
                        }
                      }
                    }
                  } else if (res.benchmarkName.find("RayPayload") != std::string::npos) {
                    if (res.configIndex == 0) isLocalBase = true;
                    else {
                      for (const auto &bp : subcat.benchmarks) {
                        if (bp.second.count(backend)) {
                          const auto &br = bp.second.at(backend);
                          if (br.benchmarkName.find("RayPayload") != std::string::npos && br.configIndex == 0 && br.time_ms > 0.0) {
                            localBaseVal = static_cast<double>(br.operations) / (br.time_ms / 1000.0) / 1e6;
                            localBaseMetric = br.metric;
                            break;
                          }
                        }
                      }
                    }
                  } else if (res.benchmarkName.find("RayAnyHit") != std::string::npos) {
                    if (res.configIndex == 0) isLocalBase = true;
                    else {
                      for (const auto &bp : subcat.benchmarks) {
                        if (bp.second.count(backend)) {
                          const auto &br = bp.second.at(backend);
                          if (br.benchmarkName.find("RayAnyHit") != std::string::npos && br.configIndex == 0 && br.time_ms > 0.0) {
                            localBaseVal = static_cast<double>(br.operations) / (br.time_ms / 1000.0) / 1e6;
                            localBaseMetric = br.metric;
                            break;
                          }
                        }
                      }
                    }
                  } else if (res.baselineConfigIndex >= 0) {
                    for (const auto &bp : subcat.benchmarks) {
                      if (bp.second.count(backend)) {
                        const auto &br = bp.second.at(backend);
                        if (br.configIndex == static_cast<uint32_t>(res.baselineConfigIndex) && br.time_ms > 0.0 && br.isValid) {
                          localBaseVal = computeResultValue(br);
                          localBaseMetric = br.metric;
                          break;
                        }
                      }
                    }
                  } else if (fullName == baselineKeyName ||
                             (((fullName.find("Megakernel") != std::string::npos && fullName.find("RTP") == std::string::npos) ||
                               fullName.find("Traditional") != std::string::npos || fullName.find("Baseline") != std::string::npos ||
                               fullName.find("Scanline") != std::string::npos || fullName.find("Single-Pass") != std::string::npos))) {
                    isLocalBase = true;
                  } else if (hasComparison && baselineVal > 0.0 && baselineMetric == res.metric && res.baselineConfigIndex != -1) {
                    localBaseVal = baselineVal;
                    localBaseMetric = baselineMetric;
                  }

                  if (isLocalBase) {
                    noteStr = "[Baseline]" + (fpsStr.empty() ? "" : " [" + fpsStr + "]");
                  } else if (localBaseVal > 0.0 && value > 0.0 && (localBaseMetric.empty() || localBaseMetric == res.metric)) {
                    const bool lowerIsBetter =
                        (res.metric == "us" || res.metric == "ns" || res.metric == "ms");
                    double ratio = lowerIsBetter ? (localBaseVal / value) : (value / localBaseVal);
                    double pct = (ratio - 1.0) * 100.0;
                    std::string pctStr = (std::abs(pct) >= 0.1) ? (std::string(" (") + (pct >= 0 ? "+" : "") + formatDouble(pct, 1) + "%)") : "";
                    noteStr = "└──> " + formatDouble(ratio, 2) + "x" + pctStr + (fpsStr.empty() ? "" : " [" + fpsStr + "]");
                  } else if (!fpsStr.empty()) {
                    noteStr = fpsStr;
                  }
                }

                if (!res.supportNote.empty() &&
                    res.benchmarkName.find("RayRawTraversal") == std::string::npos &&
                    res.subcategory != "Hardware BVH Traversal") {
                  noteStr += (noteStr.empty() ? "" : " ") + ("[" + cleanSupportNote(res.supportNote) + "]");
                }
              }

              if (showPower && res.hasPowerTelemetry && res.powerWatts > 0.0f && !res.isUnsupported && res.isValid && res.time_ms > 0.0) {
                std::string pwrStr = formatDouble(res.powerWatts, 0) + "W";
                if (res.joulesPerUnit > 0.0 && res.unitPerWatt > 0.0) {
                  if (res.efficiencyUnit == "J/TFLOP") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 1) + " GFLOPS/W";
                  } else if (res.efficiencyUnit == "J/TOP") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 1) + " GOPS/W";
                  } else if (res.efficiencyUnit == "J/GB") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 2) + " GB/J";
                  } else if (res.efficiencyUnit == "J/MRay") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 0) + " kRays/J";
                  } else if (res.efficiencyUnit == "J/MTri") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 0) + " kTris/J";
                  } else if (res.efficiencyUnit == "J/MInst") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 0) + " kInst/J";
                  } else if (res.efficiencyUnit == "J/MRecord") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 0) + " kRecords/J";
                  } else if (res.efficiencyUnit == "J/MHit") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 0) + " kHits/J";
                  } else if (res.efficiencyUnit == "J/GIS") {
                    pwrStr += " | " + formatDouble(res.unitPerWatt, 0) + " MIS/J";
                  }
                }
                noteStr += (noteStr.empty() ? "" : " ") + ("[" + pwrStr + "]");
              }

              std::string backendStr = truncateUtf8(backend, w2, "");
              valStr = truncateUtf8(valStr, w3, "");
              noteStr = truncateUtf8(noteStr, w4);
              size_t noteVis = visualLength(noteStr);
              std::string notePad = (w4 > noteVis) ? std::string(w4 - noteVis, ' ') : "";

              std::string nameStr = firstBackend ? displayName : "";
              size_t nameVis = visualLength(nameStr);
              std::string namePad = (w1 > nameVis) ? std::string(w1 - nameVis, ' ') : "";

              std::cout << BOLD << CYAN << "  │ " << RESET
                        << nameStr << namePad
                        << BOLD << CYAN << " │ " << RESET << YELLOW << std::left << std::setw(w2) << backendStr << RESET
                        << BOLD << CYAN << " │ " << RESET << statusColor << BOLD << std::right << std::setw(w3) << valStr << RESET
                        << BOLD << CYAN << " │ " << RESET << DIM << noteStr << notePad << RESET
                        << BOLD << CYAN << " │" << RESET << "\n";

              bool hasRenderablePerf = (res.hasPerfQueryTelemetry && res.perfCounters.available);
              bool hasRenderableReg = (res.hasRegisterTelemetry && res.vgprCount > 0);
              if (showTelemetry && (hasRenderablePerf || hasRenderableReg) && !res.isUnsupported && res.isValid && res.time_ms > 0.0) {
                std::string telemInfo = "   └─ ";
                bool firstItem = true;
                if (hasRenderablePerf) {
                  telemInfo += "Cycles: " + formatNumber(res.perfCounters.gpuActiveCycles) +
                               " | Waves: " + formatNumber(res.perfCounters.waves) +
                               " | SALU: " + formatDouble(res.perfCounters.saluBusyPct, 1) + "%";
                  if (res.perfCounters.vramReadBytes > 0 || res.perfCounters.l0CacheHitRatio > 0.0f) {
                    telemInfo += " | VRAM Rd: " + formatDouble(static_cast<double>(res.perfCounters.vramReadBytes) / (1024.0 * 1024.0), 1) + " MB" +
                                 " | L0 Hit: " + formatDouble(res.perfCounters.l0CacheHitRatio, 1) + "%" +
                                 " | L1 Hit: " + formatDouble(res.perfCounters.l1CacheHitRatio, 1) + "%";
                  }
                  firstItem = false;
                }
                if (hasRenderableReg) {
                  if (!firstItem) telemInfo += " | ";
                  telemInfo += std::to_string(res.vgprCount) + " VGPRs";
                  if (res.maxWavesPerSimd > 0) {
                    telemInfo += " (" + std::to_string(res.maxWavesPerSimd) + "w/SIMD)";
                  }
                }
                std::string truncTelem = truncateUtf8(telemInfo, cardInnerWidth);
                size_t tVis = visualLength(truncTelem);
                std::string tPad = (cardInnerWidth > tVis) ? std::string(cardInnerWidth - tVis, ' ') : "";
                std::cout << BOLD << CYAN << "  │ " << RESET << DIM << truncTelem << tPad << RESET
                          << BOLD << CYAN << " │" << RESET << "\n";
              }

              firstBackend = false;
            }
          }
        }

        // Bottom Border
        std::cout << BOLD << CYAN << "  ╰─" << repeatUtf8("─", w1)
                  << "─┴─" << repeatUtf8("─", w2)
                  << "─┴─" << repeatUtf8("─", w3)
                  << "─┴─" << repeatUtf8("─", w4)
                  << "─╯" << RESET << "\n";
      }
    }
  }

  // 4. Executive Summary Card (if Ray Tracing or Power was evaluated)
  bool hasAnyPower = false;
  double totalPowerSum = 0.0;
  size_t powerCount = 0;
  double maxComputeEff = 0.0;
  std::string maxComputeWl;
  std::string maxComputeUnit;
  for (const auto &r : results) {
    if (r.hasPowerTelemetry && r.powerWatts > 0.0f) {
      hasAnyPower = true;
      totalPowerSum += r.powerWatts;
      powerCount++;
      if ((r.efficiencyUnit == "J/TFLOP" || r.efficiencyUnit == "J/TOP") && r.unitPerWatt > maxComputeEff) {
        maxComputeEff = r.unitPerWatt;
        maxComputeWl = cleanWorkloadName(r.benchmarkName, r.subcategory);
        maxComputeUnit = (r.efficiencyUnit == "J/TFLOP") ? "GFLOPS/W" : "GOPS/W";
      }
    }
  }

  if (totalWorkloadsEvaluated > 0 && (maxRayRate > 0.0 || (showPower && hasAnyPower))) {
    std::cout << "\n";
    std::string cardTitle = " Executive Performance Summary & Architectural Takeaways ";
    size_t dashCount = (cardBoxWidth > cardTitle.length() + 3)
                           ? (cardBoxWidth - 3 - cardTitle.length())
                           : 10;

    std::cout << BOLD << CYAN << "  ╭─" << RESET << BOLD << cardTitle << RESET
              << BOLD << CYAN << repeatUtf8("─", dashCount) << "╮" << RESET << "\n";

    auto printSummaryRow = [&](const std::string &label, const std::string &val, const std::string &extra) {
      std::string visualStr = "• " + label + " : " + val + extra;
      size_t visLen = visualLength(visualStr);
      size_t pad = (cardInnerWidth > visLen) ? (cardInnerWidth - visLen) : 0;
      std::cout << BOLD << CYAN << "  │ " << RESET
                << BOLD << "• " << label << RESET << " : "
                << val << extra
                << std::string(pad, ' ')
                << BOLD << CYAN << " │" << RESET << "\n";
    };

    if (megakernelPBRRate > 0.0 && dgcPBRRate > 0.0) {
      double pbrSpeedup = dgcPBRRate / megakernelPBRRate;
      std::string val = BOLD + GREEN + formatDouble(pbrSpeedup, 2) + "x" + RESET;
      std::string sceneTag = pbrSceneName.empty() ? "" : (" [" + pbrSceneName + "]");
      std::string extra = " in PBR Ray Tracing" + sceneTag + " (" + formatDouble(dgcPBRRate, 1) + " vs " +
                          formatDouble(megakernelPBRRate, 1) + " MRays/s)";
      printSummaryRow("Wavefront Scheduling Speedup", val, extra);
    }
    if (scanlineRate > 0.0 && tiledRate > 0.0) {
      double cacheGain = ((tiledRate - scanlineRate) / scanlineRate) * 100.0;
      std::string val = BOLD + CYAN + (cacheGain >= 0.0 ? "+" : "") + formatDouble(cacheGain, 1) + "%" + RESET;
      std::string extra = " with 2D Screen Tiling (" + formatDouble(tiledRate, 1) + " vs " +
                          formatDouble(scanlineRate, 1) + " MRays/s)";
      printSummaryRow("BVH Traversal Cache Locality", val, extra);
    }
    if (maxBlasUpdateRate > 0.0 || maxTlasRate > 0.0) {
      std::string val = BOLD + YELLOW + formatDouble(maxBlasUpdateRate, 1) + " MTris/s" + RESET + " (BLAS Update) | " +
                        BOLD + YELLOW + formatDouble(maxTlasRate, 1) + " MInst/s" + RESET + " (TLAS Construction)";
      printSummaryRow("Acceleration Build Peak Rates", val, "");
    }
    if (rawBoxGis > 0.0) {
      std::string val = BOLD + GREEN + formatDouble(rawBoxGis, 1) + " GIS/s" + RESET;
      printSummaryRow("Hardware BVH Traversal (Deep)", val, " (Multi-Layer Stress)");
    }
    if (rawTriangleGis > 0.0) {
      std::string val = BOLD + GREEN + formatDouble(rawTriangleGis, 1) + " GIS/s" + RESET;
      printSummaryRow("Hardware BVH Traversal (Coherent)", val, " (Max Occupancy)");
    }
    if (maxRayRate > 0.0) {
      std::string val = BOLD + GREEN + formatDouble(maxRayRate, 1) + " MRays/s" + RESET;
      std::string extra = " (" + maxRayWorkload + ")";
      printSummaryRow("Peak Measured Ray Rate       ", val, extra);
    }
    if (showPower && hasAnyPower && powerCount > 0) {
      double avgPower = totalPowerSum / powerCount;
      std::string val = BOLD + GREEN + formatDouble(avgPower, 1) + " W" + RESET + " Average Power";
      std::string extra = (maxComputeEff > 0.0)
          ? (" | Peak Efficiency: " + formatDouble(maxComputeEff, 1) + " " + maxComputeUnit + " (" + maxComputeWl + ")")
          : "";
      printSummaryRow("Power & Energy Telemetry     ", val, extra);
    }

    std::cout << BOLD << CYAN << "  ╰─" << repeatUtf8("─", cardBoxWidth - 3) << "╯" << RESET << "\n";
  }

  // 5. Backend Parity Attribution (Optimization O-4)
  std::map<std::string, std::map<std::string, ResultData>> parityMap;
  for (const auto &r : results) {
    if (r.component == "Compute" && r.isValid && r.time_ms > 0.0) {
      parityMap[r.benchmarkName][r.backendName] = r;
    }
  }

  bool hasMultiBackendParity = false;
  for (const auto &[bName, bMap] : parityMap) {
    if (bMap.size() > 1 && (bMap.count("Vulkan") || bMap.count("ROCm"))) {
      hasMultiBackendParity = true;
      break;
    }
  }

  if (hasMultiBackendParity) {
    auto computeParityVal = [](const ResultData &r) -> double {
      if (r.time_ms <= 0.0 || r.operations == 0)
        return 0.0;
      if (r.metric == "MItems/s") {
        return (static_cast<double>(r.operations) / (r.time_ms / 1000.0)) / 1e6;
      } else if (r.metric == "GB/s") {
        return (static_cast<double>(r.operations) / (r.time_ms / 1000.0)) / 1e9;
      } else if (r.metric == "us") {
        return (r.time_ms * 1000.0) / static_cast<double>(r.operations);
      } else if (r.metric == "ns") {
        return (r.time_ms * 1e6) / static_cast<double>(r.operations);
      }
      return (static_cast<double>(r.operations) / (r.time_ms / 1000.0)) / 1e12;
    };

    std::cout << "\n";
    std::cout << BOLD << CYAN << "  ╭─ Compiler Telemetry & Backend Parity Attribution ───────────────────────────╮" << RESET << "\n";
    for (const auto &[bName, bMap] : parityMap) {
      if (bMap.size() <= 1) continue;
      std::cout << BOLD << CYAN << "  │ " << RESET << BOLD << "Workload: " << bName << RESET << "\n";
      for (const auto &[backend, r] : bMap) {
        double val = computeParityVal(r);
        std::string telemStr;
        if (r.hasRegisterTelemetry && r.vgprCount > 0) {
          telemStr = " | " + std::to_string(r.vgprCount) + " VGPRs" +
                     (r.sgprCount > 0 ? (", " + std::to_string(r.sgprCount) + " SGPRs") : "") +
                     ", " + std::to_string(r.maxWavesPerSimd) + " waves/SIMD" +
                     (r.scratchSizeBytes > 0 ? (", " + std::to_string(r.scratchSizeBytes) + " B spill") : "");
        }
        if (r.hasPerfQueryTelemetry && r.perfCounters.waves > 0) {
          telemStr += " [HW: " + std::to_string(r.perfCounters.waves) + " waves, " +
                      formatDouble(r.perfCounters.valuBusyPct, 1) + "% VALU busy]";
        }
        std::cout << BOLD << CYAN << "  │ " << RESET << " • " << std::left << std::setw(8) << backend << ": "
                  << BOLD << GREEN << formatDouble(val, 2) << " " << r.metric << RESET
                  << DIM << telemStr << " (" << (!r.compilerTarget.empty() ? r.compilerTarget : backend) << ")" << RESET << "\n";
      }

      if (bMap.count("Vulkan") && bMap.count("ROCm")) {
        const auto &v = bMap.at("Vulkan");
        const auto &roc = bMap.at("ROCm");
        double vVal = computeParityVal(v);
        double rVal = computeParityVal(roc);
        if (vVal > 0.0 && rVal > 0.0) {
          bool lowerIsBetter = (v.metric == "us" || v.metric == "ns");
          double deltaPct = lowerIsBetter ? (((rVal - vVal) / rVal) * 100.0)
                                          : (((vVal - rVal) / vVal) * 100.0);
          std::cout << BOLD << CYAN << "  │ " << RESET << " • " << BOLD << "Parity Delta: " << RESET;
          if (std::abs(deltaPct) < 3.0) {
            std::cout << GREEN << "Within " << formatDouble(std::abs(deltaPct), 1) << "% (Full Parity)" << RESET << "\n";
          } else {
            std::cout << YELLOW << (deltaPct > 0 ? "Vulkan +" : "ROCm +") << formatDouble(std::abs(deltaPct), 1) << "% delta" << RESET;
            if (v.hasRegisterTelemetry && roc.hasRegisterTelemetry) {
              if (v.vgprCount != roc.vgprCount) {
                std::cout << " — attributed to compiler register allocation (ACO "
                          << v.vgprCount << " VGPRs vs LLVM " << roc.vgprCount << " VGPRs; "
                          << v.maxWavesPerSimd << " vs " << roc.maxWavesPerSimd << " waves/SIMD occupancy)";
              } else {
                std::cout << " — equal register allocation (" << v.vgprCount << " VGPRs); attributed to instruction scheduling / dual-issue packing";
              }
            }
            std::cout << "\n";
          }
        }
      }
    }
    std::cout << BOLD << CYAN << "  ╰──────────────────────────────────────────────────────────────────────────────╯" << RESET << "\n";
  }

  std::cout << std::endl;
}

void ResultFormatter::printComparison(const ImportedRun &runA, const ImportedRun &runB) {
  printComparison(std::vector<ImportedRun>{runA, runB});
}

void ResultFormatter::printComparison(const std::vector<ImportedRun> &runs) {
  if (runs.empty()) return;
  if (runs.size() == 1) {
    ResultFormatter fmt;
    for (const auto &r : runs[0].results) fmt.addResult(r);
    fmt.print();
    return;
  }

  const bool useColor = shouldUseColor();
  const std::string RESET = useColor ? "\033[0m" : "";
  const std::string BOLD = useColor ? "\033[1m" : "";
  const std::string DIM = useColor ? "\033[2m" : "";
  const std::string CYAN = useColor ? "\033[36m" : "";
  const std::string GREEN = useColor ? "\033[32m" : "";
  const std::string RED = useColor ? "\033[31m" : "";
  const std::string YELLOW = useColor ? "\033[33m" : "";
  const std::string MAGENTA = useColor ? "\033[35m" : "";
  const std::string BLUE = useColor ? "\033[34m" : "";

  const size_t N = runs.size();

  auto getShortNick = [](const std::string &fullName) -> std::string {
    std::string n = fullName;
    const std::vector<std::string> prefixes = {
      "AMD Radeon AI PRO ", "AMD Radeon RX ", "AMD Radeon ", "AMD ",
      "NVIDIA GeForce RTX ", "NVIDIA GeForce ", "NVIDIA RTX ", "NVIDIA "
    };
    for (const auto &p : prefixes) {
      if (n.rfind(p, 0) == 0) {
        n = n.substr(p.length());
        break;
      }
    }
    return n;
  };

  std::vector<std::string> devNames(N);
  std::vector<std::string> devNicks(N);
  const std::vector<std::string> devColors = useColor ? std::vector<std::string>{
    YELLOW, MAGENTA, CYAN, BLUE, "\033[38;5;208m", "\033[38;5;141m", GREEN
  } : std::vector<std::string>(7, "");

  for (size_t i = 0; i < N; ++i) {
    char letter = static_cast<char>('A' + i);
    devNames[i] = !runs[i].deviceProfile.deviceName.empty() ? runs[i].deviceProfile.deviceName : ("Device " + std::string(1, letter));
    devNicks[i] = getShortNick(devNames[i]);
  }

  const size_t w_wl = (N <= 2 ? 36 : (N == 3 ? 30 : 26));
  const size_t w_col = (N <= 2 ? 22 : (N == 3 ? 19 : 17));
  const size_t w_delta = (N <= 2 ? 38 : (N == 3 ? 42 : 36));

  std::vector<size_t> w_devs(N, w_col);
  size_t cardBoxWidth = w_wl + w_delta + 3 * (N + 1) + 1;
  for (size_t w : w_devs) cardBoxWidth += w;
  size_t cardInnerWidth = cardBoxWidth - 4;

  std::vector<std::string> shortHeaders(N);
  for (size_t i = 0; i < N; ++i) {
    char letter = static_cast<char>('A' + i);
    std::string sh = std::string(1, letter) + ": " + devNames[i];
    shortHeaders[i] = truncateUtf8(sh, w_devs[i]);
  }

  std::cout << std::endl;
  std::string reportTitle = (N == 2)
      ? " GPUBench Side-by-Side Comparative Benchmark Report "
      : " GPUBench Multi-Device Comparative Benchmark Report (" + std::to_string(N) + " GPUs) ";
  size_t dashCount = (cardBoxWidth > reportTitle.length() + 3) ? (cardBoxWidth - 3 - reportTitle.length()) : 10;

  std::cout << BOLD << CYAN << "  ╭─" << RESET << BOLD << reportTitle << RESET
            << BOLD << CYAN << repeatUtf8("─", dashCount) << "╮" << RESET << "\n";

  auto printHeaderLine = [&](const std::string &label, const std::string &val) {
    std::string line = label + val;
    size_t vLen = visualLength(line);
    size_t pad = (cardInnerWidth > vLen) ? (cardInnerWidth - vLen) : 0;
    std::cout << BOLD << CYAN << "  │ " << RESET
              << BOLD << label << RESET << MAGENTA << val << RESET
              << std::string(pad, ' ')
              << BOLD << CYAN << " │" << RESET << "\n";
  };

  for (size_t i = 0; i < N; ++i) {
    char letter = static_cast<char>('A' + i);
    std::string info = devNames[i];
    if (!runs[i].osName.empty() || !runs[i].cpuModel.empty()) {
      info += " [" + runs[i].osName + (runs[i].cpuModel.empty() ? "" : (" / " + runs[i].cpuModel)) + "]";
    }
    std::string lbl = "Run (" + std::string(1, letter) + ") : ";
    printHeaderLine(lbl, info);
  }
  printHeaderLine("Preset / API : ", runs[0].backend + " @ " + runs[0].resolution);

  std::cout << BOLD << CYAN << "  ╰─" << repeatUtf8("─", cardBoxWidth - 3) << "╯" << RESET << "\n";

  // Build aligned comparison structure
  struct CompWorkload {
    std::string cleanName;
    int sortWeight = 999;
    uint32_t configIndex = 0;
    std::vector<ResultData> results;
    std::vector<bool> hasResult;
    CompWorkload() {}
    CompWorkload(size_t n) : results(n), hasResult(n, false) {}
  };

  struct CompSubcategory {
    std::string name;
    int minSortWeight = 999;
    std::map<std::pair<int, uint32_t>, CompWorkload> workloads;
  };

  std::map<std::pair<int, std::string>, std::map<std::string, CompSubcategory>> compData;

  for (size_t runIdx = 0; runIdx < N; ++runIdx) {
    for (const auto &res : runs[runIdx].results) {
      int compWeight = 5;
      if (res.component == "Compute") compWeight = 1;
      else if (res.component == "Memory") compWeight = 2;
      else if (res.component == "Scheduling") compWeight = 3;
      else if (res.component == "Rasterization & ROP") compWeight = 4;
      else if (res.component == "Ray Tracing") compWeight = 5;

      std::string cName = cleanWorkloadName(res.benchmarkName, res.subcategory);
      std::string effectiveSubcat = res.subcategory;
      std::string sceneName = extractSceneName(res.benchmarkName);
      if (!sceneName.empty()) {
        effectiveSubcat = res.subcategory + " [" + sceneName + "]";
      }
      auto &subcat = compData[std::make_pair(compWeight, res.component)][effectiveSubcat];
      subcat.name = effectiveSubcat;
      subcat.minSortWeight = (std::min)(subcat.minSortWeight, res.sortWeight);

      auto key = std::make_pair(res.sortWeight, res.configIndex);
      if (subcat.workloads.find(key) == subcat.workloads.end()) {
        subcat.workloads[key] = CompWorkload(N);
      }
      auto &wl = subcat.workloads[key];
      wl.cleanName = cName;
      wl.sortWeight = res.sortWeight;
      wl.configIndex = res.configIndex;
      if (wl.results.size() < N) {
        wl.results.resize(N);
        wl.hasResult.resize(N, false);
      }
      wl.results[runIdx] = res;
      wl.hasResult[runIdx] = true;
    }
  }

  struct WorkloadVal {
    double val = 0.0;
    std::string formattedStr = "—";
    std::string metric;
    bool isValid = false;
    bool isUnsupported = false;
    bool isLatency = false;
  };

  auto getWorkloadVal = [](const ResultData &res, bool present) -> WorkloadVal {
    WorkloadVal wv;
    if (!present) return wv;
    if (res.isUnsupported) {
      wv.isUnsupported = true;
      wv.formattedStr = "UNSUPPORTED";
      return wv;
    }
    if (!res.isValid) {
      wv.formattedStr = "INVALID";
      wv.isValid = false;
      return wv;
    }
    if (res.time_ms <= 0.0 || res.operations == 0) {
      wv.formattedStr = "N/A";
      return wv;
    }

    double time_s = res.time_ms / 1000.0;
    double rawRate = static_cast<double>(res.operations) / time_s;
    int prec = 2;

    if (res.component == "Compute") {
      if (res.metric == "MItems/s" || res.metric == "MSamples/s" || res.metric == "MBVH/s") {
        wv.val = rawRate / 1e6;
        wv.metric = res.metric;
      } else if (res.metric == "GB/s" || res.metric == "GOPS" || res.metric == "GFLOPS") {
        wv.val = rawRate / 1e9;
        wv.metric = res.metric;
      } else if (res.metric == "us") {
        wv.val = (res.time_ms * 1000.0) / static_cast<double>(res.operations);
        wv.isLatency = true;
        wv.metric = "us";
        wv.isValid = true;
        wv.formattedStr = formatDouble(wv.val, 2) + " us";
        return wv;
      } else {
        wv.val = rawRate / 1e12;
        wv.metric = res.metric.empty() ? "TFLOPS" : res.metric;
      }
    } else if (res.component == "Memory") {
      if (res.subcategory == "Latency" || res.subcategory == "Cache Latency Curve" || res.metric == "ns" || res.benchmarkName.find("Latency") != std::string::npos) {
        wv.val = (res.time_ms * 1e6) / static_cast<double>(res.operations);
        wv.isLatency = true;
        wv.metric = "ns";
        wv.isValid = true;
        wv.formattedStr = formatDouble(wv.val, 2) + " ns";
        return wv;
      } else if (res.metric == "TB/s") {
        wv.val = rawRate / 1e12;
        wv.metric = "TB/s";
      } else {
        wv.val = rawRate / 1e9;
        wv.metric = "GB/s";
      }
    } else {
      if (res.metric == "us") {
        wv.val = (res.time_ms * 1000.0) / static_cast<double>(res.operations);
        wv.isLatency = true;
        wv.metric = "us";
      } else if (res.metric == "ns") {
        wv.val = (res.time_ms * 1e6) / static_cast<double>(res.operations);
        wv.isLatency = true;
        wv.metric = "ns";
      } else if (res.metric == "ms") {
        wv.val = res.time_ms / static_cast<double>(res.operations);
        wv.isLatency = true;
        wv.metric = "ms";
      } else if (res.metric == "TB/s" || res.metric == "TFLOPS" || res.metric == "TOPS") {
        wv.val = rawRate / 1e12;
        wv.metric = res.metric;
      } else if (res.metric == "GIS/s" || res.metric == "GRays/s" || res.metric == "GPixels/s" ||
                 res.metric == "GB/s" || res.metric == "GOPS" || res.metric == "GFLOPS") {
        wv.val = rawRate / 1e9;
        if (res.metric == "GRays/s") prec = 3;
        wv.metric = res.metric;
      } else {
        wv.val = rawRate / 1e6;
        wv.metric = res.metric;
      }
    }
    wv.isValid = true;
    wv.formattedStr = formatDouble(wv.val, prec) + " " + wv.metric;
    return wv;
  };

  // Tracking metrics for Executive Summary
  std::vector<int> winCount(N, 0);
  int totalContestedWorkloads = 0;

  std::vector<double> fp32Rates(N, 0.0);
  std::vector<double> int8Rates(N, 0.0);
  std::vector<double> vramBwRates(N, 0.0);
  std::vector<double> l0LatRates(N, 0.0);
  std::vector<double> ptTradRates(N, 0.0);
  std::vector<double> matTradRates(N, 0.0);
  std::vector<double> matWlRates(N, 0.0);

  for (const auto &compPair : compData) {
    const std::string &compName = compPair.first.second;
    std::cout << "\n  [" << BOLD << CYAN << compName << RESET << "]" << std::endl;

    std::vector<CompSubcategory> sortedSubcats;
    for (const auto &sp : compPair.second) sortedSubcats.push_back(sp.second);
    std::sort(sortedSubcats.begin(), sortedSubcats.end(),
              [](const CompSubcategory &a, const CompSubcategory &b) {
                if (a.minSortWeight != b.minSortWeight)
                  return a.minSortWeight < b.minSortWeight;
                int orderA = getSceneOrder(a.name);
                int orderB = getSceneOrder(b.name);
                if (orderA != orderB)
                  return orderA < orderB;
                return a.name < b.name;
              });

    for (const auto &subcat : sortedSubcats) {
      std::string title = " " + subcat.name + " ";
      size_t subcatDashCount = (cardBoxWidth > title.length() + 3)
                                   ? (cardBoxWidth - 3 - title.length())
                                   : 10;

      // Top Border
      std::cout << BOLD << CYAN << "  ╭─" << RESET << BOLD << title << RESET
                << BOLD << CYAN << repeatUtf8("─", subcatDashCount) << "╮" << RESET << "\n";

      // Header Row
      std::cout << BOLD << CYAN << "  │ " << RESET << BOLD << std::left << std::setw(w_wl) << "Workload" << RESET;
      for (size_t i = 0; i < N; ++i) {
        std::string color = devColors[i % devColors.size()];
        std::cout << BOLD << CYAN << " │ " << RESET << color << std::right << std::setw(w_devs[i]) << shortHeaders[i] << RESET;
      }
      std::cout << BOLD << CYAN << " │ " << RESET << BOLD << std::left << std::setw(w_delta) << "Fastest / Comparison (Lead)" << RESET
                << BOLD << CYAN << " │" << RESET << "\n";

      // Divider Row
      std::cout << BOLD << CYAN << "  ├─" << repeatUtf8("─", w_wl);
      for (size_t i = 0; i < N; ++i) {
        std::cout << "─┼─" << repeatUtf8("─", w_devs[i]);
      }
      std::cout << "─┼─" << repeatUtf8("─", w_delta) << "─┤" << RESET << "\n";

      for (const auto &wlPair : subcat.workloads) {
        const auto &wl = wlPair.second;
        std::string dispName = truncateUtf8(wl.cleanName, w_wl);

        std::vector<WorkloadVal> vals(N);
        int bestIdx = -1;
        int secondIdx = -1;
        double bestVal = 0.0;
        double secondVal = 0.0;
        int validCount = 0;
        bool isLatency = false;

        for (size_t i = 0; i < N; ++i) {
          vals[i] = getWorkloadVal(wl.results[i], wl.hasResult[i]);
          if (!vals[i].isValid) continue;
          validCount++;
          if (vals[i].isLatency) isLatency = true;

          if (bestIdx == -1) {
            bestIdx = static_cast<int>(i);
            bestVal = vals[i].val;
          } else {
            bool isBetter = isLatency ? (vals[i].val < bestVal) : (vals[i].val > bestVal);
            if (isBetter) {
              secondIdx = bestIdx;
              secondVal = bestVal;
              bestIdx = static_cast<int>(i);
              bestVal = vals[i].val;
            } else if (secondIdx == -1) {
              secondIdx = static_cast<int>(i);
              secondVal = vals[i].val;
            } else {
              bool isSecondBetter = isLatency ? (vals[i].val < secondVal) : (vals[i].val > secondVal);
              if (isSecondBetter) {
                secondIdx = static_cast<int>(i);
                secondVal = vals[i].val;
              }
            }
          }

          // Key metrics tracking for executive summary
          if (wl.cleanName == "FP32") fp32Rates[i] = vals[i].val;
          if (wl.cleanName.find("INT8") != std::string::npos || (subcat.name.find("INT8") != std::string::npos && wl.configIndex == 1)) {
            int8Rates[i] = vals[i].val;
          }
          if (subcat.name == "Device Memory Bandwidth" && wl.configIndex == 0) vramBwRates[i] = vals[i].val;
          if (subcat.name == "Latency" && wl.cleanName.find("L0") != std::string::npos) l0LatRates[i] = vals[i].val;
          if (subcat.name.find("Scene Path Tracing (Multi-Bounce)") != std::string::npos && wl.configIndex == 3) {
            if (ptTradRates[i] == 0.0 || subcat.name.find("Indoor") != std::string::npos) {
              ptTradRates[i] = vals[i].val;
            }
          }
          if (subcat.name.find("Material Shading") != std::string::npos) {
            if (wl.configIndex == 0 && (matTradRates[i] == 0.0 || subcat.name.find("Showroom") != std::string::npos)) {
              matTradRates[i] = vals[i].val;
            } else if (wl.configIndex == 2 && (matWlRates[i] == 0.0 || subcat.name.find("Showroom") != std::string::npos)) {
              matWlRates[i] = vals[i].val;
            }
          }
        }

        std::string deltaStr;
        std::string deltaColor = RESET;

        if (validCount >= 2 && bestIdx >= 0 && secondIdx >= 0) {
          char winLetter = static_cast<char>('A' + bestIdx);
          char secLetter = static_cast<char>('A' + secondIdx);

          double realDelta = 0.0;
          double ratio = 1.0;

          if (isLatency) {
            realDelta = secondVal - bestVal;
            ratio = (bestVal > 0.0) ? (secondVal / bestVal) : 1.0;
          } else {
            realDelta = bestVal - secondVal;
            ratio = (secondVal > 0.0) ? (bestVal / secondVal) : 1.0;
          }

          if (ratio <= 1.005 && std::abs(realDelta) < 0.01) {
            deltaStr = "Parity / Tie (" + std::string(1, winLetter) + " & " + std::string(1, secLetter) + ")";
            deltaColor = DIM;
          } else {
            winCount[bestIdx]++;
            totalContestedWorkloads++;

            std::string unit = vals[bestIdx].metric;
            int prec = (std::abs(realDelta) >= 100.0) ? 1 : 2;
            std::string realStr = (isLatency ? "-" : "+") + formatDouble(std::abs(realDelta), prec) + " " + unit;
            std::string relRatioStr = formatDouble(ratio, 2) + "x";

            if (N == 2) {
              deltaStr = std::string(1, winLetter) + " is " + relRatioStr + " (" + realStr + ")";
            } else {
              deltaStr = std::string(1, winLetter) + " leads: " + realStr + " (" + relRatioStr + " vs " + std::string(1, secLetter) + ")";
            }
            deltaColor = GREEN;
          }
        } else if (validCount == 1 && bestIdx >= 0) {
          char winLetter = static_cast<char>('A' + bestIdx);
          deltaStr = "Only on Run " + std::string(1, winLetter);
          deltaColor = DIM;
        } else {
          deltaStr = "—";
          deltaColor = DIM;
        }

        // Print row
        size_t nameVis = visualLength(dispName);
        std::string namePad = (w_wl > nameVis) ? std::string(w_wl - nameVis, ' ') : "";

        std::cout << BOLD << CYAN << "  │ " << RESET << dispName << namePad;

        for (size_t i = 0; i < N; ++i) {
          std::string str = truncateUtf8(vals[i].formattedStr, w_devs[i], "");

          std::string cellColor = devColors[i % devColors.size()];
          if (vals[i].isUnsupported) {
            cellColor = RED;
          } else if (vals[i].isValid) {
            if (static_cast<int>(i) == bestIdx && validCount >= 2 && deltaColor == GREEN) {
              cellColor = BOLD + GREEN;
            } else {
              cellColor = devColors[i % devColors.size()];
            }
          } else {
            cellColor = DIM;
          }

          std::cout << BOLD << CYAN << " │ " << RESET << cellColor << std::right << std::setw(w_devs[i]) << str << RESET;
        }

        deltaStr = truncateUtf8(deltaStr, w_delta);
        size_t dVis = visualLength(deltaStr);
        std::string dPad = (w_delta > dVis) ? std::string(w_delta - dVis, ' ') : "";

        std::cout << BOLD << CYAN << " │ " << RESET << deltaColor << BOLD << deltaStr << dPad << RESET
                  << BOLD << CYAN << " │" << RESET << "\n";
      }

      // Bottom Border
      std::cout << BOLD << CYAN << "  ╰─" << repeatUtf8("─", w_wl);
      for (size_t i = 0; i < N; ++i) {
        std::cout << "─┴─" << repeatUtf8("─", w_devs[i]);
      }
      std::cout << "─┴─" << repeatUtf8("─", w_delta) << "─╯" << RESET << "\n";
    }
  }

  // Executive Comparison Summary Card
  std::cout << "\n";
  std::string cardTitle = " Executive Comparison Summary & Head-to-Head Takeaways ";
  size_t dashC = (cardBoxWidth > cardTitle.length() + 3) ? (cardBoxWidth - 3 - cardTitle.length()) : 10;

  std::cout << BOLD << CYAN << "  ╭─" << RESET << BOLD << cardTitle << RESET
            << BOLD << CYAN << repeatUtf8("─", dashC) << "╮" << RESET << "\n";

  auto printRow = [&](const std::string &label, const std::string &val, const std::string &extra) {
    std::string line = "• " + label + " : " + val + extra;
    size_t vLen = visualLength(line);
    size_t pad = (cardInnerWidth > vLen) ? (cardInnerWidth - vLen) : 0;
    std::cout << BOLD << CYAN << "  │ " << RESET
              << BOLD << "• " << label << RESET << " : "
              << val << extra
              << std::string(pad, ' ')
              << BOLD << CYAN << " │" << RESET << "\n";
  };

  // 1. Overall Tally Scoreboard
  if (totalContestedWorkloads > 0) {
    std::string tallyStr = "";
    for (size_t i = 0; i < N; ++i) {
      char letter = static_cast<char>('A' + i);
      std::string color = devColors[i % devColors.size()];
      tallyStr += (i > 0 ? " | " : "") + color + BOLD + std::string(1, letter) + " (" + devNicks[i] + "): " +
                  std::to_string(winCount[i]) + " wins" + RESET;
    }
    printRow("Overall Workload Win Tally  ", tallyStr, " (of " + std::to_string(totalContestedWorkloads) + " compared tests)");
  }

  // Helper for pillar comparisons across N runs
  auto comparePillar = [&](const std::string &label, const std::vector<double> &rates, const std::string &unit, bool lowerIsBetter = false) {
    int best = -1, sec = -1;
    double bVal = 0.0, sVal = 0.0;
    for (size_t i = 0; i < N; ++i) {
      if (rates[i] <= 0.0) continue;
      if (best == -1) {
        best = static_cast<int>(i);
        bVal = rates[i];
      } else {
        bool better = lowerIsBetter ? (rates[i] < bVal) : (rates[i] > bVal);
        if (better) {
          sec = best; sVal = bVal;
          best = static_cast<int>(i); bVal = rates[i];
        } else if (sec == -1) {
          sec = static_cast<int>(i); sVal = rates[i];
        } else {
          bool sBetter = lowerIsBetter ? (rates[i] < sVal) : (rates[i] > sVal);
          if (sBetter) { sec = static_cast<int>(i); sVal = rates[i]; }
        }
      }
    }
    if (best >= 0 && sec >= 0 && bVal > 0.0 && sVal > 0.0) {
      char bLetter = static_cast<char>('A' + best);
      double ratio = lowerIsBetter ? (sVal / bVal) : (bVal / sVal);
      double realLead = lowerIsBetter ? (sVal - bVal) : (bVal - sVal);
      std::string color = devColors[best % devColors.size()];
      std::string valStr = BOLD + color + std::string(1, bLetter) + " (" + devNicks[best] + ")" + RESET +
                           " leads by " + BOLD + GREEN + formatDouble(ratio, 2) + "x" + RESET +
                           " (" + (lowerIsBetter ? "-" : "+") + formatDouble(std::abs(realLead), 1) + " " + unit + ")";

      std::string details = " [";
      for (size_t i = 0; i < N; ++i) {
        if (rates[i] <= 0.0) continue;
        char l = static_cast<char>('A' + i);
        if (details.length() > 2) details += " vs ";
        details += std::string(1, l) + ": " + formatDouble(rates[i], 1);
      }
      details += " " + unit + "]";
      printRow(label, valStr, details);
    }
  };

  comparePillar("Raw FP32 Compute Throughput ", fp32Rates, "TFLOPS");
  comparePillar("INT8 Tensor Matrix Throughput", int8Rates, "TOPS");
  comparePillar("Device VRAM Memory Bandwidth ", vramBwRates, "GB/s");
  comparePillar("Lowest L0 Cache Latency      ", l0LatRates, "ns", true);
  comparePillar("Path Tracing (Megakernel)    ", ptTradRates, "MRays/s");

  // Wavefront Scheduling Scaling
  std::string schedStr = "";
  for (size_t i = 0; i < N; ++i) {
    if (matTradRates[i] > 0.0 && matWlRates[i] > 0.0) {
      char l = static_cast<char>('A' + i);
      double scale = (matWlRates[i] / matTradRates[i] - 1.0) * 100.0;
      std::string color = devColors[i % devColors.size()];
      schedStr += (schedStr.empty() ? "" : " | ") + color + std::string(1, l) + ": " +
                  (scale >= 0 ? "+" : "") + formatDouble(scale, 1) + "%" + RESET;
    }
  }
  if (!schedStr.empty()) {
    printRow("Wavefront Scheduling Scaling ", schedStr, " (Material Shading)");
  }

  std::cout << BOLD << CYAN << "  ╰─" << repeatUtf8("─", cardBoxWidth - 3) << "╯" << RESET << "\n\n";
}

namespace {

void addFixed6(yyjson_mut_doc *doc, yyjson_mut_val *obj, const char *key,
               double val) {
  char buf[64];
  const double safeVal = std::isfinite(val) ? val : 0.0;
  int len = std::snprintf(buf, sizeof(buf), "%.6f", safeVal);
  if (len <= 0 || static_cast<size_t>(len) >= sizeof(buf)) {
    yyjson_mut_obj_add_val(doc, obj, key, yyjson_mut_rawncpy(doc, "0.000000", 8));
    return;
  }
  yyjson_mut_obj_add_val(doc, obj, key,
                         yyjson_mut_rawncpy(doc, buf, static_cast<size_t>(len)));
}

void addFixed6DynamicKey(yyjson_mut_doc *doc, yyjson_mut_val *obj,
                         const std::string &key, double val) {
  char buf[64];
  const double safeVal = std::isfinite(val) ? val : 0.0;
  int len = std::snprintf(buf, sizeof(buf), "%.6f", safeVal);
  yyjson_mut_val *k = yyjson_mut_strcpy(doc, key.c_str());
  yyjson_mut_val *v = (len > 0 && static_cast<size_t>(len) < sizeof(buf))
                          ? yyjson_mut_rawncpy(doc, buf, static_cast<size_t>(len))
                          : yyjson_mut_rawncpy(doc, "0.000000", 8);
  yyjson_mut_obj_add(obj, k, v);
}

struct HostSystemSummary {
  std::string osName;
  std::string cpuModel;
  uint32_t cpuLogicalCores = 0;
  double totalRamGb = 0.0;
};

HostSystemSummary collectHostSystemSummary() {
  HostSystemSummary s;
  s.cpuLogicalCores = std::thread::hardware_concurrency();
#if defined(_WIN32)
  s.osName = "Windows";
  MEMORYSTATUSEX mem{};
  mem.dwLength = sizeof(mem);
  if (GlobalMemoryStatusEx(&mem)) {
    s.totalRamGb = static_cast<double>(mem.ullTotalPhys) /
                   (1024.0 * 1024.0 * 1024.0);
  }
  char cpuName[256] = {0};
  DWORD bufSize = sizeof(cpuName);
  if (RegGetValueA(HKEY_LOCAL_MACHINE,
                   "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                   "ProcessorNameString", RRF_RT_REG_SZ, nullptr, cpuName,
                   &bufSize) == ERROR_SUCCESS) {
    s.cpuModel = cpuName;
    while (!s.cpuModel.empty() &&
           std::isspace(static_cast<unsigned char>(s.cpuModel.back()))) {
      s.cpuModel.pop_back();
    }
  }
#else
  s.osName = "Linux";
  long pages = sysconf(_SC_PHYS_PAGES);
  long pageSize = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && pageSize > 0) {
    s.totalRamGb = (static_cast<double>(pages) * static_cast<double>(pageSize)) /
                   (1024.0 * 1024.0 * 1024.0);
  }
  if (FILE *f = std::fopen("/proc/cpuinfo", "r")) {
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
      std::string ln(line);
      if (ln.rfind("model name", 0) == 0) {
        size_t colon = ln.find(':');
        if (colon != std::string::npos) {
          std::string model = ln.substr(colon + 1);
          size_t first = model.find_first_not_of(" \t\r\n");
          size_t last = model.find_last_not_of(" \t\r\n");
          if (first != std::string::npos && last != std::string::npos) {
            s.cpuModel = model.substr(first, last - first + 1);
          }
          break;
        }
      }
    }
    std::fclose(f);
  }
#endif
  if (s.cpuModel.empty()) {
    s.cpuModel = "Host CPU";
  }
  return s;
}

} // namespace

double computeResultValue(const ResultData &r) {
  double value = 0.0;
  if (r.time_ms > 0.0 && r.operations > 0) {
    double seconds = r.time_ms / 1000.0;
    if (r.metric == "TFLOPS" || r.metric == "TOPS" || r.metric == "TB/s") {
      value = (static_cast<double>(r.operations) / seconds) / 1e12;
    } else if (r.metric == "GB/s" || r.metric == "GIS/s" ||
               r.metric == "GRays/s" || r.metric == "GPixels/s" ||
               r.metric == "GOPS" || r.metric == "GFLOPS") {
      value = (static_cast<double>(r.operations) / seconds) / 1e9;
    } else if (r.metric == "MRays/s" || r.metric == "MHits/s" ||
               r.metric == "MTris/s" || r.metric == "MInst/s" ||
               r.metric == "MRecords/s" || r.metric == "MItems/s" ||
               r.metric == "MSamples/s" || r.metric == "MBVH/s" ||
               r.metric == "MPixels/s") {
      value = (static_cast<double>(r.operations) / seconds) / 1e6;
    } else if (r.metric == "us") {
      value = (r.time_ms * 1000.0) / static_cast<double>(r.operations);
    } else if (r.metric == "ns") {
      value = (r.time_ms * 1e6) / static_cast<double>(r.operations);
    } else if (r.metric == "ms") {
      value = r.time_ms / static_cast<double>(r.operations);
    }
  }
  return value;
}

std::string getDefaultJsonFilename() {
  char hostname[256] = {0};
#if defined(_WIN32)
  DWORD size = sizeof(hostname);
  if (!GetComputerNameA(hostname, &size)) {
    hostname[0] = '\0';
  }
#else
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    hostname[0] = '\0';
  }
#endif
  std::string host(hostname);
  for (char &c : host) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
      c = '_';
    }
  }

  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif

  char timeStr[64];
  std::strftime(timeStr, sizeof(timeStr), "%Y%m%d_%H%M%S", &tm);

  if (!host.empty() && host != "localhost") {
    return "gpubench_" + host + "_" + std::string(timeStr) + ".json";
  }
  return "gpubench_" + std::string(timeStr) + ".json";
}

std::string resultsToJson(const std::vector<ResultData> &results) {
  return resultsToJson(results, GetDeviceProfilesAPI(), nullptr);
}

std::string resultsToJson(const std::vector<ResultData> &results,
                          const std::vector<DeviceProfile> &profiles) {
  return resultsToJson(results, profiles, nullptr);
}

std::string resultsToJson(const std::vector<ResultData> &results,
                          const std::vector<DeviceProfile> &profiles,
                          const UnifiedSessionMetadata *sessionMeta) {
  yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
  if (!doc) {
    return "{}";
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc);
  yyjson_mut_doc_set_root(doc, root);

  const uint64_t nowTs = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  std::string topBackend = "Vulkan";
  for (const auto &r : results) {
    if (!r.backendName.empty() && r.backendName != "System") {
      topBackend = r.backendName;
      break;
    }
  }
  if (topBackend == "Vulkan" && !profiles.empty() && !profiles[0].backend.empty()) {
    topBackend = profiles[0].backend;
  }

  uint32_t topW = 0, topH = 0;
  for (const auto &r : results) {
    if (r.width > 0 && r.height > 0) {
      topW = r.width;
      topH = r.height;
      break;
    }
  }
  const std::string topRes =
      (topW > 0 && topH > 0)
          ? (std::to_string(topW) + "x" + std::to_string(topH))
          : "3840x2160";

  yyjson_mut_obj_add_strcpy(doc, root, "schema_version", "2.0");
  yyjson_mut_obj_add_strcpy(doc, root, "version", GPUBENCH_VERSION);
  yyjson_mut_obj_add_strcpy(doc, root, "app_version", GPUBENCH_VERSION);
  yyjson_mut_obj_add_uint(doc, root, "timestamp", nowTs);
  yyjson_mut_obj_add_strcpy(doc, root, "backend", topBackend.c_str());
  yyjson_mut_obj_add_strcpy(doc, root, "resolution", topRes.c_str());

  const HostSystemSummary hostSys = collectHostSystemSummary();
  yyjson_mut_val *sysObj = yyjson_mut_obj_add_obj(doc, root, "system_info");
  yyjson_mut_obj_add_strcpy(doc, sysObj, "os_name", hostSys.osName.c_str());
  yyjson_mut_obj_add_strcpy(doc, sysObj, "cpu_model", hostSys.cpuModel.c_str());
  yyjson_mut_obj_add_uint(doc, sysObj, "cpu_logical_cores", hostSys.cpuLogicalCores);
  addFixed6(doc, sysObj, "total_ram_gb", hostSys.totalRamGb);

  if (sessionMeta != nullptr) {
    yyjson_mut_val *sessObj = yyjson_mut_obj_add_obj(doc, root, "gui_session");
    yyjson_mut_obj_add_strcpy(doc, sessObj, "scene", sessionMeta->scene.c_str());
    yyjson_mut_obj_add_uint(doc, sessObj, "samples_per_pixel", sessionMeta->samplesPerPixel);
    yyjson_mut_obj_add_bool(doc, sessObj, "dump_renders", sessionMeta->dumpRenders);
    addFixed6(doc, sessObj, "ui_scale", sessionMeta->uiScale);
  }

  yyjson_mut_val *profArr = yyjson_mut_obj_add_arr(doc, root, "device_profiles");
  for (const auto &dp : profiles) {
    char vendorHex[16], deviceHex[16];
    std::snprintf(vendorHex, sizeof(vendorHex), "0x%04X", dp.vendorID);
    std::snprintf(deviceHex, sizeof(deviceHex), "0x%04X", dp.deviceID);

    yyjson_mut_val *pObj = yyjson_mut_arr_add_obj(doc, profArr);
    yyjson_mut_obj_add_strcpy(doc, pObj, "backend", dp.backend.c_str());
    yyjson_mut_obj_add_uint(doc, pObj, "device_index", dp.deviceIndex);
    yyjson_mut_obj_add_strcpy(doc, pObj, "device_name", dp.deviceName.c_str());
    yyjson_mut_obj_add_strcpy(doc, pObj, "vendor_id", vendorHex);
    yyjson_mut_obj_add_strcpy(doc, pObj, "device_id", deviceHex);
    yyjson_mut_obj_add_strcpy(doc, pObj, "driver_name", dp.driverName.c_str());
    yyjson_mut_obj_add_strcpy(doc, pObj, "driver_info", dp.driverInfo.c_str());
    yyjson_mut_obj_add_strcpy(doc, pObj, "driver_version", dp.driverVersion.c_str());
    yyjson_mut_obj_add_strcpy(doc, pObj, "api_version", dp.apiVersion.c_str());
    yyjson_mut_obj_add_uint(doc, pObj, "vram_total_mb", dp.vramTotalMb);
    yyjson_mut_obj_add_uint(doc, pObj, "compute_units", dp.computeUnits);
    yyjson_mut_obj_add_uint(doc, pObj, "subgroup_size", dp.subgroupSize);
    yyjson_mut_obj_add_uint(doc, pObj, "max_workgroup_size", dp.maxWorkGroupSize);
    yyjson_mut_obj_add_bool(doc, pObj, "is_apu", dp.isApu);
    yyjson_mut_obj_add_bool(doc, pObj, "rebar_supported", dp.rebarSupported);
    yyjson_mut_obj_add_uint(doc, pObj, "rebar_size_mb", dp.rebarSizeBytes / (1024ULL * 1024ULL));
    yyjson_mut_obj_add_bool(doc, pObj, "ray_tracing_supported", dp.rayTracingSupported);
    yyjson_mut_obj_add_bool(doc, pObj, "ser_supported", dp.serSupported);
    yyjson_mut_obj_add_bool(doc, pObj, "opacity_micromap_supported", dp.opacityMicromapSupported);
    yyjson_mut_obj_add_bool(doc, pObj, "work_graphs_supported", dp.workGraphsSupported);
    yyjson_mut_obj_add_bool(doc, pObj, "cooperative_matrix_supported", dp.cooperativeMatrixSupported);
    yyjson_mut_obj_add_bool(doc, pObj, "structured_sparsity_supported", dp.structuredSparsitySupported);
    yyjson_mut_obj_add_bool(doc, pObj, "cooperative_vector_supported", dp.cooperativeVectorSupported);
    yyjson_mut_obj_add_bool(doc, pObj, "float16_supported", dp.float16Supported);
    yyjson_mut_obj_add_bool(doc, pObj, "int8_supported", dp.int8Supported);
    yyjson_mut_obj_add_bool(doc, pObj, "ocp_fp6_supported", dp.ocpFp6Supported);
    yyjson_mut_obj_add_bool(doc, pObj, "ocp_fp4_supported", dp.ocpFp4Supported);
    yyjson_mut_obj_add_bool(doc, pObj, "performance_query_supported", dp.performanceQuerySupported);

    const auto &prof = DeviceDatabase::lookup(dp.vendorID, dp.deviceID, dp.deviceName);
    yyjson_mut_obj_add_strcpy(doc, pObj, "architecture", prof.archName.c_str());
    yyjson_mut_obj_add_strcpy(doc, pObj, "memory_type", prof.memoryType.c_str());
    if (prof.theoreticalFp32Tflops > 0.0) {
      addFixed6(doc, pObj, "theoretical_fp32_tflops", prof.theoreticalFp32Tflops);
    }
    if (prof.theoreticalBandwidthGBps > 0.0) {
      addFixed6(doc, pObj, "theoretical_bandwidth_gbps", prof.theoreticalBandwidthGBps);
    }
    if (prof.theoreticalTriangleGis > 0.0) {
      addFixed6(doc, pObj, "theoretical_triangle_gis", prof.theoreticalTriangleGis);
      addFixed6(doc, pObj, "theoretical_box_gis", prof.theoreticalBoxGis);
    }

    const GpuContentionInfo cont = HardwareTelemetry::checkContention(dp.deviceIndex);
    yyjson_mut_val *contObj = yyjson_mut_obj_add_obj(doc, pObj, "hardware_contention");
    yyjson_mut_obj_add_bool(doc, contObj, "detected", cont.hasContention);
    yyjson_mut_obj_add_bool(doc, contObj, "is_critical", cont.isCritical);
    yyjson_mut_obj_add_uint(doc, contObj, "gpu_busy_percent", cont.gpuBusyPct);
    yyjson_mut_obj_add_uint(doc, contObj, "vram_used_mb", cont.vramUsedMb);
    yyjson_mut_obj_add_uint(doc, contObj, "vram_total_mb", cont.vramTotalMb);
    yyjson_mut_obj_add_uint(doc, contObj, "gtt_used_mb", cont.gttUsedMb);
    addFixed6(doc, contObj, "cpu_load_1min", cont.cpuLoad1Min);
    yyjson_mut_val *reasonsArr = yyjson_mut_obj_add_arr(doc, contObj, "reasons");
    for (const auto &reason : cont.reasons) {
      yyjson_mut_arr_add_strcpy(doc, reasonsArr, reason.c_str());
    }
  }

  yyjson_mut_val *resArr = yyjson_mut_obj_add_arr(doc, root, "results");
  for (const ResultData &r : results) {
    yyjson_mut_val *rObj = yyjson_mut_arr_add_obj(doc, resArr);
    const double value = computeResultValue(r);

    yyjson_mut_obj_add_strcpy(doc, rObj, "backend", r.backendName.c_str());
    yyjson_mut_obj_add_strcpy(doc, rObj, "device", r.deviceName.c_str());
    if (r.deviceIndex == 0xFFFFFFFF || r.backendName == "System") {
      yyjson_mut_obj_add_null(doc, rObj, "device_index");
    } else {
      yyjson_mut_obj_add_uint(doc, rObj, "device_index", r.deviceIndex);
    }
    yyjson_mut_obj_add_strcpy(doc, rObj, "benchmark", r.benchmarkName.c_str());
    yyjson_mut_obj_add_strcpy(doc, rObj, "component", r.component.c_str());
    yyjson_mut_obj_add_strcpy(doc, rObj, "subcategory", r.subcategory.c_str());
    yyjson_mut_obj_add_strcpy(doc, rObj, "metric", r.metric.c_str());
    yyjson_mut_obj_add_strcpy(doc, rObj, "metric_unit", r.metric.c_str());
    addFixed6(doc, rObj, "value", value);
    if (r.operationalIntensity > 0.0) {
      addFixed6(doc, rObj, "operational_intensity", r.operationalIntensity);
    }

    if (r.benchmarkName.find("RayScheduling") != std::string::npos && r.metric == "MRays/s") {
      const uint32_t w = r.width ? r.width : 1920;
      const uint32_t h = r.height ? r.height : 1080;
      const double fps = (value * 1e6) / static_cast<double>(w * h);
      addFixed6(doc, rObj, "fps", fps);
      const std::string resStr = std::to_string(w) + "x" + std::to_string(h);
      yyjson_mut_obj_add_strcpy(doc, rObj, "resolution", resStr.c_str());
    }

    if (r.benchmarkName.find("RayRawTraversal") != std::string::npos ||
        r.subcategory == "Hardware BVH Traversal") {
      const double time_s = r.time_ms / 1000.0;
      double throughputGis = 0.0;
      if (time_s > 0.0) {
        const uint64_t ops = (r.configIndex == 0) ? r.operations : (r.operations * 64);
        throughputGis = (static_cast<double>(ops) / time_s) / 1e9;
      }
      yyjson_mut_obj_add_strcpy(doc, rObj, "peak_type", r.configIndex == 0 ? "Triangle" : "Box");
      yyjson_mut_obj_add_strcpy(doc, rObj, "traversal_model",
                                r.configIndex == 0 ? "coherent_single_layer_triangle"
                                                   : "synthetic_64_layer_aabb");
      yyjson_mut_obj_add_strcpy(doc, rObj, "synthetic_model",
                                r.configIndex == 0 ? "coherent_single_layer_triangle"
                                                   : "synthetic_64_layer_aabb");
      yyjson_mut_obj_add_uint(doc, rObj, "model_box_tests_per_ray",
                              r.configIndex == 0 ? 1u : 64u);
      yyjson_mut_obj_add_strcpy(
          doc, rObj, "synthetic_disclosure",
          r.configIndex == 0
              ? "Measures sustained hardware triangle intersection rate terminating on Layer 0."
              : "Measures synthetic 64-layer AABB traversal (assuming 8 BVH8 nodes x 8 box tests = 64 box tests/ray) under idealized occupancy; does not represent real triangle mesh geometry.");
      addFixed6(doc, rObj, "throughput_gis", throughputGis);

      const auto &prof = DeviceDatabase::lookup(r.vendorId, r.deviceId, r.deviceName);
      const double peakGis =
          (r.configIndex == 0) ? prof.theoreticalTriangleGis : prof.theoreticalBoxGis;
      if (peakGis > 0.0) {
        double pctPeak = (throughputGis / peakGis) * 100.0;
        if (!std::isfinite(pctPeak)) pctPeak = 0.0;
        char buf[96];
        if (r.configIndex == 0) {
          std::snprintf(buf, sizeof(buf), "%.1f%% of %.1f GIS/s %s Boost Peak", pctPeak,
                        peakGis, prof.archName.c_str());
        } else {
          std::snprintf(buf, sizeof(buf), "%.1f%% of %.2f TIS/s %s Boost Peak", pctPeak,
                        peakGis / 1000.0, prof.archName.c_str());
        }
        yyjson_mut_obj_add_strcpy(doc, rObj, "target_architecture", prof.archName.c_str());
        addFixed6(doc, rObj, "theoretical_peak_gis", peakGis);
        addFixed6(doc, rObj, "pct_theoretical_peak", pctPeak);
        yyjson_mut_obj_add_strcpy(doc, rObj, "details_speedup", buf);
      }
    }

    if ((r.benchmarkName.find("FP32") != std::string::npos ||
         r.benchmarkName.find("Dual-Issue") != std::string::npos) &&
        r.metric == "TFLOPS") {
      const auto &prof = DeviceDatabase::lookup(r.vendorId, r.deviceId, r.deviceName);
      if (prof.theoreticalFp32Tflops > 0.0) {
        double pctPeak = (value / prof.theoreticalFp32Tflops) * 100.0;
        if (!std::isfinite(pctPeak)) pctPeak = 0.0;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%.1f%% of %.2f TFLOPS %s Peak", pctPeak,
                      prof.theoreticalFp32Tflops, prof.archName.c_str());
        yyjson_mut_obj_add_strcpy(doc, rObj, "target_architecture", prof.archName.c_str());
        addFixed6(doc, rObj, "theoretical_peak_tflops", prof.theoreticalFp32Tflops);
        addFixed6(doc, rObj, "pct_theoretical_peak", pctPeak);
        yyjson_mut_obj_add_strcpy(doc, rObj, "details_speedup", buf);
      }
    }

    if ((r.benchmarkName.find("Device Memory Bandwidth") != std::string::npos ||
         r.benchmarkName.find("MemBandwidth") != std::string::npos) &&
        r.metric == "GB/s") {
      const auto &prof = DeviceDatabase::lookup(r.vendorId, r.deviceId, r.deviceName);
      if (prof.theoreticalBandwidthGBps > 0.0) {
        double pctPeak = (value / prof.theoreticalBandwidthGBps) * 100.0;
        if (!std::isfinite(pctPeak)) pctPeak = 0.0;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%.1f%% of %.1f GB/s %s Peak", pctPeak,
                      prof.theoreticalBandwidthGBps, prof.archName.c_str());
        yyjson_mut_obj_add_strcpy(doc, rObj, "target_architecture", prof.archName.c_str());
        addFixed6(doc, rObj, "theoretical_peak_gbps", prof.theoreticalBandwidthGBps);
        addFixed6(doc, rObj, "pct_theoretical_peak", pctPeak);
        yyjson_mut_obj_add_strcpy(doc, rObj, "details_speedup", buf);
      }
    }

    yyjson_mut_obj_add_uint(doc, rObj, "operations", r.operations);
    addFixed6(doc, rObj, "time_ms", r.time_ms);

    if (r.sample_count > 0) {
      yyjson_mut_val *statObj = yyjson_mut_obj_add_obj(doc, rObj, "statistics");
      yyjson_mut_obj_add_uint(doc, statObj, "samples", r.sample_count);
      addFixed6(doc, statObj, "min_time_ms", r.min_time_ms);
      addFixed6(doc, statObj, "median_time_ms", r.median_time_ms);
      addFixed6(doc, statObj, "mean_time_ms", r.mean_time_ms);
      addFixed6(doc, statObj, "p95_time_ms", r.p95_time_ms);
    }

    if (r.hasRegisterTelemetry) {
      yyjson_mut_val *compObj = yyjson_mut_obj_add_obj(doc, rObj, "compiler_telemetry");
      yyjson_mut_obj_add_bool(doc, compObj, "available", true);
      yyjson_mut_obj_add_uint(doc, compObj, "vgpr_count", r.vgprCount);
      yyjson_mut_obj_add_uint(doc, compObj, "sgpr_count", r.sgprCount);
      yyjson_mut_obj_add_uint(doc, compObj, "lds_size_bytes", r.ldsSizeBytes);
      yyjson_mut_obj_add_uint(doc, compObj, "scratch_spill_bytes", r.scratchSizeBytes);
      yyjson_mut_obj_add_uint(doc, compObj, "code_size_bytes", r.codeSizeBytes);
      yyjson_mut_obj_add_uint(doc, compObj, "max_waves_per_simd", r.maxWavesPerSimd);
      yyjson_mut_obj_add_strcpy(doc, compObj, "compiler", r.compilerTarget.c_str());
    }

    if (r.hasPowerTelemetry) {
      yyjson_mut_val *pwrObj = yyjson_mut_obj_add_obj(doc, rObj, "power_telemetry");
      yyjson_mut_obj_add_bool(doc, pwrObj, "available", true);
      addFixed6(doc, pwrObj, "power_watts", r.powerWatts);
      addFixed6(doc, pwrObj, "energy_joules", r.energyJoules);
      addFixed6(doc, pwrObj, "joules_per_unit", r.joulesPerUnit);
      addFixed6(doc, pwrObj, "unit_per_watt", r.unitPerWatt);
      yyjson_mut_obj_add_strcpy(doc, pwrObj, "efficiency_unit", r.efficiencyUnit.c_str());
    }

    if (r.hasPerfQueryTelemetry) {
      yyjson_mut_val *pqObj = yyjson_mut_obj_add_obj(doc, rObj, "performance_query");
      yyjson_mut_obj_add_bool(doc, pqObj, "available", true);
      yyjson_mut_obj_add_uint(doc, pqObj, "gpu_active_cycles", r.perfCounters.gpuActiveCycles);
      yyjson_mut_obj_add_uint(doc, pqObj, "waves", r.perfCounters.waves);
      yyjson_mut_obj_add_uint(doc, pqObj, "valu_instructions", r.perfCounters.valuInstructions);
      yyjson_mut_obj_add_uint(doc, pqObj, "salu_instructions", r.perfCounters.saluInstructions);
      yyjson_mut_obj_add_uint(doc, pqObj, "vmem_load_instructions", r.perfCounters.vmemLoadInstructions);
      yyjson_mut_obj_add_uint(doc, pqObj, "smem_load_instructions", r.perfCounters.smemLoadInstructions);
      yyjson_mut_obj_add_uint(doc, pqObj, "vmem_store_instructions", r.perfCounters.vmemStoreInstructions);
      yyjson_mut_obj_add_uint(doc, pqObj, "lds_instructions", r.perfCounters.ldsInstructions);
      addFixed6(doc, pqObj, "valu_busy_pct", r.perfCounters.valuBusyPct);
      addFixed6(doc, pqObj, "salu_busy_pct", r.perfCounters.saluBusyPct);
      yyjson_mut_obj_add_uint(doc, pqObj, "vram_read_bytes", r.perfCounters.vramReadBytes);
      addFixed6(doc, pqObj, "l0_cache_hit_ratio", r.perfCounters.l0CacheHitRatio);
      addFixed6(doc, pqObj, "l1_cache_hit_ratio", r.perfCounters.l1CacheHitRatio);
      yyjson_mut_val *rawObj = yyjson_mut_obj_add_obj(doc, pqObj, "raw_counters");
      for (const auto &[cName, cVal] : r.perfCounters.rawCounters) {
        addFixed6DynamicKey(doc, rawObj, cName, cVal);
      }
    }

    if (r.time_ms == -3.0) {
      yyjson_mut_obj_add_strcpy(doc, rObj, "status", "ABORTED");
      yyjson_mut_obj_add_strcpy(doc, rObj, "error", r.errorString.c_str());
    } else if (r.time_ms < 0.0) {
      yyjson_mut_obj_add_strcpy(doc, rObj, "status", "FAILED");
      yyjson_mut_obj_add_strcpy(doc, rObj, "error", r.errorString.c_str());
    } else if (r.isUnsupported) {
      yyjson_mut_obj_add_strcpy(doc, rObj, "status", "UNSUPPORTED");
    } else if (!r.isValid) {
      yyjson_mut_obj_add_strcpy(doc, rObj, "status", "INVALID");
      yyjson_mut_obj_add_strcpy(
          doc, rObj, "error",
          "Result validation failed (numerical divergence or NaN/Inf detected)");
    } else {
      yyjson_mut_obj_add_strcpy(doc, rObj, "status", "SUCCESS");
    }

    yyjson_mut_obj_add_bool(doc, rObj, "is_valid", r.isValid);
    yyjson_mut_obj_add_bool(doc, rObj, "is_emulated", r.isEmulated);
    yyjson_mut_obj_add_bool(doc, rObj, "emulated", r.isEmulated);
    const std::string emulationReason =
        r.isEmulated
            ? (r.supportNote.empty() ? "Emulated via software unpack" : r.supportNote)
            : "";
    yyjson_mut_obj_add_strcpy(doc, rObj, "emulation_reason", emulationReason.c_str());
    yyjson_mut_obj_add_bool(doc, rObj, "unsupported", r.isUnsupported);
    if (r.isUnsupported) {
      yyjson_mut_obj_add_strcpy(doc, rObj, "unsupported_category", r.supportCategory.c_str());
      yyjson_mut_obj_add_strcpy(doc, rObj, "unsupported_reason", r.supportNote.c_str());
    } else if (!r.supportNote.empty()) {
      yyjson_mut_obj_add_strcpy(doc, rObj, "support_note", r.supportNote.c_str());
      yyjson_mut_obj_add_strcpy(doc, rObj, "caveat", r.supportNote.c_str());
    }
    yyjson_mut_obj_add_strcpy(doc, rObj, "notes", r.supportNote.c_str());
    yyjson_mut_obj_add_uint(doc, rObj, "max_workgroup_size", r.maxWorkGroupSize);
    yyjson_mut_obj_add_sint(doc, rObj, "sort_weight", r.sortWeight);
    yyjson_mut_obj_add_sint(doc, rObj, "baseline_config_index", r.baselineConfigIndex);
    yyjson_mut_obj_add_uint(doc, rObj, "width", r.width);
    yyjson_mut_obj_add_uint(doc, rObj, "height", r.height);
    yyjson_mut_obj_add_uint(doc, rObj, "config_index", r.configIndex);
  }

  const std::vector<gpubench::DeviceRoofline> rooflines =
      gpubench::RooflineModel::analyze(results);
  if (!rooflines.empty()) {
    yyjson_mut_val *rfArr = yyjson_mut_obj_add_arr(doc, root, "roofline");
    for (const auto &dr : rooflines) {
      yyjson_mut_val *rfObj = yyjson_mut_arr_add_obj(doc, rfArr);
      yyjson_mut_obj_add_strcpy(doc, rfObj, "device_name", dr.deviceName.c_str());
      yyjson_mut_obj_add_strcpy(doc, rfObj, "backend_name", dr.backendName.c_str());
      yyjson_mut_obj_add_uint(doc, rfObj, "device_index", dr.deviceIndex);
      addFixed6(doc, rfObj, "primary_vram_bw_gbps", dr.primaryVramGBps());
      addFixed6(doc, rfObj, "primary_fp32_tflops", dr.primaryFp32Gops() / 1000.0);
      addFixed6(doc, rfObj, "highest_compute_tflops", dr.highestComputeGops() / 1000.0);

      yyjson_mut_val *bwArr = yyjson_mut_obj_add_arr(doc, rfObj, "bandwidth_ceilings");
      for (const auto &bw : dr.bandwidthCeilings) {
        yyjson_mut_val *bwObj = yyjson_mut_arr_add_obj(doc, bwArr);
        yyjson_mut_obj_add_strcpy(doc, bwObj, "level", bw.level.c_str());
        yyjson_mut_obj_add_strcpy(doc, bwObj, "source", bw.sourceLabel.c_str());
        addFixed6(doc, bwObj, "bandwidth_gbps", bw.bandwidthGBps);
      }

      yyjson_mut_val *ccArr = yyjson_mut_obj_add_arr(doc, rfObj, "compute_ceilings");
      for (const auto &cc : dr.computeCeilings) {
        yyjson_mut_val *ccObj = yyjson_mut_arr_add_obj(doc, ccArr);
        yyjson_mut_obj_add_strcpy(doc, ccObj, "name", cc.name.c_str());
        yyjson_mut_obj_add_strcpy(doc, ccObj, "metric", cc.metric.c_str());
        yyjson_mut_obj_add_strcpy(doc, ccObj, "source", cc.sourceLabel.c_str());
        addFixed6(doc, ccObj, "peak_tops", cc.peakTops);
      }

      yyjson_mut_val *rpArr = yyjson_mut_obj_add_arr(doc, rfObj, "ridge_points");
      for (const auto &rp : dr.ridgePoints) {
        yyjson_mut_val *rpObj = yyjson_mut_arr_add_obj(doc, rpArr);
        yyjson_mut_obj_add_strcpy(doc, rpObj, "memory_level", rp.memoryLevel.c_str());
        yyjson_mut_obj_add_strcpy(doc, rpObj, "compute_ceiling", rp.computeCeiling.c_str());
        addFixed6(doc, rpObj, "ridge_ops_per_byte", rp.ridgeOpsPerByte);
      }
    }
  }

  double totalPowerSumInJson = 0.0;
  double totalEnergySumInJson = 0.0;
  size_t powerCountInJson = 0;
  for (const auto &r : results) {
    if (r.hasPowerTelemetry && std::isfinite(r.powerWatts) && r.powerWatts > 0.0f) {
      totalPowerSumInJson += r.powerWatts;
      totalEnergySumInJson += std::isfinite(r.energyJoules) ? r.energyJoules : 0.0;
      powerCountInJson++;
    }
  }
  if (powerCountInJson > 0) {
    yyjson_mut_val *pwrSumObj = yyjson_mut_obj_add_obj(doc, root, "power_summary");
    addFixed6(doc, pwrSumObj, "average_power_watts",
              totalPowerSumInJson / static_cast<double>(powerCountInJson));
    addFixed6(doc, pwrSumObj, "total_energy_joules", totalEnergySumInJson);
  }

  size_t outLen = 0;
  char *rawJson = yyjson_mut_write_opts(
      doc, YYJSON_WRITE_PRETTY_TWO_SPACES | YYJSON_WRITE_NEWLINE_AT_END, nullptr,
      &outLen, nullptr);
  std::string out;
  if (rawJson != nullptr) {
    out.assign(rawJson, outLen);
    std::free(rawJson);
  }
  yyjson_mut_doc_free(doc);
  return out;
}
