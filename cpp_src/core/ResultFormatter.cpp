#include "ResultFormatter.h"
#include "ResultImporter.h"
#include "RunnerAPI.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

std::string ResultFormatter::formatDouble(double value, int precision) {
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

std::string cleanWorkloadName(const std::string &rawName, const std::string &subcat) {
  std::string name = rawName;

  // Handle "RayScheduling (Indoor Atrium) (Config Name)"
  if (name.rfind("RayScheduling (", 0) == 0) {
    size_t secondOpen = name.find(") (");
    if (secondOpen != std::string::npos && name.back() == ')') {
      name = name.substr(secondOpen + 3, name.length() - (secondOpen + 4));
    }
  } else {
    // Handle "RayASBuild (BLAS Build (1M Tris))" or "RayIntersect (Ray-Triangle)"
    size_t firstOpen = name.find(" (");
    if (firstOpen != std::string::npos && name.back() == ')') {
      name = name.substr(firstOpen + 2, name.length() - (firstOpen + 3));
    }
  }

  if (name == "Raw BVH Traversal - Coherent Triangle Traversal (Max Occupancy)") {
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
  if (subcat.find("Path Tracing") != std::string::npos) {
    size_t dash = name.rfind(" - ");
    if (dash != std::string::npos) {
      name = name.substr(dash + 3);
    }
  }

  // Strip repeated scene ray tracing patterns
  if (subcat.find("Scene Ray Tracing") != std::string::npos) {
    size_t dash = name.rfind(" - ");
    if (dash != std::string::npos) {
      name = name.substr(dash + 3);
    }
  }

  // Clean verbose divergence string
  if (name.find("Isolated Microbenchmark") != std::string::npos) {
    // Preserve unified taxonomy naming
  } else if (name.find("0% Coherence (Diffuse)") != std::string::npos ||
      name.find("0% Diffuse") != std::string::npos) {
    name = "Secondary bounce rays - 0% Diffuse";
  }

  // Clean long pipeline breakdown names
  if (name == "BVH Traversal - Linear 1D Scanline (Baseline)") {
    name = "BVH Traversal - Linear 1D Scanline";
  } else if (name == "Queue Compaction - Single-Pass Unified Stream") {
    name = "Queue Compaction - Single-Pass Stream";
  } else if (name.find("Queue Memory - VRAM Round-Trip") != std::string::npos) {
    name = "Queue Memory - VRAM Round-Trip Bandwidth";
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
  double megakernelPT16Rate = 0.0;
  double dgcPT16Rate = 0.0;
  double scanlineRate = 0.0;
  double tiledRate = 0.0;
  double maxBlasUpdateRate = 0.0;
  double maxTlasRate = 0.0;
  double rawBoxGis = 0.0;
  double rawTriangleGis = 0.0;
  size_t totalWorkloadsEvaluated = 0;

  for (const auto &res : results) {
    totalWorkloadsEvaluated++;
    int compWeight = 4;
    if (res.component == "Compute") compWeight = 1;
    else if (res.component == "Memory") compWeight = 2;
    else if (res.component == "Ray Tracing") compWeight = 3;

    std::string cleanName = cleanWorkloadName(res.benchmarkName, res.subcategory);
    if (res.isEmulated) cleanName += " (Emulated)";

    auto &subcatGroup = organizedData[res.deviceIndex][{compWeight, res.component}][res.subcategory];
    subcatGroup.name = res.subcategory;
    subcatGroup.minSortWeight = (std::min)(subcatGroup.minSortWeight, res.sortWeight);
    subcatGroup.benchmarks[{res.sortWeight, res.configIndex, cleanName}][res.backendName] = res;

    // Track metrics for executive summary
    if (!res.isUnsupported && res.time_ms > 0.0) {
      double rate = (static_cast<double>(res.operations) / (res.time_ms / 1000.0)) / 1e6;
      if (res.benchmarkName.find("RayRawTraversal") != std::string::npos) {
        double time_s = res.time_ms / 1000.0;
        if (res.configIndex == 0) {
          rawTriangleGis = (time_s > 0.0) ? ((static_cast<double>(res.operations) / time_s) / 1e9) : 0.0;
        } else {
          uint64_t boxOps = res.operations * 64;
          rawBoxGis = (time_s > 0.0) ? ((static_cast<double>(boxOps) / time_s) / 1e9) : 0.0;
        }
      }
      if (res.metric == "MRays/s") {
        if (rate > maxRayRate) {
          maxRayRate = rate;
          maxRayWorkload = cleanName;
        }
        if (res.benchmarkName.find("Primary Rays (Megakernel)") != std::string::npos ||
            res.benchmarkName.find("Full Scene Ray Tracing (PBR) - Megakernel") != std::string::npos) {
          megakernelPBRRate = rate;
        } else if (res.benchmarkName.find("Primary Rays (DGC)") != std::string::npos ||
                   res.benchmarkName.find("Full Scene Ray Tracing (PBR) - DGC") != std::string::npos ||
                   res.benchmarkName.find("Full Scene Ray Tracing (PBR) - Work Lists") != std::string::npos ||
                   res.benchmarkName.find("Full Scene Ray Tracing (PBR) - Device-Generated Commands") != std::string::npos) {
          dgcPBRRate = rate;
        } else if (res.benchmarkName.find("16 SPP) (Megakernel)") != std::string::npos ||
                   res.benchmarkName.find("16 SPP) - Traditional Megakernel") != std::string::npos) {
          megakernelPT16Rate = rate;
        } else if (res.benchmarkName.find("16 SPP) (DGC)") != std::string::npos ||
                   res.benchmarkName.find("16 SPP) - Work Lists") != std::string::npos ||
                   res.benchmarkName.find("16 SPP) - Device-Generated Commands") != std::string::npos) {
          dgcPT16Rate = rate;
        } else if (res.benchmarkName.find("1D Scanline") != std::string::npos) {
          scanlineRate = rate;
        } else if (res.benchmarkName.find("2D Screen Tiled") != std::string::npos) {
          tiledRate = rate;
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

  const std::string RESET = "\033[0m";
  const std::string BOLD = "\033[1m";
  const std::string DIM = "\033[2m";
  const std::string CYAN = "\033[36m";
  const std::string GREEN = "\033[32m";
  const std::string RED = "\033[31m";
  const std::string YELLOW = "\033[33m";
  const std::string MAGENTA = "\033[35m";
  const std::string BLUE = "\033[34m";

  const size_t cardBoxWidth = 128;
  const size_t cardInnerWidth = cardBoxWidth - 4; // 124

  const size_t w1 = 44;
  const size_t w2 = 8;
  const size_t w3 = 22;
  const size_t w4 = 41;

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
                  return a.minSortWeight < b.minSortWeight;
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
          }

          if (hasMegakernel || (hasVector && subcat.benchmarks.size() >= 2) || hasScanline) {
            hasComparison = true;
            for (const auto &benchPair : subcat.benchmarks) {
              std::string cName = std::get<2>(benchPair.first);
              bool isCandidate = false;
              if (hasMegakernel && (cName.find("Megakernel") != std::string::npos || cName.find("Traditional") != std::string::npos)) {
                isCandidate = true;
              } else if (!hasMegakernel && hasVector && cName.find("Vector") != std::string::npos) {
                isCandidate = true;
              } else if (!hasMegakernel && !hasVector && hasScanline && (cName.find("Scanline") != std::string::npos || cName.find("Baseline") != std::string::npos)) {
                isCandidate = true;
              }

              if (isCandidate) {
                for (const auto &backend : backends) {
                  if (benchPair.second.count(backend)) {
                    const auto &res = benchPair.second.at(backend);
                    if (!res.isUnsupported && res.time_ms > 0.0) {
                      double val = static_cast<double>(res.operations) / (res.time_ms / 1000.0);
                      if (res.component == "Compute") val /= 1e12;
                      else if (res.component == "Memory") val = (res.subcategory == "Latency") ? ((res.time_ms * 1e6) / res.operations) : (val / 1e9);
                      else val /= (res.metric == "GIS/s" || res.metric == "GB/s") ? 1e9 : 1e6;

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
          if (displayName.length() > w1) {
            displayName = displayName.substr(0, w1 - 2) + "..";
          }

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
              } else if (res.component == "Compute") {
                double value = (static_cast<double>(res.operations) /
                               (res.time_ms / 1000.0)) / 1e12;
                valStr = formatDouble(value, 2) + " " + res.metric;
                if (hasComparison && baselineVal > 0.0 && value > 0.0) {
                  if (fullName == baselineKeyName || fullName.find("Vector") != std::string::npos) {
                    noteStr = "[Baseline]";
                  } else {
                    double ratio = value / baselineVal;
                    double pct = (ratio - 1.0) * 100.0;
                    noteStr = "└──> " + formatDouble(ratio, 2) + "x (" + (pct >= 0 ? "+" : "") + formatDouble(pct, 1) + "%)";
                  }
                }
              } else if (res.component == "Memory") {
                if (res.subcategory == "Latency" || res.metric == "ns") {
                  double value = (res.time_ms * 1e6) / res.operations; // ns
                  valStr = formatDouble(value, 2) + " ns";
                } else {
                  double value = (static_cast<double>(res.operations) /
                                 (res.time_ms / 1000.0)) / 1e9; // GB/s
                  valStr = formatDouble(value, 2) + " GB/s";
                }
              } else {
                double value = static_cast<double>(res.operations) / (res.time_ms / 1000.0);
                int precision = 2;
                if (res.metric == "GIS/s" || res.metric == "GRays/s" ||
                    res.metric == "GPixels/s" || res.metric == "GB/s") {
                  value /= 1e9;
                  if (res.metric == "GRays/s") precision = 3;
                } else if (res.metric == "MTris/s" || res.metric == "MInst/s" ||
                           res.metric == "MRays/s" || res.metric == "MHits/s" ||
                           res.metric == "MRecords/s") {
                  value /= 1e6;
                }
                valStr = formatDouble(value, precision) + " " + res.metric;

                std::string fpsStr = "";
                if (res.benchmarkName.find("RayScheduling") != std::string::npos && res.metric == "MRays/s") {
                  uint32_t w = res.width ? res.width : 1920;
                  uint32_t h = res.height ? res.height : 1080;
                  double fps = (value * 1e6) / static_cast<double>(w * h);
                  fpsStr = formatDouble(fps, 1) + " FPS";
                }

                if (res.benchmarkName.find("RayRawTraversal") != std::string::npos) {
                  double time_s = res.time_ms / 1000.0;
                  if (res.configIndex == 0) {
                    double gis_s = (time_s > 0.0) ? ((static_cast<double>(res.operations) / time_s) / 1e9) : 0.0;
                    double pct = (gis_s / 300.8) * 100.0;
                    noteStr = formatDouble(pct, 1) + "% of 300.8 GIS/s Boost Peak";
                  } else {
                    uint64_t boxOps = res.operations * 64;
                    double box_gis_s = (time_s > 0.0) ? ((static_cast<double>(boxOps) / time_s) / 1e9) : 0.0;
                    double pct = (box_gis_s / 1203.2) * 100.0;
                    noteStr = formatDouble(pct, 1) + "% of 1.20 TIS/s Boost Peak";
                  }
                } else {
                  double localBaseVal = 0.0;
                  std::string localBaseMetric = "";
                  bool isLocalBase = false;

                  if (res.benchmarkName.find("RayDivergence") != std::string::npos) {
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
                  } else if (fullName == baselineKeyName || fullName.find("Megakernel") != std::string::npos || fullName.find("Traditional") != std::string::npos || fullName.find("Baseline") != std::string::npos) {
                    isLocalBase = true;
                  } else if (hasComparison && baselineVal > 0.0 && baselineMetric == res.metric) {
                    localBaseVal = baselineVal;
                    localBaseMetric = baselineMetric;
                  }

                  if (isLocalBase) {
                    noteStr = "[Baseline]" + (fpsStr.empty() ? "" : " [" + fpsStr + "]");
                  } else if (localBaseVal > 0.0 && value > 0.0 && (localBaseMetric.empty() || localBaseMetric == res.metric)) {
                    double ratio = value / localBaseVal;
                    double pct = (ratio - 1.0) * 100.0;
                    std::string pctStr = (std::abs(pct) >= 0.1) ? (std::string(" (") + (pct >= 0 ? "+" : "") + formatDouble(pct, 1) + "%)") : "";
                    noteStr = "└──> " + formatDouble(ratio, 2) + "x" + pctStr + (fpsStr.empty() ? "" : " [" + fpsStr + "]");
                  } else if (!fpsStr.empty()) {
                    noteStr = fpsStr;
                  }
                }

                if (!res.supportNote.empty() && res.benchmarkName.find("RayRawTraversal") == std::string::npos) {
                  noteStr += (noteStr.empty() ? "" : " ") + ("[" + cleanSupportNote(res.supportNote) + "]");
                }
              }

              std::string backendStr = backend;
              if (backendStr.length() > w2) {
                backendStr = backendStr.substr(0, w2);
              }
              if (valStr.length() > w3) {
                valStr = valStr.substr(0, w3);
              }
              size_t noteVis = visualLength(noteStr);
              if (noteVis > w4) {
                while (!noteStr.empty() && visualLength(noteStr) > w4 - 2) {
                  noteStr.pop_back();
                }
                noteStr += "..";
                noteVis = visualLength(noteStr);
              }
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

  // 4. Executive Summary Card (if Ray Tracing was evaluated)
  if (totalWorkloadsEvaluated > 0 && maxRayRate > 0.0) {
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
      std::string extra = " in PBR Ray Tracing (" + formatDouble(dgcPBRRate, 1) + " vs " +
                          formatDouble(megakernelPBRRate, 1) + " MRays/s)";
      printSummaryRow("Wavefront Scheduling Speedup", val, extra);
    }
    if (megakernelPT16Rate > 0.0 && dgcPT16Rate > 0.0) {
      double ptSpeedup = dgcPT16Rate / megakernelPT16Rate;
      std::string val = BOLD + GREEN + formatDouble(ptSpeedup, 2) + "x" + RESET;
      std::string extra = " in 16 SPP Stress (" + formatDouble(dgcPT16Rate, 1) + " vs " +
                          formatDouble(megakernelPT16Rate, 1) + " MRays/s)";
      printSummaryRow("Multi-Bounce Path Tracing   ", val, extra);
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
      double pct = (rawBoxGis / 1203.2) * 100.0;
      std::string val = BOLD + GREEN + formatDouble(rawBoxGis, 1) + " GIS/s" + RESET;
      std::string extra = " (" + formatDouble(pct, 1) + "% of 1.20 TIS/s Boost Peak)";
      printSummaryRow("Hardware BVH8 Box Peak Rate  ", val, extra);
    }
    if (rawTriangleGis > 0.0) {
      double pct = (rawTriangleGis / 300.8) * 100.0;
      std::string val = BOLD + GREEN + formatDouble(rawTriangleGis, 1) + " GIS/s" + RESET;
      std::string extra = " (" + formatDouble(pct, 1) + "% of 300.8 GIS/s Boost Peak)";
      printSummaryRow("Hardware Triangle Peak Rate  ", val, extra);
    }
    std::string val = BOLD + GREEN + formatDouble(maxRayRate, 1) + " MRays/s" + RESET;
    std::string extra = " (" + maxRayWorkload + ")";
    printSummaryRow("Peak Measured Ray Rate       ", val, extra);

    std::cout << BOLD << CYAN << "  ╰─" << repeatUtf8("─", cardBoxWidth - 3) << "╯" << RESET << "\n";
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

  const std::string RESET = "\033[0m";
  const std::string BOLD = "\033[1m";
  const std::string DIM = "\033[2m";
  const std::string CYAN = "\033[36m";
  const std::string GREEN = "\033[32m";
  const std::string RED = "\033[31m";
  const std::string YELLOW = "\033[33m";
  const std::string MAGENTA = "\033[35m";
  const std::string BLUE = "\033[34m";

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
  const std::vector<std::string> devColors = {
    YELLOW, MAGENTA, CYAN, BLUE, "\033[38;5;208m", "\033[38;5;141m", GREEN
  };

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
    if (sh.length() > w_devs[i]) sh = sh.substr(0, w_devs[i] - 2) + "..";
    shortHeaders[i] = sh;
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
      int compWeight = 4;
      if (res.component == "Compute") compWeight = 1;
      else if (res.component == "Memory") compWeight = 2;
      else if (res.component == "Rasterization & ROP") compWeight = 3;
      else if (res.component == "Ray Tracing") compWeight = 4;

      std::string cName = cleanWorkloadName(res.benchmarkName, res.subcategory);
      auto &subcat = compData[std::make_pair(compWeight, res.component)][res.subcategory];
      subcat.name = res.subcategory;
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
    if (res.time_ms <= 0.0 || res.operations == 0) {
      wv.formattedStr = "N/A";
      return wv;
    }

    double time_s = res.time_ms / 1000.0;
    double rawRate = static_cast<double>(res.operations) / time_s;
    int prec = 2;

    if (res.component == "Compute") {
      wv.val = rawRate / 1e12;
      wv.metric = res.metric.empty() ? "TFLOPS" : res.metric;
    } else if (res.component == "Memory") {
      if (res.subcategory == "Latency" || res.metric == "ns" || res.benchmarkName.find("Latency") != std::string::npos) {
        wv.val = (res.time_ms * 1e6) / static_cast<double>(res.operations);
        wv.isLatency = true;
        wv.metric = "ns";
        wv.isValid = true;
        wv.formattedStr = formatDouble(wv.val, 2) + " ns";
        return wv;
      } else {
        wv.val = rawRate / 1e9;
        wv.metric = "GB/s";
      }
    } else {
      if (res.metric == "GIS/s" || res.metric == "GRays/s" || res.metric == "GPixels/s" || res.metric == "GB/s") {
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
                return a.minSortWeight < b.minSortWeight;
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
        std::string dispName = wl.cleanName;
        if (dispName.length() > w_wl) dispName = dispName.substr(0, w_wl - 2) + "..";

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
          if (subcat.name.find("Indoor") != std::string::npos) {
            if (wl.configIndex == 0) ptTradRates[i] = vals[i].val;
          }
          if (subcat.name.find("Material") != std::string::npos || subcat.name.find("Showroom") != std::string::npos) {
            if (wl.configIndex == 0) matTradRates[i] = vals[i].val;
            else if (wl.configIndex == 1) matWlRates[i] = vals[i].val;
          }
        }

        std::string deltaStr;
        std::string deltaColor = RESET;

        if (validCount >= 2 && bestIdx >= 0 && secondIdx >= 0) {
          char winLetter = static_cast<char>('A' + bestIdx);
          char secLetter = static_cast<char>('A' + secondIdx);

          double realDelta = 0.0;
          double ratio = 1.0;
          double pct = 0.0;

          if (isLatency) {
            realDelta = secondVal - bestVal;
            ratio = (bestVal > 0.0) ? (secondVal / bestVal) : 1.0;
            pct = (secondVal > 0.0) ? ((secondVal - bestVal) / secondVal) * 100.0 : 0.0;
          } else {
            realDelta = bestVal - secondVal;
            ratio = (secondVal > 0.0) ? (bestVal / secondVal) : 1.0;
            pct = (ratio - 1.0) * 100.0;
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
          std::string str = vals[i].formattedStr;
          if (str.length() > w_devs[i]) str = str.substr(0, w_devs[i]);

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

        size_t dVis = visualLength(deltaStr);
        if (dVis > w_delta) {
          while (!deltaStr.empty() && visualLength(deltaStr) > w_delta - 2) deltaStr.pop_back();
          deltaStr += "..";
          dVis = visualLength(deltaStr);
        }
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
std::string jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
}
} // namespace

double computeResultValue(const ResultData &r) {
  double value = 0.0;
  if (r.time_ms > 0.0 && r.operations > 0) {
    double seconds = r.time_ms / 1000.0;
    if (r.metric == "TFLOPS" || r.metric == "TOPS") {
      value = (static_cast<double>(r.operations) / seconds) / 1e12;
    } else if (r.metric == "GB/s") {
      value = (static_cast<double>(r.operations) / seconds) / 1e9;
    } else if (r.metric == "MRays/s" || r.metric == "MHits/s" ||
               r.metric == "MTris/s" || r.metric == "MInst/s" ||
               r.metric == "MRecords/s") {
      value = (static_cast<double>(r.operations) / seconds) / 1e6;
    } else if (r.metric == "GIS/s") {
      value = (static_cast<double>(r.operations) / seconds) / 1e9;
    } else if (r.metric == "GPixels/s") {
      value = (static_cast<double>(r.operations) / seconds) / 1e9;
    } else if (r.metric == "ns") {
      value = (r.time_ms * 1e6) / static_cast<double>(r.operations);
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
  auto profiles = GetDeviceProfilesAPI();
  std::string out = "{\n";
  out += "  \"version\": \"" + std::string(GPUBENCH_VERSION) + "\",\n";
  out += "  \"device_profiles\": [\n";
  for (size_t d = 0; d < profiles.size(); ++d) {
    const auto &dp = profiles[d];
    char vendorHex[16], deviceHex[16];
    std::snprintf(vendorHex, sizeof(vendorHex), "0x%04X", dp.vendorID);
    std::snprintf(deviceHex, sizeof(deviceHex), "0x%04X", dp.deviceID);

    out += "    {\n";
    out += "      \"backend\": \"" + jsonEscape(dp.backend) + "\",\n";
    out += "      \"device_index\": " + std::to_string(dp.deviceIndex) + ",\n";
    out += "      \"device_name\": \"" + jsonEscape(dp.deviceName) + "\",\n";
    out += "      \"vendor_id\": \"" + std::string(vendorHex) + "\",\n";
    out += "      \"device_id\": \"" + std::string(deviceHex) + "\",\n";
    out += "      \"driver_name\": \"" + jsonEscape(dp.driverName) + "\",\n";
    out += "      \"driver_info\": \"" + jsonEscape(dp.driverInfo) + "\",\n";
    out += "      \"driver_version\": \"" + jsonEscape(dp.driverVersion) + "\",\n";
    out += "      \"api_version\": \"" + jsonEscape(dp.apiVersion) + "\",\n";
    out += "      \"vram_total_mb\": " + std::to_string(dp.vramTotalMb) + ",\n";
    out += "      \"subgroup_size\": " + std::to_string(dp.subgroupSize) + ",\n";
    out += "      \"max_workgroup_size\": " + std::to_string(dp.maxWorkGroupSize) + ",\n";
    out += "      \"ray_tracing_supported\": " + std::string(dp.rayTracingSupported ? "true" : "false") + ",\n";
    out += "      \"ser_supported\": " + std::string(dp.serSupported ? "true" : "false") + ",\n";
    out += "      \"work_graphs_supported\": " + std::string(dp.workGraphsSupported ? "true" : "false") + ",\n";
    out += "      \"cooperative_matrix_supported\": " + std::string(dp.cooperativeMatrixSupported ? "true" : "false") + ",\n";
    out += "      \"float16_supported\": " + std::string(dp.float16Supported ? "true" : "false") + ",\n";
    out += "      \"int8_supported\": " + std::string(dp.int8Supported ? "true" : "false") + "\n";
    out += (d + 1 < profiles.size()) ? "    },\n" : "    }\n";
  }
  out += "  ],\n";
  out += "  \"results\": [\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const ResultData &r = results[i];
    std::string devIdxStr = (r.deviceIndex == 0xFFFFFFFF || r.backendName == "System")
                                ? "null"
                                : std::to_string(r.deviceIndex);
    double value = computeResultValue(r);

    out += "    {\n";
    out += "      \"backend\": \"" + jsonEscape(r.backendName) + "\",\n";
    out += "      \"device\": \"" + jsonEscape(r.deviceName) + "\",\n";
    out += "      \"device_index\": " + devIdxStr + ",\n";
    out += "      \"benchmark\": \"" + jsonEscape(r.benchmarkName) + "\",\n";
    out += "      \"component\": \"" + jsonEscape(r.component) + "\",\n";
    out += "      \"subcategory\": \"" + jsonEscape(r.subcategory) + "\",\n";
    out += "      \"metric\": \"" + jsonEscape(r.metric) + "\",\n";
    out += "      \"value\": " + std::to_string(value) + ",\n";
    if (r.benchmarkName.find("RayScheduling") != std::string::npos && r.metric == "MRays/s") {
      uint32_t w = r.width ? r.width : 1920;
      uint32_t h = r.height ? r.height : 1080;
      double fps = (value * 1e6) / static_cast<double>(w * h);
      out += "      \"fps\": " + std::to_string(fps) + ",\n";
      out += "      \"resolution\": \"" + std::to_string(w) + "x" + std::to_string(h) + "\",\n";
    }
    if (r.benchmarkName.find("RayRawTraversal") != std::string::npos) {
      double peakGis = (r.configIndex == 0) ? 300.8 : 1203.2;
      double time_s = r.time_ms / 1000.0;
      double throughputGis = 0.0;
      if (time_s > 0.0) {
        uint64_t ops = (r.configIndex == 0) ? r.operations : (r.operations * 64);
        throughputGis = (static_cast<double>(ops) / time_s) / 1e9;
      }
      double pctPeak = (peakGis > 0.0) ? ((throughputGis / peakGis) * 100.0) : 0.0;
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.1f%% of %s Boost Peak", pctPeak,
                    (r.configIndex == 0 ? "300.8 GIS/s" : "1.20 TIS/s"));
      std::string detailsStr(buf);
      out += "      \"peak_type\": \"" + std::string(r.configIndex == 0 ? "Triangle" : "Box") + "\",\n";
      out += "      \"theoretical_peak_gis\": " + std::to_string(peakGis) + ",\n";
      out += "      \"throughput_gis\": " + std::to_string(throughputGis) + ",\n";
      out += "      \"pct_theoretical_peak\": " + std::to_string(pctPeak) + ",\n";
      out += "      \"details_speedup\": \"" + jsonEscape(detailsStr) + "\",\n";
    }
    out += "      \"operations\": " + std::to_string(r.operations) + ",\n";
    out += "      \"time_ms\": " + std::to_string(r.time_ms) + ",\n";
    out += std::string("      \"is_emulated\": ") +
           (r.isEmulated ? "true" : "false") + ",\n";
    out += std::string("      \"unsupported\": ") +
           (r.isUnsupported ? "true" : "false") + ",\n";
    if (r.isUnsupported) {
      out += "      \"unsupported_category\": \"" +
             jsonEscape(r.supportCategory) + "\",\n";
      out += "      \"unsupported_reason\": \"" + jsonEscape(r.supportNote) +
             "\",\n";
    } else if (!r.supportNote.empty()) {
      out += "      \"support_note\": \"" + jsonEscape(r.supportNote) + "\",\n";
      out += "      \"caveat\": \"" + jsonEscape(r.supportNote) + "\",\n";
    }
    out += "      \"max_workgroup_size\": " +
           std::to_string(r.maxWorkGroupSize) + ",\n";
    out += "      \"config_index\": " + std::to_string(r.configIndex) + "\n";
    out += (i + 1 < results.size()) ? "    },\n" : "    }\n";
  }
  out += "  ]\n";
  out += "}\n";
  return out;
}
