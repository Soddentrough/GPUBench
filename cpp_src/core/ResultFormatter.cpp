#include "ResultFormatter.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <vector>

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

static std::string cleanWorkloadName(const std::string &rawName, const std::string &subcat) {
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
  if (name.find("0% Coherence (Diffuse)") != std::string::npos ||
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
  double worklistPBRRate = 0.0;
  double megakernelPT16Rate = 0.0;
  double worklistPT16Rate = 0.0;
  double scanlineRate = 0.0;
  double tiledRate = 0.0;
  double maxBlasUpdateRate = 0.0;
  double maxTlasRate = 0.0;
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
    subcatGroup.minSortWeight = std::min(subcatGroup.minSortWeight, res.sortWeight);
    subcatGroup.benchmarks[{res.sortWeight, res.configIndex, cleanName}][res.backendName] = res;

    // Track metrics for executive summary
    if (!res.isUnsupported && res.time_ms > 0.0) {
      double rate = (static_cast<double>(res.operations) / (res.time_ms / 1000.0)) / 1e6;
      if (res.metric == "MRays/s") {
        if (rate > maxRayRate) {
          maxRayRate = rate;
          maxRayWorkload = cleanName;
        }
        if (res.benchmarkName.find("Full Scene Ray Tracing (PBR) - Megakernel") != std::string::npos) {
          megakernelPBRRate = rate;
        } else if (res.benchmarkName.find("Full Scene Ray Tracing (PBR) - Work Lists") != std::string::npos) {
          worklistPBRRate = rate;
        } else if (res.benchmarkName.find("16 SPP) - Traditional Megakernel") != std::string::npos) {
          megakernelPT16Rate = rate;
        } else if (res.benchmarkName.find("16 SPP) - Work Lists") != std::string::npos) {
          worklistPT16Rate = rate;
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

  const size_t cardBoxWidth = 124;
  const size_t cardInnerWidth = cardBoxWidth - 4; // 120

  const size_t w1 = 44;
  const size_t w2 = 8;
  const size_t w3 = 17;
  const size_t w4 = 42;

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

        // Determine baseline for speedup calculations within this subcategory
        double baselineVal = 0.0;
        std::string baselineMetric = "";
        for (const auto &benchPair : subcat.benchmarks) {
          for (const auto &backend : backends) {
            if (benchPair.second.count(backend)) {
              const auto &res = benchPair.second.at(backend);
              if (!res.isUnsupported && res.time_ms > 0.0) {
                double val = static_cast<double>(res.operations) / (res.time_ms / 1000.0);
                if (res.component == "Compute") val /= 1e12;
                else if (res.component == "Memory") val = (res.subcategory == "Latency") ? ((res.time_ms * 1e6) / res.operations) : (val / 1e9);
                else val /= (res.metric == "GIS/s" || res.metric == "GB/s") ? 1e9 : 1e6;

                std::string cName = std::get<2>(benchPair.first);
                if (cName.find("Megakernel") != std::string::npos ||
                    cName.find("Baseline") != std::string::npos ||
                    baselineVal == 0.0) {
                  baselineVal = val;
                  baselineMetric = res.metric;
                  break;
                }
              }
            }
          }
          if (baselineVal > 0.0) break;
        }

        // Print benchmark rows
        for (const auto &benchPair : subcat.benchmarks) {
          std::string displayName = std::get<2>(benchPair.first);
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
                if (baselineVal > 0.0 && value > 0.0) {
                  double ratio = value / baselineVal;
                  if (std::abs(ratio - 1.0) < 0.01) {
                    noteStr = "1.00x baseline";
                  } else {
                    noteStr = formatDouble(ratio, 2) + "x (" + (ratio >= 1.0 ? "+" : "") + formatDouble((ratio - 1.0) * 100.0, 1) + "%)";
                  }
                }
              } else if (res.component == "Memory") {
                if (res.subcategory == "Latency") {
                  double value = (res.time_ms * 1e6) / res.operations; // ns
                  valStr = formatDouble(value, 2) + " ns";
                  if (baselineVal > 0.0 && value > 0.0) {
                    double ratio = baselineVal / value;
                    noteStr = (std::abs(ratio - 1.0) < 0.01) ? "1.00x baseline" : (formatDouble(ratio, 2) + "x");
                  }
                } else {
                  double value = (static_cast<double>(res.operations) /
                                 (res.time_ms / 1000.0)) / 1e9; // GB/s
                  valStr = formatDouble(value, 2) + " GB/s";
                  if (baselineVal > 0.0 && value > 0.0) {
                    double ratio = value / baselineVal;
                    noteStr = (std::abs(ratio - 1.0) < 0.01) ? "1.00x baseline" : (formatDouble(ratio, 2) + "x");
                  }
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

                if (baselineVal > 0.0 && value > 0.0 && baselineMetric == res.metric) {
                  double ratio = value / baselineVal;
                  if (std::abs(ratio - 1.0) < 0.01) {
                    noteStr = "1.00x base" + (fpsStr.empty() ? "" : " (" + fpsStr + ")");
                  } else {
                    noteStr = formatDouble(ratio, 2) + "x" + (fpsStr.empty() ? "" : " (" + fpsStr + ")");
                  }
                } else if (!fpsStr.empty()) {
                  noteStr = fpsStr;
                }

                if (!res.supportNote.empty()) {
                  noteStr += (noteStr.empty() ? "" : " ") + ("[" + cleanSupportNote(res.supportNote) + "]");
                }
              }

              if (noteStr.length() > w4) {
                noteStr = noteStr.substr(0, w4 - 2) + "..";
              }

              std::cout << BOLD << CYAN << "  │ " << RESET
                        << (firstBackend ? displayName : std::string(w1, ' '))
                        << std::setw(w1 - (firstBackend ? displayName.length() : 0)) << ""
                        << BOLD << CYAN << " │ " << RESET << YELLOW << std::left << std::setw(w2) << backend << RESET
                        << BOLD << CYAN << " │ " << RESET << statusColor << BOLD << std::right << std::setw(w3) << valStr << RESET
                        << BOLD << CYAN << " │ " << RESET << DIM << std::left << std::setw(w4) << noteStr << RESET
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

    if (megakernelPBRRate > 0.0 && worklistPBRRate > 0.0) {
      double pbrSpeedup = worklistPBRRate / megakernelPBRRate;
      std::string val = BOLD + GREEN + formatDouble(pbrSpeedup, 2) + "x" + RESET;
      std::string extra = " in PBR Ray Tracing (" + formatDouble(worklistPBRRate, 1) + " vs " +
                          formatDouble(megakernelPBRRate, 1) + " MRays/s)";
      printSummaryRow("Wavefront Scheduling Speedup", val, extra);
    }
    if (megakernelPT16Rate > 0.0 && worklistPT16Rate > 0.0) {
      double ptSpeedup = worklistPT16Rate / megakernelPT16Rate;
      std::string val = BOLD + GREEN + formatDouble(ptSpeedup, 2) + "x" + RESET;
      std::string extra = " in 16 SPP Stress (" + formatDouble(worklistPT16Rate, 1) + " vs " +
                          formatDouble(megakernelPT16Rate, 1) + " MRays/s)";
      printSummaryRow("Multi-Bounce Path Tracing   ", val, extra);
    }
    if (scanlineRate > 0.0 && tiledRate > 0.0) {
      double cacheGain = ((tiledRate - scanlineRate) / scanlineRate) * 100.0;
      std::string val = BOLD + CYAN + "+" + formatDouble(cacheGain, 1) + "%" + RESET;
      std::string extra = " with 2D Screen Tiling (" + formatDouble(tiledRate, 1) + " vs " +
                          formatDouble(scanlineRate, 1) + " MRays/s)";
      printSummaryRow("BVH Traversal Cache Locality", val, extra);
    }
    if (maxBlasUpdateRate > 0.0 || maxTlasRate > 0.0) {
      std::string val = BOLD + YELLOW + formatDouble(maxBlasUpdateRate, 1) + " MTris/s" + RESET + " (BLAS Update) | " +
                        BOLD + YELLOW + formatDouble(maxTlasRate, 1) + " MInst/s" + RESET + " (TLAS Construction)";
      printSummaryRow("Acceleration Build Peak Rates", val, "");
    }
    std::string val = BOLD + GREEN + formatDouble(maxRayRate, 1) + " MRays/s" + RESET;
    std::string extra = " (" + maxRayWorkload + ")";
    printSummaryRow("Peak Measured Ray Rate       ", val, extra);

    std::cout << BOLD << CYAN << "  ╰─" << repeatUtf8("─", cardBoxWidth - 3) << "╯" << RESET << "\n";
  }

  std::cout << std::endl;
}
