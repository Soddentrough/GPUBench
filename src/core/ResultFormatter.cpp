#include "ResultFormatter.h"
#include <algorithm>
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
  std::string integerPart = str.substr(0, str.find('.'));
  std::string fractionalPart = str.substr(str.find('.'));
  int insertPosition = integerPart.length() - 3;
  while (insertPosition > 0) {
    integerPart.insert(insertPosition, ",");
    insertPosition -= 3;
  }
  return integerPart + fractionalPart;
}

std::string ResultFormatter::formatNumber(uint64_t value) {
  std::string numWithCommas = std::to_string(value);
  int insertPosition = numWithCommas.length() - 3;
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

enum class BenchmarkCategory { Compute, Memory, Latency, Unknown };

// Helper to lowercase a string
static std::string to_lower_static(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

BenchmarkCategory getBenchmarkCategory(const std::string &name) {
  std::string lower_name = to_lower_static(name);
  if (lower_name.find("fp") != std::string::npos ||
      lower_name.find("int") != std::string::npos) {
    return BenchmarkCategory::Compute;
  }
  if (lower_name.find("bandwidth") != std::string::npos) {
    return BenchmarkCategory::Memory;
  }
  if (lower_name.find("latency") != std::string::npos) {
    return BenchmarkCategory::Latency;
  }
  return BenchmarkCategory::Unknown;
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

  // 3. Organize data:
  // [DeviceIndex][Component][SubCategory][BenchmarkName][Backend] -> ResultData
  // We use std::map to keep things sorted alphabetically (Component,
  // SubCategory) For benchmarks within a subcategory, we might want a specific
  // order.
  std::map<
      uint32_t,
      std::map<
          std::string,
          std::map<std::string,
                   std::map<std::string, std::map<std::string, ResultData>>>>>
      organizedData;

  size_t maxBenchNameLen = 30; // Minimum width

  for (const auto &res : results) {
    std::string name = res.benchmarkName;
    if (res.isEmulated)
      name += " (Emulated)";
    organizedData[res.deviceIndex][res.component][res.subcategory][name]
                 [res.backendName] = res;
    
    if (name.length() > maxBenchNameLen) {
        maxBenchNameLen = name.length();
    }
  }

  const std::string RESET = "\033[0m";
  const std::string BOLD = "\033[1m";
  const std::string CYAN = "\033[36m";
  const std::string GREEN = "\033[32m";
  const std::string YELLOW = "\033[33m";
  const std::string MAGENTA = "\033[35m";

  std::cout << std::endl;
  std::cout << BOLD << CYAN
            << "==============================================================="
               "================="
            << RESET << std::endl;
  std::cout << BOLD << CYAN
            << "                         GPUBench HIERARCHICAL REPORT" << RESET
            << std::endl;
  std::cout << BOLD << CYAN
            << "==============================================================="
               "================="
            << RESET << std::endl;

  for (uint32_t devIdx : deviceIndices) {
    std::cout << std::endl;
    if (devIdx == 0xFFFFFFFF) {
      std::cout << BOLD << "Device: " << MAGENTA << "System" << RESET
                << " (Host CPU)" << std::endl;
    } else {
      std::cout << BOLD << "Device: " << MAGENTA << deviceNames[devIdx] << RESET
                << " (ID: " << devIdx << ")" << std::endl;
    }
    std::cout << "-------------------------------------------------------------"
                 "-------------------"
              << std::endl;

    const auto &components = organizedData[devIdx];
    for (const auto &compPair : components) {
      const std::string &compName = compPair.first;
      std::cout << "  [" << BOLD << CYAN << compName << RESET << "]"
                << std::endl;

      const auto &subcats = compPair.second;
      for (const auto &subcatPair : subcats) {
        const std::string &subcatName = subcatPair.first;
        std::cout << "    > " << YELLOW << subcatName << RESET << std::endl;

        const auto &benchmarks = subcatPair.second;
        for (const auto &benchPair : benchmarks) {
          const std::string &benchName = benchPair.first;
          std::cout << "      - " << std::left << std::setw(maxBenchNameLen) << benchName;

          const auto &backendData = benchPair.second;
          bool firstBackend = true;
          for (const auto &backend : backends) {
            if (backendData.count(backend)) {
              if (!firstBackend) {
                std::cout << std::endl << std::setw(8 + maxBenchNameLen) << "";
              }
              const auto &res = backendData.at(backend);

              double value = 0;
              std::string valStr;
              std::string unit = res.metric;

              if (res.component == "Compute") {
                // TFLOPS or TOPS
                value = (static_cast<double>(res.operations) /
                         (res.time_ms / 1000.0)) /
                        1e12;
                valStr = formatDouble(value, 2);
                if (unit == "TFLOPS")
                  unit = " TFLOPS";
                else if (unit == "TOPS")
                  unit = " TOPS";
                else
                  unit = " " + unit;
              } else if (res.component == "Memory") {
                if (res.subcategory == "Latency") {
                  value = (res.time_ms * 1e6) / res.operations; // ns
                  valStr = formatDouble(value, 2);
                  unit = " ns";
                } else {
                  value = (static_cast<double>(res.operations) /
                           (res.time_ms / 1000.0)) /
                          1e9; // GB/s
                  valStr = formatDouble(value, 2);
                  unit = " GB/s";
                }
              } else {
                value = (static_cast<double>(res.operations) /
                         (res.time_ms / 1000.0));
                valStr = formatDouble(value, 2);
                unit = " " + res.metric;
              }

              std::cout << " : " << std::setw(12) << std::right << YELLOW
                        << backend << RESET << " | " << BOLD << GREEN
                        << std::setw(10) << valStr << RESET << unit;
              firstBackend = false;
            }
          }
          std::cout << std::endl;
        }
      }
    }
  }
  std::cout << BOLD << CYAN
            << "==============================================================="
               "================="
            << RESET << std::endl;
  std::cout << std::endl;
}
