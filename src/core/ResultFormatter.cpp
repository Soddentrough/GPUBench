#include "ResultFormatter.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <locale>
#include <map>
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

void ResultFormatter::addResult(const ResultData& result) {
    results.push_back(result);
}

enum class BenchmarkCategory {
    Compute,
    Memory,
    Latency,
    Unknown
};

// Helper to lowercase a string
static std::string to_lower_static(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

BenchmarkCategory getBenchmarkCategory(const std::string& name) {
    std::string lower_name = to_lower_static(name);
    if (lower_name.find("fp") != std::string::npos || lower_name.find("int") != std::string::npos) {
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

    // Group results by backend and device
    std::map<std::string, std::map<std::string, std::vector<ResultData>>> groupedResults;
    for (const auto& result : results) {
        groupedResults[result.backendName][result.deviceName].push_back(result);
    }

    for (const auto& backend_pair : groupedResults) {
        for (const auto& device_pair : backend_pair.second) {
            std::cout << "----------------------------------------------------------------" << std::endl;
            std::cout << "Backend: " << backend_pair.first << std::endl;
            std::cout << "Device:  " << device_pair.first << std::endl;
            // Get max workgroup size from first result (all results from same device have same value)
            if (!device_pair.second.empty()) {
                std::cout << "Maximum workgroup size: " << device_pair.second[0].maxWorkGroupSize << std::endl;
            }
            std::cout << "----------------------------------------------------------------" << std::endl;

            std::map<BenchmarkCategory, std::vector<ResultData>> categorizedResults;
            for (const auto& result : device_pair.second) {
                categorizedResults[getBenchmarkCategory(result.benchmarkName)].push_back(result);
            }

            if (!categorizedResults[BenchmarkCategory::Compute].empty()) {
                std::cout << "\n[Compute Benchmarks]" << std::endl;
                for (const auto& result : categorizedResults[BenchmarkCategory::Compute]) {
                    double value = (static_cast<double>(result.operations) / (result.time_ms / 1000.0)) / 1e12;
                    std::string unit = "TFLOPs";
                    if (result.benchmarkName.find("Int") != std::string::npos) {
                        unit = "TOPS";
                    }
                    std::string benchmarkName = result.benchmarkName;
                     if (result.isEmulated) {
                        benchmarkName += " (Emulated)";
                    }
                    std::cout << "  - " << std::left << std::setw(50) << benchmarkName + ":"
                              << std::right << std::setw(10) << formatDouble(value, 2) << " " << unit << std::endl;
                }
            }

            if (!categorizedResults[BenchmarkCategory::Memory].empty()) {
                std::cout << "\n[Memory Benchmarks]" << std::endl;
                for (const auto& result : categorizedResults[BenchmarkCategory::Memory]) {
                    double value = (static_cast<double>(result.operations) / (result.time_ms / 1000.0)) / 1e9;
                    std::cout << "  - " << std::left << std::setw(50) << result.benchmarkName + ":"
                              << std::right << std::setw(10) << formatDouble(value, 2) << " GB/s" << std::endl;
                }
            }

            if (!categorizedResults[BenchmarkCategory::Latency].empty()) {
                std::cout << "\n[Latency Benchmarks]" << std::endl;
                for (const auto& result : categorizedResults[BenchmarkCategory::Latency]) {
                    double value = (result.time_ms * 1e6) / result.operations;
                    std::cout << "  - " << std::left << std::setw(50) << result.benchmarkName + ":"
                              << std::right << std::setw(10) << formatDouble(value, 2) << " ns" << std::endl;
                }
            }
            std::cout << std::endl;
        }
    }
}
