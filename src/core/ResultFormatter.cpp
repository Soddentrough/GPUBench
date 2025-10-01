#include "ResultFormatter.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <locale>

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

void ResultFormatter::print() {
    if (results.empty()) {
        return;
    }

    // Print header
    std::cout << std::left << std::setw(12) << "Backend"
              << std::setw(30) << "Device"
              << std::setw(20) << "Benchmark"
              << std::setw(15) << "TFLOPS"
              << std::setw(25) << "Operations"
              << std::setw(15) << "Time (ms)" << std::endl;
    
    std::cout << std::string(117, '-') << std::endl;

    // Print results
    for (const auto& result : results) {
        double tflops = (static_cast<double>(result.operations) / (result.time_ms / 1000.0)) / 1e12;
        std::string benchmarkName = result.benchmarkName;
        if (result.isEmulated) {
            benchmarkName += " (Emulated)";
        }

        std::cout << std::left << std::setw(12) << result.backendName
                  << std::setw(30) << result.deviceName
                  << std::setw(20) << benchmarkName
                  << std::fixed << std::setprecision(3) << std::setw(15) << tflops
                  << std::setw(25) << formatNumber(result.operations)
                  << std::fixed << std::setprecision(3) << std::setw(15) << result.time_ms << std::endl;
    }
}
