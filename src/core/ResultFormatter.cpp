#include "ResultFormatter.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <locale>

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

void ResultFormatter::print() {
    if (results.empty()) {
        return;
    }

    // Print header
    std::cout << std::left << std::setw(12) << "Backend"
              << std::setw(40) << "Device"
              << std::setw(30) << "Benchmark"
              << std::setw(20) << "Result"
              << std::setw(25) << "Operations"
              << std::setw(15) << "Time (ms)" << std::endl;
    
    std::cout << std::string(142, '-') << std::endl;

    // Print results
    for (const auto& result : results) {
        double value = 0;
        std::string unit;
        if (result.metric == "TFLOPS") {
            value = (static_cast<double>(result.operations) / (result.time_ms / 1000.0)) / 1e12;
            unit = " TFLOPs";
        } else if (result.metric == "GB/s") {
            value = (static_cast<double>(result.operations) / (result.time_ms / 1000.0)) / 1e9;
            unit = " GB/s";
        } else if (result.metric == "ns") {
            value = (result.time_ms * 1e6) / result.operations;
            unit = " ns";
        }

        std::string benchmarkName = result.benchmarkName;
        if (result.isEmulated) {
            benchmarkName += " (Emulated)";
        }

        std::string formatted_value = formatDouble(value, 3);
        std::stringstream resultStream;
        resultStream << formatted_value << unit;

        std::cout << std::left << std::setw(12) << result.backendName
                  << std::setw(40) << result.deviceName
                  << std::setw(30) << benchmarkName
                  << std::setw(20) << resultStream.str()
                  << std::setw(25) << formatNumber(result.operations)
                  << std::fixed << std::setprecision(3) << std::setw(15) << result.time_ms << std::endl;
    }
}
