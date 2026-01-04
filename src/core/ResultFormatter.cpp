#include "ResultFormatter.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <locale>
#include <map>
#include <vector>
#include <set>

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

    // 1. Identify all unique devices (by index) and map to names
    std::set<uint32_t> deviceIndices;
    std::map<uint32_t, std::string> deviceNames;
    for (const auto& res : results) {
        deviceIndices.insert(res.deviceIndex);
        if (deviceNames.find(res.deviceIndex) == deviceNames.end()) {
            deviceNames[res.deviceIndex] = res.deviceName;
        }
    }

    // 2. Identify all unique backends
    std::set<std::string> backends;
    for (const auto& res : results) {
        backends.insert(res.backendName);
    }

    // 3. Identify all unique benchmarks and categorize them
    std::vector<std::string> benchmarkOrder;
    std::map<std::string, BenchmarkCategory> benchmarkCategories;
    for (const auto& result : results) {
        std::string name = result.benchmarkName;
        if (result.isEmulated) name += " (Emulated)";
        
        if (benchmarkCategories.find(name) == benchmarkCategories.end()) {
            benchmarkCategories[name] = getBenchmarkCategory(result.benchmarkName);
            benchmarkOrder.push_back(name);
        }
    }

    // Sort benchmarks
    auto get_type_rank = [](const std::string& name) {
        std::string lower = to_lower_static(name);
        if (lower.find("fp64") != std::string::npos) return 0;
        if (lower.find("fp32") != std::string::npos) return 1;
        if (lower.find("fp16") != std::string::npos) return 2;
        if (lower.find("fp8") != std::string::npos) return 3;
        if (lower.find("fp6") != std::string::npos) return 4;
        if (lower.find("fp4") != std::string::npos) return 5;
        if (lower.find("int8") != std::string::npos) return 6;
        if (lower.find("int4") != std::string::npos) return 7;
        return 100;
    };

    auto get_sub_rank = [](const std::string& name) {
        std::string lower = to_lower_static(name);
        if (lower.find("matrix") != std::string::npos) return 1;
        if (lower.find("vector") != std::string::npos) return 0;
        return 2;
    };

    std::sort(benchmarkOrder.begin(), benchmarkOrder.end(), 
        [&](const std::string& a, const std::string& b) {
            if (benchmarkCategories[a] != benchmarkCategories[b]) {
                return (int)benchmarkCategories[a] < (int)benchmarkCategories[b];
            }
            if (benchmarkCategories[a] == BenchmarkCategory::Compute) {
                int rankA = get_type_rank(a);
                int rankB = get_type_rank(b);
                if (rankA != rankB) return rankA < rankB;
                
                int subA = get_sub_rank(a);
                int subB = get_sub_rank(b);
                return subA < subB;
            }
            return a < b;
        });

    // 4. Organize data: [Benchmark][Backend][DeviceIndex] -> ResultData
    std::map<std::string, std::map<std::string, std::map<uint32_t, ResultData>>> data;
    for (const auto& res : results) {
        std::string name = res.benchmarkName;
        if (res.isEmulated) name += " (Emulated)";
        data[name][res.backendName][res.deviceIndex] = res;
    }

    // 5. Print Table
    int nameWidth = 45;
    int colWidth = 25;
    for (const auto& pair : deviceNames) {
        colWidth = std::max(colWidth, (int)pair.second.length() + 2);
    }

    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string CYAN = "\033[36m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";

    auto repeat = [](const std::string& s, int n) {
        std::string res;
        for (int i = 0; i < n; ++i) res += s;
        return res;
    };

    auto print_line = [&](const std::string& start, const std::string& mid, const std::string& sep, const std::string& end) {
        std::cout << start << repeat(mid, nameWidth);
        for (size_t i = 0; i < deviceIndices.size(); ++i) {
            std::cout << sep << repeat(mid, colWidth + 2);
        }
        std::cout << end << std::endl;
    };

    auto print_row_start = [&](const std::string& text, const std::string& color = "", bool bold = false) {
        std::cout << "│ " << (bold ? BOLD : "") << color << std::left << std::setw(nameWidth - 1) << text << RESET;
    };

    auto print_row_cell = [&](const std::string& text, const std::string& unit = "", const std::string& color = "", bool right = true) {
        std::cout << "│ ";
        if (!color.empty()) std::cout << color;
        if (right) {
            std::cout << std::right << std::setw(colWidth - (int)unit.length()) << text << RESET << unit;
        } else {
            std::cout << std::left << std::setw(colWidth) << text << RESET;
        }
        std::cout << " ";
    };

    std::cout << std::endl;
    print_line("┌", "─", "┬", "┐");
    print_row_start("BENCHMARK RESULTS", "", true);
    for (size_t i = 0; i < deviceIndices.size(); ++i) print_row_cell("", "", "", false);
    std::cout << "│" << std::endl;
    print_line("├", "─", "┼", "┤");

    // Header Row
    print_row_start("Benchmark", "", true);
    for (uint32_t idx : deviceIndices) {
        print_row_cell(deviceNames[idx], "", "", true);
    }
    std::cout << "│" << std::endl;
    print_line("├", "─", "┼", "┤");

    BenchmarkCategory currentCat = BenchmarkCategory::Unknown;
    for (const auto& benchName : benchmarkOrder) {
        BenchmarkCategory cat = benchmarkCategories[benchName];
        if (cat != currentCat) {
            currentCat = cat;
            std::string catName;
            switch(cat) {
                case BenchmarkCategory::Compute: catName = "Compute"; break;
                case BenchmarkCategory::Memory: catName = "Memory Bandwidth"; break;
                case BenchmarkCategory::Latency: catName = "Latency"; break;
                default: catName = "Other"; break;
            }
            print_row_start("[ " + catName + " ]", CYAN, true);
            for (size_t i = 0; i < deviceIndices.size(); ++i) print_row_cell("", "", "", false);
            std::cout << "│" << std::endl;
        }

        // Print Benchmark Name Row
        print_row_start(benchName);
        for (size_t i = 0; i < deviceIndices.size(); ++i) print_row_cell("", "", "", false);
        std::cout << "│" << std::endl;

        // Print Backend Rows
        for (const auto& backend : backends) {
            if (data[benchName].find(backend) == data[benchName].end()) continue;

            print_row_start("  " + backend, YELLOW);
            
            for (uint32_t devIdx : deviceIndices) {
                if (data[benchName][backend].count(devIdx)) {
                    const auto& res = data[benchName][backend][devIdx];
                    std::string valStr;
                    std::string unit;
                    
                    if (cat == BenchmarkCategory::Compute) {
                        double val = (static_cast<double>(res.operations) / (res.time_ms / 1000.0)) / 1e12;
                        valStr = formatDouble(val, 2);
                        unit = (res.benchmarkName.find("Int") != std::string::npos || res.benchmarkName.find("INT") != std::string::npos) ? " TOPS" : " TFLOPs";
                    } else if (cat == BenchmarkCategory::Memory) {
                        double val = (static_cast<double>(res.operations) / (res.time_ms / 1000.0)) / 1e9;
                        valStr = formatDouble(val, 2);
                        unit = " GB/s";
                    } else if (cat == BenchmarkCategory::Latency) {
                        double val = (res.time_ms * 1e6) / res.operations;
                        valStr = formatDouble(val, 2);
                        unit = " ns";
                    }
                    print_row_cell(valStr, unit, GREEN, true);
                } else {
                    print_row_cell("-", "", "", true);
                }
            }
            std::cout << "│" << std::endl;
        }
    }
    print_line("└", "─", "┴", "┘");
    std::cout << std::endl;
}
