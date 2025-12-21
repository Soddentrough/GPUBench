#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct ResultData {
    std::string backendName;
    std::string deviceName;
    std::string benchmarkName;
    std::string metric;
    uint64_t operations;
    double time_ms;
    bool isEmulated;
    uint32_t maxWorkGroupSize;
    uint32_t deviceIndex;
};

class ResultFormatter {
public:
    ResultFormatter();
    ~ResultFormatter();

    void addResult(const ResultData& result);
    void print();

private:
    std::string formatNumber(uint64_t n);
    std::string formatDouble(double value, int precision);
    std::vector<ResultData> results;
};
