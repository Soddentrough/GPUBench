#pragma once

#include "core/IComputeContext.h"
#include "benchmarks/IBenchmark.h"
#include <vector>
#include <memory>

class ResultFormatter;

class BenchmarkRunner {
public:
    BenchmarkRunner(const std::vector<IComputeContext*>& contexts);
    ~BenchmarkRunner();

    void run(const std::vector<std::string>& benchmarks_to_run);

private:
    void discoverBenchmarks();

    std::vector<IComputeContext*> contexts;
    std::vector<std::unique_ptr<IBenchmark>> benchmarks;
    std::unique_ptr<ResultFormatter> formatter;
};
