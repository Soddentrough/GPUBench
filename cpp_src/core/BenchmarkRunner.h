#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include "core/ResultFormatter.h"
#include <memory>
#include <vector>
#include <functional>

class BenchmarkRunner {
public:
  BenchmarkRunner(const std::vector<IComputeContext *> &contexts,
                  bool verbose = false, bool debug = false,
                  bool dumpGeometry = false);
  ~BenchmarkRunner();

  void run(const std::vector<std::string> &benchmarks_to_run);

  std::function<void(const ResultData&)> onResult;

  std::vector<std::string> getAvailableBenchmarks() const;
  const std::vector<ResultData>& getResults() const;

private:
  void discoverBenchmarks();

  std::vector<IComputeContext *> contexts;
  std::vector<std::unique_ptr<IBenchmark>> benchmarks;
  std::unique_ptr<ResultFormatter> formatter;
  bool verbose;
  bool debug;
  bool dumpGeometry;
};
