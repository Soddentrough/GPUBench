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

  // Names passed to run() that did not match any benchmark (populated by
  // the most recent run() call).
  const std::vector<std::string> &getUnmatchedBenchmarks() const {
    return unmatchedBenchmarks;
  }
  // Number of benchmark configs that produced a result in the most recent
  // run() call.
  uint32_t getNumBenchmarksRun() const { return numBenchmarksRun; }

private:
  void discoverBenchmarks();

  std::vector<IComputeContext *> contexts;
  std::vector<std::unique_ptr<IBenchmark>> benchmarks;
  std::unique_ptr<ResultFormatter> formatter;
  std::vector<std::string> unmatchedBenchmarks;
  uint32_t numBenchmarksRun = 0;
  bool verbose;
  bool debug;
  bool dumpGeometry;
};
