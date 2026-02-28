#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include <memory>
#include <vector>

class ResultFormatter;

class BenchmarkRunner {
public:
  BenchmarkRunner(const std::vector<IComputeContext *> &contexts,
                  bool verbose = false, bool debug = false,
                  bool dumpGeometry = false);
  ~BenchmarkRunner();

  void run(const std::vector<std::string> &benchmarks_to_run);

  std::vector<std::string> getAvailableBenchmarks() const;

private:
  void discoverBenchmarks();

  std::vector<IComputeContext *> contexts;
  std::vector<std::unique_ptr<IBenchmark>> benchmarks;
  std::unique_ptr<ResultFormatter> formatter;
  bool verbose;
  bool debug;
  bool dumpGeometry;
};
