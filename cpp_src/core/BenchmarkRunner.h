#pragma once

#include "benchmarks/IBenchmark.h"
#include "core/IComputeContext.h"
#include "core/ResultFormatter.h"
#include <memory>
#include <vector>
#include <functional>

struct BenchmarkGroupInfo {
  std::string name;
  std::string id;
  std::vector<std::string> aliases;
  std::string description;
  std::vector<std::string> benchmarks;
};

class BenchmarkRunner {
public:
  BenchmarkRunner(const std::vector<IComputeContext *> &contexts,
                  bool verbose = false, bool debug = false,
                  bool dumpGeometry = false, bool dumpRenders = true,
                  const std::string &scene = "indoor");
  ~BenchmarkRunner();

  void run(const std::vector<std::string> &benchmarks_to_run);
  void runForContext(IComputeContext *context, const std::vector<std::string> &benchmarks_to_run);
  void runHostBenchmarks(const std::vector<std::string> &benchmarks_to_run);
  void printReport();
  void initRunConfig(const std::vector<std::string> &benchmarks_to_run);

  std::function<void(const ResultData&)> onResult;

  std::vector<std::string> getAvailableBenchmarks() const;
  std::vector<BenchmarkGroupInfo> getAvailableGroups() const;
  std::vector<std::string> expandGroups(const std::vector<std::string> &inputs) const;
  const std::vector<ResultData>& getResults() const;

  // Names passed to run() that did not match any benchmark (populated by
  // the most recent run() call).
  const std::vector<std::string> &getUnmatchedBenchmarks() const {
    return unmatchedBenchmarks;
  }
  // Number of benchmark configs that produced a result in the most recent
  // run() call.
  uint32_t getNumBenchmarksRun() const { return numBenchmarksRun; }

  void setResolution(uint32_t w, uint32_t h) {
    renderWidth = w;
    renderHeight = h;
  }
  uint32_t getRenderWidth() const { return renderWidth; }
  uint32_t getRenderHeight() const { return renderHeight; }

  void setScene(const std::string &scene) { sceneName = scene; }
  const std::string &getScene() const { return sceneName; }

  void setBounceDepth(uint32_t b);
  uint32_t getBounceDepth() const { return bounceDepth; }

  void setTargetConfig(int config) { targetConfig = config; }
  int getTargetConfig() const { return targetConfig; }
  void setProfileSnapshot(bool enable) { profileSnapshot = enable; }
  bool getProfileSnapshot() const { return profileSnapshot; }

private:
  void discoverBenchmarks();
  void printBanner();

  std::vector<IComputeContext *> contexts;
  std::vector<std::unique_ptr<IBenchmark>> benchmarks;
  std::unique_ptr<ResultFormatter> formatter;
  std::vector<std::string> unmatchedBenchmarks;
  std::vector<std::string> effective_benchmarks;
  std::vector<std::string> lower_benchmarks_to_run;
  bool runConfigInitialized = false;
  bool bannerPrinted = false;
  uint32_t numBenchmarksRun = 0;
  bool verbose;
  bool debug;
  bool dumpGeometry;
  bool dumpRenders;
  std::string sceneName = "indoor";
  uint32_t renderWidth = 0;
  uint32_t renderHeight = 0;
  uint32_t bounceDepth = 2;
  int targetConfig = -1;
  bool profileSnapshot = false;
};
