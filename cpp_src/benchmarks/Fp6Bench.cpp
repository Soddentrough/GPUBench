#include "Fp6Bench.h"
#include <stdexcept>

bool Fp6Bench::IsSupported(const DeviceInfo &info,
                           IComputeContext *context) const {
  return info.fp6Support;
}

void Fp6Bench::Setup(IComputeContext &context, const std::string &build_dir) {
  (void)context;
  (void)build_dir;
  throw std::runtime_error("FP6 benchmark is unsupported on this hardware/API (NVIDIA SPV_NV_float6 only)");
}

void Fp6Bench::Run(uint32_t config_idx) {
  (void)config_idx;
  throw std::runtime_error("FP6 benchmark is unsupported on this hardware/API (NVIDIA SPV_NV_float6 only)");
}

BenchmarkResult Fp6Bench::GetResult(uint32_t config_idx) const {
  // Implementation will be added in a future step.
  return {0, 0};
}

void Fp6Bench::Teardown() {
  // Implementation will be added in a future step.
}
