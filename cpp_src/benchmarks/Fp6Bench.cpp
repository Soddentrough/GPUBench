#include "Fp6Bench.h"
#include <stdexcept>

bool Fp6Bench::IsSupported(const DeviceInfo &info,
                           IComputeContext *context) const {
  return info.fp6Support;
}

void Fp6Bench::Setup(IComputeContext &context, const std::string &build_dir) {
  this->context = &context;

  DeviceInfo info = context.getCurrentDeviceInfo();
  // The logic for emulation based on device name is removed as per the
  // instruction's implied change.

  // Implementation will be added in a future step.
}

void Fp6Bench::Run(uint32_t config_idx) {
  // Implementation will be added in a future step.
}

BenchmarkResult Fp6Bench::GetResult(uint32_t config_idx) const {
  // Implementation will be added in a future step.
  return {0, 0};
}

void Fp6Bench::Teardown() {
  // Implementation will be added in a future step.
}
