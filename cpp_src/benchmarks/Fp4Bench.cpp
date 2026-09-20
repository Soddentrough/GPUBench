#include "benchmarks/Fp4Bench.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

bool Fp4Bench::IsSupported(const DeviceInfo &info,
                           IComputeContext *context) const {
  // FP4 is completely emulated on all current backends via FP16.
  // Disabled to prevent wildly inaccurate results.
  return false;
}

void Fp4Bench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;
  this->kernel = nullptr;
}

void Fp4Bench::Run(uint32_t config_idx) {
  if (kernel) {
    context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
  }
}

void Fp4Bench::Teardown() {
  if (kernel) {
    context->releaseKernel(kernel);
    kernel = nullptr;
  }
  if (buffer) {
    context->releaseBuffer(buffer);
    buffer = nullptr;
  }
}

BenchmarkResult Fp4Bench::GetResult(uint32_t config_idx) const {
  if (!kernel) return {0, 0.0};
  
  // Shader (fp4_emulated.comp): 16 f16vec4 FMAs per iteration.
  // Each FMA on f16vec4 = 4 components × 2 ops (mul+add) = 8 ops per FMA.
  // 16 FMAs × 8 = 128 FP4-equivalent operations per iteration.
  // 16384 iterations × 128 ops × 8192 workgroups × 64 threads
  uint64_t num_ops = (uint64_t)16384 * 128 * 8192 * 64;
  return {num_ops, 0.0};
}

uint32_t Fp4Bench::GetNumConfigs() const {
  return kernel ? 1 : 0;
}

std::string Fp4Bench::GetConfigName(uint32_t config_idx) const {
  return "Vector";
}
