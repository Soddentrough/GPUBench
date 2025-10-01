#include "Fp6Bench.h"
#include <stdexcept>

Fp6Bench::Fp6Bench() {}

Fp6Bench::~Fp6Bench() {}

const char* Fp6Bench::GetName() const {
    return "FP6";
}

bool Fp6Bench::IsSupported(const DeviceInfo& device, IComputeContext* context) const {
    // FP6 is not a standard data type, so we will assume it requires emulation.
    // The logic for emulation will be added in a future step.
    return false;
}

void Fp6Bench::Setup(IComputeContext& context, const std::string& build_dir) {
    this->context = &context;
    // Implementation will be added in a future step.
}

void Fp6Bench::Run() {
    // Implementation will be added in a future step.
}

BenchmarkResult Fp6Bench::GetResult() const {
    // Implementation will be added in a future step.
    return {0, 0};
}

void Fp6Bench::Teardown() {
    // Implementation will be added in a future step.
}
