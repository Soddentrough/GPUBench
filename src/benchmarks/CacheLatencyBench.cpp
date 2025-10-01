#include "CacheLatencyBench.h"
#include <stdexcept>

CacheLatencyBench::CacheLatencyBench() {}

CacheLatencyBench::~CacheLatencyBench() {}

const char* CacheLatencyBench::GetName() const {
    return "Cache Latency";
}

bool CacheLatencyBench::IsSupported(const DeviceInfo& device, IComputeContext* context) const {
    // For now, let's assume it's supported on all devices.
    // We can add more specific checks later if needed.
    return true;
}

void CacheLatencyBench::Setup(IComputeContext& context, const std::string& build_dir) {
    this->context = &context;
    // Implementation will be added in a future step.
}

void CacheLatencyBench::Run() {
    // Implementation will be added in a future step.
}

BenchmarkResult CacheLatencyBench::GetResult() const {
    // Implementation will be added in a future step.
    return {0, 0};
}

void CacheLatencyBench::Teardown() {
    // Implementation will be added in a future step.
}
