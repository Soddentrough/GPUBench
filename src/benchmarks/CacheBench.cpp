#include "benchmarks/CacheBench.h"
#include <stdexcept>

CacheBench::CacheBench(std::string name, std::string metric, uint64_t bufferSize, std::string kernelFile)
    : name(name), metric(metric), bufferSize(bufferSize), kernelFile(kernelFile) {}

CacheBench::~CacheBench() {}

bool CacheBench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return true;
}

void CacheBench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    if (bufferSize > 0) {
        buffer = context.createBuffer(bufferSize);
    }

    std::string full_kernel_path;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        full_kernel_path = kernel_dir + "/" + kernelFile + ".spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        full_kernel_path = kernel_dir + "/hip_kernels/" + kernelFile + ".o";
    } else {
        full_kernel_path = kernel_dir + "/" + kernelFile + ".cl";
    }
    
    kernel = context.createKernel(full_kernel_path, "main", 1);
    if (buffer) {
        context.setKernelArg(kernel, 0, buffer);
    }
}

void CacheBench::Run() {
    context->dispatch(kernel, 1, 1, 1, 1, 1, 1);
}

void CacheBench::Teardown() {
    if (kernel) {
        context->releaseKernel(kernel);
    }
    if (buffer) {
        context->releaseBuffer(buffer);
    }
}

const char* CacheBench::GetName() const {
    return name.c_str();
}

const char* CacheBench::GetMetric() const {
    return metric.c_str();
}

BenchmarkResult CacheBench::GetResult() const {
    uint64_t operations = 0;
    if (name == "L0 Cache Bandwidth") {
        // 16 registers are read and written in a loop of 1024 iterations.
        // Each register is 4 bytes.
        operations = 16 * 1024 * 4 * 2;
    } else if (name == "L0 Cache Latency") {
        // 16 dependent operations in a loop of 1024 iterations.
        operations = 16 * 1024;
    } else if (metric == "GB/s") {
        operations = bufferSize * 1024 * 2; // 1024 iterations, read/write
    } else if (metric == "ns") {
        operations = 32 * 1024; // 32 dependent latency reads * 1024 iterations
    }
    return {operations, 0.0};
}
