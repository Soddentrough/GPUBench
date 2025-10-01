#include "benchmarks/CacheBench.h"
#include <stdexcept>

CacheBench::CacheBench(std::string name, std::string metric, uint64_t bufferSize, std::string kernelFile, std::vector<uint32_t> initData)
    : name(name), metric(metric), bufferSize(bufferSize), kernelFile(kernelFile), initData(initData) {}

CacheBench::~CacheBench() {}

bool CacheBench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return true;
}

void CacheBench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    if (bufferSize > 0) {
        buffer = context.createBuffer(bufferSize);
        if (!initData.empty()) {
            context.writeBuffer(buffer, 0, initData.size() * sizeof(uint32_t), initData.data());
        }
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
    if (metric == "GB/s") {
        // For bandwidth, we want to saturate the GPU with threads.
        // Dispatch a large number of workgroups.
        // The shader uses a workgroup size of 256.
        context->dispatch(kernel, 65536, 1, 1, 256, 1, 1);
    } else {
        // For latency, we run a single thread to measure the dependency chain.
        context->dispatch(kernel, 1, 1, 1, 1, 1, 1);
    }
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
    const uint64_t num_threads_bw = 65536 * 256;

    if (name == "L0 Cache Bandwidth") {
        // 16 independent ops * 1024 iterations * num_threads
        operations = 16 * 1024 * num_threads_bw;
    } else if (name == "L0 Cache Latency") {
        // 16 dependent ops * 128 iterations
        operations = 16 * 128;
    } else if (metric == "GB/s") {
        // Each thread reads 1024 times.
        // We multiply by sizeof(uint) to get bytes.
        operations = num_threads_bw * 1024 * sizeof(uint32_t);
    } else if (metric == "ns") {
        // 1024 dependent reads
        operations = 1024;
    }
    return {operations, 0.0};
}
