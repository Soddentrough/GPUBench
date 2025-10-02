#include "benchmarks/MemBandwidthBench.h"
#include <stdexcept>

bool MemBandwidthBench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return true;
}

void MemBandwidthBench::createKernel(BandwidthConfig& config, const std::string& kernel_dir) {
    std::string kernel_file;
    if (context->getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/" + config.kernelFile + ".spv";
    } else if (context->getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/hip_kernels/" + config.kernelFile + ".o";
    } else {
        kernel_file = kernel_dir + "/" + config.kernelFile + ".cl";
    }
    
    std::string kernel_name = (context->getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
    config.kernel = context->createKernel(kernel_file, kernel_name, 2);
    context->setKernelArg(config.kernel, 0, inputBuffer);
    context->setKernelArg(config.kernel, 1, outputBuffer);
}

void MemBandwidthBench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    size_t bufferSize = 256 * 1024 * 1024; // 256 MB
    inputBuffer = context.createBuffer(bufferSize);
    outputBuffer = context.createBuffer(bufferSize);

    configs.push_back({"128 threads/group", "membw_128", 128, 4096, nullptr});
    configs.push_back({"256 threads/group", "membw_256", 256, 2048, nullptr});
    configs.push_back({"1024 threads/group", "membw_1024", 1024, 512, nullptr});
    
    for (auto& config : configs) {
        createKernel(config, kernel_dir);
    }
    
    currentConfigIndex = 0;
}

void MemBandwidthBench::Run() {
    if (currentConfigIndex >= configs.size()) {
        currentConfigIndex = 0;
    }
    
    auto& config = configs[currentConfigIndex];
    context->dispatch(config.kernel, config.numWorkgroups, 1, 1, config.workgroupSize, 1, 1);
    
    currentConfigIndex++;
}

void MemBandwidthBench::Teardown() {
    for (auto& config : configs) {
        if (config.kernel) {
            context->releaseKernel(config.kernel);
        }
    }
    configs.clear();
    
    if (inputBuffer) {
        context->releaseBuffer(inputBuffer);
    }
    if (outputBuffer) {
        context->releaseBuffer(outputBuffer);
    }
}

const char* MemBandwidthBench::GetName() const {
    return "Memory Bandwidth";
}

const char* MemBandwidthBench::GetMetric() const {
    return "GB/s";
}

BenchmarkResult MemBandwidthBench::GetResult() const {
    if (currentConfigIndex == 0 || currentConfigIndex > configs.size()) {
        return {0, 0.0};
    }
    
    const auto& config = configs[currentConfigIndex - 1];
    // Each thread transfers 16 vec4s (16*16=256 bytes) per iteration, for 1024 iterations, times 2 for read/write
    uint64_t bytes_transferred = (uint64_t)config.workgroupSize * config.numWorkgroups * 256 * 1024 * 2;
    return {bytes_transferred, 0.0};
}
