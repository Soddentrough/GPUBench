#include "benchmarks/MemBandwidthBench.h"
#include <stdexcept>

bool MemBandwidthBench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return true;
}

void MemBandwidthBench::createKernel(BandwidthConfig& config, const std::string& kernel_dir) {
        std::string kernel_file;
        if (context->getBackend() == ComputeBackend::Vulkan) {
            kernel_file = kernel_dir + "/vulkan/" + config.kernelFile + ".spv";
        } else if (context->getBackend() == ComputeBackend::ROCm) {
            kernel_file = kernel_dir + "/rocm/" + config.kernelFile + ".o";
        } else {
            kernel_file = kernel_dir + "/opencl/" + config.kernelFile + ".cl";
        }
    
    std::string kernel_name;
    if (context->getBackend() == ComputeBackend::Vulkan) {
        kernel_name = "main";
    } else if (context->getBackend() == ComputeBackend::ROCm) {
        kernel_name = "run_benchmark";
    } else {
        kernel_name = "run_benchmark";
    }
    config.kernel = this->context->createKernel(kernel_file, kernel_name, 4);
    this->context->setKernelArg(config.kernel, 0, inputBuffer);
    this->context->setKernelArg(config.kernel, 1, outputBuffer);
    uint32_t mode = static_cast<uint32_t>(config.mode);
    this->context->setKernelArg(config.kernel, 2, sizeof(mode), &mode);
    uint32_t bufferSizeParam = static_cast<uint32_t>(bufferSize);
    this->context->setKernelArg(config.kernel, 3, sizeof(bufferSizeParam), &bufferSizeParam);
}

void MemBandwidthBench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Get device info first
    DeviceInfo deviceInfo = context.getCurrentDeviceInfo();
    
    // Use a fixed 2GB buffer size per buffer
    // This is large enough to measure memory bandwidth while avoiding OOM errors
    // Ensure buffer size is a multiple of 16 bytes (vec4 size)
    this->bufferSize = 2UL * 1024 * 1024 * 1024; // 2 GB
    this->bufferSize = (this->bufferSize / 16) * 16; // Round down to multiple of 16
    
    inputBuffer = this->context->createBuffer(bufferSize);
    outputBuffer = this->context->createBuffer(bufferSize);

    // Initialize input buffer with test data to prevent reading uninitialized memory
    std::vector<float> testData(bufferSize / sizeof(float), 1.0f);
    this->context->writeBuffer(inputBuffer, 0, bufferSize, testData.data());
    this->context->waitIdle();

    // Get device max workgroup size and clamp configurations to it
    uint32_t maxWorkgroupSize = deviceInfo.maxWorkGroupSize;
    
    configs.push_back({"Read 128 threads/group", "membw_128", 128, 4096, TestMode::Read, nullptr});
    configs.push_back({"Write 128 threads/group", "membw_128", 128, 4096, TestMode::Write, nullptr});
    configs.push_back({"R/W 128 threads/group", "membw_128", 128, 4096, TestMode::ReadWrite, nullptr});

    configs.push_back({"Read 256 threads/group", "membw_256", std::min(256u, maxWorkgroupSize), 2048, TestMode::Read, nullptr});
    configs.push_back({"Write 256 threads/group", "membw_256", std::min(256u, maxWorkgroupSize), 2048, TestMode::Write, nullptr});
    configs.push_back({"R/W 256 threads/group", "membw_256", std::min(256u, maxWorkgroupSize), 2048, TestMode::ReadWrite, nullptr});
    
    // Only add 1024 config if device supports it
    if (maxWorkgroupSize >= 1024) {
        configs.push_back({"Read 1024 threads/group", "membw_1024", 1024, 512, TestMode::Read, nullptr});
        configs.push_back({"Write 1024 threads/group", "membw_1024", 1024, 512, TestMode::Write, nullptr});
        configs.push_back({"R/W 1024 threads/group", "membw_1024", 1024, 512, TestMode::ReadWrite, nullptr});
    }
    
    for (auto& config : configs) {
        createKernel(config, kernel_dir);
    }
}

void MemBandwidthBench::Run(uint32_t config_idx) {
    if (config_idx >= configs.size()) {
        throw std::runtime_error("Invalid config index in MemBandwidthBench::Run");
    }
    
    auto& config = configs[config_idx];
    context->dispatch(config.kernel, config.numWorkgroups, 1, 1, config.workgroupSize, 1, 1);
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

BenchmarkResult MemBandwidthBench::GetResult(uint32_t config_idx) const {
    if (config_idx >= configs.size()) {
        return {0, 0.0};
    }
    
    const auto& config = configs[config_idx];
    // Each thread transfers 32 vec4s (32*16=512 bytes) per iteration, for 32 iterations
    uint64_t bytes_transferred = (uint64_t)config.workgroupSize * config.numWorkgroups * 512 * 32;
    if (config.mode == TestMode::ReadWrite) {
        bytes_transferred *= 2;
    }
    return {bytes_transferred, 0.0};
}

uint32_t MemBandwidthBench::GetNumConfigs() const {
    return configs.size();
}

std::string MemBandwidthBench::GetConfigName(uint32_t config_idx) const {
    if (config_idx >= configs.size()) {
        return "Invalid Config";
    }
    return configs[config_idx].name;
}
