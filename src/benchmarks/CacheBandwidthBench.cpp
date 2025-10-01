#include "benchmarks/CacheBandwidthBench.h"
#include <stdexcept>

bool CacheBandwidthBench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return true;
}

void CacheBandwidthBench::createKernel(CacheConfig& config, const std::string& kernel_dir) {
    std::string kernel_file;
    if (context->getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/" + config.kernelFile + ".spv";
    } else if (context->getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/hip_kernels/" + config.kernelFile + ".o";
    } else {
        kernel_file = kernel_dir + "/" + config.kernelFile + ".cl";
    }
    
    config.kernel = context->createKernel(kernel_file, "main", 2);
    context->setKernelArg(config.kernel, 0, buffer);
    context->setKernelArg(config.kernel, 1, sizeof(uint32_t), &config.iterations);
}

void CacheBandwidthBench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Use a working set size large enough to test Infinity Cache (128 MB)
    size_t bufferSize = 64 * 1024 * 1024; // 64 MB max working set
    buffer = context.createBuffer(bufferSize);

    // Setup different cache level configurations
    configs.push_back({"L1", "cachebw_l1", 256, 512, 16 * 1024, 1000, nullptr});
    configs.push_back({"L2", "cachebw_l2", 256, 512, 2 * 1024 * 1024, 500, nullptr});
    configs.push_back({"L3", "cachebw_l3", 256, 512, 64 * 1024 * 1024, 200, nullptr});
    
    // Create kernels for all configurations
    for (auto& config : configs) {
        createKernel(config, kernel_dir);
    }
    
    currentConfigIndex = 0;
}

void CacheBandwidthBench::Run() {
    if (currentConfigIndex >= configs.size()) {
        currentConfigIndex = 0;
    }
    
    auto& config = configs[currentConfigIndex];
    context->dispatch(config.kernel, config.numWorkgroups, 1, 1, config.workgroupSize, 1, 1);
    
    // Move to next configuration for next run
    currentConfigIndex++;
}

void CacheBandwidthBench::Teardown() {
    for (auto& config : configs) {
        if (config.kernel) {
            context->releaseKernel(config.kernel);
        }
    }
    configs.clear();
    
    if (buffer) {
        context->releaseBuffer(buffer);
    }
}

BenchmarkResult CacheBandwidthBench::GetResult() const {
    if (currentConfigIndex == 0 || currentConfigIndex > configs.size()) {
        return {0, 0.0};
    }
    
    // Get the config that just ran
    const auto& config = configs[currentConfigIndex - 1];
    
    uint64_t total_threads = (uint64_t)config.workgroupSize * config.numWorkgroups;
    uint64_t vec4_per_thread = (config.name.find("L3") != std::string::npos) ? 32 : 1;
    uint64_t bytes_per_iteration = total_threads * vec4_per_thread * 16; // vec4 = 16 bytes
    uint64_t bytes_accessed = bytes_per_iteration * config.iterations;
    
    return {bytes_accessed, 0.0};
}
