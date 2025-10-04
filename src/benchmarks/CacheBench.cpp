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
        full_kernel_path = kernel_dir + "/vulkan/" + kernelFile + ".spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        full_kernel_path = kernel_dir + "/rocm/" + kernelFile + ".o";
    } else {
        full_kernel_path = kernel_dir + "/opencl/" + kernelFile + ".cl";
    }
    
    std::string kernel_name;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_name = "main";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_name = "run_benchmark";
    } else {
        kernel_name = "run_benchmark";
    }
    kernel = context.createKernel(full_kernel_path, kernel_name, 1);
    if (buffer) {
        context.setKernelArg(kernel, 0, buffer);
    }
}

void CacheBench::Run() {
    if (!context) {
        throw std::runtime_error("Context is not set up");
    }
    if (metric == "GB/s") {
        // For bandwidth, we want to saturate the GPU with threads.
        // The shader uses a workgroup size of 256.
        uint32_t numWorkgroups = 65536;
        
        // For L3 cache bandwidth with the cachebw_l3 kernel, reduce workgroups to prevent 
        // out-of-bounds access. The kernel uses workgroupOffset = get_group_id(0) * 8192,
        // so with a 16MB buffer (1M float4 elements), we can only safely use ~122 workgroups.
        if (kernelFile == "cachebw_l3" && bufferSize == 16 * 1024 * 1024) {
            // bufferSize / sizeof(float4) / 8192 = 16MB / 16 / 8192 = ~122
            numWorkgroups = 100;  // Use 100 to be safe
        }
        
        context->dispatch(kernel, numWorkgroups, 1, 1, 256, 1, 1);
    } else {
        // For latency, we run a single thread to measure the dependency chain.
        context->dispatch(kernel, 1, 1, 1, 1, 1, 1);
    }
}

void CacheBench::Teardown() {
    if (kernel) {
        context->releaseKernel(kernel);
        kernel = nullptr;
    }
    if (buffer) {
        context->releaseBuffer(buffer);
        buffer = nullptr;
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
