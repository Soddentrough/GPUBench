#include "benchmarks/Fp16Bench.h"
#include <stdexcept>

bool Fp16Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return info.fp16Support;
}

void Fp16Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (f16vec2)
    buffer = context.createBuffer(bufferSize);
    
    // Initialize buffer
    // Use uint32_t to fill with zeros (size is in bytes)
    std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0);
    context.writeBuffer(buffer, 0, bufferSize, initData.data());

    // Create kernel
    std::string kernel_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/vulkan/fp16.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/rocm/fp16.co";
    } else {
        kernel_file = kernel_dir + "/opencl/fp16.cl";
    }
    
    std::string kernel_name;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_name = "main";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_name = "run_benchmark";
    } else {
        kernel_name = "run_benchmark";
    }
    kernel = context.createKernel(kernel_file, kernel_name, 1);
    context.setKernelArg(kernel, 0, buffer);
}

void Fp16Bench::Run(uint32_t config_idx) {
    context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
}

void Fp16Bench::Teardown() {
    if (kernel) {
        context->releaseKernel(kernel);
    }
    if (buffer) {
        context->releaseBuffer(buffer);
    }
}

BenchmarkResult Fp16Bench::GetResult(uint32_t config_idx) const {
    // 8 f16vec2 FMAs per iteration = 8 * 2 * 2 = 32 FP16 operations per iteration
    // 16384 iterations * 32 ops * 8192 workgroups * 64 threads
    uint64_t num_ops = (uint64_t)16384 * 32 * 8192 * 64;
    return {num_ops, 0.0};
}
