#include "benchmarks/Fp32Bench.h"
#include <stdexcept>

bool Fp32Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    // FP32 is universally supported
    return true;
}

void Fp32Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * sizeof(float); // 8192 workgroups * 64 threads * 4 bytes
    buffer = context.createBuffer(bufferSize);

    // Create kernel
    std::string kernel_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/fp32.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/hip_kernels/fp32.o";
    } else {
        kernel_file = kernel_dir + "/fp32.cl";
    }
    
    std::string kernel_name = (context.getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
    kernel = context.createKernel(kernel_file, kernel_name, 1);
    context.setKernelArg(kernel, 0, buffer);
}

void Fp32Bench::Run() {
    // Increase to 8192 workgroups for better GPU saturation
    context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
}

void Fp32Bench::Teardown() {
    if (kernel) {
        context->releaseKernel(kernel);
        kernel = nullptr;
    }
    if (buffer) {
        context->releaseBuffer(buffer);
        buffer = nullptr;
    }
}

BenchmarkResult Fp32Bench::GetResult() const {
    // 4 vec4 FMAs per iteration = 4 * 4 * 2 = 32 FP32 operations per iteration
    // 16384 iterations * 32 ops * 8192 workgroups * 64 threads
    uint64_t num_ops = (uint64_t)16384 * 32 * 8192 * 64;
    return {num_ops, 0.0};
}
