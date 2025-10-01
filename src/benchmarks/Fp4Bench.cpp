#include "benchmarks/Fp4Bench.h"
#include <stdexcept>

#include <iostream>

bool Fp4Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    // Check for gfx942, the first architecture to support FP4/FP8
    if (info.name.find("gfx942") != std::string::npos) {
        return true;
    }
    
    if (info.verbose) {
        std::cout << "FP4 benchmark not supported on device: " << info.name << std::endl;
    }
    return false;
}

void Fp4Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (u8vec4)
    buffer = context.createBuffer(bufferSize);

    // Create kernel
    std::string kernel_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/fp4.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/hip_kernels/fp4.o";
    } else {
        kernel_file = kernel_dir + "/fp4.cl";
    }
    
    kernel = context.createKernel(kernel_file, "compute", 1);
    context.setKernelArg(kernel, 0, buffer);
}

void Fp4Bench::Run() {
    context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
}

void Fp4Bench::Teardown() {
    if (kernel) {
        context->releaseKernel(kernel);
    }
    if (buffer) {
        context->releaseBuffer(buffer);
    }
}

BenchmarkResult Fp4Bench::GetResult() const {
    // 12 u8vec4 operations per iteration, each with multiply-add + AND = 3 ops per component
    // 12 * 4 * 3 = 144 FP4-equivalent operations per iteration
    // 16384 iterations * 144 ops * 8192 workgroups * 64 threads
    uint64_t num_ops = (uint64_t)16384 * 144 * 8192 * 64;
    return {num_ops, 0.0};
}
