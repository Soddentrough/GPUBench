#include "benchmarks/Int8Bench.h"
#include <stdexcept>
#include <vulkan/vulkan.h>

bool Int8Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return info.int8Support;
}

void Int8Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (i8vec4)
    buffer = context.createBuffer(bufferSize);

    // Load Vector Kernel
    std::string vector_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        vector_file = kernel_dir + "/vulkan/int8.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        vector_file = kernel_dir + "/rocm/int8.co";
    } else {
        vector_file = kernel_dir + "/opencl/int8.cl";
    }
    
    std::string kernel_name = (context.getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
    vectorKernel = context.createKernel(vector_file, kernel_name, 1);
    context.setKernelArg(vectorKernel, 0, buffer);

    // Optionally load Matrix Kernel
    if (context.getCurrentDeviceInfo().cooperativeMatrixSupport && context.getBackend() == ComputeBackend::Vulkan) {
        std::string matrix_file = kernel_dir + "/vulkan/coop_matrix_int8.spv";
        matrixKernel = context.createKernel(matrix_file, "main", 2);
        context.setKernelArg(matrixKernel, 0, buffer); // Binding 0: int8 (A/B)
        context.setKernelArg(matrixKernel, 1, buffer); // Binding 1: int32 (C)
    }
}

void Int8Bench::Run(uint32_t config_idx) {
    if (config_idx == 0) {
        context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
    } else if (matrixKernel) {
        // Matrix mode uses 32x1x1 workgroups for cooperative matrices
        context->dispatch(matrixKernel, 32768, 1, 1, 32, 1, 1);
    }
}

void Int8Bench::Teardown() {
    if (vectorKernel) {
        context->releaseKernel(vectorKernel);
        vectorKernel = nullptr;
    }
    if (matrixKernel) {
        context->releaseKernel(matrixKernel);
        matrixKernel = nullptr;
    }
    if (buffer) {
        context->releaseBuffer(buffer);
        buffer = nullptr;
    }
}

BenchmarkResult Int8Bench::GetResult(uint32_t config_idx) const {
    if (config_idx == 0) {
        // 8 i8vec4 multiply-adds per iteration = 8 * 4 * 2 = 64 INT8 operations per iteration
        // 16384 iterations * 64 ops * 8192 workgroups * 64 threads
        uint64_t num_ops = (uint64_t)16384 * 64 * 8192 * 64;
        return {num_ops, 0.0};
    } else {
        // Matrix mode: 16x16x16 matmul = 16 * 16 * 16 * 2 ops = 8192 ops per iteration
        // 16384 iterations * 8192 ops * 32768 workgroups
        uint64_t num_ops = (uint64_t)16384 * 8192 * 32768;
        return {num_ops, 0.0};
    }
}

uint32_t Int8Bench::GetNumConfigs() const {
    return context && context->getCurrentDeviceInfo().cooperativeMatrixSupport ? 2 : 1;
}

std::string Int8Bench::GetConfigName(uint32_t config_idx) const {
    return config_idx == 0 ? "Vector" : "Matrix";
}
