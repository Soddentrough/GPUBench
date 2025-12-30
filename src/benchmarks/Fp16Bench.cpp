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
    std::vector<uint32_t> initData(bufferSize / sizeof(uint32_t), 0);
    context.writeBuffer(buffer, 0, bufferSize, initData.data());

    // Load Vector Kernel
    std::string vector_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        vector_file = kernel_dir + "/vulkan/fp16.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        vector_file = kernel_dir + "/rocm/fp16.co";
    } else {
        vector_file = kernel_dir + "/opencl/fp16.cl";
    }
    
    std::string kernel_name = (context.getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
    vectorKernel = context.createKernel(vector_file, kernel_name, 1);
    context.setKernelArg(vectorKernel, 0, buffer);

    // Optionally load Matrix Kernel if supported
    if (context.getCurrentDeviceInfo().cooperativeMatrixSupport && context.getBackend() == ComputeBackend::Vulkan) {
        std::string matrix_file = kernel_dir + "/vulkan/coop_matrix_fp16.spv";
        matrixKernel = context.createKernel(matrix_file, "main", 1);
        context.setKernelArg(matrixKernel, 0, buffer);
    }
}

void Fp16Bench::Run(uint32_t config_idx) {
    if (config_idx == 0) {
        context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
    } else if (matrixKernel) {
        // Matrix mode uses 32x1x1 workgroups for cooperative matrices
        context->dispatch(matrixKernel, 32768, 1, 1, 32, 1, 1);
    }
}

void Fp16Bench::Teardown() {
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

BenchmarkResult Fp16Bench::GetResult(uint32_t config_idx) const {
    if (config_idx == 0) {
        // 32 f16vec2 FMAs per iteration = 32 * 2 * 2 = 128 FP16 operations per iteration
        // 65536 iterations * 128 ops * 8192 workgroups * 64 threads
        uint64_t num_ops = (uint64_t)65536 * 128 * 8192 * 64;
        return {num_ops, 0.0};
    } else {
        // Matrix mode: 16x16x16 matmul = 16 * 16 * 16 * 2 ops = 8192 ops per iteration
        // 16384 iterations * 8192 ops * 32768 workgroups
        uint64_t num_ops = (uint64_t)16384 * 8192 * 32768;
        return {num_ops, 0.0};
    }
}

uint32_t Fp16Bench::GetNumConfigs() const {
    return context && context->getCurrentDeviceInfo().cooperativeMatrixSupport ? 2 : 1;
}

std::string Fp16Bench::GetConfigName(uint32_t config_idx) const {
    return config_idx == 0 ? "Vector" : "Matrix";
}
