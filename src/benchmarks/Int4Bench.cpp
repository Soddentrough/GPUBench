#include "benchmarks/Int4Bench.h"
#include <stdexcept>
#include <fstream>

bool Int4Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return info.int4Support;
}

void Int4Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (i8vec4)
    buffer = context.createBuffer(bufferSize);

    // Load Vector Kernel
    std::string vector_file = kernel_dir + "/vulkan/int4.spv";
    vectorKernel = context.createKernel(vector_file, "main", 1);
    context.setKernelArg(vectorKernel, 0, buffer);

    // Helper to check if file exists
    auto file_exists = [](const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    };

    // Load Matrix Kernel
    bool is_rdna4 = context.getCurrentDeviceInfo().name.find("gfx12") != std::string::npos;
    if (context.getCurrentDeviceInfo().cooperativeMatrixSupport && context.getBackend() == ComputeBackend::Vulkan && is_rdna4) {
        std::string matrix_file = kernel_dir + "/vulkan/coop_matrix_int4.spv";
        if (file_exists(matrix_file)) {
            matrixKernel = context.createKernel(matrix_file, "main", 1);
            context.setKernelArg(matrixKernel, 0, buffer);
        }
    }
}

void Int4Bench::Run(uint32_t config_idx) {
    if (config_idx == 0) {
        context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
    } else if (matrixKernel) {
        context->dispatch(matrixKernel, 32768, 1, 1, 32, 1, 1);
    }
}

void Int4Bench::Teardown() {
    if (vectorKernel) context->releaseKernel(vectorKernel);
    if (matrixKernel) context->releaseKernel(matrixKernel);
    if (buffer) context->releaseBuffer(buffer);
    vectorKernel = nullptr;
    matrixKernel = nullptr;
    buffer = nullptr;
}

BenchmarkResult Int4Bench::GetResult(uint32_t config_idx) const {
    if (config_idx == 0) { // Vector
        // 12 i8vec4 operations per iteration, each with multiply-add + AND = 3 ops per component
        // 12 * 4 * 3 = 144 INT4-equivalent operations per iteration
        uint64_t num_ops = (uint64_t)16384 * 144 * 8192 * 64;
        return {num_ops, 0.0};
    } else { // Matrix
        // 16x16x16 matmul = 8192 ops per iteration
        // 16384 iterations * 8192 ops * 32768 subgroups
        uint64_t num_ops = (uint64_t)16384 * 8192 * 32768;
        return {num_ops, 0.0};
    }
}

uint32_t Int4Bench::GetNumConfigs() const {
    return (matrixKernel != nullptr) ? 2 : 1;
}

std::string Int4Bench::GetConfigName(uint32_t config_idx) const {
    return (config_idx == 0) ? "Vector" : "Matrix";
}
