#include "benchmarks/Int4Bench.h"
#include <stdexcept>
#include <fstream>
#include <iostream>

bool Int4Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return info.int4Support;
}

void Int4Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (i8vec4)
    buffer = context.createBuffer(bufferSize);

    // Helper to check if file exists
    auto file_exists = [](const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    };

    if (context.getBackend() == ComputeBackend::ROCm) {
         // HIP Path
         std::string kernel_file = kernel_dir + "/rocm/int4.co";
         if (file_exists(kernel_file)) {
             vectorKernel = context.createKernel(kernel_file, "run_benchmark", 1);
             context.setKernelArg(vectorKernel, 0, buffer);
             is_native_vector = true;
             is_emulated = false;
         } else {
             std::cerr << "Native INT4 HIP kernel missing: " << kernel_file << std::endl;
             is_native_vector = false; // Not supported
             // We can return here or let it be null.
         }
         is_native_matrix = false;
         return;
    }

    // Vulkan Path
    // Load Vector Kernel
    std::string native_vector = kernel_dir + "/vulkan/int4_native.spv";
    std::string emulated_vector = kernel_dir + "/vulkan/int4.spv"; // Assuming int4.spv is emulated/fallback
    
    std::string vector_file = emulated_vector;
    is_native_vector = false;
    is_emulated = true;

    // Check for native
    if (file_exists(native_vector)) {
        vector_file = native_vector;
        is_native_vector = true;
        is_emulated = false;
    }
    
    try {
        vectorKernel = context.createKernel(vector_file, "main", 1);
        context.setKernelArg(vectorKernel, 0, buffer);
    } catch (...) {
         if (is_native_vector) {
            std::cerr << "Native INT4 vector kernel failed to load, falling back to emulation." << std::endl;
            vector_file = emulated_vector;
            vectorKernel = context.createKernel(vector_file, "main", 1);
            context.setKernelArg(vectorKernel, 0, buffer);
            is_native_vector = false;
            is_emulated = true;
        } else {
            throw;
        }
    }

    bool is_rdna4 = context.getCurrentDeviceInfo().name.find("gfx12") != std::string::npos;
    if (context.getCurrentDeviceInfo().cooperativeMatrixSupport && context.getBackend() == ComputeBackend::Vulkan && is_rdna4) {
        std::string matrix_file = kernel_dir + "/vulkan/coop_matrix_int4.spv";
        if (file_exists(matrix_file)) {
            try {
                matrixKernel = context.createKernel(matrix_file, "main", 1);
                context.setKernelArg(matrixKernel, 0, buffer);
                is_native_matrix = true;
            } catch (...) {
                is_native_matrix = false;
            }
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
