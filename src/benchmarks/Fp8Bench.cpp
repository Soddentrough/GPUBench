#include "benchmarks/Fp8Bench.h"
#include <stdexcept>
#include <fstream>
#include <iostream>

const char* Fp8Bench::GetName() const {
    return "FP8";
}

bool Fp8Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return info.fp8Support;
}

void Fp8Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;
    
    DeviceInfo info = context.getCurrentDeviceInfo();
    is_emulated = (info.name.find("gfx942") == std::string::npos && 
                   info.name.find("gfx11") == std::string::npos &&
                   info.name.find("gfx12") == std::string::npos); // RDNA4 is native

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (u8vec4)
    if (context.getBackend() == ComputeBackend::OpenCL) {
        // The OpenCL kernel uses half4, which is 8 bytes per thread
        bufferSize = 8192 * 64 * 8;
    }
    buffer = context.createBuffer(bufferSize);

    // Helper to check if file exists
    auto file_exists = [](const std::string& path) {
        std::ifstream f(path.c_str());
        return f.good();
    };

    if (context.getBackend() == ComputeBackend::ROCm) {
         // HIP Path
         std::string kernel_file = kernel_dir + "/rocm/fp8.co";
         std::string matrix_file = kernel_dir + "/rocm/fp8_matrix.co";
         
         if (file_exists(kernel_file)) {
             vectorKernel = context.createKernel(kernel_file, "run_benchmark", 1);
             context.setKernelArg(vectorKernel, 0, buffer);
             is_native_vector = true;
             is_emulated = false; // "Native" placeholder
         } else {
             is_native_vector = false;
         }

         if (file_exists(matrix_file)) {
             matrixKernel = context.createKernel(matrix_file, "run_benchmark", 1);
             context.setKernelArg(matrixKernel, 0, buffer);
             is_native_matrix = true; 
         } else {
             is_native_matrix = false;
         }
         return;
    }

    // Vulkan Path
    // Load Vector Kernel
    std::string native_vector = kernel_dir + "/vulkan/fp8_native.spv";
    std::string emulated_vector = kernel_dir + "/vulkan/fp8_emulated.spv";
    
    // Default to emulated
    std::string vector_file = emulated_vector;
    is_native_vector = false;
    is_emulated = true;

    // Try native if file exists and we are on compatible hardware (or forcing check)
    // FORCE DISABLE: The native shader has known issues and ghosts, forcing emulation for consistency.
    /*
    if (file_exists(native_vector)) {
        vector_file = native_vector;
        is_native_vector = true;
        is_emulated = false;
    }
    */
    
    try {
        vectorKernel = context.createKernel(vector_file, "main", 1);
        context.setKernelArg(vectorKernel, 0, buffer);
    } catch (...) {
        // Fallback if native failed to load even if file existed
        if (is_native_vector) {
            std::cerr << "Native FP8 vector kernel failed to load, falling back to emulation." << std::endl;
            vector_file = emulated_vector;
            vectorKernel = context.createKernel(vector_file, "main", 1);
            context.setKernelArg(vectorKernel, 0, buffer);
            is_native_vector = false;
            is_emulated = true;
        } else {
            throw;
        }
    }

    // Optionally load Matrix Kernel
    is_native_matrix = false;
    if (info.cooperativeMatrixSupport && context.getBackend() == ComputeBackend::Vulkan) {
        std::string matrix_file = kernel_dir + "/vulkan/coop_matrix_fp8.spv";
        if (file_exists(matrix_file)) {
            try {
                matrixKernel = context.createKernel(matrix_file, "main", 1);
                context.setKernelArg(matrixKernel, 0, buffer);
                is_native_matrix = true;
            } catch (...) {
                // Ignore failure, just don't enable matrix mode
                is_native_matrix = false;
            }
        }
    }
    
    // std::cout << "DEBUG: Setup Complete. vectorKernel=" << vectorKernel << ", buffer=" << buffer << std::endl;
}

void Fp8Bench::Run(uint32_t config_idx) {
    if (config_idx == 0) {
        if (!vectorKernel) {
             throw std::runtime_error("vectorKernel is NULL in Run!");
        }
        context->dispatch(vectorKernel, 8192, 1, 1, 64, 1, 1);
    } else if (matrixKernel) {
        context->dispatch(matrixKernel, 32768, 1, 1, 32, 1, 1);
    }
}

void Fp8Bench::Teardown() {
    if (vectorKernel) context->releaseKernel(vectorKernel);
    if (matrixKernel) context->releaseKernel(matrixKernel);
    if (buffer) context->releaseBuffer(buffer);
    vectorKernel = nullptr;
    matrixKernel = nullptr;
    buffer = nullptr;
}

BenchmarkResult Fp8Bench::GetResult(uint32_t config_idx) const {
    if (config_idx == 0) { // Vector
        // 8 fma operations per iteration, each is 2 ops (multiply, add)
        // 8 * 2 * 4 = 64 FP8-equivalent operations per iteration
        uint64_t num_ops = (uint64_t)16384 * 64 * 8192 * 64;
        return {num_ops, 0.0};
    } else { // Matrix
        // 16x16x16 matrix multiply = 8192 ops
        // 16384 iterations * 8192 ops * 32768 subgroups
        uint64_t num_ops = (uint64_t)16384 * 8192 * 32768; 
        return {num_ops, 0.0};
    }
}

uint32_t Fp8Bench::GetNumConfigs() const {
    return (matrixKernel != nullptr) ? 2 : 1;
}

std::string Fp8Bench::GetConfigName(uint32_t config_idx) const {
    return (config_idx == 0) ? "Vector" : "Matrix";
}
