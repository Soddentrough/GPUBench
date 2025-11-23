#include "benchmarks/CacheBench.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm> // for std::min

CacheBench::CacheBench(std::string name, std::string metric, uint64_t bufferSize, std::string kernelFile, std::vector<uint32_t> initData, std::vector<std::string> aliases)
    : name(name), metric(metric), bufferSize(bufferSize), kernelFile(kernelFile), initData(initData), aliases(aliases) {}

CacheBench::~CacheBench() {}

bool CacheBench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return true;
}

void CacheBench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    if (bufferSize > 0) {
        // Allocate page-aligned host memory for Zero Copy / Pinned access.
        // This is critical for stability on Unified Memory (APU) platforms like Strix Halo
        // to ensure the driver can map the memory without page faults or copying.
        hostMem = aligned_alloc(4096, bufferSize);
        
        if (hostMem) {
            // Initialize memory
            if (!initData.empty()) {
                size_t copySize = std::min((size_t)bufferSize, initData.size() * sizeof(uint32_t));
                memcpy(hostMem, initData.data(), copySize);
                // Zero fill the rest if any
                if (copySize < bufferSize) {
                    memset((char*)hostMem + copySize, 0, bufferSize - copySize);
                }
            } else {
                memset(hostMem, 0, bufferSize);
            }
            
            // Create buffer using the aligned host pointer (CL_MEM_USE_HOST_PTR)
            // Print buffer range for debugging
            std::cout << "  [DEBUG] CacheBench Buffer: " << hostMem << " - " << (void*)((char*)hostMem + bufferSize) << std::endl;
            buffer = context.createBuffer(bufferSize, hostMem);
        } else {
            // Fallback to standard allocation if host allocation fails (unlikely)
            buffer = context.createBuffer(bufferSize);
            if (!initData.empty()) {
                context.writeBuffer(buffer, 0, initData.size() * sizeof(uint32_t), initData.data());
            }
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

void CacheBench::Run(uint32_t config_idx) {
    if (!context) {
        throw std::runtime_error("Context is not set up");
    }
    if (metric == "GB/s") {
        // For bandwidth, we want to saturate the GPU with threads.
        // The shader uses a workgroup size of 256.
        // Reduced to 32 to eliminate wrapping/aliasing on 4MB buffers (L3).
        // 32 workgroups * 8192 vec4s/wg = 262144 vec4s = 4MB.
        // This ensures 1:1 mapping with no contention.
        uint32_t numWorkgroups = 32;
        
        // For L3 cache bandwidth with the cachebw_l3 kernel, we use a mask in the kernel
        // to ensure we stay within bounds (wrapping around the buffer).
        // This allows us to saturate the GPU with many workgroups.
        
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
    if (hostMem) {
        free(hostMem);
        hostMem = nullptr;
    }
}

const char* CacheBench::GetName() const {
    return name.c_str();
}

std::vector<std::string> CacheBench::GetAliases() const {
    return aliases;
}

const char* CacheBench::GetMetric() const {
    return metric.c_str();
}

BenchmarkResult CacheBench::GetResult(uint32_t config_idx) const {
    uint64_t operations = 0;
    // Must match numWorkgroups in Run()
    const uint64_t num_threads_bw = 32 * 256;

    if (name == "L0 Cache Bandwidth") {
        // 16 independent ops * 1024 iterations * num_threads
        operations = 16 * 1024 * num_threads_bw;
    } else if (name == "L0 Cache Latency") {
        // 16 dependent ops * 128 iterations
        operations = 16 * 128;
    } else if (name == "L1 Cache Bandwidth") {
        // L1: 1024 iterations × 1 float4 read × sizeof(float4)
        operations = num_threads_bw * 1024 * sizeof(float) * 4;
    } else if (name == "L2 Cache Bandwidth") {
        // L2: 500 iterations × 1 float4 read × sizeof(float4)
        operations = num_threads_bw * 500 * sizeof(float) * 4;
    } else if (name == "L3 Cache Bandwidth") {
        // L3: 200 iterations × 32 float4 reads × sizeof(float4)
        operations = num_threads_bw * 200 * 32 * sizeof(float) * 4;
    } else if (metric == "GB/s") {
        // Generic bandwidth: assume 1024 reads
        operations = num_threads_bw * 1024 * sizeof(uint32_t);
    } else if (metric == "ns") {
        // 1024 dependent reads
        operations = 1024;
    }
    return {operations, 0.0};
}
