#include "benchmarks/CacheBench.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm> // for std::min

CacheBench::CacheBench(std::string name, std::string metric, uint64_t bufferSize, std::string kernelFile, std::vector<uint32_t> initData, std::vector<std::string> aliases, int targetCacheLevel)
    : name(name), metric(metric), bufferSize(bufferSize), kernelFile(kernelFile), initData(initData), aliases(aliases), targetCacheLevel(targetCacheLevel) {}

CacheBench::~CacheBench() {}

bool CacheBench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    return true;
}

// Helper to round down to nearest power of 2
static uint64_t roundDownToPowerOf2(uint64_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v >> 1;
}

void CacheBench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;
    DeviceInfo info = context.getCurrentDeviceInfo();

    // Adjust buffer size based on cache level if available
    if (targetCacheLevel == 3 && info.l3CacheSize > 0) {
        // Use 75% of L3 size, rounded down to power of 2
        bufferSize = roundDownToPowerOf2(info.l3CacheSize * 3 / 4);
        if (bufferSize < 1024 * 1024) bufferSize = 1024 * 1024; // Min 1MB
    } else if (targetCacheLevel == 2 && info.l2CacheSize > 0) {
        // Use 75% of L2 size, rounded down to power of 2
        bufferSize = roundDownToPowerOf2(info.l2CacheSize * 3 / 4);
    }

    // Determine numWorkgroups to cover the buffer exactly once.
    // Each thread accesses 32 vec4s for Bandwidth tests (except L1 which is different but we'll adapt).
    // Let's standardize on the L3 pattern: 256 threads * 32 vec4s/thread = 8192 vec4s per workgroup.
    if (metric == "GB/s") {
        uint64_t vec4_count = bufferSize / 16;
        uint64_t elements_per_wg = 8192; // Default for L3
        if (targetCacheLevel == 1) elements_per_wg = 2;
        else if (targetCacheLevel == 2) elements_per_wg = 256;
        
        numWorkgroups = static_cast<uint32_t>(std::max<uint64_t>(1, vec4_count / elements_per_wg));
        // Clamp to avoid excessive dispatch on small buffers
        if (numWorkgroups > 65536) numWorkgroups = 65536; 
    } else {
        numWorkgroups = 1;
    }

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
            if (debug) {
                std::cout << "  [DEBUG] CacheBench Buffer: " << hostMem << " - " << (void*)((char*)hostMem + bufferSize) << " (" << (bufferSize/1024/1024) << " MB)" << std::endl;
            }
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
        full_kernel_path = kernel_dir + "/rocm/" + kernelFile + ".co";
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
    
    // We now pass 2 arguments: buffer and mask
    kernel = context.createKernel(full_kernel_path, kernel_name, 2);
    if (buffer) {
        context.setKernelArg(kernel, 0, buffer);
        
        // Calculate mask (element count - 1)
        // bufferSize is in bytes, vec4 is 16 bytes
        uint32_t mask = (bufferSize / 16) - 1;
        context.setKernelArg(kernel, 1, sizeof(uint32_t), &mask);
    }
}

void CacheBench::Run(uint32_t config_idx) {
    if (!context) {
        throw std::runtime_error("Context is not set up");
    }
    if (metric == "GB/s") {
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
    uint64_t num_threads_bw = (uint64_t)numWorkgroups * 256;

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
