#include "benchmarks/Fp16Bench.h"
#include <stdexcept>
#include <vulkan/vulkan.h>

bool Fp16Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
        VkPhysicalDevice16BitStorageFeatures features16bit = {};
        features16bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
        
        VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &features16bit;
        
        vkGetPhysicalDeviceFeatures2(context->getVulkanPhysicalDevice(), &deviceFeatures2);
        
        return features16bit.storageBuffer16BitAccess;
    }
    return true;
}

void Fp16Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (f16vec2)
    buffer = context.createBuffer(bufferSize);

    // Create kernel
    std::string kernel_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/fp16.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/hip_kernels/fp16.o";
    } else {
        kernel_file = kernel_dir + "/fp16.cl";
    }
    
    std::string kernel_name = (context.getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
    kernel = context.createKernel(kernel_file, kernel_name, 1);
    context.setKernelArg(kernel, 0, buffer);
}

void Fp16Bench::Run() {
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

BenchmarkResult Fp16Bench::GetResult() const {
    // 8 f16vec2 FMAs per iteration = 8 * 2 * 2 = 32 FP16 operations per iteration
    // 16384 iterations * 32 ops * 8192 workgroups * 64 threads
    uint64_t num_ops = (uint64_t)16384 * 32 * 8192 * 64;
    return {num_ops, 0.0};
}
