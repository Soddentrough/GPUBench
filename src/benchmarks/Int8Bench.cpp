#include "benchmarks/Int8Bench.h"
#include <stdexcept>
#include <vulkan/vulkan.h>

bool Int8Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
        VkPhysicalDevice8BitStorageFeatures features8bit = {};
        features8bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
        
        VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &features8bit;
        
        vkGetPhysicalDeviceFeatures2(context->getVulkanPhysicalDevice(), &deviceFeatures2);
        
        return features8bit.storageBuffer8BitAccess;
    }
    return true;
}

void Int8Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 8192 * 64 * 4; // 8192 workgroups * 64 threads * 4 bytes (i8vec4)
    buffer = context.createBuffer(bufferSize);

    // Create kernel
    std::string kernel_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/vulkan/int8.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/rocm/int8.o";
    } else {
        kernel_file = kernel_dir + "/opencl/int8.cl";
    }
    
    std::string kernel_name;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_name = "main";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_name = "run_benchmark";
    } else {
        kernel_name = "run_benchmark";
    }
    kernel = context.createKernel(kernel_file, kernel_name, 1);
    context.setKernelArg(kernel, 0, buffer);
}

void Int8Bench::Run(uint32_t config_idx) {
    context->dispatch(kernel, 8192, 1, 1, 64, 1, 1);
}

void Int8Bench::Teardown() {
    if (kernel) {
        context->releaseKernel(kernel);
    }
    if (buffer) {
        context->releaseBuffer(buffer);
    }
}

BenchmarkResult Int8Bench::GetResult(uint32_t config_idx) const {
    // 8 i8vec4 multiply-adds per iteration = 8 * 4 * 2 = 64 INT8 operations per iteration
    // 16384 iterations * 64 ops * 8192 workgroups * 64 threads
    uint64_t num_ops = (uint64_t)16384 * 64 * 8192 * 64;
    return {num_ops, 0.0};
}
