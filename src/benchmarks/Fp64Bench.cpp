#include "benchmarks/Fp64Bench.h"
#include "core/VulkanContext.h"
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

bool Fp64Bench::IsSupported(const DeviceInfo& info, IComputeContext* context) const {
    if (context && context->getBackend() == ComputeBackend::Vulkan) {
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(context->getVulkanPhysicalDevice(), &features);
        return features.shaderFloat64;
    }
    return true;
}

void Fp64Bench::Setup(IComputeContext& context, const std::string& kernel_dir) {
    this->context = &context;

    // Create storage buffer
    size_t bufferSize = 4096 * 64 * sizeof(double); // 4096 workgroups * 64 threads * 8 bytes
    buffer = context.createBuffer(bufferSize);

    // Create kernel
    std::string kernel_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/fp64.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/hip_kernels/fp64.o";
    } else {
        kernel_file = kernel_dir + "/fp64.cl";
    }
    
    kernel = context.createKernel(kernel_file, "run_benchmark", 1);
    context.setKernelArg(kernel, 0, buffer);
}

void Fp64Bench::Run() {
    context->dispatch(kernel, 4096, 1, 1, 64, 1, 1);
}

void Fp64Bench::Teardown() {
    if (kernel) {
        context->releaseKernel(kernel);
        kernel = nullptr;
    }
    if (buffer) {
        context->releaseBuffer(buffer);
        buffer = nullptr;
    }
}

BenchmarkResult Fp64Bench::GetResult() const {
    // 2 operations per loop iteration (FMA)
    uint64_t num_threads = 4096 * 64;
    uint64_t num_ops = (uint64_t)65536 * 2 * num_threads;
    return {num_ops, 0.0};
}
