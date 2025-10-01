#include "benchmarks/Fp64Bench.h"
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
    size_t bufferSize = 1024 * 64 * sizeof(double); // 1024 workgroups * 64 threads * 8 bytes
    std::vector<double> initialData(1024 * 64, 0.0);
    buffer = context.createBuffer(bufferSize, initialData.data());

    // Create kernel
    std::string kernel_file;
    if (context.getBackend() == ComputeBackend::Vulkan) {
        kernel_file = kernel_dir + "/fp64.spv";
    } else if (context.getBackend() == ComputeBackend::ROCm) {
        kernel_file = kernel_dir + "/hip_kernels/fp64.o";
    } else {
        kernel_file = kernel_dir + "/fp64.cl";
    }
    
    kernel = context.createKernel(kernel_file, "main", 1);
    context.setKernelArg(kernel, 0, buffer);
}

void Fp64Bench::Run() {
    context->dispatch(kernel, 1024, 1, 1, 64, 1, 1);
}

void Fp64Bench::Teardown() {
    if (kernel) {
        context->releaseKernel(kernel);
    }
    if (buffer) {
        context->releaseBuffer(buffer);
    }
}

BenchmarkResult Fp64Bench::GetResult() const {
    // 2 operations per loop iteration (FMA)
    uint64_t num_ops = (uint64_t)65536 * 2 * 1024 * 64;
    return {num_ops, 0.0};
}
