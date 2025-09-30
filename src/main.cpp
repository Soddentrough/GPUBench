#include "core/ComputeBackendFactory.h"
#include "core/BenchmarkRunner.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "CLI11.hpp"

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

int main(int argc, char** argv) {
    CLI::App app{"GPUBench"};

    std::vector<std::string> benchmarks_to_run;
    app.add_option("-b,--benchmarks", benchmarks_to_run, "Benchmarks to run")->expected(1, -1);

    uint32_t device_index = 0;
    app.add_option("-d,--device", device_index, "Device to use");

    bool list_devices = false;
    app.add_flag("-l,--list-devices", list_devices, "List available devices");

    std::string backend_str = "auto";
    app.add_option("-k,--backend", backend_str, "Backend to use: auto, vulkan, opencl (default: auto)");

    CLI11_PARSE(app, argc, argv);

    try {
        // Create compute context with specified backend
        std::unique_ptr<IComputeContext> context;
        
        if (backend_str == "auto") {
            std::cout << "Auto-detecting compute backend..." << std::endl;
            context = ComputeBackendFactory::createWithFallback();
        } else if (backend_str == "vulkan") {
            if (!ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
                std::cerr << "Vulkan backend not available" << std::endl;
                return EXIT_FAILURE;
            }
            context = ComputeBackendFactory::create(ComputeBackend::Vulkan);
        } else if (backend_str == "opencl") {
            if (!ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
                std::cerr << "OpenCL backend not available" << std::endl;
                return EXIT_FAILURE;
            }
            context = ComputeBackendFactory::create(ComputeBackend::OpenCL);
        } else {
            std::cerr << "Unknown backend: " << backend_str << std::endl;
            std::cerr << "Valid options: auto, vulkan, opencl" << std::endl;
            return EXIT_FAILURE;
        }
        
        std::cout << "Using backend: " << ComputeBackendFactory::getBackendName(context->getBackend()) << std::endl;

        if (list_devices) {
            std::cout << "Available devices:" << std::endl;
            const auto& devices = context->getDevices();
            for (size_t i = 0; i < devices.size(); ++i) {
                std::cout << "  " << i << ": " << devices[i].name << std::endl;
            }
            return EXIT_SUCCESS;
        }

        context->pickDevice(device_index);
        
        DeviceInfo info = context->getCurrentDeviceInfo();
        std::cout << "Using device: " << info.name << std::endl;
        std::cout << "  VRAM: " << info.memorySize / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Subgroup size (wavefront/warp): " << info.subgroupSize << " threads" << std::endl;
        std::cout << "  Max work group size: " << info.maxWorkGroupSize << " threads" << std::endl;
        std::cout << "  Max work group count: " << info.maxComputeWorkGroupCountX 
                  << " x " << info.maxComputeWorkGroupCountY 
                  << " x " << info.maxComputeWorkGroupCountZ << std::endl;
        std::cout << "  Shared memory per work group: " << info.maxComputeSharedMemorySize / 1024 << " KB" << std::endl;

        BenchmarkRunner runner(*context);
        runner.run(benchmarks_to_run);
    } catch (const std::runtime_error& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
