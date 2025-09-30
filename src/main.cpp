#include "core/VulkanContext.h"
#include "core/BenchmarkRunner.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "CLI11.hpp"

int main(int argc, char** argv) {
    CLI::App app{"GPUBench"};

    std::vector<std::string> benchmarks_to_run;
    app.add_option("-b,--benchmarks", benchmarks_to_run, "Benchmarks to run")->expected(1, -1);

    uint32_t device_index = 0;
    app.add_option("-d,--device", device_index, "Device to use");

    bool list_devices = false;
    app.add_flag("-l,--list-devices", list_devices, "List available devices");

    CLI11_PARSE(app, argc, argv);

    try {
        VulkanContext context;

        if (list_devices) {
            std::cout << "Available devices:" << std::endl;
            for (uint32_t i = 0; i < context.getPhysicalDevices().size(); ++i) {
                VkPhysicalDeviceProperties properties;
                vkGetPhysicalDeviceProperties(context.getPhysicalDevices()[i], &properties);
                std::cout << "  " << i << ": " << properties.deviceName << std::endl;
            }
            return EXIT_SUCCESS;
        }

        context.pickPhysicalDevice(device_index);

        VkPhysicalDeviceProperties properties = context.getPhysicalDeviceProperties();
        std::cout << "Using device: " << properties.deviceName << std::endl;

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(context.getPhysicalDevice(), &memProperties);
        uint64_t vram_size = 0;
        for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
            if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                vram_size = memProperties.memoryHeaps[i].size;
                break;
            }
        }

        std::cout << "  VRAM: " << vram_size / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shader cores: " << properties.limits.maxComputeWorkGroupInvocations << std::endl;


        BenchmarkRunner runner(context);
        runner.run(benchmarks_to_run);
    } catch (const std::runtime_error& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
