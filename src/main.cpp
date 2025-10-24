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
    app.add_option("-b,--benchmarks,--benchmark", benchmarks_to_run, "Benchmarks to run (comma-separated)")->delimiter(',');

    bool list_benchmarks = false;
    app.add_flag("--list-benchmarks", list_benchmarks, "List available benchmarks");

    uint32_t device_index = 0;
    app.add_option("-d,--device", device_index, "Device to use");

    bool list_devices = false;
    app.add_flag("-l,--list-devices", list_devices, "List available devices");

    bool list_backends = false;
    app.add_flag("--list-backends", list_backends, "List available backends");

    std::vector<std::string> backend_strs;
    app.add_option("-k,--backend", backend_strs, "Backend to use: auto, vulkan, opencl, rocm (default: auto)")->delimiter(',');

    bool verbose = false;
    app.add_flag("--verbose", verbose, "Enable verbose logging");

    bool debug = false;
    app.add_flag("--debug", debug, "Enable debug logging (implies verbose)");

    CLI11_PARSE(app, argc, argv);
    
    // Debug implies verbose
    if (debug) {
        verbose = true;
    }

    if (list_benchmarks) {
        BenchmarkRunner runner({});
        auto available_benchmarks = runner.getAvailableBenchmarks();
        std::cout << "Available benchmarks:" << std::endl;
        for (const auto& name : available_benchmarks) {
            std::cout << "- " << name << std::endl;
        }
        return EXIT_SUCCESS;
    }

    if (verbose) {
        std::cout << "Benchmarks to run: " << std::endl;
        for (const auto& name : benchmarks_to_run) {
            std::cout << "- " << name << std::endl;
        }
    }

    try {
        // Create compute contexts for specified backends
        std::vector<std::unique_ptr<IComputeContext>> contexts;
        if (backend_strs.empty() || (backend_strs.size() == 1 && backend_strs[0] == "auto")) {
            // Default to Vulkan, fall back to OpenCL, then ROCm
            if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
                contexts.push_back(ComputeBackendFactory::create(ComputeBackend::Vulkan));
            } else if (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
                contexts.push_back(ComputeBackendFactory::create(ComputeBackend::OpenCL));
            } else if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
                contexts.push_back(ComputeBackendFactory::create(ComputeBackend::ROCm, verbose));
            } else {
                std::cerr << "No compute backend available." << std::endl;
                return EXIT_FAILURE;
            }
        } else {
            for (const auto& backend_str : backend_strs) {
                if (backend_str == "vulkan") {
                    if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
                        contexts.push_back(ComputeBackendFactory::create(ComputeBackend::Vulkan));
                    }
                } else if (backend_str == "opencl") {
                    if (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
                        contexts.push_back(ComputeBackendFactory::create(ComputeBackend::OpenCL));
                    }
                } else if (backend_str == "rocm") {
                    if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
                        contexts.push_back(ComputeBackendFactory::create(ComputeBackend::ROCm, verbose));
                    }
                } else {
                    std::cerr << "Unknown or unavailable backend: " << backend_str << std::endl;
                }
            }
        }

        if (contexts.empty() && !list_backends) {
            std::cerr << "No valid compute backends found." << std::endl;
            return EXIT_FAILURE;
        }

        if (list_backends) {
            // This part is a bit of a placeholder as we don't have a formal way 
            // to query data type support without creating a context and checking benchmarks.
            std::cout << "Available backends:" << std::endl;
            std::cout << "- vulkan: " << (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan) ? "Supported" : "Not Supported") << std::endl;
            std::cout << "- opencl: " << (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL) ? "Supported" : "Not Supported") << std::endl;
            std::cout << "- rocm: " << (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm) ? "Supported" : "Not Supported") << std::endl;
            return EXIT_SUCCESS;
        }

        if (list_devices) {
            for (const auto& context : contexts) {
                std::cout << "Backend: " << ComputeBackendFactory::getBackendName(context->getBackend()) << std::endl;
                const auto& devices = context->getDevices();
                for (size_t i = 0; i < devices.size(); ++i) {
                    std::cout << "  " << i << ": " << devices[i].name << std::endl;
                }
            }
            return EXIT_SUCCESS;
        }

        std::vector<IComputeContext*> context_ptrs;
        for (const auto& context : contexts) {
            context->pickDevice(device_index);
            context_ptrs.push_back(context.get());
        }

        BenchmarkRunner runner(context_ptrs, verbose, debug);
        runner.run(benchmarks_to_run);
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
