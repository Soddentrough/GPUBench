#include "CLI11.hpp"
#include "core/BenchmarkRunner.h"
#include "core/ComputeBackendFactory.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char **argv) {
#ifdef _WIN32
  // Set console output to UTF-8
  SetConsoleOutputCP(CP_UTF8);
#endif

#ifdef __linux__
  // Suppress Mesa/RADV conformance warnings to keep the output clean
  setenv("MESA_VK_IGNORE_CONFORMANCE_WARNING", "1", 1);
#endif
  CLI::App app{"GPUBench"};
  app.set_version_flag("--version", GPUBENCH_VERSION);

  std::vector<std::string> benchmarks_to_run;
  app.add_option("-b,--benchmarks,--benchmark", benchmarks_to_run,
                 "Benchmarks to run (comma-separated)")
      ->delimiter(',');

  bool list_benchmarks = false;
  app.add_flag("--list-benchmarks", list_benchmarks,
               "List available benchmarks");

  std::vector<uint32_t> device_indices;
  app.add_option("-d,--device", device_indices,
                 "Device(s) to use (comma-separated)")
      ->delimiter(',');

  bool list_devices = false;
  app.add_flag("-l,--list-devices", list_devices, "List available devices");

  bool list_backends = false;
  app.add_flag("--list-backends", list_backends, "List available backends");

  std::vector<std::string> backend_strs;
  app.add_option("-k,--backend", backend_strs,
                 "Backend to use: auto, vulkan, opencl, rocm (default: auto)")
      ->delimiter(',');

  bool verbose = false;
  app.add_flag("--verbose", verbose, "Enable verbose logging");

  bool debug = false;
  app.add_flag("--debug", debug, "Enable debug logging (implies verbose)");

  CLI11_PARSE(app, argc, argv);

  // Default to device 0 if none specified
  // if (device_indices.empty()) {
  //     device_indices.push_back(0);
  // }

  // Debug implies verbose
  if (debug) {
    verbose = true;
  }

  if (list_benchmarks) {
    BenchmarkRunner runner({});
    auto available_benchmarks = runner.getAvailableBenchmarks();
    std::cout << "Available benchmarks:" << std::endl;
    for (const auto &name : available_benchmarks) {
      std::cout << "- " << name << std::endl;
    }
    return EXIT_SUCCESS;
  }

  if (verbose) {
    std::cout << "Benchmarks to run: " << std::endl;
    for (const auto &name : benchmarks_to_run) {
      std::cout << "- " << name << std::endl;
    }
  }

  try {
    std::cout << "GPUBench version " << GPUBENCH_VERSION << std::endl
              << std::endl;
    // Create compute contexts for specified backends
    std::vector<std::unique_ptr<IComputeContext>> contexts;
    if (backend_strs.empty() ||
        (backend_strs.size() == 1 && backend_strs[0] == "auto")) {
      // Default to Vulkan, fall back to OpenCL, then ROCm
      if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
        contexts.push_back(
            ComputeBackendFactory::create(ComputeBackend::Vulkan));
      } else if (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
        contexts.push_back(
            ComputeBackendFactory::create(ComputeBackend::OpenCL));
      } else if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
        contexts.push_back(
            ComputeBackendFactory::create(ComputeBackend::ROCm, verbose));
      } else {
        std::cerr << "No compute backend available." << std::endl;
        return EXIT_FAILURE;
      }
    } else {
      for (const auto &backend_str : backend_strs) {
        if (backend_str == "vulkan") {
          if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
            contexts.push_back(
                ComputeBackendFactory::create(ComputeBackend::Vulkan));
          }
        } else if (backend_str == "opencl") {
          if (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
            contexts.push_back(
                ComputeBackendFactory::create(ComputeBackend::OpenCL));
          }
        } else if (backend_str == "rocm") {
          if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
            contexts.push_back(
                ComputeBackendFactory::create(ComputeBackend::ROCm, verbose));
          }
        } else {
          std::cerr << "Unknown or unavailable backend: " << backend_str
                    << std::endl;
        }
      }
    }

    if (contexts.empty() && !list_backends) {
      std::cerr << "No valid compute backends found." << std::endl;
      return EXIT_FAILURE;
    }

    if (list_backends) {
      // This part is a bit of a placeholder as we don't have a formal way
      // to query data type support without creating a context and checking
      // benchmarks.
      std::cout << "Available backends:" << std::endl;
      std::cout << "- vulkan: "
                << (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)
                        ? "Supported"
                        : "Not Supported")
                << std::endl;
      std::cout << "- opencl: "
                << (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)
                        ? "Supported"
                        : "Not Supported")
                << std::endl;
      std::cout << "- rocm: "
                << (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)
                        ? "Supported"
                        : "Not Supported")
                << std::endl;
      return EXIT_SUCCESS;
    }

    if (list_devices) {
      for (const auto &context : contexts) {
        std::cout << "Backend: "
                  << ComputeBackendFactory::getBackendName(
                         context->getBackend())
                  << std::endl;
        const auto &devices = context->getDevices();
        for (size_t i = 0; i < devices.size(); ++i) {
          std::cout << "  " << i << ": " << devices[i].name << std::endl;
        }
      }
      return EXIT_SUCCESS;
    }

    std::vector<IComputeContext *> context_ptrs;

    // For each context (backend), we need to create a separate instance for
    // each selected device But wait, IComputeContext is stateful (selected
    // device). We can't reuse the same context pointer for multiple devices
    // simultaneously if they share state. However, looking at the
    // implementations (VulkanContext, OpenCLContext, ROCmContext), they seem to
    // hold a single 'device' or 'physicalDevice'. So we need to duplicate the
    // context for each device we want to test.

    // Actually, the current design seems to assume one context = one backend
    // instance. And pickDevice() sets the active device for that context. If we
    // want to test multiple devices on the same backend, we need multiple
    // context instances.

    // Let's rebuild the context list based on the requested devices.
    std::vector<std::unique_ptr<IComputeContext>> execution_contexts;

    for (auto &proto_context : contexts) {
      ComputeBackend backend = proto_context->getBackend();
      const auto &devices = proto_context->getDevices();

      std::vector<uint32_t> target_indices = device_indices;
      if (target_indices.empty()) {
        target_indices.push_back(0);
      }

      for (uint32_t device_idx : target_indices) {
        if (device_idx < devices.size()) {
          // Create a new context for this device
          std::unique_ptr<IComputeContext> new_context =
              ComputeBackendFactory::create(backend, verbose);

          if (new_context) {
            new_context->pickDevice(device_idx);
            execution_contexts.push_back(std::move(new_context));
          }
        } else {
          if (verbose) {
            std::cerr << "Warning: Device index " << device_idx
                      << " out of range for backend "
                      << ComputeBackendFactory::getBackendName(backend)
                      << std::endl;
          }
        }
      }
    }

    // Now populate context_ptrs for the runner
    for (const auto &ctx : execution_contexts) {
      context_ptrs.push_back(ctx.get());
    }

    // We need to keep execution_contexts alive until runner finishes
    BenchmarkRunner runner(context_ptrs, verbose, debug);
    runner.run(benchmarks_to_run);

    // execution_contexts will be destroyed here, cleaning up resources

  } catch (const std::exception &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
