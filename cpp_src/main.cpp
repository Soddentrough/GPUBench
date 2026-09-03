#include "CLI11.hpp"
#include "benchmarks/RayAnyHitBench.h"
#include "benchmarks/RayDivergenceBench.h"
#include "benchmarks/RayTracingBench.h"
#include "benchmarks/RayPathTracingBench.h"
#include "core/BenchmarkRunner.h"
#include "core/ComputeBackendFactory.h"
#include "core/ResultFormatter.h"
#include <cstdlib>
#include <fstream>
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

namespace {

// Escape a string for inclusion in a JSON double-quoted value.
std::string jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
    }
  }
  return out;
}

// Quote a field for CSV (RFC 4180 style: quote when it contains comma,
// quote, or newline; double up embedded quotes).
std::string csvQuote(const std::string &s) {
  if (s.find_first_of(",\"\n\r") == std::string::npos) {
    return s;
  }
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

static double computeResultValue(const ResultData &r) {
  if (r.metric == "ns") {
    return (r.operations > 0) ? ((r.time_ms * 1e6) / r.operations) : 0.0;
  }
  double time_s = r.time_ms / 1000.0;
  if (time_s <= 0.0) return 0.0;

  if (r.metric == "TFLOPS" || r.metric == "TOPS") {
    return (r.operations / time_s) / 1e12;
  } else if (r.metric == "GB/s") {
    return (r.operations / time_s) / 1e9;
  } else if (r.metric == "MRays/s" || r.metric == "MHits/s" || r.metric == "MRecords/s" ||
             r.metric == "MTris/s" || r.metric == "MInst/s") {
    return (r.operations / time_s) / 1e6;
  } else if (r.metric == "GRays/s" || r.metric == "GIS/s" || r.metric == "GPixels/s") {
    return (r.operations / time_s) / 1e9;
  }
  return r.operations / time_s;
}

std::string resultsToJson(const std::vector<ResultData> &results) {
  std::string out = "[\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const ResultData &r = results[i];
    std::string devIdxStr = (r.deviceIndex == 0xFFFFFFFF || r.backendName == "System")
                                ? "null"
                                : std::to_string(r.deviceIndex);
    double value = computeResultValue(r);

    out += "  {\n";
    out += "    \"backend\": \"" + jsonEscape(r.backendName) + "\",\n";
    out += "    \"device\": \"" + jsonEscape(r.deviceName) + "\",\n";
    out += "    \"device_index\": " + devIdxStr + ",\n";
    out += "    \"benchmark\": \"" + jsonEscape(r.benchmarkName) + "\",\n";
    out += "    \"component\": \"" + jsonEscape(r.component) + "\",\n";
    out += "    \"subcategory\": \"" + jsonEscape(r.subcategory) + "\",\n";
    out += "    \"metric\": \"" + jsonEscape(r.metric) + "\",\n";
    out += "    \"value\": " + std::to_string(value) + ",\n";
    if (r.benchmarkName.find("RayScheduling") != std::string::npos && r.metric == "MRays/s") {
      uint32_t w = r.width ? r.width : 1920;
      uint32_t h = r.height ? r.height : 1080;
      double fps = (value * 1e6) / static_cast<double>(w * h);
      out += "    \"fps\": " + std::to_string(fps) + ",\n";
      out += "    \"resolution\": \"" + std::to_string(w) + "x" + std::to_string(h) + "\",\n";
    }
    out += "    \"operations\": " + std::to_string(r.operations) + ",\n";
    out += "    \"time_ms\": " + std::to_string(r.time_ms) + ",\n";
    out += std::string("    \"is_emulated\": ") +
           (r.isEmulated ? "true" : "false") + ",\n";
    out += std::string("    \"unsupported\": ") +
           (r.isUnsupported ? "true" : "false") + ",\n";
    if (r.isUnsupported) {
      out += "    \"unsupported_category\": \"" +
             jsonEscape(r.supportCategory) + "\",\n";
      out += "    \"unsupported_reason\": \"" + jsonEscape(r.supportNote) +
             "\",\n";
    }
    out += "    \"max_workgroup_size\": " +
           std::to_string(r.maxWorkGroupSize) + ",\n";
    out += "    \"config_index\": " + std::to_string(r.configIndex) + "\n";
    out += (i + 1 < results.size()) ? "  },\n" : "  }\n";
  }
  out += "]\n";
  return out;
}

std::string resultsToCsv(const std::vector<ResultData> &results) {
  std::string out =
      "backend,device,device_index,benchmark,component,subcategory,metric,"
      "value,operations,time_ms,is_emulated,max_workgroup_size,config_index\n";
  for (const ResultData &r : results) {
    std::string devIdxStr = (r.deviceIndex == 0xFFFFFFFF || r.backendName == "System")
                                ? ""
                                : std::to_string(r.deviceIndex);
    double value = computeResultValue(r);
    out += csvQuote(r.backendName) + "," + csvQuote(r.deviceName) + "," +
           devIdxStr + "," + csvQuote(r.benchmarkName) +
           "," + csvQuote(r.component) + "," + csvQuote(r.subcategory) + "," +
           csvQuote(r.metric) + "," + std::to_string(value) + "," +
           std::to_string(r.operations) + "," + std::to_string(r.time_ms) +
           "," + (r.isEmulated ? "true" : "false") + "," +
           std::to_string(r.maxWorkGroupSize) + "," +
           std::to_string(r.configIndex) + "\n";
  }
  return out;
}

} // namespace

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
                 "Benchmarks to run (comma-separated, can also be a group name)")
      ->delimiter(',');

  std::vector<std::string> groups_to_run;
  app.add_option("-g,--groups,--group", groups_to_run,
                 "Benchmark group(s) to run: compute, memory, raytracing, graphics, system")
      ->delimiter(',');

  bool list_benchmarks = false;
  app.add_flag("--list-benchmarks", list_benchmarks,
               "List available benchmarks (organized by group)");

  bool list_groups = false;
  app.add_flag("--list-groups", list_groups,
               "List available benchmark groups");

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

  bool dump_geometry = false;
  app.add_flag("--dump-geometry", dump_geometry,
               "Dump ray tracing geometry to OBJ files");

  bool dump_renders = false;
  app.add_flag("--dump-renders,--dump", dump_renders,
               "Dump and analytically compare rendered frames between Megakernel and Work Lists");

  std::string resolution_str = "1080p";
  app.add_option("-r,--resolution", resolution_str,
                 "Resolution preset (720p, 1080p, 1440p, 4k, 1024x1024) or custom WxH (default: 1080p)");

  std::string output_format;
  app.add_option("--output", output_format,
                 "Machine-readable output format: json or csv")
      ->check(CLI::IsMember({"json", "csv"}));

  std::string output_file;
  app.add_option("--output-file", output_file,
                 "Write machine-readable output to this file instead of "
                 "stdout (requires --output)");

  CLI11_PARSE(app, argc, argv);

  if (!output_file.empty() && output_format.empty()) {
    std::cerr << "Error: --output-file requires --output (json or csv)"
              << std::endl;
    return EXIT_FAILURE;
  }

  // Parse resolution
  uint32_t render_width = 1920;
  uint32_t render_height = 1080;
  std::string res_lower;
  for (char c : resolution_str) res_lower.push_back(std::tolower(static_cast<unsigned char>(c)));

  if (res_lower == "720p") {
    render_width = 1280;
    render_height = 720;
  } else if (res_lower == "1080p" || res_lower == "fhd") {
    render_width = 1920;
    render_height = 1080;
  } else if (res_lower == "1440p" || res_lower == "2k" || res_lower == "qhd") {
    render_width = 2560;
    render_height = 1440;
  } else if (res_lower == "4k" || res_lower == "2160p" || res_lower == "uhd") {
    render_width = 3840;
    render_height = 2160;
  } else if (res_lower == "1024x1024") {
    render_width = 1024;
    render_height = 1024;
  } else {
    auto xPos = res_lower.find('x');
    if (xPos != std::string::npos) {
      try {
        render_width = std::stoul(res_lower.substr(0, xPos));
        render_height = std::stoul(res_lower.substr(xPos + 1));
      } catch (...) {
        std::cerr << "Warning: Invalid resolution string '" << resolution_str
                  << "', defaulting to 1080p (1920x1080)" << std::endl;
        render_width = 1920;
        render_height = 1080;
      }
    } else {
      std::cerr << "Warning: Unrecognized resolution preset '" << resolution_str
                << "', defaulting to 1080p (1920x1080)" << std::endl;
      render_width = 1920;
      render_height = 1080;
    }
  }

  // Debug implies verbose
  if (debug) {
    verbose = true;
  }

  if (list_groups) {
    BenchmarkRunner runner({});
    std::cout << "Available benchmark groups:" << std::endl << std::endl;
    for (const auto &grp : runner.getAvailableGroups()) {
      std::cout << "  " << grp.name << "  (flag: -g " << grp.id << ")" << std::endl;
      std::cout << "    Description: " << grp.description << std::endl;
      std::cout << "    Benchmarks:  ";
      for (size_t i = 0; i < grp.benchmarks.size(); ++i) {
        std::cout << grp.benchmarks[i] << (i + 1 < grp.benchmarks.size() ? ", " : "");
      }
      std::cout << std::endl << std::endl;
    }
    return EXIT_SUCCESS;
  }

  if (list_benchmarks) {
    BenchmarkRunner runner({});
    std::cout << "Available benchmarks (grouped):" << std::endl;
    for (const auto &grp : runner.getAvailableGroups()) {
      std::cout << std::endl << "[" << grp.name << "]  (run group with: -g " << grp.id << ")" << std::endl;
      for (const auto &name : grp.benchmarks) {
        std::cout << "  - " << name << std::endl;
      }
    }
    std::cout << std::endl;
    return EXIT_SUCCESS;
  }

  for (const auto &grp : groups_to_run) {
    benchmarks_to_run.push_back(grp);
  }

  if (verbose) {
    std::cout << "Benchmarks to run: " << std::endl;
    for (const auto &name : benchmarks_to_run) {
      std::cout << "- " << name << std::endl;
    }
  }

  // If machine-readable output is requested to stdout, divert diagnostic
  // logging (banners, progress, tables) to stderr so stdout is pure JSON/CSV.
  std::streambuf *orig_cout = nullptr;
  if (!output_format.empty() && output_file.empty()) {
    orig_cout = std::cout.rdbuf(std::cerr.rdbuf());
  }

  try {
    std::cout << "GPUBench version " << GPUBENCH_VERSION << std::endl
              << std::endl;
    // Create compute contexts for specified backends
    std::vector<std::unique_ptr<IComputeContext>> contexts;
    if (backend_strs.empty() ||
        (backend_strs.size() == 1 && backend_strs[0] == "auto")) {
      // Default to Vulkan, fall back to OpenCL, then ROCm. A backend can be
      // compiled in but fail at runtime (missing driver/GPU), so attempt
      // creation in order and fall through on failure.
      const ComputeBackend auto_order[] = {
          ComputeBackend::Vulkan, ComputeBackend::OpenCL, ComputeBackend::ROCm};
      for (ComputeBackend backend : auto_order) {
        if (!ComputeBackendFactory::isAvailable(backend)) {
          continue;
        }
        try {
          contexts.push_back(
              ComputeBackendFactory::create(backend, verbose, debug));
          break;
        } catch (const std::exception &e) {
          std::cerr << "Backend "
                    << ComputeBackendFactory::getBackendName(backend)
                    << " failed to initialize (" << e.what()
                    << "), trying next backend..." << std::endl;
        }
      }
      if (contexts.empty()) {
        std::cerr << "No compute backend available." << std::endl;
        return EXIT_FAILURE;
      }
    } else {
      for (const auto &backend_str : backend_strs) {
        if (backend_str == "vulkan") {
          if (ComputeBackendFactory::isAvailable(ComputeBackend::Vulkan)) {
            contexts.push_back(
                ComputeBackendFactory::create(ComputeBackend::Vulkan, verbose, debug));
          }
        } else if (backend_str == "opencl") {
          if (ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
            contexts.push_back(
                ComputeBackendFactory::create(ComputeBackend::OpenCL, verbose, debug));
          }
        } else if (backend_str == "rocm") {
          if (ComputeBackendFactory::isAvailable(ComputeBackend::ROCm)) {
            contexts.push_back(
                ComputeBackendFactory::create(ComputeBackend::ROCm, verbose, debug));
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
      // Report both compile-time support and runtime availability (a
      // lightweight context creation probe) for each backend.
      auto reportBackend = [](const char *name, ComputeBackend backend) {
        if (!ComputeBackendFactory::isAvailable(backend)) {
          std::cout << "- " << name << ": Not Supported (not compiled in)"
                    << std::endl;
          return;
        }
        bool runtime = ComputeBackendFactory::isRuntimeAvailable(backend);
        std::cout << "- " << name << ": Supported, runtime "
                  << (runtime ? "available" : "UNAVAILABLE (driver/GPU "
                                             "missing or init failed)")
                  << std::endl;
      };
      std::cout << "Available backends:" << std::endl;
      reportBackend("vulkan", ComputeBackend::Vulkan);
      reportBackend("opencl", ComputeBackend::OpenCL);
      reportBackend("rocm", ComputeBackend::ROCm);
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
              ComputeBackendFactory::create(backend, verbose, debug);

          if (new_context) {
            new_context->pickDevice(device_idx);
            execution_contexts.push_back(std::move(new_context));
          }
        } else {
          std::cerr << "Warning: Device index " << device_idx
                    << " out of range for backend "
                    << ComputeBackendFactory::getBackendName(backend)
                    << std::endl;
        }
      }
    }

    // Now populate context_ptrs for the runner
    for (const auto &ctx : execution_contexts) {
      context_ptrs.push_back(ctx.get());
    }

    // We need to keep execution_contexts alive until runner finishes
    BenchmarkRunner runner(context_ptrs, verbose, debug, dump_geometry, dump_renders);
    runner.setResolution(render_width, render_height);
    runner.run(benchmarks_to_run);

    // Warn about requested benchmark names that matched nothing
    bool hadUnmatched = false;
    for (const auto &name : runner.getUnmatchedBenchmarks()) {
      std::cerr << "Warning: no benchmark matched '" << name
                << "' (see --list-benchmarks)" << std::endl;
      hadUnmatched = true;
    }

    // Machine-readable output (in addition to the human report above)
    if (!output_format.empty()) {
      std::string payload = (output_format == "json")
                                ? resultsToJson(runner.getResults())
                                : resultsToCsv(runner.getResults());
      if (!output_file.empty()) {
        std::ofstream ofs(output_file, std::ios::out | std::ios::trunc);
        if (!ofs) {
          std::cerr << "Error: could not open output file '" << output_file
                    << "'" << std::endl;
          return EXIT_FAILURE;
        }
        ofs << payload;
      } else {
        if (orig_cout) {
          std::cout.rdbuf(orig_cout);
          orig_cout = nullptr;
        }
        std::cout << payload;
      }
    }

    if (orig_cout) {
      std::cout.rdbuf(orig_cout);
      orig_cout = nullptr;
    }

    // Exit non-zero when nothing ran (bogus benchmark names, out-of-range
    // device indices, etc.) so scripts can detect failure.
    if (runner.getNumBenchmarksRun() == 0) {
      std::cerr << "Error: no benchmarks were run." << std::endl;
      return EXIT_FAILURE;
    }
    if (hadUnmatched) {
      return EXIT_FAILURE;
    }

    // execution_contexts will be destroyed here, cleaning up resources

  } catch (const std::exception &e) {
    if (orig_cout) {
      std::cout.rdbuf(orig_cout);
    }
    std::cerr << "An error occurred: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
