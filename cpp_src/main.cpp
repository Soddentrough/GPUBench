#include "CLI11.hpp"
#include "benchmarks/RayAnyHitBench.h"
#include "benchmarks/RayDivergenceBench.h"
#include "benchmarks/RayIntersectBench.h"
#include "benchmarks/RayRawTraversalBench.h"
#include "core/BenchmarkRunner.h"
#include "core/ComputeBackendFactory.h"
#include "core/ResultFormatter.h"
#include "core/ResultImporter.h"
#include "core/RunnerAPI.h"

void SetRunnerTargetConfigs(const std::vector<int> &configs);
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

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
                 "Benchmarks to run (comma-separated, can also be a group name)")
      ->delimiter(',');

  std::vector<std::string> groups_to_run;
  app.add_option("-g,--groups,--group", groups_to_run,
                 "Benchmark group(s) to run: compute, memory, graphics, raster, raytracing, system (or all)")
      ->delimiter(',');

  app.footer(
      "\nBENCHMARK GROUPS & INCLUDED TESTS:\n"
      "  compute     Compute arithmetic units (vector & matrix tensor operations):\n"
      "              FP64, FP32, FP16, BF16, FP8, INT8, INT4\n"
      "  memory      VRAM and GPU cache hierarchy:\n"
      "              Device Memory Bandwidth, L0/L1/L2/L3 Cache Bandwidth & Latency\n"
      "  graphics    All 3D graphics rendering pipelines (combines 'raster' and 'raytracing', alias: 'gfx'):\n"
      "              Runs both fixed-function rasterization (ROP) and hardware ray tracing\n"
      "  raster      Fixed-function rasterization & ROP pixel fill rates (subset of graphics):\n"
      "              Pixel Fill Rate (RGBA8, RGBA16F HDR, Alpha Blending)\n"
      "  raytracing  Hardware BVH traversal, intersection & scheduling (subset of graphics, alias: 'rt'):\n"
      "              RayRawTraversal (Raw BVH Traversal: Coherent Triangles & Deep Multi-Layer BVH8),\n"
      "              RayIntersect, RayAnyHit, RayProcedural, RayDivergence,\n"
      "              RayPayload, RayASBuild, RayScheduling (Megakernel vs DGC / SER),\n"
      "              Pipeline Breakdown (Linear vs 2D Tiled vs Morton Z-Curve, Queue Compaction)\n"
      "  system      Host CPU & RAM system memory:\n"
      "              System Memory Bandwidth (Multi & Single-Threaded), System Memory Latency\n"
      "  all         Run all benchmark groups across enabled devices\n"
  );

  bool list_benchmarks = false;
  app.add_flag("--list-benchmarks,--list", list_benchmarks,
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
               "Dump and analytically compare rendered frames between Megakernel and DGC (default: disabled)");
  bool no_dump_renders = false;
  app.add_flag("--no-dump-renders,--no-dump", no_dump_renders,
               "Disable render dumping and image comparisons");

  bool verify_parity = false;
  app.add_flag("--verify-parity", verify_parity,
               "Enforce visual parity gating between Megakernel and DGC (fails if PSNR < 45 dB or discrepancy > 0.01%)");

  std::string scene_str = "all";
  app.add_option("-s,--scene", scene_str,
                 "Ray tracing benchmark scenario: showroom, indoor, outdoor, forest, all (default: all)")
      ->check(CLI::IsMember({"showroom", "indoor", "outdoor", "forest", "aaa_forest", "all"}));

  std::string resolution_str = "auto";
  app.add_option("-r,--resolution", resolution_str,
                 "Resolution preset (auto, 720p, 1080p, 1440p, 4k, 1024x1024) or custom WxH (default: auto)");

  std::vector<int> config_targets;
  app.add_option("-c,--config", config_targets,
                 "Run specific benchmark configuration index(es) (0-based, comma-separated)")
      ->delimiter(',');

  uint32_t bounce_depth = 2;
  app.add_option("--bounces", bounce_depth,
                 "Ray tracing path tracing bounce depth (1..8, default: 2)")
      ->check(CLI::Range(1u, 8u));

  uint32_t samples_per_pixel = 1;
  app.add_option("--spp", samples_per_pixel,
                 "Ray tracing path tracing samples per pixel (1..256, default: 1)")
      ->check(CLI::Range(1u, 256u));

  bool profile_snapshot = false;
  app.add_flag("--profile-snapshot", profile_snapshot,
               "Run in profiling snapshot mode (1 warmup, 1 timed submit for clean profiler traces)");

  bool rra_trace = false;
  app.add_flag("--rra", rra_trace,
               "Enable Radeon Raytracing Analyzer (RRA) trace capture (implies --profile-snapshot)");

  std::string output_json_path;
  CLI::Option *opt_output_json = app.add_option(
      "-o,--output-json", output_json_path,
      "Write benchmark results to JSON file (optional file path, defaults to gpubench_<hostname>_<timestamp>.json)")
      ->expected(0, 1);

  std::string legacy_output_format;
  CLI::Option *opt_legacy_output = app.add_option(
      "--output", legacy_output_format,
      "Legacy output format: json (deprecated, use --output-json)")
      ->check(CLI::IsMember({"json"}))
      ->group("");

  std::string legacy_output_file;
  CLI::Option *opt_legacy_file = app.add_option(
      "--output-file", legacy_output_file,
      "Legacy output file path (deprecated, use --output-json [FILE])")
      ->group("");

  std::vector<std::string> import_files;
  app.add_option("-i,--import,--input", import_files,
                 "Load and display results from benchmark JSON report file(s)")
      ->expected(1, -1);

  std::vector<std::string> compare_files;
  app.add_option("--compare", compare_files,
                 "Compare benchmark JSON report files side-by-side (e.g. --compare fileA.json fileB.json [fileC.json ...])")
      ->expected(1, -1);

  CLI11_PARSE(app, argc, argv);

  bool want_json_output = false;
  if (opt_output_json->count() > 0) {
    want_json_output = true;
    if (output_json_path.empty()) {
      output_json_path = getDefaultJsonFilename();
    }
  } else if (!legacy_output_file.empty() || !legacy_output_format.empty()) {
    want_json_output = true;
    if (!legacy_output_file.empty()) {
      output_json_path = legacy_output_file;
    } else {
      output_json_path = "-";
    }
  }

  // Collect all files specified across --compare and -i/--import
  std::vector<std::string> all_compare_files;
  for (const auto &f : compare_files) {
    if (!f.empty()) all_compare_files.push_back(f);
  }
  for (const auto &f : import_files) {
    if (!f.empty() && std::find(all_compare_files.begin(), all_compare_files.end(), f) == all_compare_files.end()) {
      all_compare_files.push_back(f);
    }
  }

  // Handle multi-run comparison mode (>= 2 files)
  if (all_compare_files.size() >= 2) {
    std::vector<ImportedRun> runs;
    std::string err;
    if (!ResultImporter::loadFromFiles(all_compare_files, runs, err)) {
      std::cerr << "Error: " << err << std::endl;
      return EXIT_FAILURE;
    }
    ResultFormatter::printComparison(runs);
    return EXIT_SUCCESS;
  }

  if (!compare_files.empty() && all_compare_files.size() < 2) {
    std::cerr << "Error: --compare requires at least two JSON report files (e.g. --compare fileA.json fileB.json [fileC.json ...])" << std::endl;
    return EXIT_FAILURE;
  }

  // Handle single imported result file
  if (!import_files.empty()) {
    ImportedRun run;
    std::string err;
    if (!ResultImporter::loadFromFile(import_files[0], run, err)) {
      std::cerr << "Error: " << err << std::endl;
      return EXIT_FAILURE;
    }

    ResultFormatter formatter;
    for (const auto &res : run.results) {
      formatter.addResult(res);
    }
    formatter.print();

    // Also support re-exporting to json if requested
    if (want_json_output) {
      std::string payload = resultsToJson(run.results);
      if (output_json_path == "-" || output_json_path == "stdout") {
        std::cout << payload;
      } else {
        std::ofstream ofs(output_json_path, std::ios::out | std::ios::trunc);
        if (!ofs) {
          std::cerr << "Error: could not open output file '" << output_json_path << "'" << std::endl;
          return EXIT_FAILURE;
        }
        ofs << payload;
        std::cout << "\n  [JSON Report] Saved to: " << output_json_path << "\n" << std::endl;
      }
    }
    return EXIT_SUCCESS;
  }

  if (rra_trace) {
    profile_snapshot = true;
#ifdef __linux__
    setenv("MESA_VK_TRACE", "rra", 0);
    setenv("MESA_VK_TRACE_FRAME", "1", 0);
#elif defined(_WIN32)
    _putenv("MESA_VK_TRACE=rra");
    _putenv("MESA_VK_TRACE_FRAME=1");
#endif
  }

  // Parse resolution
  uint32_t render_width = 0;
  uint32_t render_height = 0;
  std::string res_lower;
  for (char c : resolution_str) res_lower.push_back(std::tolower(static_cast<unsigned char>(c)));

  if (res_lower == "auto") {
    render_width = 0;
    render_height = 0;
  } else if (res_lower == "720p") {
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
                  << "', defaulting to auto" << std::endl;
        render_width = 0;
        render_height = 0;
      }
    } else {
      std::cerr << "Warning: Unrecognized resolution preset '" << resolution_str
                << "', defaulting to auto" << std::endl;
      render_width = 0;
      render_height = 0;
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

  if (list_backends) {
    auto apiList = GetAllComputeApiSupportAPI();
    std::cout << "Compute API Support & Diagnostics:" << std::endl << std::endl;
    for (const auto &api : apiList) {
      std::cout << "  [" << (api.isSupported ? "+" : "-") << "] " << api.label << ": "
                << (api.isSupported ? "SUPPORTED / AVAILABLE" : "UNSUPPORTED") << std::endl;
      std::cout << "      Status Note: " << api.reason << std::endl;
      if (!api.missingRequirement.empty()) {
        std::cout << "      Missing:     " << api.missingRequirement << std::endl;
      }
      std::cout << std::endl;
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
        if (name == "RayRawTraversal") {
          RayRawTraversalBench rawBench;
          for (uint32_t c = 0; c < rawBench.GetNumConfigs(); ++c) {
            std::cout << "      [" << c << "] " << rawBench.GetConfigName(c) << std::endl;
          }
        }
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
  // logging (banners, progress, tables) to stderr so stdout is pure JSON.
  std::streambuf *orig_cout = nullptr;
  if (want_json_output && (output_json_path == "-" || output_json_path == "stdout")) {
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
          const auto &d = devices[i];
          std::cout << "  " << i << ": " << d.name;
          if (!d.driverName.empty() || !d.driverVersionStr.empty()) {
            std::cout << " [Driver: " << d.driverName << " " << d.driverVersionStr << "]";
          }
          std::cout << std::endl;
        }
      }
      return EXIT_SUCCESS;
    }

    if (no_dump_renders) {
      dump_renders = false;
    }
    if (verify_parity) {
      dump_renders = true;
      no_dump_renders = false;
    }

    BenchmarkRunner runner({}, verbose, debug, dump_geometry, dump_renders, scene_str);
    runner.setResolution(render_width, render_height);
    runner.setBounceDepth(bounce_depth);
    runner.setSamplesPerPixel(samples_per_pixel);
    if (!config_targets.empty()) {
      if (config_targets.size() == 1) {
        runner.setTargetConfig(config_targets[0]);
      }
      SetRunnerTargetConfigs(config_targets);
    }
    runner.setProfileSnapshot(profile_snapshot);
    runner.setVerifyParity(verify_parity);

    std::vector<uint32_t> target_indices = device_indices;
    if (target_indices.empty()) {
      target_indices.push_back(0);
    }

    std::vector<ComputeBackend> target_backends;
    for (const auto &proto_context : contexts) {
      target_backends.push_back(proto_context->getBackend());
    }
    // Drop prototype contexts to free any early probe allocations before benchmark execution
    contexts.clear();

    // Execute each backend and device sequentially.
    // Each context is instantiated, executed, and immediately destroyed to prevent
    // cross-runtime resource contention (e.g. HIP vs Vulkan on display GPU).
    for (ComputeBackend backend : target_backends) {
      for (uint32_t device_idx : target_indices) {
        std::unique_ptr<IComputeContext> new_context =
            ComputeBackendFactory::create(backend, verbose, debug);

        if (new_context) {
          if (device_idx < new_context->getDevices().size()) {
            new_context->pickDevice(device_idx);
            runner.runForContext(new_context.get(), benchmarks_to_run);
          } else {
            std::cerr << "Warning: Device index " << device_idx
                      << " out of range for backend "
                      << ComputeBackendFactory::getBackendName(backend)
                      << std::endl;
          }
        }
        // Context is destroyed here before next device/backend initializes
      }
    }

    runner.runHostBenchmarks(benchmarks_to_run);
    if (!runner.onResult) {
      runner.printReport();
    }

    // Warn about requested benchmark names that matched nothing
    bool hadUnmatched = false;
    for (const auto &name : runner.getUnmatchedBenchmarks()) {
      std::cerr << "Warning: no benchmark matched '" << name
                << "' (see --list-benchmarks)" << std::endl;
      hadUnmatched = true;
    }

    // Machine-readable output (in addition to the human report above)
    if (want_json_output) {
      std::string payload = resultsToJson(runner.getResults());
      if (output_json_path == "-" || output_json_path == "stdout") {
        if (orig_cout) {
          std::cout.rdbuf(orig_cout);
          orig_cout = nullptr;
        }
        std::cout << payload;
      } else {
        std::ofstream ofs(output_json_path, std::ios::out | std::ios::trunc);
        if (!ofs) {
          std::cerr << "\nError: could not write JSON output to '" << output_json_path
                    << "'" << std::endl;
          return EXIT_FAILURE;
        }
        ofs << payload;
        std::cout << "\n  [JSON Report] Saved to: " << output_json_path << "\n" << std::endl;
      }
    }

    if (orig_cout) {
      std::cout.rdbuf(orig_cout);
      orig_cout = nullptr;
    }

    // Exit non-zero when nothing ran (bogus benchmark names, out-of-range
    // device indices, etc.) so scripts can detect failure.
    if (runner.getNumBenchmarksRun() == 0 && runner.getResults().empty()) {
      std::cerr << "Error: no benchmarks were run." << std::endl;
      return EXIT_FAILURE;
    }
    if (hadUnmatched) {
      return EXIT_FAILURE;
    }
    if (runner.hasParityFailure()) {
      std::cerr << "Error: Visual parity verification failed." << std::endl;
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
