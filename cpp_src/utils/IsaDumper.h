#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct IsaDumperOptions {
  // Target filter(s) from --dump-isa or -b/--benchmark
  std::vector<std::string> targets;

  // Backend filter ("auto", "vulkan", "rocm", "opencl").
  // Empty or "auto" defaults to "vulkan" (native VK_KHR_pipeline_executable_properties).
  std::vector<std::string> backends;

  // Config index filter (-c / --config), e.g., 0 for Vector, 1 for Matrix
  std::vector<int> configs;

  // Vulkan device index (-d / --device, defaults to 0)
  uint32_t deviceIndex = 0;

  // Target ASIC for offline ROCm/OpenCL fallback (defaults to "gfx1201")
  std::string asic = "gfx1201";

  // Print full ISA (and IR) instead of just the hot loop + statistics summary
  bool fullIsa = false;
};

class IsaDumper {
public:
  // Extracts shader/kernel hardware ISA and register statistics.
  // For Vulkan shaders, uses native VK_KHR_pipeline_executable_properties directly
  // from the active Vulkan driver (zero external tools required).
  static int dump(const IsaDumperOptions &opts);
};
