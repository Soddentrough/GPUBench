#pragma once

#include <cstdint>
#include <string>
#include <utility>
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

  // Target ASIC for offline ROCm/OpenCL fallback (empty = auto-detect from device)
  std::string asic = "";

  // Print full ISA (and IR) instead of just the hot loop + statistics summary
  bool fullIsa = false;
};

struct IsaStatItem {
  std::string name;
  std::string value;
};

struct IsaInstructionMix {
  int wmma = 0;
  int valu = 0;
  int vmem = 0;
  int lds = 0;
  int bvh = 0;
  int salu = 0;
  int other = 0;
  int total = 0;
};

struct IsaExecutableReport {
  std::string name;
  std::string description;
  uint32_t subgroupSize = 0;
  uint32_t vgprCount = 0;
  uint32_t sgprCount = 0;
  uint32_t ldsBytes = 0;
  uint32_t scratchBytes = 0;
  std::vector<IsaStatItem> stats;
  IsaInstructionMix mix;
  std::vector<std::pair<std::string, int>> topOpcodes;
  int hotLoopStartLine = -1; // 1-based inclusive line in fullIsaLines (-1 if none)
  int hotLoopEndLine = -1;   // 1-based inclusive line in fullIsaLines
  std::vector<std::string> hotLoopLines;
  std::vector<std::string> fullIsaLines;
  std::string unavailableReason;
};

struct IsaArtifactReport {
  std::string benchName;
  std::string configLabel;
  int configIndex = 0;
  std::string backend;
  std::string artifactFile;
  std::string skipReason;
  std::string errorMessage;
  std::vector<IsaExecutableReport> executables;
};

struct IsaInspectionResult {
  bool ok = false;
  std::string deviceName;
  std::string driverInfo;
  std::string resolvedAsic;
  std::string errorSummary;
  std::vector<IsaArtifactReport> artifacts;
};

class IsaDumper {
public:
  // Extracts shader/kernel hardware ISA and register statistics.
  // For Vulkan shaders, uses native VK_KHR_pipeline_executable_properties directly
  // from the active Vulkan driver (zero external tools required).
  static int dump(const IsaDumperOptions &opts);

  // Structured in-memory ISA inspection for IDE/GUI hardware inspectors without stdout spam.
  static IsaInspectionResult inspect(const IsaDumperOptions &opts);
};
