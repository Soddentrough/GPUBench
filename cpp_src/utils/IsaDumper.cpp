#include "utils/IsaDumper.h"
#include "core/DeviceDatabase.h"
#include "third_party/yyjson.h"
#include "utils/KernelPath.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <system_error>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

#ifdef HAVE_VULKAN
#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS
#endif
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_beta.h>
#ifndef VK_EXT_shader_ocp_microscaling_types
#define VK_EXT_shader_ocp_microscaling_types 1
#define VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME "VK_EXT_shader_ocp_microscaling_types"
static constexpr VkStructureType
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT =
        static_cast<VkStructureType>(1000672000);
typedef struct VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT {
  VkStructureType sType;
  void *pNext;
  VkBool32 shaderFloat4;
  VkBool32 shaderFloat6;
  VkBool32 shaderFloat8UnsignedE8M0;
  VkBool32 shaderMXInt8;
} VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT;
#endif
static constexpr VkComponentTypeKHR kIsaComponentTypeFloat6E3M2EXT =
    static_cast<VkComponentTypeKHR>(1000672001);
static constexpr VkComponentTypeKHR kIsaComponentTypeFloat4E2M1EXT =
    static_cast<VkComponentTypeKHR>(1000672002);
#endif

namespace fs = std::filesystem;

namespace {

struct SpecConstEntry {
  uint32_t id = 0;
  uint32_t value = 0;
};

enum class RasterColorFormat {
  RGBA8_UNORM,
  RGBA16F_SFLOAT,
};

struct RasterPipelineConfig {
  bool custom = false;
  RasterColorFormat colorFormat = RasterColorFormat::RGBA8_UNORM;
  bool blendEnable = false;
  bool depthEnable = false;
  uint32_t pushConstantSize = 16u;
  bool hasStorageBufferBinding = false;
};

struct KernelArtifact {
  std::string benchmarkName;
  std::string configName;
  int configIndex = 0;
  std::string backend; // "vulkan", "rocm", "opencl"
  fs::path filePath;
  std::string rocmSymbol;
  std::string skipReason;
  std::vector<SpecConstEntry> specConstants;
  RasterPipelineConfig rasterConfig;
};

static std::string normalizeKey(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c != ' ' && c != '_' && c != '-' && c != '(' && c != ')' && c != '&') {
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
  }
  return out;
}

struct BenchmarkMapping {
  std::string benchName;
  std::vector<std::string> aliases;
  std::vector<int> configIndices;
  std::string configLabel;
  std::string vulkanSpv;
  std::string rocmCo;
  std::string openclCl;
  std::string rocmSymbol = "";
  std::string fixedSkipReason = "";
  std::vector<SpecConstEntry> specConstants = {};
  RasterPipelineConfig rasterConfig = {};
  std::string rocmConfigLabel = "";
};

static std::vector<BenchmarkMapping> getBenchmarkMappings() {
  const std::vector<std::string> rsAliases = {
      "rayscheduling",
      "rtscheduling",
      "rayschedulingopenworldforest",
      "rayschedulingoutdoorlandscape",
      "rayschedulingindooratrium",
      "rayschedulingshowroomstudio",
      "rayexecutionparadigm"};
  const RasterPipelineConfig pixelFillRgba8{
      true, RasterColorFormat::RGBA8_UNORM, false, false, 16u, false};
  const RasterPipelineConfig pixelFillRgba16f{
      true, RasterColorFormat::RGBA16F_SFLOAT, false, false, 16u, false};
  const RasterPipelineConfig pixelFillBlend{
      true, RasterColorFormat::RGBA8_UNORM, true, false, 16u, false};
  const RasterPipelineConfig asyncGfxRaster{
      true, RasterColorFormat::RGBA16F_SFLOAT, false, true, 8u, false};

  return {
      {"BF16", {"bf16"}, {0}, "BF16 (Vector)", "bf16.comp.spv", "bf16.co", ""},
      {"BF16", {"bf16", "bf16matrix", "coopmatrixbf16"}, {1}, "BF16 (Matrix)", "coop_matrix_bf16.comp.spv", "bf16_matrix.co", ""},
      {"FP16", {"fp16"}, {0}, "FP16 (Vector)", "fp16.comp.spv", "fp16.co", "fp16.cl"},
      {"FP16", {"fp16", "fp16matrix", "coopmatrixfp16"}, {1}, "FP16 (Matrix)", "coop_matrix_fp16.comp.spv", "fp16_matrix.co", ""},
      {"FP32", {"fp32"}, {0}, "FP32 (Vector)", "fp32.comp.spv", "fp32.co", "fp32.cl"},
      {"FP64", {"fp64"}, {0}, "FP64 (Vector)", "fp64.comp.spv", "fp64.co", "fp64.cl"},
      {"FP8", {"fp8", "coopmatrixfp8"}, {0}, "FP8 (Matrix)", "coop_matrix_fp8.comp.spv", "fp8_matrix.co", ""},
      {"FP6", {"fp6", "mxfp6", "coopmatrixfp6"}, {0}, "FP6 (Matrix)", "coop_matrix_fp6.comp.spv", "fp6_matrix.co", ""},
      {"FP4", {"fp4", "mxfp4", "coopmatrixfp4"}, {0}, "FP4 (Matrix)", "coop_matrix_fp4.comp.spv", "fp4_matrix.co", ""},
      {"INT8", {"int8"}, {0}, "INT8 (Vector)", "int8.comp.spv", "int8.co", "int8.cl"},
      {"INT8", {"int8", "int8matrix", "coopmatrixint8"}, {1}, "INT8 (Matrix)", "coop_matrix_int8.comp.spv", "int8_matrix.co", ""},
      {"INT4", {"int4", "int4matrix"}, {0}, "INT4 (Matrix)", "", "int4_matrix.co", ""},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, {0}, "FP32 Serialized (1 Chain / Low ILP)", "dual_issue_ilp4.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_ilp4"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, {1}, "FP32 Moderate ILP (8 Chains)", "dual_issue_ilp8.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_ilp8"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, {2}, "FP32 Peak ILP / Dual-Issue (16 Chains)", "dual_issue_ilp16.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_ilp16"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, {3}, "INT32 Serialized (1 Chain / Low ILP)", "dual_issue_int32_4.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_int32_4"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, {4}, "INT32 Moderate ILP (8 Chains)", "dual_issue_int32_8.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_int32_8"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, {5}, "INT32 Peak ILP / Dual-Issue (16 Chains)", "dual_issue_int32.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_int32"},
      {"ILP & Dual-Issue", {"dualissue", "ilp"}, {6}, "Concurrent Mixed (FP32 + INT32)", "dual_issue_mixed.comp.spv", "dual_issue.co", "dual_issue.cl", "run_dual_issue_mixed"},
      {"LDS Bank Conflicts", {"lds", "ldsbankconflicts"}, {0, 1, 2, 3, 4, 5, 6, 7}, "LDS Bank Conflicts", "lds_bank_conflicts.comp.spv", "lds_bank_conflicts.co", "lds_bank_conflicts.cl"},
      {"GPU-Driven Compute Dispatch", {"indirect", "inshaderindirect"}, {0, 1}, "Direct Compute Dispatch", "direct_worker.comp.spv", "direct_worker.co", ""},
      {"GPU-Driven Compute Dispatch", {"indirect", "inshaderindirect"}, {2, 3, 4}, "Indirect Classify", "indirect_classify.comp.spv", "indirect_classify.co", "", "", "", {}, {}, "Persistent Work-Queue Classify"},
      {"GPU-Driven Compute Dispatch", {"indirect", "inshaderindirect"}, {2, 3, 4}, "Indirect Worker", "indirect_worker.comp.spv", "indirect_worker.co", "", "", "", {}, {}, "Persistent Work-Queue Worker"},
      {"GPU-Driven Compute Dispatch", {"indirect", "inshaderindirect"}, {5}, "1024-Bucket Direct Dispatch", "indirect_multibucket_direct.comp.spv", "", ""},
      {"GPU-Driven Compute Dispatch", {"indirect", "inshaderindirect"}, {6, 7}, "1024-Bucket Classify", "indirect_multibucket_classify.comp.spv", "", ""},
      {"GPU-Driven Compute Dispatch", {"indirect", "inshaderindirect"}, {6, 7}, "1024-Bucket Resolve", "indirect_multibucket_resolve.comp.spv", "", ""},
      {"GPU-Driven Compute Dispatch", {"indirect", "inshaderindirect"}, {6, 7}, "1024-Bucket Worker", "indirect_multibucket_worker.comp.spv", "", "", "", "", {{0u, 0u}}},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {0}, "MemBW 128 — Read", "membw_128.comp.spv", "membw_128.co", "membw_128.cl", "", "", {{0u, 0u}}, {}, "MemBW 128 (Runtime Mode)"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {1}, "MemBW 128 — Write", "membw_128.comp.spv", "membw_128.co", "membw_128.cl", "", "", {{0u, 1u}}, {}, "MemBW 128 (Runtime Mode)"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {2}, "MemBW 128 — R/W", "membw_128.comp.spv", "membw_128.co", "membw_128.cl", "", "", {{0u, 2u}}, {}, "MemBW 128 (Runtime Mode)"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {3}, "MemBW 256 — Read", "membw_256.comp.spv", "membw_256.co", "membw_256.cl", "", "", {{0u, 0u}}, {}, "MemBW 256 (Runtime Mode)"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {4}, "MemBW 256 — Write", "membw_256.comp.spv", "membw_256.co", "membw_256.cl", "", "", {{0u, 1u}}, {}, "MemBW 256 (Runtime Mode)"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {5}, "MemBW 256 — R/W", "membw_256.comp.spv", "membw_256.co", "membw_256.cl", "", "", {{0u, 2u}}, {}, "MemBW 256 (Runtime Mode)"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {6}, "MemBW 1024 — Read", "membw_1024.comp.spv", "membw_1024.co", "membw_1024.cl", "", "", {{0u, 0u}}, {}, "MemBW 1024 (Runtime Mode)"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {7}, "MemBW 1024 — Write", "membw_1024.comp.spv", "membw_1024.co", "membw_1024.cl", "", "", {{0u, 1u}}, {}, "MemBW 1024 (Runtime Mode)"},
      {"Device Memory Bandwidth", {"membw", "devicememorybandwidth"}, {8}, "MemBW 1024 — R/W", "membw_1024.comp.spv", "membw_1024.co", "membw_1024.cl", "", "", {{0u, 2u}}, {}, "MemBW 1024 (Runtime Mode)"},
      {"Cache Bandwidth", {"cachebw", "l0cachebandwidth", "l1cachebandwidth", "l2cachebandwidth", "l3cachebandwidth", "l0b", "l1b", "l2b", "l3b", "l0bw", "l1bw", "l2bw", "l3bw"}, {0}, "Cache Bandwidth", "cache_bw_robust.comp.spv", "cache_bw_robust.co", "cache_bw_robust.cl"},
      {"L0 Vector Cache Latency", {"l0cache", "l0cachelatency", "l0vectorcachelatency16kb", "l0l", "cache"}, {0}, "L0 Cache Latency", "l0_cache_latency.comp.spv", "l0_cache_latency.co", "l0_cache_latency.cl", "run_benchmark"},
      {"Cache Latency Curve", {"cachelatencycurve", "cachecurve"}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, "Cache Latency Curve", "cache_latency_curve.comp.spv", "cache_latency_curve.co", "cache_latency_curve.cl"},
      {"Pixel Fill Rate", {"pixelfill", "pixelfillrate", "fillrate", "raster", "rop"}, {0}, "Pixel Fill Rate — RGBA8 Color Fill", "pixel_fill.frag.spv", "", "", "", "", {}, pixelFillRgba8},
      {"Pixel Fill Rate", {"pixelfill", "pixelfillrate", "fillrate", "raster", "rop"}, {1}, "Pixel Fill Rate — RGBA16F HDR Fill", "pixel_fill.frag.spv", "", "", "", "", {}, pixelFillRgba16f},
      {"Pixel Fill Rate", {"pixelfill", "pixelfillrate", "fillrate", "raster", "rop"}, {2}, "Pixel Fill Rate — Alpha Blending Fill", "pixel_fill.frag.spv", "", "", "", "", {}, pixelFillBlend},
      {"RayRawTraversal", {"rayrawtraversal", "rawtraversal"}, {0, 1}, "Raw Traversal", "rt_raw_traversal.comp.spv", "", ""},
      {"RayASBuild", {"rayasbuild", "asbuild", "bvhbuild"}, {0, 1, 2, 3}, "Acceleration Structure Build", "", "", "", "", "Fixed-function Vulkan Acceleration Structure Builder (vkCmdBuildAccelerationStructuresKHR) — executes on hardware BVH build unit with no programmable shader ISA."},
      {"RayIntersect", {"rayintersect"}, {0, 1}, "Ray Intersect", "rt_benchmark.comp.spv", "", ""},
      {"RayAnyHit", {"rayanyhit"}, {0, 1}, "Ray Any-Hit (RTP)", "rayanyhit_pipeline.rgen.spv", "", ""},
      {"RayDivergence", {"raydivergence", "raydiv"}, {0, 1, 2, 3, 4}, "Ray Divergence (RTP)", "raydiv_pipeline.rgen.spv", "", ""},
      {"RayProcedural", {"rayprocedural"}, {0}, "Ray Procedural AABB (RTP)", "rayprocedural.rgen.spv", "", ""},
      {"RayPayload", {"raypayload"}, {0}, "Ray Payload 16B (RTP)", "raypayload_16b.rgen.spv", "", ""},
      {"RayPayload", {"raypayload"}, {1}, "Ray Payload 128B (RTP)", "raypayload_128b.rgen.spv", "", ""},
      {"RayPayload", {"raypayload"}, {2}, "Ray Payload 256B (RTP)", "raypayload_256b.rgen.spv", "", ""},
      // RayScheduling: exact specialization constant_id=0 per config (0..26) matching RaySchedulingBench.cpp
      {"RayScheduling", rsAliases, {0}, "RT Scheduling — Material Shading (Megakernel)", "rt_scheduling_traditional_megakernel.comp.spv", "", "", "", "", {{0u, 4u}}},
      {"RayScheduling", rsAliases, {4, 7}, "RT Scheduling — Path Tracing (Megakernel)", "rt_scheduling_traditional_megakernel.comp.spv", "", "", "", "", {{0u, 1u}}},
      {"RayScheduling", rsAliases, {9}, "RT Scheduling — Primary Rays (Megakernel)", "rt_scheduling_traditional_megakernel.comp.spv", "", "", "", "", {{0u, 7u}}},
      {"RayScheduling", rsAliases, {12}, "RT Scheduling — Incoherent Diffuse GI (Megakernel)", "rt_scheduling_traditional_megakernel.comp.spv", "", "", "", "", {{0u, 2u}}},
      {"RayScheduling", rsAliases, {14}, "RT Scheduling — Multi-Light 128 (Megakernel)", "rt_scheduling_traditional_megakernel.comp.spv", "", "", "", "", {{0u, 11u}}},
      {"RayScheduling", rsAliases, {17, 18}, "RT Scheduling — BVH Traversal (Megakernel)", "rt_scheduling_traditional_megakernel.comp.spv", "", "", "", "", {{0u, 3u}}},
      {"RayScheduling", rsAliases, {16}, "RT Scheduling — Wavefront Classify / Stream Compaction (Alpha Cutout)", "rt_scheduling_device_generated_commands_classify.comp.spv", "", "", "", "", {{0u, 7u}}},
      {"RayScheduling", rsAliases, {19}, "RT Scheduling — Wavefront Classify / Stream Compaction (128B Stream)", "rt_scheduling_device_generated_commands_classify.comp.spv", "", "", "", "", {{0u, 6u}}},
      {"RayScheduling", rsAliases, {20}, "RT Scheduling — Wavefront Classify / Stream Compaction (16B Ballot)", "rt_scheduling_device_generated_commands_classify.comp.spv", "", "", "", "", {{0u, 3u}}},
      {"RayScheduling", rsAliases, {1}, "RT Scheduling — Wavefront Material (Legacy Indirect)", "rt_scheduling_device_generated_commands_material.comp.spv", "", "", "", "", {{0u, 0u}}},
      {"RayScheduling", rsAliases, {2}, "RT Scheduling — DGC Material (ExecutionSet)", "rt_scheduling_device_generated_commands_material.comp.spv", "", "", "", "", {{0u, 0u}}},
      {"RayScheduling", rsAliases, {5, 8}, "RT Scheduling — DGC Path Tracing Bounce", "rt_scheduling_device_generated_commands_bounce.comp.spv", "", "", "", "", {{0u, 0u}}},
      {"RayScheduling", rsAliases, {13}, "RT Scheduling — DGC Octant Bounce", "rt_scheduling_device_generated_commands_bounce.comp.spv", "", "", "", "", {{0u, 2u}}},
      {"RayScheduling", rsAliases, {15}, "RT Scheduling — DGC Multi-Light", "rt_scheduling_device_generated_commands_light.comp.spv", "", ""},
      {"RayScheduling", rsAliases, {10}, "RT Scheduling — RTP Baseline", "rt_scheduling.rgen.spv", "", "", "", "", {{0u, 7u}}},
      {"RayScheduling", rsAliases, {3, 6, 11}, "RT Scheduling — RTP + SER", "rt_scheduling_ser.rgen.spv", "", "", "", "", {{0u, 0u}}},
      {"Interconnect", {"interconnect", "pciehostdeviceinterconnect", "pcie", "rebar", "p2p"}, {0, 1, 2, 3, 5}, "Interconnect — DMA / ReBAR / P2P Transfer", "", "", "", "", "PCIe DMA / Resizable BAR / Peer-to-Peer transfer benchmark — executes via Vulkan/ROCm DMA copy engines and host CPU streaming with no timed compute shader ISA."},
      {"Interconnect", {"interconnect", "pciehostdeviceinterconnect", "pcie", "rebar", "p2p"}, {4}, "Interconnect — Unified Memory Coherency Latency", "interconnect_latency.comp.spv", "interconnect_latency.co", ""},
      {"AsyncOverlap", {"asyncoverlap", "asynccomputecopyoverlap", "async", "multiqueue", "ace", "sdma"}, {0, 1}, "AsyncOverlap — Graphics Pass", "async_gfx.frag.spv", "", "", "", "", {}, asyncGfxRaster},
      {"AsyncOverlap", {"asyncoverlap", "asynccomputecopyoverlap", "async", "multiqueue", "ace", "sdma"}, {0, 1}, "AsyncOverlap — Async Compute Pass", "async_compute_heavy.comp.spv", "", ""},
      {"AsyncOverlap", {"asyncoverlap", "asynccomputecopyoverlap", "async", "multiqueue", "ace", "sdma"}, {2, 3}, "AsyncOverlap — Async Compute Pass", "async_compute_heavy.comp.spv", "async_compute_heavy.co", ""},
      {"SparseMatrix", {"sparsematrix", "2:4structuredsparsematrix", "sparse", "sparsity"}, {0}, "Sparse FP16 2:4 (Matrix)", "", "sparse_fp16_matrix.co", ""},
      {"SparseMatrix", {"sparsematrix", "2:4structuredsparsematrix", "sparse", "sparsity"}, {1}, "Sparse FP8 2:4 (Matrix)", "", "sparse_fp8_matrix.co", ""},
      {"SparseMatrix", {"sparsematrix", "2:4structuredsparsematrix", "sparse", "sparsity"}, {2}, "Sparse INT8 2:4 (Matrix)", "", "sparse_int8_matrix.co", ""},
      {"CoopVector", {"coopvector", "cooperativevector", "cooperativevectorneuralshading", "neural", "mlp"}, {0}, "CoopVector FP16 Shader Baseline", "coop_vector_fp16_baseline.comp.spv", "", ""},
      {"CoopVector", {"coopvector", "cooperativevector", "cooperativevectorneuralshading", "neural", "mlp"}, {1}, "CoopVector FP16 MatVec", "coop_vector_fp16.comp.spv", "", ""},
      {"CoopVector", {"coopvector", "cooperativevector", "cooperativevectorneuralshading", "neural", "mlp"}, {2}, "CoopVector INT8 Shader Baseline", "coop_vector_int8_baseline.comp.spv", "", ""},
      {"CoopVector", {"coopvector", "cooperativevector", "cooperativevectorneuralshading", "neural", "mlp"}, {3}, "CoopVector INT8 MatVec", "coop_vector_int8.comp.spv", "", ""},
#ifdef GPUBENCH_ENABLE_EXPERIMENTAL_WORKGRAPHS
      {"WorkGraphs", {"workgraphs", "gpuworkgraphsproducerconsumer", "shaderenqueue"}, {0}, "WorkGraphs — Indirect / Persistent Producer", "workgraphs_producer.comp.spv", "workgraphs_producer.co", ""},
      {"WorkGraphs", {"workgraphs", "gpuworkgraphsproducerconsumer", "shaderenqueue"}, {1}, "WorkGraphs — Indirect / Persistent Consumer", "workgraphs_consumer.comp.spv", "workgraphs_consumer.co", ""},
      {"WorkGraphs", {"workgraphs", "gpuworkgraphsproducerconsumer", "shaderenqueue"}, {2}, "WorkGraphs — VK_AMDX_shader_enqueue", "workgraphs_amdx.comp.spv", "", ""},
#endif
      {"RayOMM", {"rayomm", "raytracingopacitymicromap", "opacitymicromap", "omm", "micromap"}, {0}, "RayOMM — Compute RayQuery Alpha-Cutout", "rayomm_rq_alpha.comp.spv", "", ""},
      {"RayOMM", {"rayomm", "raytracingopacitymicromap", "opacitymicromap", "omm", "micromap"}, {1}, "RayOMM — RTP Any-Hit Baseline", "rayomm.rgen.spv", "", "", "", "", {{0u, 0u}}},
      {"RayOMM", {"rayomm", "raytracingopacitymicromap", "opacitymicromap", "omm", "micromap"}, {2}, "RayOMM — Hardware 2-State OMM", "rayomm_omm.rgen.spv", "", "", "", "", {{0u, 1u}}},
      {"RayOMM", {"rayomm", "raytracingopacitymicromap", "opacitymicromap", "omm", "micromap"}, {3}, "RayOMM — Hardware 4-State OMM", "rayomm_omm.rgen.spv", "", "", "", "", {{0u, 0u}}},
      {"RayOMM", {"rayomm", "raytracingopacitymicromap", "opacitymicromap", "omm", "micromap"}, {4}, "RayOMM — RTP Force-Opaque Reference", "rayomm.rgen.spv", "", "", "", "", {{0u, 1u}}},
      {"System Memory Bandwidth", {"sysmembandwidth", "rambandwidth", "sysmembw", "sysmem"}, {0, 1, 2, 3, 4, 5}, "Host SIMD Memory Bandwidth", "", "", "", "", "Host CPU SIMD benchmark (AVX-512 / AVX2 / NEON) — executes on system RAM with no GPU shader ISA."},
      {"System Memory Latency", {"sysmemlatency", "ramlatency", "sysmemlat", "sl"}, {0}, "Host Pointer-Chasing Latency", "", "", "", "", "Host CPU pointer-chasing latency benchmark — executes on CPU caches/DRAM with no GPU shader ISA."},
  };
}

static std::string toLowerCopy(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

static void classifyMnemonic(const std::string &mnemonic, IsaInstructionMix &mix) {
  std::string low = toLowerCopy(mnemonic);
  mix.total++;
  if (low.find("wmma") != std::string::npos ||
      low.find("mfma") != std::string::npos ||
      low.find("smfmac") != std::string::npos ||
      low.find("swmmac") != std::string::npos ||
      low.find("mma") != std::string::npos ||
      low.rfind("v_dot", 0) == 0) {
    mix.wmma++;
  } else if (low.find("bvh") != std::string::npos) {
    mix.bvh++;
  } else if (low.rfind("ds_", 0) == 0 || low.rfind("lds_", 0) == 0) {
    mix.lds++;
  } else if (low.rfind("global_", 0) == 0 ||
             low.rfind("buffer_", 0) == 0 ||
             low.rfind("flat_", 0) == 0 ||
             low.rfind("scratch_", 0) == 0 ||
             low.rfind("image_", 0) == 0 ||
             low.rfind("tbuffer_", 0) == 0) {
    mix.vmem++;
  } else if (low.rfind("s_", 0) == 0) {
    mix.salu++;
  } else if (low.rfind("v_", 0) == 0) {
    mix.valu++;
  } else {
    mix.other++;
  }
}

static void analyzeIsaText(const std::string &isaText, IsaExecutableReport &outExec) {
  std::istringstream iss(isaText);
  std::vector<std::string> lines;
  std::string line;
  std::map<std::string, int> opcodeHistogram;

  while (std::getline(iss, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.find("s_code_end") != std::string::npos)
      continue;
    lines.push_back(line);

    if (!line.empty() && (line[0] == '\t' || line[0] == ' ')) {
      size_t start = line.find_first_not_of(" \t");
      if (start != std::string::npos && line[start] != ';' && line[start] != '/' &&
          line[start] != '(') {
        size_t end = line.find_first_of(" \t", start);
        std::string mnemonic = line.substr(start, end - start);
        if (!mnemonic.empty() && mnemonic.back() != ':') {
          opcodeHistogram[mnemonic]++;
          classifyMnemonic(mnemonic, outExec.mix);
        }
      }
    }
  }

  outExec.fullIsaLines = lines;

  if (!opcodeHistogram.empty()) {
    std::vector<std::pair<std::string, int>> sortedOps(opcodeHistogram.begin(),
                                                       opcodeHistogram.end());
    std::sort(sortedOps.begin(), sortedOps.end(),
              [](const auto &a, const auto &b) { return b.second < a.second; });
    for (const auto &[op, count] : sortedOps) {
      if (op == "s_wait_alu" || op == "s_delay_alu" || op == "s_nop")
        continue;
      outExec.topOpcodes.emplace_back(op, count);
      if (outExec.topOpcodes.size() >= 12)
        break;
    }
  }

  // Locate hot loop: look for backward branch (s_cbranch_* jumping to an earlier label)
  std::map<std::string, int> labelLines;
  std::map<std::string, uint64_t> symbolBaseAddr;
  std::map<uint64_t, int> addrToLine;

  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    std::string trimmed = lines[i];
    while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\t'))
      trimmed.pop_back();
    if (!trimmed.empty() && trimmed.back() == ':') {
      if (trimmed[0] != '\t' && trimmed[0] != ' ') {
        labelLines[trimmed.substr(0, trimmed.size() - 1)] = i;
      }
      size_t lt = trimmed.find('<');
      size_t gt = trimmed.find('>', lt != std::string::npos ? lt : 0);
      if (lt != std::string::npos && gt != std::string::npos && gt > lt) {
        labelLines[trimmed.substr(lt, gt - lt + 1)] = i;
        if (lt > 0) {
          std::string addrPart = trimmed.substr(0, lt);
          while (!addrPart.empty() &&
                 (addrPart.back() == ' ' || addrPart.back() == '\t')) {
            addrPart.pop_back();
          }
          if (!addrPart.empty() &&
              std::all_of(addrPart.begin(), addrPart.end(),
                          [](unsigned char ch) { return std::isxdigit(ch) != 0; })) {
            try {
              uint64_t symAddr = std::stoull(addrPart, nullptr, 16);
              std::string symName = trimmed.substr(lt + 1, gt - lt - 1);
              symbolBaseAddr[symName] = symAddr;
            } catch (...) {
            }
          }
        }
      }
    }
    // Parse llvm-objdump instruction address comment: "// 000000001934: ..."
    size_t cmtPos = lines[i].find("// ");
    if (cmtPos != std::string::npos) {
      size_t hexStart = cmtPos + 3;
      size_t hexEnd = hexStart;
      while (hexEnd < lines[i].size() &&
             std::isxdigit(static_cast<unsigned char>(lines[i][hexEnd]))) {
        ++hexEnd;
      }
      if (hexEnd - hexStart >= 4 && hexEnd < lines[i].size() &&
          lines[i][hexEnd] == ':') {
        try {
          uint64_t instAddr =
              std::stoull(lines[i].substr(hexStart, hexEnd - hexStart), nullptr, 16);
          addrToLine.emplace(instAddr, i);
        } catch (...) {
        }
      }
    }
  }

  auto findBranchTargetLine = [&labelLines, &symbolBaseAddr,
                               &addrToLine](const std::string &line) -> int {
    size_t pos = line.find("s_cbranch_");
    if (pos == std::string::npos) pos = line.find("s_branch");
    if (pos == std::string::npos) return -1;
    size_t spacePos = line.find_first_of(" \t", pos);
    if (spacePos == std::string::npos) return -1;
    size_t targetStart = line.find_first_not_of(" \t", spacePos);
    if (targetStart == std::string::npos) return -1;
    size_t targetEnd = line.find_first_of(" \t;/", targetStart);
    std::string targetLabel = line.substr(targetStart, targetEnd - targetStart);
    auto it = labelLines.find(targetLabel);
    if (it == labelLines.end()) {
      size_t lt = line.find('<', targetEnd);
      size_t gt = line.find('>', lt != std::string::npos ? lt : 0);
      if (lt != std::string::npos && gt != std::string::npos && gt > lt) {
        std::string angleTok = line.substr(lt, gt - lt + 1);
        it = labelLines.find(angleTok);
        if (it == labelLines.end()) {
          std::string inner = line.substr(lt + 1, gt - lt - 1);
          size_t plusPos = inner.find('+');
          std::string symName =
              (plusPos != std::string::npos) ? inner.substr(0, plusPos) : inner;
          uint64_t offset = 0;
          if (plusPos != std::string::npos) {
            try {
              offset = std::stoull(inner.substr(plusPos + 1), nullptr, 0);
            } catch (...) {
              offset = 0;
            }
          }
          auto baseIt = symbolBaseAddr.find(symName);
          if (baseIt != symbolBaseAddr.end()) {
            auto addrIt = addrToLine.find(baseIt->second + offset);
            if (addrIt != addrToLine.end()) {
              return addrIt->second;
            }
          }
        }
      }
    }
    return (it != labelLines.end()) ? it->second : -1;
  };

  auto isExitStubTarget = [&lines, &findBranchTargetLine](int labelLineIdx, int branchLineIdx) -> bool {
    int seenInsns = 0;
    // Start at labelLineIdx (not labelLineIdx + 1) because llvm-objdump <sym+offset>
    // targets resolve directly to the instruction line (e.g. s_endpgm) rather than a
    // preceding label line; label lines ending in ':' are skipped below.
    for (int j = labelLineIdx; j < branchLineIdx && seenInsns < 8; ++j) {
      std::string s = lines[j];
      size_t first = s.find_first_not_of(" \t\r");
      if (first == std::string::npos) continue;
      s = s.substr(first);
      if (s.rfind(";", 0) == 0 || s.rfind("//", 0) == 0) continue;
      while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
        s.pop_back();
      if (!s.empty() && s.back() == ':') continue;
      ++seenInsns;
      if (s.rfind("s_endpgm", 0) == 0) return true;
      if (s.rfind("s_cbranch_", 0) == 0 || s.rfind("s_branch", 0) == 0) {
        int dest = findBranchTargetLine(s);
        if (dest <= branchLineIdx) return false;
        continue;
      }
      if (s.rfind("s_cmp_", 0) == 0 || s.rfind("s_waitcnt", 0) == 0 ||
          s.rfind("s_delay_alu", 0) == 0 || s.rfind("s_and_", 0) == 0 ||
          s.rfind("s_or_", 0) == 0 || s.rfind("s_mov_", 0) == 0) {
        continue;
      }
      return false;
    }
    return false;
  };

  int loopStartIdx = -1;
  int loopEndIdx = -1;
  int maxLoopSpan = 0;
  for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
    if (lines[i].find("s_cbranch_") != std::string::npos) {
      int targetIdx = findBranchTargetLine(lines[i]);
      // Skip backward branches targeting shared s_endpgm exit/epilogue stubs
      if (targetIdx != -1 && targetIdx < i && !isExitStubTarget(targetIdx, i)) {
        int span = i - targetIdx + 1;
        if (span >= maxLoopSpan) {
          maxLoopSpan = span;
          loopStartIdx = targetIdx;
          loopEndIdx = i;
        }
      }
    }
  }

  if (loopStartIdx != -1 && loopEndIdx != -1) {
    outExec.hotLoopStartLine = loopStartIdx + 1;
    outExec.hotLoopEndLine = loopEndIdx + 1;
    for (int i = loopStartIdx; i <= loopEndIdx; ++i) {
      outExec.hotLoopLines.push_back(lines[i]);
    }
  }
}

static void inferRocmRegisterCounts(const std::vector<std::string> &lines,
                                    uint32_t &outVgpr, uint32_t &outSgpr) {
  int maxV = -1;
  int maxS = -1;
  auto parseRegToken = [&](const std::string &tok) {
    if (tok.size() < 2) return;
    const char prefix = tok[0];
    if (prefix != 'v' && prefix != 's') return;
    int &targetMax = (prefix == 'v') ? maxV : maxS;
    if (tok[1] == '[' && tok.back() == ']') {
      size_t colon = tok.find(':', 2);
      if (colon != std::string::npos) {
        try {
          int lo = std::stoi(tok.substr(2, colon - 2));
          int hi = std::stoi(tok.substr(colon + 1, tok.size() - colon - 2));
          if (lo >= 0 && hi >= lo && hi < 512) {
            targetMax = std::max(targetMax, hi);
          }
        } catch (...) {}
      }
    } else if (std::isdigit(static_cast<unsigned char>(tok[1]))) {
      bool allDigits = true;
      for (size_t k = 2; k < tok.size(); ++k) {
        if (!std::isdigit(static_cast<unsigned char>(tok[k]))) {
          allDigits = false;
          break;
        }
      }
      if (allDigits) {
        try {
          int idx = std::stoi(tok.substr(1));
          if (idx >= 0 && idx < 512) {
            targetMax = std::max(targetMax, idx);
          }
        } catch (...) {}
      }
    }
  };

  for (const auto &line : lines) {
    if (line.empty() || (line[0] != '\t' && line[0] != ' ')) continue;
    size_t commentPos = std::min(line.find(';'), line.find("//"));
    std::string code = (commentPos == std::string::npos) ? line : line.substr(0, commentPos);
    size_t i = 0;
    while (i < code.size()) {
      char c = code[i];
      if ((c == 'v' || c == 's') &&
          (i == 0 || (!std::isalnum(static_cast<unsigned char>(code[i - 1])) && code[i - 1] != '_'))) {
        if (i + 1 < code.size() && (code[i + 1] == '[' || std::isdigit(static_cast<unsigned char>(code[i + 1])))) {
          size_t end = i + 1;
          if (code[end] == '[') {
            while (end < code.size() && code[end] != ']') ++end;
            if (end < code.size()) ++end;
          } else {
            while (end < code.size() && std::isdigit(static_cast<unsigned char>(code[end]))) ++end;
          }
          if (end >= code.size() || !std::isalnum(static_cast<unsigned char>(code[end]))) {
            parseRegToken(code.substr(i, end - i));
          }
          i = end;
          continue;
        }
      }
      ++i;
    }
  }
  if (outVgpr == 0 && maxV >= 0) outVgpr = static_cast<uint32_t>(maxV + 1);
  if (outSgpr == 0 && maxS >= 0) outSgpr = static_cast<uint32_t>(maxS + 1);
}

static void printIsaExecSummary(const IsaExecutableReport &exec, bool fullIsa) {
  if (!exec.topOpcodes.empty()) {
    std::cout << "  Instruction Mix (Top Opcodes):\n    ";
    int printed = 0;
    for (const auto &[op, count] : exec.topOpcodes) {
      if (printed > 0 && printed % 3 == 0)
        std::cout << "\n    ";
      std::cout << std::left << std::setw(32) << (op + ": " + std::to_string(count));
      if (++printed >= 12)
        break;
    }
    std::cout << "\n\n";
  }

  const auto &lines = exec.fullIsaLines;
  if (lines.empty()) {
    const std::string reason = exec.unavailableReason.empty()
                                   ? "Driver did not return text ISA disassembly for this executable."
                                   : exec.unavailableReason;
    std::cout << "  Disassembly Unavailable (0 text lines):\n";
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n";
    std::cout << "  " << reason << "\n";
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n\n";
    return;
  }

  if (fullIsa) {
    std::cout << "  Full Disassembly (" << lines.size() << " lines):\n";
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n";
    for (const auto &l : lines) {
      std::cout << "  " << l << "\n";
    }
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n\n";
    return;
  }

  if (exec.hotLoopStartLine > 0 && !exec.hotLoopLines.empty()) {
    int totalLoopLines = static_cast<int>(exec.hotLoopLines.size());
    std::cout << "  Hot Loop Disassembly (lines " << exec.hotLoopStartLine << "–" << exec.hotLoopEndLine
              << ", " << totalLoopLines << " instructions — pass --full-isa for entire shader):\n";
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n";
    const int maxPreview = 110;
    if (totalLoopLines <= maxPreview) {
      for (const auto &l : exec.hotLoopLines) {
        std::cout << "  " << l << "\n";
      }
    } else {
      for (int i = 0; i < 50; ++i) {
        std::cout << "  " << exec.hotLoopLines[i] << "\n";
      }
      std::cout << "  \t... (" << (totalLoopLines - 70)
                << " instructions omitted; use --full-isa to view all) ...\n";
      for (int i = totalLoopLines - 20; i < totalLoopLines; ++i) {
        std::cout << "  " << exec.hotLoopLines[i] << "\n";
      }
    }
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n\n";
  } else {
    std::cout << "  Disassembly Preview (" << lines.size() << " total lines):\n";
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n";
    int limit = std::min(static_cast<int>(lines.size()), 80);
    for (int i = 0; i < limit; ++i) {
      std::cout << "  " << lines[i] << "\n";
    }
    if (static_cast<int>(lines.size()) > limit) {
      std::cout << "  \t... (" << (lines.size() - limit)
                << " lines omitted; use --full-isa to view all) ...\n";
    }
    std::cout << "  ──────────────────────────────────────────────────────────────────────────────\n\n";
  }
}

#ifdef HAVE_VULKAN
class NativeVulkanIsaInspector {
public:
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  std::string deviceName;
  std::string driverInfo;
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  bool subgroupSize32Supported = false;
  bool rtPipelineSupported = false;
  bool rayQuerySupported = false;
  bool serSupported = false;
  bool serHardwareSupported = false;
  bool coopMatrixSupported = false;
  bool bf16DotProductSupported = false;
  bool bf16DotProductEmulatedOnArch = false;
  std::string bf16ArchFamily;
  bool bf16CoopMatrixSupported = false;
  bool fp8CoopMatrixSupported = false;
  bool fp6CoopMatrixSupported = false;
  bool fp4CoopMatrixSupported = false;
  bool fp16Supported = false;
  bool fp64Supported = false;
  bool int8Supported = false;
  bool coopVectorSupported = false;
#ifdef GPUBENCH_ENABLE_EXPERIMENTAL_WORKGRAPHS
  bool workGraphsSupported = false;
#endif
  bool opacityMicromapSupported = false;

  PFN_vkGetPipelineExecutablePropertiesKHR pfnGetProps = nullptr;
  PFN_vkGetPipelineExecutableStatisticsKHR pfnGetStats = nullptr;
  PFN_vkGetPipelineExecutableInternalRepresentationsKHR pfnGetIrs = nullptr;
  PFN_vkCreateRayTracingPipelinesKHR pfnCreateRTPipelines = nullptr;

  bool selectCoopTile(VkComponentTypeKHR abType, VkComponentTypeKHR cType,
                      uint32_t &outM, uint32_t &outN, uint32_t &outK) const {
    outM = 16;
    outN = 16;
    outK = 16;
    auto pfnGetCoopProps =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(
                instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!pfnGetCoopProps) {
      return false;
    }
    uint32_t propCount = 0;
    if (pfnGetCoopProps(physicalDevice, &propCount, nullptr) != VK_SUCCESS ||
        propCount == 0) {
      return false;
    }
    std::vector<VkCooperativeMatrixPropertiesKHR> props(propCount);
    for (auto &p : props) {
      p = {};
      p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    }
    if (pfnGetCoopProps(physicalDevice, &propCount, props.data()) != VK_SUCCESS) {
      return false;
    }
    uint64_t bestVol = 0;
    for (const auto &p : props) {
      if (p.AType == abType && p.BType == abType && p.CType == cType &&
          p.ResultType == cType && p.scope == VK_SCOPE_SUBGROUP_KHR &&
          p.MSize > 0 && p.NSize > 0 && p.KSize > 0) {
        if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16) {
          outM = 16;
          outN = 16;
          outK = 16;
          return true;
        }
        const uint64_t vol = static_cast<uint64_t>(p.MSize) * p.NSize * p.KSize;
        if (vol > bestVol) {
          bestVol = vol;
          outM = p.MSize;
          outN = p.NSize;
          outK = p.KSize;
        }
      }
    }
    return bestVol > 0;
  }

  bool quiet = false;
  std::string lastError;

  bool init(uint32_t deviceIdx, bool quietMode = false) {
    quiet = quietMode;
    lastError.clear();
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPUBench ISA Inspector";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "GPUBench";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instInfo{};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) {
      lastError = "Failed to create Vulkan 1.4 instance for ISA inspection.";
      if (!quiet) std::cerr << "Error: " << lastError << "\n";
      return false;
    }

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    if (devCount == 0) {
      lastError = "No Vulkan physical devices found.";
      if (!quiet) std::cerr << "Error: " << lastError << "\n";
      return false;
    }
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, devs.data());
    std::stable_sort(devs.begin(), devs.end(),
                     [](VkPhysicalDevice a, VkPhysicalDevice b) {
                       VkPhysicalDeviceProperties pa{}, pb{};
                       vkGetPhysicalDeviceProperties(a, &pa);
                       vkGetPhysicalDeviceProperties(b, &pb);
                       auto rank = [](VkPhysicalDeviceType t) -> int {
                         if (t == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 0;
                         if (t == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return 1;
                         if (t == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) return 2;
                         return 3;
                       };
                       return rank(pa.deviceType) < rank(pb.deviceType);
                     });
    if (deviceIdx >= devCount) {
      deviceIdx = 0;
    }
    physicalDevice = devs[deviceIdx];

    VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT serProps{};
    serProps.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT;
    VkPhysicalDeviceSubgroupSizeControlProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    subgroupProps.pNext = &serProps;
    VkPhysicalDeviceDriverProperties driverProps{};
    driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    driverProps.pNext = &subgroupProps;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &driverProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
    deviceName = props2.properties.deviceName;
    vendorId = props2.properties.vendorID;
    deviceId = props2.properties.deviceID;
    driverInfo = std::string(driverProps.driverName) + " (" + driverProps.driverInfo + ")";

    // Enumerate supported device extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availExts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, availExts.data());

    auto hasExt = [&](const char *name) {
      for (const auto &e : availExts) {
        if (std::strcmp(e.extensionName, name) == 0)
          return true;
      }
      return false;
    };

    if (!hasExt(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
      lastError = std::string("Active Vulkan driver does not support ") +
                  VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME + ".";
      if (!quiet) std::cerr << "Error: " << lastError << "\n";
      return false;
    }

    std::vector<const char *> enabledExts = {
        VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME};
    const char *optionalExts[] = {
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        "VK_KHR_shader_bfloat16",
        "VK_KHR_cooperative_matrix",
        "VK_NV_cooperative_vector",
        "VK_EXT_shader_float8",
        VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME,
        "VK_KHR_shader_float16_int8",
        "VK_KHR_16bit_storage",
        "VK_KHR_8bit_storage",
        "VK_KHR_shader_subgroup_extended_types",
        "VK_KHR_shader_integer_dot_product",
        "VK_KHR_buffer_device_address",
        "VK_KHR_shader_atomic_int64",
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_query",
        "VK_KHR_ray_tracing_pipeline",
        "VK_EXT_opacity_micromap",
        "VK_KHR_synchronization2",
        "VK_EXT_ray_tracing_invocation_reorder",
        "VK_NV_ray_tracing_invocation_reorder",
        "VK_KHR_deferred_host_operations",
        "VK_KHR_spirv_1_4",
        "VK_KHR_shader_float_controls",
        "VK_KHR_dynamic_rendering"};
    for (const char *opt : optionalExts) {
      if (hasExt(opt)) {
        enabledExts.push_back(opt);
      }
    }

    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR execFeatures{};
    execFeatures.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;

    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16Features{};
    bf16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    execFeatures.pNext = &bf16Features;

    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8Features{};
    fp8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    bf16Features.pNext = &fp8Features;

    VkPhysicalDeviceShaderOCPMicroscalingTypesFeaturesEXT ocpFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OCP_MICROSCALING_TYPES_FEATURES_EXT,
        nullptr, VK_FALSE, VK_FALSE, VK_FALSE, VK_FALSE};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures{};
    coopFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    if (hasExt(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME)) {
      fp8Features.pNext = &ocpFeatures;
      ocpFeatures.pNext = &coopFeatures;
    } else {
      fp8Features.pNext = &coopFeatures;
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{};
    rqFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    coopFeatures.pNext = &rqFeatures;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtpFeatures{};
    rtpFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rqFeatures.pNext = &rtpFeatures;

    VkPhysicalDeviceRayTracingInvocationReorderFeaturesEXT serFeatures{};
    serFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT;
    rtpFeatures.pNext = &serFeatures;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    serFeatures.pNext = &asFeatures;

    VkPhysicalDeviceOpacityMicromapFeaturesEXT ommFeatures{};
    ommFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT;

    VkPhysicalDeviceVulkan11Features v11Features{};
    v11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    if (hasExt("VK_EXT_opacity_micromap")) {
      asFeatures.pNext = &ommFeatures;
      ommFeatures.pNext = &v11Features;
    } else {
      asFeatures.pNext = &v11Features;
    }

    VkPhysicalDeviceVulkan12Features v12Features{};
    v12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v11Features.pNext = &v12Features;

    VkPhysicalDeviceVulkan13Features v13Features{};
    v13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v12Features.pNext = &v13Features;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &execFeatures;

    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
    execFeatures.pipelineExecutableInfo = VK_TRUE;

    subgroupSize32Supported =
        (v13Features.subgroupSizeControl == VK_TRUE) &&
        (subgroupProps.minSubgroupSize <= 32 && subgroupProps.maxSubgroupSize >= 32) &&
        ((subgroupProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0);
    rtPipelineSupported = hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
                          (rtpFeatures.rayTracingPipeline == VK_TRUE);
    rayQuerySupported = hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                        hasExt(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                        (rqFeatures.rayQuery == VK_TRUE);
    serSupported = (hasExt("VK_EXT_ray_tracing_invocation_reorder") &&
                    (serFeatures.rayTracingInvocationReorder == VK_TRUE)) ||
                   hasExt("VK_NV_ray_tracing_invocation_reorder");
    serHardwareSupported =
        (vendorId != 0x1002) &&
        ((hasExt("VK_EXT_ray_tracing_invocation_reorder") &&
          (serFeatures.rayTracingInvocationReorder == VK_TRUE) &&
          (serProps.rayTracingInvocationReorderReorderingHint ==
           VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT)) ||
         hasExt("VK_NV_ray_tracing_invocation_reorder"));
    coopMatrixSupported = hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) &&
                          (coopFeatures.cooperativeMatrix == VK_TRUE);
    coopVectorSupported = hasExt("VK_NV_cooperative_vector");
#ifdef GPUBENCH_ENABLE_EXPERIMENTAL_WORKGRAPHS
    workGraphsSupported = hasExt("VK_AMDX_shader_enqueue");
#endif
    opacityMicromapSupported = hasExt("VK_EXT_opacity_micromap") &&
                               (ommFeatures.micromap == VK_TRUE);
    bool hasBf16 = hasExt("VK_KHR_shader_bfloat16") || hasExt("VK_EXT_shader_bfloat16");
    bf16DotProductSupported = hasBf16 && (bf16Features.shaderBFloat16DotProduct == VK_TRUE);
    bf16CoopMatrixSupported = hasBf16 && coopMatrixSupported &&
                              (bf16Features.shaderBFloat16CooperativeMatrix == VK_TRUE);
    const auto &dbProf = DeviceDatabase::lookup(vendorId, deviceId, deviceName);
    bf16ArchFamily = dbProf.archFamily;
    // RDNA 3/3.5 drivers expose shaderBFloat16DotProduct via FP32 bit-manipulation
    // emulation rather than native BF16 VALU dot2; preserve driver support so ISA
    // inspection continues while flagging the hardware discrepancy.
    if (bf16ArchFamily == "RDNA 3" || bf16ArchFamily == "RDNA 3.5") {
      if (bf16DotProductSupported) {
        bf16DotProductEmulatedOnArch = true;
      }
    }
    fp8CoopMatrixSupported = hasExt("VK_EXT_shader_float8") && coopMatrixSupported &&
                             (fp8Features.shaderFloat8 == VK_TRUE) &&
                             (fp8Features.shaderFloat8CooperativeMatrix == VK_TRUE);
    bool hasOcp = hasExt(VK_EXT_SHADER_OCP_MICROSCALING_TYPES_EXTENSION_NAME);
    uint32_t tm = 16, tn = 16, tk = 16;
    fp6CoopMatrixSupported = hasOcp && coopMatrixSupported && subgroupSize32Supported &&
                             (ocpFeatures.shaderFloat6 == VK_TRUE) &&
                             selectCoopTile(kIsaComponentTypeFloat6E3M2EXT,
                                            VK_COMPONENT_TYPE_FLOAT32_KHR, tm, tn, tk);
    fp4CoopMatrixSupported = hasOcp && coopMatrixSupported && subgroupSize32Supported &&
                             (ocpFeatures.shaderFloat4 == VK_TRUE) &&
                             selectCoopTile(kIsaComponentTypeFloat4E2M1EXT,
                                            VK_COMPONENT_TYPE_FLOAT32_KHR, tm, tn, tk);
    fp16Supported = (v12Features.shaderFloat16 == VK_TRUE);
    fp64Supported = (features2.features.shaderFloat64 == VK_TRUE);
    int8Supported = (v12Features.shaderInt8 == VK_TRUE);

    float qPrio = 1.0f;
    VkDeviceQueueCreateInfo qInfo{};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = 0;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &qPrio;

    VkDeviceCreateInfo devInfo{};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.pNext = &features2;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;
    devInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
    devInfo.ppEnabledExtensionNames = enabledExts.data();

    if (vkCreateDevice(physicalDevice, &devInfo, nullptr, &device) != VK_SUCCESS) {
      std::cerr << "Error: Failed to create logical VkDevice for ISA inspection.\n";
      return false;
    }

    pfnGetProps = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
    pfnGetStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR"));
    pfnGetIrs = reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(
        vkGetDeviceProcAddr(device, "vkGetPipelineExecutableInternalRepresentationsKHR"));
    pfnCreateRTPipelines = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
        vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));

    return pfnGetProps && pfnGetStats && pfnGetIrs;
  }

  ~NativeVulkanIsaInspector() {
    if (device != VK_NULL_HANDLE)
      vkDestroyDevice(device, nullptr);
    if (instance != VK_NULL_HANDLE)
      vkDestroyInstance(instance, nullptr);
  }

  VkShaderModule loadSpvModule(const fs::path &spvPath, std::vector<uint32_t> *outWords = nullptr) {
    std::ifstream ifs(spvPath, std::ios::binary | std::ios::ate);
    if (!ifs) {
      return VK_NULL_HANDLE;
    }
    size_t fileSize = static_cast<size_t>(ifs.tellg());
    if (fileSize == 0 || fileSize % 4 != 0) {
      return VK_NULL_HANDLE;
    }
    ifs.seekg(0, std::ios::beg);
    std::vector<uint32_t> spvCode(fileSize / sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(spvCode.data()), static_cast<std::streamsize>(fileSize));
    if (outWords) {
      *outWords = spvCode;
    }

    VkShaderModuleCreateInfo modInfo{};
    modInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    modInfo.codeSize = fileSize;
    modInfo.pCode = spvCode.data();

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &modInfo, nullptr, &mod) != VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    return mod;
  }

  static bool spirvHasAccelerationStructure(const std::vector<uint32_t> &spvCode) {
    if (spvCode.size() >= 5 && spvCode[0] == 0x07230203) {
      size_t idx = 5;
      while (idx < spvCode.size()) {
        uint32_t word = spvCode[idx];
        uint16_t opcode = static_cast<uint16_t>(word & 0xFFFF);
        uint16_t len = static_cast<uint16_t>((word >> 16) & 0xFFFF);
        if (len == 0 || idx + len > spvCode.size())
          break;
        if (opcode == 5341 /* OpTypeAccelerationStructureKHR */) {
          return true;
        }
        if (opcode == 54 /* OpFunction */) {
          break;
        }
        idx += len;
      }
    }
    return false;
  }

  static uint32_t spirvGetLocalSizeX(const std::vector<uint32_t> &spvCode) {
    if (spvCode.size() >= 5 && spvCode[0] == 0x07230203) {
      size_t idx = 5;
      while (idx < spvCode.size()) {
        uint32_t word = spvCode[idx];
        uint16_t opcode = static_cast<uint16_t>(word & 0xFFFF);
        uint16_t len = static_cast<uint16_t>((word >> 16) & 0xFFFF);
        if (len == 0 || idx + len > spvCode.size())
          break;
        if (opcode == 16 /* OpExecutionMode */ && len >= 6 &&
            spvCode[idx + 2] == 17 /* ExecutionModeLocalSize */) {
          return spvCode[idx + 3];
        }
        if (opcode == 54 /* OpFunction */) {
          break;
        }
        idx += len;
      }
    }
    return 0;
  }

  void capturePipelineExecutables(VkPipeline pipeline, IsaArtifactReport &outArt) {
    VkPipelineInfoKHR pInfo{};
    pInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
    pInfo.pipeline = pipeline;

    uint32_t execCount = 0;
    pfnGetProps(device, &pInfo, &execCount, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> execProps(execCount);
    for (auto &ep : execProps) {
      ep.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    }
    pfnGetProps(device, &pInfo, &execCount, execProps.data());

    for (uint32_t e = 0; e < execCount; ++e) {
      IsaExecutableReport execRep;
      execRep.name = execProps[e].name;
      execRep.description = execProps[e].description;
      execRep.subgroupSize = execProps[e].subgroupSize;

      VkPipelineExecutableInfoKHR execInfo{};
      execInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
      execInfo.pipeline = pipeline;
      execInfo.executableIndex = e;

      uint32_t statCount = 0;
      pfnGetStats(device, &execInfo, &statCount, nullptr);
      std::vector<VkPipelineExecutableStatisticKHR> stats(statCount);
      for (auto &s : stats) {
        s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
      }
      pfnGetStats(device, &execInfo, &statCount, stats.data());

      for (const auto &st : stats) {
        if (std::strcmp(st.name, "Driver pipeline hash") == 0 ||
            std::strcmp(st.name, "Hash") == 0) {
          continue;
        }
        std::ostringstream valStream;
        uint32_t numericVal = 0;
        switch (st.format) {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
          valStream << (st.value.b32 ? "true" : "false");
          numericVal = st.value.b32 ? 1u : 0u;
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
          valStream << st.value.i64;
          numericVal = (st.value.i64 > 0) ? static_cast<uint32_t>(st.value.i64) : 0u;
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
          valStream << st.value.u64;
          numericVal = static_cast<uint32_t>(st.value.u64);
          break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
          valStream << st.value.f64;
          numericVal = (st.value.f64 > 0.0) ? static_cast<uint32_t>(st.value.f64) : 0u;
          break;
        default:
          break;
        }
        execRep.stats.push_back({st.name, valStream.str()});

        std::string lowName = toLowerCopy(st.name);
        if (lowName == "vgprs" || (execRep.vgprCount == 0 && lowName.find("vgpr") != std::string::npos &&
                                   lowName.find("spill") == std::string::npos &&
                                   lowName.find("pre") == std::string::npos)) {
          execRep.vgprCount = numericVal;
        } else if (lowName == "sgprs" || (execRep.sgprCount == 0 && lowName.find("sgpr") != std::string::npos &&
                                          lowName.find("spill") == std::string::npos &&
                                          lowName.find("pre") == std::string::npos)) {
          execRep.sgprCount = numericVal;
        } else if (lowName == "lds size" || lowName.find("lds") != std::string::npos ||
                   lowName.find("shared") != std::string::npos) {
          if (execRep.ldsBytes == 0) execRep.ldsBytes = numericVal;
        } else if (lowName == "scratch size" || lowName.find("scratch") != std::string::npos ||
                   lowName.find("private") != std::string::npos) {
          if (execRep.scratchBytes == 0) execRep.scratchBytes = numericVal;
        }
      }

      uint32_t irCount = 0;
      pfnGetIrs(device, &execInfo, &irCount, nullptr);
      std::vector<VkPipelineExecutableInternalRepresentationKHR> irs(irCount);
      for (auto &ir : irs) {
        ir.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        ir.pData = nullptr;
        ir.dataSize = 0;
      }

      // AMD LLPC on Ray Tracing pipelines often leaves later IR entries at dataSize=0 until
      // earlier entries have non-null pData buffers. Iterate until all advertised IR sizes are fetched.
      std::vector<std::vector<char>> irBuffers(irCount);
      const uint32_t maxIrPasses = irCount + 2;
      for (uint32_t pass = 0; pass < maxIrPasses && irCount > 0; ++pass) {
        for (uint32_t i = 0; i < irCount; ++i) {
          if (irs[i].dataSize > irBuffers[i].size()) {
            irBuffers[i].resize(irs[i].dataSize);
          }
          irs[i].pData = irBuffers[i].empty() ? nullptr : irBuffers[i].data();
          irs[i].dataSize = irBuffers[i].size();
        }
        VkResult irRes = pfnGetIrs(device, &execInfo, &irCount, irs.data());
        bool needRetry = false;
        for (uint32_t i = 0; i < irCount; ++i) {
          if (irs[i].dataSize > irBuffers[i].size()) {
            needRetry = true;
          }
        }
        if (!needRetry && irRes != VK_INCOMPLETE) {
          break;
        }
      }

      auto looksLikeIsaText = [](const char *data, size_t rawSize) -> size_t {
        if (!data || rawSize == 0) return 0;
        // Trim trailing NUL padding if present
        size_t len = rawSize;
        while (len > 0 && data[len - 1] == '\0') {
          --len;
        }
        if (len == 0) return 0;
        // Reject ELF binaries (\x7FELF) or binary headers
        if (len >= 4 && static_cast<unsigned char>(data[0]) == 0x7F &&
            data[1] == 'E' && data[2] == 'L' && data[3] == 'F') {
          return 0;
        }
        size_t printable = 0;
        size_t newlines = 0;
        for (size_t k = 0; k < len; ++k) {
          unsigned char ch = static_cast<unsigned char>(data[k]);
          if (ch == '\n') {
            ++newlines;
            ++printable;
          } else if (ch == '\r' || ch == '\t' || (ch >= 0x20 && ch < 0x7F) || ch >= 0x80) {
            ++printable;
          } else {
            return 0; // Binary control byte / embedded NUL
          }
        }
        return (printable == len && newlines > 0) ? len : 0;
      };

      int chosenIr = -1;
      for (uint32_t i = 0; i < irCount; ++i) {
        std::string irName = toLowerCopy(irs[i].name);
        if (irName.find("disassembly") != std::string::npos ||
            irName.find("isa") != std::string::npos ||
            irName.find("assembly") != std::string::npos) {
          chosenIr = static_cast<int>(i);
          break;
        }
      }
      if (chosenIr == -1 && irCount > 0) {
        chosenIr = static_cast<int>(irCount) - 1;
      }

      if (chosenIr != -1 && irs[chosenIr].pData) {
        const char *rawIr = static_cast<const char *>(irs[chosenIr].pData);
        const size_t availBytes = std::min(irs[chosenIr].dataSize, irBuffers[chosenIr].size());
        size_t textLen = 0;
        if (irs[chosenIr].isText) {
          textLen = strnlen(rawIr, availBytes);
        } else {
          // Some drivers (e.g. AMD proprietary LLPC for RTP ISA.ray-gen) emit ASCII ISA with isText=VK_FALSE
          textLen = looksLikeIsaText(rawIr, availBytes);
        }
        if (textLen > 0) {
          std::string isaStr(rawIr, textLen);
          analyzeIsaText(isaStr, execRep);
          if (execRep.fullIsaLines.empty()) {
            execRep.unavailableReason =
                "Driver returned empty text internal representation ('" +
                std::string(irs[chosenIr].name) + "').";
          }
        } else {
          execRep.unavailableReason =
              "Driver returned non-text / binary internal representation ('" +
              std::string(irs[chosenIr].name) + "', " +
              std::to_string(irs[chosenIr].dataSize) +
              " bytes); text ISA disassembly unavailable.";
        }
      } else if (irCount == 0) {
        execRep.unavailableReason =
            "Vulkan driver exposed pipeline statistics but returned 0 internal representations "
            "(no text ISA emitted for this pipeline/stage).";
      } else {
        execRep.unavailableReason =
            "No usable ISA internal representation found in " +
            std::to_string(irCount) + " driver entries.";
      }

      outArt.executables.push_back(std::move(execRep));
    }
  }

  void printPipelineExecutables(VkPipeline pipeline, const KernelArtifact &art, bool fullIsa,
                                IsaArtifactReport *outReport) {
    IsaArtifactReport localRep;
    IsaArtifactReport &rep = outReport ? *outReport : localRep;
    rep.benchName = art.benchmarkName;
    rep.configLabel = art.configName;
    rep.configIndex = art.configIndex;
    rep.backend = art.backend;
    rep.artifactFile = art.filePath.filename().string();

    capturePipelineExecutables(pipeline, rep);

    if (quiet || outReport) {
      return;
    }

    std::cout << "━━━ [vulkan] " << art.configName
              << " (" << art.filePath.filename().string() << ") ━━━\n";
    for (size_t e = 0; e < rep.executables.size(); ++e) {
      const auto &ex = rep.executables[e];
      std::string subgroupDesc;
      if (ex.subgroupSize == 32 || ex.subgroupSize == 64) {
        subgroupDesc = "Wave" + std::to_string(ex.subgroupSize);
      } else if (ex.subgroupSize == 0) {
        subgroupDesc = "N/A (driver reported 0)";
      } else {
        subgroupDesc = std::to_string(ex.subgroupSize) +
                       " (driver-reported non-wavefront / workgroup value)";
      }
      std::cout << "  Executable [" << e << "] : " << ex.name << " — "
                << ex.description
                << " (Subgroup Size: " << subgroupDesc << ")\n";
      std::cout << "  Hardware Statistics (VK_KHR_pipeline_executable_properties):\n    ";
      int col = 0;
      for (const auto &st : ex.stats) {
        if (col > 0 && col % 3 == 0)
          std::cout << "\n    ";
        std::cout << std::left << std::setw(32) << (st.name + ": " + st.value);
        ++col;
      }
      std::cout << "\n\n";
      printIsaExecSummary(ex, fullIsa);
    }
  }

  bool inspectRasterSpv(const KernelArtifact &art, bool fullIsa, IsaArtifactReport *outReport = nullptr) {
    fs::path dir = art.filePath.parent_path();
    std::string fname = art.filePath.filename().string();
    const bool isAsyncGfx = (fname.rfind("async_gfx.", 0) == 0);
    fs::path vertPath = dir / (isAsyncGfx ? "async_gfx.vert.spv" : "pixel_fill.vert.spv");
    fs::path fragPath = dir / (isAsyncGfx ? "async_gfx.frag.spv" : "pixel_fill.frag.spv");
    if (fname.find(".vert.spv") != std::string::npos) {
      vertPath = art.filePath;
    } else if (fname.find(".frag.spv") != std::string::npos) {
      fragPath = art.filePath;
    }

    RasterPipelineConfig rc = art.rasterConfig;
    if (!rc.custom) {
      if (isAsyncGfx) {
        rc.colorFormat = RasterColorFormat::RGBA16F_SFLOAT;
        rc.blendEnable = false;
        rc.depthEnable = true;
        rc.pushConstantSize = 8u;
        rc.hasStorageBufferBinding = false;
      } else {
        rc.colorFormat = RasterColorFormat::RGBA8_UNORM;
        rc.blendEnable = false;
        rc.depthEnable = false;
        rc.pushConstantSize = 16u;
        rc.hasStorageBufferBinding = false;
      }
    }

    VkShaderModule vertMod = loadSpvModule(vertPath);
    VkShaderModule fragMod = loadSpvModule(fragPath);
    if (!vertMod || !fragMod) {
      if (vertMod) vkDestroyShaderModule(device, vertMod, nullptr);
      if (fragMod) vkDestroyShaderModule(device, fragMod, nullptr);
      std::string err = "Failed to load vertex/fragment SPIR-V modules for " + art.filePath.string();
      if (outReport) outReport->errorMessage = err;
      if (!quiet) std::cerr << "  [Error] " << err << "\n";
      return false;
    }

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    if (rc.hasStorageBufferBinding) {
      VkDescriptorSetLayoutBinding b{};
      b.binding = 0;
      b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b.descriptorCount = 1;
      b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
      VkDescriptorSetLayoutCreateInfo dslInfo{};
      dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      dslInfo.bindingCount = 1;
      dslInfo.pBindings = &b;
      vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl);
    }

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = isAsyncGfx
                             ? (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
                             : VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = rc.pushConstantSize;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    if (dsl != VK_NULL_HANDLE) {
      plInfo.setLayoutCount = 1;
      plInfo.pSetLayouts = &dsl;
    }
    if (rc.pushConstantSize > 0u) {
      plInfo.pushConstantRangeCount = 1;
      plInfo.pPushConstantRanges = &pcRange;
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{0.0f, 0.0f, 3840.0f, 2160.0f, 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {3840, 2160}};
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    if (rc.depthEnable) {
      depthStencil.depthTestEnable = VK_TRUE;
      depthStencil.depthWriteEnable = VK_TRUE;
      depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    }

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (rc.blendEnable) {
      blendAttachment.blendEnable = VK_TRUE;
      blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
      blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    } else {
      blendAttachment.blendEnable = VK_FALSE;
    }
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    VkFormat colorFormat = (rc.colorFormat == RasterColorFormat::RGBA16F_SFLOAT)
                               ? VK_FORMAT_R16G16B16A16_SFLOAT
                               : VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    if (rc.depthEnable) {
      renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    }

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.pNext = &renderingInfo;
    pipeInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                     VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    if (rc.depthEnable) {
      pipeInfo.pDepthStencilState = &depthStencil;
    }
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline);
    if (res == VK_SUCCESS && pipeline != VK_NULL_HANDLE) {
      printPipelineExecutables(pipeline, art, fullIsa, outReport);
      vkDestroyPipeline(device, pipeline, nullptr);
    } else {
      std::string err = "vkCreateGraphicsPipelines failed for " + art.filePath.string() +
                        " (VkResult=" + std::to_string(res) + ")";
      if (outReport) outReport->errorMessage = err;
      if (!quiet) std::cerr << "  [Error] " << err << "\n";
    }

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (dsl != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    }
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
    return res == VK_SUCCESS;
  }

  bool inspectRayTracingSpv(const KernelArtifact &art, bool fullIsa, IsaArtifactReport *outReport = nullptr) {
    if (!rtPipelineSupported || !pfnCreateRTPipelines) {
      std::string err = "VK_KHR_ray_tracing_pipeline not supported on this device.";
      if (outReport) outReport->skipReason = err;
      if (!quiet) std::cerr << "  [Error] " << err << "\n";
      return false;
    }

    fs::path dir = art.filePath.parent_path();
    std::string fname = art.filePath.filename().string();
    std::string prefix = fname;
    size_t dotPos = prefix.find('.');
    if (dotPos != std::string::npos) {
      prefix = prefix.substr(0, dotPos);
    }
    std::string basePrefix = prefix;
    bool isOmm = false;
    if (prefix == "rt_scheduling_ser" || prefix == "raydiv_pipeline_ser") {
      if (!serSupported) {
        std::string reason = "Hardware SER (VK_EXT_ray_tracing_invocation_reorder) is not "
                             "exposed by the active Vulkan driver (" + driverInfo + ").";
        if (outReport) outReport->skipReason = reason;
        if (!quiet) {
          std::cout << "━━━ [vulkan] " << art.configName
                    << " (" << art.filePath.filename().string() << ") ━━━\n"
                    << "  [Skipped] " << reason << "\n\n";
        }
        return true;
      }
      basePrefix = (prefix == "rt_scheduling_ser") ? "rt_scheduling" : "raydiv_pipeline";
    } else if (prefix == "rayomm_omm") {
      if (!opacityMicromapSupported) {
        std::string reason = "Hardware Opacity Micromap (VK_EXT_opacity_micromap) is not "
                             "exposed by the active Vulkan driver (" + driverInfo + ").";
        if (outReport) outReport->skipReason = reason;
        if (!quiet) {
          std::cout << "━━━ [vulkan] " << art.configName
                    << " (" << art.filePath.filename().string() << ") ━━━\n"
                    << "  [Skipped] " << reason << "\n\n";
        }
        return true;
      }
      basePrefix = "rayomm";
      isOmm = true;
    } else if (prefix.rfind("raydiv_pipeline_", 0) == 0) {
      basePrefix = "raydiv_pipeline";
    }

    fs::path rgenPath = dir / (prefix + ".rgen.spv");
    if (!fs::exists(rgenPath)) {
      rgenPath = dir / (basePrefix + ".rgen.spv");
    }
    fs::path rmissPath = dir / (basePrefix + ".rmiss.spv");
    fs::path rchitPath = dir / (prefix + ".rchit.spv");
    if (!fs::exists(rchitPath)) {
      rchitPath = dir / (basePrefix + ".rchit.spv");
    }
    if (!fs::exists(rchitPath)) {
      rchitPath = dir / (basePrefix + "_a.rchit.spv");
    }
    fs::path rahitPath = dir / (basePrefix + ".rahit.spv");
    fs::path rintPath = dir / (basePrefix + ".rint.spv");

    std::vector<VkShaderModule> modules;
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

    std::vector<VkSpecializationMapEntry> specEntries;
    std::vector<uint32_t> specValues;
    if (!art.specConstants.empty()) {
      specEntries.reserve(art.specConstants.size());
      specValues.reserve(art.specConstants.size());
      for (size_t si = 0; si < art.specConstants.size(); ++si) {
        specEntries.push_back(
            {art.specConstants[si].id, static_cast<uint32_t>(si * sizeof(uint32_t)), sizeof(uint32_t)});
        specValues.push_back(art.specConstants[si].value);
      }
    } else if (prefix == "rt_scheduling_ser" || basePrefix == "rt_scheduling") {
      specEntries.push_back({0u, 0u, sizeof(uint32_t)});
      specValues.push_back((prefix == "rt_scheduling_ser") ? 0u : 7u);
    }
    const bool useSpec = !specEntries.empty();
    VkSpecializationInfo specInfo{};
    if (useSpec) {
      specInfo.mapEntryCount = static_cast<uint32_t>(specEntries.size());
      specInfo.pMapEntries = specEntries.data();
      specInfo.dataSize = specValues.size() * sizeof(uint32_t);
      specInfo.pData = specValues.data();
    }

    auto addStage = [&](const fs::path &p, VkShaderStageFlagBits stageBit) -> int {
      if (!fs::exists(p))
        return -1;
      VkShaderModule m = loadSpvModule(p);
      if (!m)
        return -1;
      modules.push_back(m);
      VkPipelineShaderStageCreateInfo st{};
      st.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      st.stage = stageBit;
      st.module = m;
      st.pName = "main";
      if (useSpec) {
        st.pSpecializationInfo = &specInfo;
      }
      stages.push_back(st);
      return static_cast<int>(stages.size()) - 1;
    };

    int rgenIdx = addStage(rgenPath, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
    int rmissIdx = addStage(rmissPath, VK_SHADER_STAGE_MISS_BIT_KHR);
    int rchitIdx = addStage(rchitPath, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
    int rahitIdx = addStage(rahitPath, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
    int rintIdx = addStage(rintPath, VK_SHADER_STAGE_INTERSECTION_BIT_KHR);

    if (rgenIdx >= 0) {
      VkRayTracingShaderGroupCreateInfoKHR g{};
      g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      g.generalShader = static_cast<uint32_t>(rgenIdx);
      g.closestHitShader = VK_SHADER_UNUSED_KHR;
      g.anyHitShader = VK_SHADER_UNUSED_KHR;
      g.intersectionShader = VK_SHADER_UNUSED_KHR;
      groups.push_back(g);
    }
    if (rmissIdx >= 0) {
      VkRayTracingShaderGroupCreateInfoKHR g{};
      g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      g.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      g.generalShader = static_cast<uint32_t>(rmissIdx);
      g.closestHitShader = VK_SHADER_UNUSED_KHR;
      g.anyHitShader = VK_SHADER_UNUSED_KHR;
      g.intersectionShader = VK_SHADER_UNUSED_KHR;
      groups.push_back(g);
    }
    if (rchitIdx >= 0 || rahitIdx >= 0 || rintIdx >= 0) {
      VkRayTracingShaderGroupCreateInfoKHR g{};
      g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
      g.type = (rintIdx >= 0) ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                              : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      g.generalShader = VK_SHADER_UNUSED_KHR;
      g.closestHitShader = (rchitIdx >= 0) ? static_cast<uint32_t>(rchitIdx) : VK_SHADER_UNUSED_KHR;
      g.anyHitShader = (rahitIdx >= 0) ? static_cast<uint32_t>(rahitIdx) : VK_SHADER_UNUSED_KHR;
      g.intersectionShader = (rintIdx >= 0) ? static_cast<uint32_t>(rintIdx) : VK_SHADER_UNUSED_KHR;
      groups.push_back(g);
    }

    VkShaderStageFlags rtStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                  VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

    std::vector<VkDescriptorSetLayoutBinding> bindings(12);
    for (uint32_t i = 0; i < 12; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                                            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = rtStages;
    }

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    dslInfo.pBindings = bindings.data();

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl);

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = rtStages;
    pcRange.offset = 0;
    pcRange.size = 128;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dsl;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout);

    VkRayTracingPipelineCreateInfoKHR rtPipeInfo{};
    rtPipeInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rtPipeInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                       VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    if (isOmm) {
      rtPipeInfo.flags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
    }
    rtPipeInfo.stageCount = static_cast<uint32_t>(stages.size());
    rtPipeInfo.pStages = stages.data();
    rtPipeInfo.groupCount = static_cast<uint32_t>(groups.size());
    rtPipeInfo.pGroups = groups.data();
    rtPipeInfo.maxPipelineRayRecursionDepth = 1;
    rtPipeInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult res = pfnCreateRTPipelines(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtPipeInfo,
                                        nullptr, &pipeline);
    if (res == VK_SUCCESS && pipeline != VK_NULL_HANDLE) {
      printPipelineExecutables(pipeline, art, fullIsa, outReport);
      vkDestroyPipeline(device, pipeline, nullptr);
    } else {
      std::string err = "vkCreateRayTracingPipelinesKHR failed for " + art.filePath.string() +
                        " (VkResult=" + std::to_string(res) + ")";
      if (outReport) outReport->errorMessage = err;
      if (!quiet) std::cerr << "  [Error] " << err << "\n";
    }

    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    for (auto m : modules) {
      vkDestroyShaderModule(device, m, nullptr);
    }
    return res == VK_SUCCESS;
  }

  bool inspectSpv(const KernelArtifact &art, bool fullIsa, IsaArtifactReport *outReport = nullptr) {
    std::string fname = art.filePath.filename().string();
    if (outReport) {
      outReport->benchName = art.benchmarkName;
      outReport->configLabel = art.configName;
      outReport->configIndex = art.configIndex;
      outReport->backend = art.backend;
      outReport->artifactFile = fname;
    }
    if (fname.find(".rgen.spv") != std::string::npos ||
        fname.find(".rmiss.spv") != std::string::npos ||
        fname.find(".rchit.spv") != std::string::npos ||
        fname.find(".rahit.spv") != std::string::npos ||
        fname.find(".rint.spv") != std::string::npos) {
      return inspectRayTracingSpv(art, fullIsa, outReport);
    }
    if (fname.find(".vert.spv") != std::string::npos ||
        fname.find(".frag.spv") != std::string::npos) {
      return inspectRasterSpv(art, fullIsa, outReport);
    }

    auto emitSkip = [&](const std::string &reason) -> bool {
      if (outReport) outReport->skipReason = reason;
      if (!quiet) {
        std::cout << "━━━ [vulkan] " << art.configName << " (" << fname << ") ━━━\n"
                  << "  [Skipped] " << reason << "\n\n";
      }
      return true;
    };

    if (fname == "coop_vector_fp16_baseline.comp.spv" && !fp16Supported) {
      return emitSkip("FP16 shader arithmetic (shaderFloat16) is not supported on " +
                      deviceName + " (" + driverInfo + ").");
    }
    if (fname == "coop_vector_int8_baseline.comp.spv" && !int8Supported) {
      return emitSkip("INT8 shader arithmetic (shaderInt8) is not supported on " +
                      deviceName + " (" + driverInfo + ").");
    }
    if ((fname == "coop_vector_fp16.comp.spv" || fname == "coop_vector_int8.comp.spv") &&
        !coopVectorSupported) {
      return emitSkip("Cooperative vector (VK_NV_cooperative_vector) is not supported on " +
                      deviceName + " (" + driverInfo + ").");
    }
#ifdef GPUBENCH_ENABLE_EXPERIMENTAL_WORKGRAPHS
    if (fname == "workgraphs_amdx.comp.spv" && !workGraphsSupported) {
      return emitSkip("GPU Work Graphs (VK_AMDX_shader_enqueue) is not exposed on " +
                      deviceName + " (" + driverInfo + ").");
    }
#endif
    if (fname == "bf16.comp.spv" && !bf16DotProductSupported) {
      if (bf16ArchFamily == "RDNA 3" || bf16ArchFamily == "RDNA 3.5") {
        return emitSkip(
            "BF16 vector dot product (v_dot2_bf16_bf16) is not natively supported in hardware on " +
            deviceName + " (" + bf16ArchFamily + ") under Vulkan (" + driverInfo + ").");
      }
      return emitSkip("BF16 vector dot product (VK_KHR_shader_bfloat16::shaderBFloat16DotProduct / v_dot2_bf16_bf16) "
                      "is not supported on " + deviceName + " under Vulkan (" + driverInfo + ").");
    }
    if (fname == "coop_matrix_bf16.comp.spv" && !bf16CoopMatrixSupported) {
      return emitSkip("BF16 cooperative matrix (VK_KHR_shader_bfloat16::shaderBFloat16CooperativeMatrix) "
                      "is not supported on " + deviceName + " (" + driverInfo + ").");
    }
    if (fname == "coop_matrix_fp8.comp.spv" && !fp8CoopMatrixSupported) {
      return emitSkip("FP8 cooperative matrix (VK_EXT_shader_float8::shaderFloat8CooperativeMatrix) "
                      "is not supported on " + deviceName + " (" + driverInfo + ").");
    }
    if (fname == "coop_matrix_fp6.comp.spv" && !fp6CoopMatrixSupported) {
      return emitSkip("FP6 cooperative matrix (VK_EXT_shader_ocp_microscaling_types::shaderFloat6 + Float6E3M2EXT tile) "
                      "is not supported on " + deviceName + " (" + driverInfo + ").");
    }
    if (fname == "coop_matrix_fp4.comp.spv" && !fp4CoopMatrixSupported) {
      return emitSkip("FP4 cooperative matrix (VK_EXT_shader_ocp_microscaling_types::shaderFloat4 + Float4E2M1EXT tile) "
                      "is not supported on " + deviceName + " (" + driverInfo + ").");
    }
    if ((fname == "coop_matrix_fp16.comp.spv" || fname == "coop_matrix_int8.comp.spv") &&
        !coopMatrixSupported) {
      return emitSkip("Cooperative matrix (VK_KHR_cooperative_matrix) is not supported on " +
                      deviceName + " (" + driverInfo + ").");
    }
    if (fname == "fp64.comp.spv" && !fp64Supported) {
      return emitSkip("FP64 (shaderFloat64) is not supported on " + deviceName + ".");
    }

    std::vector<uint32_t> spvCode;
    VkShaderModule shaderModule = loadSpvModule(art.filePath, &spvCode);
    if (!shaderModule) {
      std::string err = "Failed to create VkShaderModule for " + art.filePath.string();
      if (outReport) outReport->errorMessage = err;
      if (!quiet) std::cerr << "  [Error] " << err << "\n";
      return false;
    }

    bool hasAccelStruct = spirvHasAccelerationStructure(spvCode);
    if (hasAccelStruct && !rayQuerySupported) {
      vkDestroyShaderModule(device, shaderModule, nullptr);
      return emitSkip("Ray query / acceleration structure (VK_KHR_ray_query) is not supported on " +
                      deviceName + " (" + driverInfo + ").");
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings(12);
    for (uint32_t i = 0; i < 12; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = (hasAccelStruct && i == 0)
                                       ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    dslInfo.pBindings = bindings.data();

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl);

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = 128;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &dsl;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &pipelineLayout);

    bool useCoopSpec = (fname.rfind("coop_matrix_", 0) == 0);
    uint32_t coopSpecVals[4] = {16u, 16u, 16u, subgroupSize32Supported ? 32u : 64u};
    if (useCoopSpec) {
      VkComponentTypeKHR abT = VK_COMPONENT_TYPE_FLOAT16_KHR;
      VkComponentTypeKHR cT = VK_COMPONENT_TYPE_FLOAT32_KHR;
      if (fname == "coop_matrix_bf16.comp.spv") {
        abT = VK_COMPONENT_TYPE_BFLOAT16_KHR;
      } else if (fname == "coop_matrix_fp8.comp.spv") {
        abT = static_cast<VkComponentTypeKHR>(1000491002);
      } else if (fname == "coop_matrix_fp6.comp.spv") {
        abT = kIsaComponentTypeFloat6E3M2EXT;
      } else if (fname == "coop_matrix_fp4.comp.spv") {
        abT = kIsaComponentTypeFloat4E2M1EXT;
      } else if (fname == "coop_matrix_int8.comp.spv") {
        abT = VK_COMPONENT_TYPE_SINT8_KHR;
        cT = VK_COMPONENT_TYPE_SINT32_KHR;
      }
      selectCoopTile(abT, cT, coopSpecVals[0], coopSpecVals[1], coopSpecVals[2]);
    }
    VkSpecializationMapEntry coopSpecEntries[4] = {
        {0, 0, sizeof(uint32_t)},
        {1, sizeof(uint32_t), sizeof(uint32_t)},
        {2, 2 * sizeof(uint32_t), sizeof(uint32_t)},
        {3, 3 * sizeof(uint32_t), sizeof(uint32_t)}};
    VkSpecializationInfo coopSpecInfo{4, coopSpecEntries, sizeof(coopSpecVals), coopSpecVals};

    std::vector<VkSpecializationMapEntry> customSpecEntries;
    std::vector<uint32_t> customSpecVals;
    VkSpecializationInfo customSpecInfo{};
    const bool useCustomSpec = (!useCoopSpec && !art.specConstants.empty());
    if (useCustomSpec) {
      customSpecEntries.reserve(art.specConstants.size());
      customSpecVals.reserve(art.specConstants.size());
      for (size_t si = 0; si < art.specConstants.size(); ++si) {
        customSpecEntries.push_back(
            {art.specConstants[si].id, static_cast<uint32_t>(si * sizeof(uint32_t)), sizeof(uint32_t)});
        customSpecVals.push_back(art.specConstants[si].value);
      }
      customSpecInfo.mapEntryCount = static_cast<uint32_t>(customSpecEntries.size());
      customSpecInfo.pMapEntries = customSpecEntries.data();
      customSpecInfo.dataSize = customSpecVals.size() * sizeof(uint32_t);
      customSpecInfo.pData = customSpecVals.data();
    }

#ifdef GPUBENCH_ENABLE_EXPERIMENTAL_WORKGRAPHS
    if (fname == "workgraphs_amdx.comp.spv") {
      auto pfnCreateGraph =
          reinterpret_cast<PFN_vkCreateExecutionGraphPipelinesAMDX>(
              vkGetDeviceProcAddr(device, "vkCreateExecutionGraphPipelinesAMDX"));
      if (!pfnCreateGraph) {
        if (outReport) outReport->skipReason = "vkCreateExecutionGraphPipelinesAMDX unavailable.";
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, dsl, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        return false;
      }
      VkPipelineShaderStageNodeCreateInfoAMDX n0{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NODE_CREATE_INFO_AMDX, nullptr,
          "producerNode", 0};
      VkPipelineShaderStageNodeCreateInfoAMDX n1{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NODE_CREATE_INFO_AMDX, nullptr,
          "consumerNode", 1};
      VkPipelineShaderStageCreateInfo wgStages[2]{};
      wgStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      wgStages[0].pNext = &n0;
      wgStages[0].stage = VK_SHADER_STAGE_COMPUTE_BIT;
      wgStages[0].module = shaderModule;
      wgStages[0].pName = "producerNode";
      wgStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      wgStages[1].pNext = &n1;
      wgStages[1].stage = VK_SHADER_STAGE_COMPUTE_BIT;
      wgStages[1].module = shaderModule;
      wgStages[1].pName = "consumerNode";

      VkExecutionGraphPipelineCreateInfoAMDX egCi{};
      egCi.sType = VK_STRUCTURE_TYPE_EXECUTION_GRAPH_PIPELINE_CREATE_INFO_AMDX;
      egCi.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                   VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
      egCi.stageCount = 2;
      egCi.pStages = wgStages;
      egCi.layout = pipelineLayout;

      VkPipeline wgPipe = VK_NULL_HANDLE;
      VkResult egRes = pfnCreateGraph(device, VK_NULL_HANDLE, 1, &egCi, nullptr, &wgPipe);
      if (egRes == VK_SUCCESS && wgPipe != VK_NULL_HANDLE) {
        printPipelineExecutables(wgPipe, art, fullIsa, outReport);
        vkDestroyPipeline(device, wgPipe, nullptr);
      }
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device, dsl, nullptr);
      vkDestroyShaderModule(device, shaderModule, nullptr);
      return egRes == VK_SUCCESS;
    }
#endif

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo reqSubgroupInfo{};
    reqSubgroupInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    reqSubgroupInfo.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                     VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = shaderModule;
    pipeInfo.stage.pName = "main";
    if (useCoopSpec) {
      pipeInfo.stage.pSpecializationInfo = &coopSpecInfo;
    } else if (useCustomSpec) {
      pipeInfo.stage.pSpecializationInfo = &customSpecInfo;
    }
    if (subgroupSize32Supported) {
      pipeInfo.stage.pNext = &reqSubgroupInfo;
      uint32_t localSizeX = spirvGetLocalSizeX(spvCode);
      if (localSizeX > 0 && (localSizeX % 32u == 0u)) {
        pipeInfo.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
      }
    }
    pipeInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline);
    if (res != VK_SUCCESS || pipeline == VK_NULL_HANDLE) {
      std::string err = "vkCreateComputePipelines failed for " + art.filePath.string() +
                        " (VkResult=" + std::to_string(res) + ")";
      if (outReport) outReport->errorMessage = err;
      if (!quiet) std::cerr << "  [Error] " << err << "\n";
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      vkDestroyDescriptorSetLayout(device, dsl, nullptr);
      vkDestroyShaderModule(device, shaderModule, nullptr);
      return false;
    }

    printPipelineExecutables(pipeline, art, fullIsa, outReport);
    if (fname == "bf16.comp.spv" && bf16DotProductEmulatedOnArch) {
      const std::string note =
          "Driver lowers BF16 dot() to FP32 on " + bf16ArchFamily +
          " (no native BF16 VALU dot2 hardware)";
      if (outReport) {
        for (auto &ex : outReport->executables) {
          ex.stats.push_back({"Hardware Note", note});
        }
      } else if (!quiet) {
        std::cout << "  [Hardware Note] " << note << "\n\n";
      }
    }

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    return true;
  }
};
#endif

static std::atomic<uint64_t> g_isaTmpCounter{0};

static bool isValidAsicToken(const std::string &s) {
  if (s.empty() || s.size() > 64 || s[0] == '-' || s == "." || s == "..") {
    return false;
  }
  for (unsigned char c : s) {
    if (!std::isalnum(c) && c != '_' && c != '-' && c != '.') {
      return false;
    }
  }
  return true;
}

static bool isValidSymbolToken(const std::string &s) {
  if (s.empty() || s.size() > 256) {
    return false;
  }
  for (unsigned char c : s) {
    if (!std::isalnum(c) && c != '_' && c != '$' && c != '.' && c != ':') {
      return false;
    }
  }
  return true;
}

struct TempFileGuard {
  fs::path path;
  ~TempFileGuard() {
    if (!path.empty()) {
      std::error_code ec;
      fs::remove(path, ec);
    }
  }
};

#ifdef _WIN32
static std::string quoteWinArg(const std::string &arg) {
  if (!arg.empty() && arg.find_first_of(" \t\n\v\"") == std::string::npos) {
    return arg;
  }
  std::string q = "\"";
  for (size_t i = 0; i < arg.size(); ++i) {
    size_t backslashes = 0;
    while (i < arg.size() && arg[i] == '\\') {
      ++backslashes;
      ++i;
    }
    if (i == arg.size()) {
      q.append(backslashes * 2, '\\');
      break;
    }
    if (arg[i] == '"') {
      q.append(backslashes * 2 + 1, '\\');
      q.push_back('"');
    } else {
      q.append(backslashes, '\\');
      q.push_back(arg[i]);
    }
  }
  q.push_back('"');
  return q;
}
#endif

static int runCommandSilent(const std::vector<std::string> &argv, std::string &output) {
  output.clear();
  if (argv.empty()) {
    return -1;
  }
#ifdef _WIN32
  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  HANDLE hRead = nullptr, hWrite = nullptr;
  if (!CreatePipe(&hRead, &hWrite, &sa, 0)) {
    return -1;
  }
  SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = hWrite;
  si.hStdError = hWrite;
  si.hStdInput = nullptr;

  std::string cmdLine;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (i > 0) cmdLine.push_back(' ');
    cmdLine += quoteWinArg(argv[i]);
  }
  std::vector<char> mutableCmd(cmdLine.begin(), cmdLine.end());
  mutableCmd.push_back('\0');

  PROCESS_INFORMATION pi{};
  BOOL ok = CreateProcessA(nullptr, mutableCmd.data(), nullptr, nullptr, TRUE,
                           CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
  CloseHandle(hWrite);
  if (!ok) {
    CloseHandle(hRead);
    return -1;
  }

  char buffer[512];
  DWORD bytesRead = 0;
  while (ReadFile(hRead, buffer, sizeof(buffer), &bytesRead, nullptr) && bytesRead > 0) {
    output.append(buffer, bytesRead);
  }
  CloseHandle(hRead);
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exitCode = 1;
  GetExitCodeProcess(pi.hProcess, &exitCode);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return static_cast<int>(exitCode);
#else
  int pipefd[2] = {-1, -1};
  if (pipe(pipefd) != 0) {
    return -1;
  }
  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions) != 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }
  bool setupOk =
      (posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0) == 0) &&
      (posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO) == 0) &&
      (posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO) == 0) &&
      (posix_spawn_file_actions_addclose(&actions, pipefd[0]) == 0) &&
      (posix_spawn_file_actions_addclose(&actions, pipefd[1]) == 0);

  std::vector<std::string> ownedArgs = argv;
  std::vector<char *> cArgs;
  cArgs.reserve(ownedArgs.size() + 1);
  for (std::string &s : ownedArgs) {
    cArgs.push_back(s.data());
  }
  cArgs.push_back(nullptr);

  pid_t pid = -1;
  int spawnRc = setupOk ? posix_spawnp(&pid, ownedArgs[0].c_str(), &actions, nullptr,
                                       cArgs.data(), environ)
                        : -1;
  posix_spawn_file_actions_destroy(&actions);
  close(pipefd[1]);

  if (spawnRc != 0) {
    close(pipefd[0]);
    return -1;
  }

  char buffer[512];
  for (;;) {
    ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
    if (n > 0) {
      output.append(buffer, static_cast<size_t>(n));
    } else if (n == 0) {
      break;
    } else if (errno != EINTR) {
      break;
    }
  }
  close(pipefd[0]);

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      return -1;
    }
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return -1;
#endif
}

static std::string extractGfxToken(const std::string &s) {
  size_t pos = s.find("gfx");
  if (pos == std::string::npos) {
    return "";
  }
  size_t end = pos + 3;
  while (end < s.size() && std::isalnum(static_cast<unsigned char>(s[end]))) {
    ++end;
  }
  if (end > pos + 3) {
    return s.substr(pos, end - pos);
  }
  return "";
}

static std::string getRocmAsicSkipReason(const std::string &fname, const std::string &asic) {
  const std::string lowAsic = toLowerCopy(asic);
  const bool isGfx950 = (lowAsic == "gfx950");
  const bool isGfx94x = (lowAsic.rfind("gfx94", 0) == 0);
  const bool isGfx90a = (lowAsic == "gfx90a");
  const bool isGfx11x = (lowAsic.rfind("gfx11", 0) == 0);
  const bool isGfx12x = (lowAsic.rfind("gfx12", 0) == 0);

  if (fname == "fp6_matrix.co" || fname == "fp4_matrix.co") {
    if (!isGfx950) {
      return "ROCm " + fname +
             " requires AMD CDNA 4 (gfx950) MXFP6/MXFP4 MFMA hardware; target ASIC '" +
             asic + "' compiles only a fallback stub.";
    }
  } else if (fname == "fp8_matrix.co" ||
             (fname.rfind("sparse_", 0) == 0 && fname.find("_matrix.co") != std::string::npos)) {
    if (!(isGfx12x || isGfx94x || isGfx950)) {
      return "ROCm " + fname +
             " requires AMD RDNA 4 (gfx12*) or CDNA 3/4 (gfx94*/gfx950) matrix hardware; target ASIC '" +
             asic + "' is unsupported.";
    }
  } else if (fname == "int4_matrix.co") {
    if (!(isGfx11x || isGfx12x)) {
      return "ROCm int4_matrix.co requires AMD RDNA 3/3.5/4 (gfx11*/gfx12*) WMMA hardware; target ASIC '" +
             asic + "' is unsupported.";
    }
  } else if (fname == "fp16_matrix.co" || fname == "bf16_matrix.co" || fname == "int8_matrix.co") {
    if (!(isGfx11x || isGfx12x || isGfx90a || isGfx94x || isGfx950)) {
      return "ROCm " + fname +
             " requires AMD RDNA 3/4 (gfx11*/gfx12*) WMMA or CDNA 2/3/4 (gfx90a/gfx94*/gfx950) MFMA hardware; target ASIC '" +
             asic + "' is unsupported.";
    }
  }
  return "";
}

static bool inspectRocmCodeObject(const KernelArtifact &art, const std::string &asic,
                                  bool fullIsa, bool quiet,
                                  IsaArtifactReport *outReport = nullptr) {
  auto failReport = [&](const std::string &err) -> bool {
    if (outReport) {
      outReport->benchName = art.benchmarkName;
      outReport->configLabel = art.configName;
      outReport->configIndex = art.configIndex;
      outReport->backend = art.backend;
      outReport->artifactFile = art.filePath.filename().string();
      outReport->errorMessage = err;
    }
    if (!quiet) {
      std::cerr << "  [Error] " << err << "\n";
    }
    return false;
  };

  if (!isValidAsicToken(asic)) {
    return failReport("Rejected invalid ROCm target ASIC token");
  }
  if (!art.rocmSymbol.empty() && !isValidSymbolToken(art.rocmSymbol)) {
    return failReport("Rejected invalid ROCm symbol token");
  }

  const std::string fname = art.filePath.filename().string();
  const std::string archSkip = getRocmAsicSkipReason(fname, asic);
  if (!archSkip.empty()) {
    if (outReport) {
      outReport->benchName = art.benchmarkName;
      outReport->configLabel = art.configName;
      outReport->configIndex = art.configIndex;
      outReport->backend = art.backend;
      outReport->artifactFile = fname;
      outReport->skipReason = archSkip;
    }
    if (!quiet) {
      std::cout << "━━━ [rocm] " << art.configName
                << " (" << fname << " — " << asic << ") ━━━\n"
                << "  [Skipped] " << archSkip << "\n\n";
    }
    return true;
  }

  // Locate clang-offload-bundler and llvm-objdump deterministically, preferring active ROCm 10.2+
  std::string bundler = "clang-offload-bundler";
  std::string objdump = "llvm-objdump";

  std::error_code ec;
  bool foundTools = false;
  auto tryToolDir = [&](const fs::path &binDir) -> bool {
    fs::path b = binDir / "clang-offload-bundler";
    fs::path d = binDir / "llvm-objdump";
    if (fs::exists(b, ec) && fs::exists(d, ec)) {
      bundler = b.string();
      objdump = d.string();
      return true;
    }
    return false;
  };

  std::vector<fs::path> rocmBinCandidates;
  if (const char *rocmEnv = std::getenv("ROCM_PATH"); rocmEnv && rocmEnv[0] != '\0') {
    rocmBinCandidates.emplace_back(fs::path(rocmEnv) / "lib" / "llvm" / "bin");
    rocmBinCandidates.emplace_back(fs::path(rocmEnv) / "llvm" / "bin");
  }
  rocmBinCandidates.emplace_back("/opt/rocm/core/lib/llvm/bin");
  rocmBinCandidates.emplace_back("/opt/rocm/llvm/bin");
  rocmBinCandidates.emplace_back("/opt/rocm/lib/llvm/bin");
  rocmBinCandidates.emplace_back("/opt/rocm/core-10.2/lib/llvm/bin");
  rocmBinCandidates.emplace_back("/opt/rocm/current/lib/llvm/bin");

  for (const auto &dir : rocmBinCandidates) {
    if (tryToolDir(dir)) {
      foundTools = true;
      break;
    }
  }
  if (!foundTools && fs::is_directory("/opt/rocm", ec)) {
    std::vector<fs::path> versionedDirs;
    for (const auto &entry : fs::directory_iterator("/opt/rocm", ec)) {
      if (entry.is_directory(ec)) {
        versionedDirs.push_back(entry.path());
      }
    }
    std::sort(versionedDirs.begin(), versionedDirs.end(), std::greater<fs::path>());
    for (const auto &dir : versionedDirs) {
      if (tryToolDir(dir / "lib" / "llvm" / "bin") ||
          tryToolDir(dir / "llvm" / "bin")) {
        foundTools = true;
        break;
      }
    }
  }

  TempFileGuard tmpGuard;
  fs::path tmpDir = fs::temp_directory_path(ec);
  if (ec || tmpDir.empty()) {
    return failReport("Failed to resolve temp directory for ROCm code object extraction");
  }
#ifdef _WIN32
  char tmpBuf[MAX_PATH] = {};
  if (GetTempFileNameA(tmpDir.string().c_str(), "gpu", 0, tmpBuf) == 0) {
    return failReport("Failed to create unique temp file for ROCm code object extraction");
  }
  tmpGuard.path = tmpBuf;
#else
  std::string tmpl = (tmpDir / "gpubench_rocm_XXXXXX").string();
  std::vector<char> tmplBuf(tmpl.begin(), tmpl.end());
  tmplBuf.push_back('\0');
  int tmpFd = mkstemp(tmplBuf.data());
  if (tmpFd < 0) {
    return failReport("Failed to create unique temp file for ROCm code object extraction");
  }
  close(tmpFd);
  tmpGuard.path = tmplBuf.data();
#endif

  std::string out;
  std::vector<std::string> unbundleArgs = {
      bundler,
      "--type=o",
      "--bundle-align=4096",
      "--targets=hipv4-amdgcn-amd-amdhsa--" + asic,
      "--input=" + art.filePath.string(),
      "--output=" + tmpGuard.path.string(),
      "--unbundle"};
  runCommandSilent(unbundleArgs, out);

  const uintmax_t tmpSz = fs::file_size(tmpGuard.path, ec);
  fs::path targetPath = (!ec && tmpSz > 0) ? tmpGuard.path : art.filePath;
  std::vector<std::string> disArgs = {objdump, "-d", "--mcpu=" + asic};
  if (!art.rocmSymbol.empty()) {
    disArgs.push_back("--disassemble-symbols=" + art.rocmSymbol);
  }
  disArgs.push_back(targetPath.string());
  int rc = runCommandSilent(disArgs, out);
  if (rc != 0 || out.empty()) {
    return failReport("Failed to disassemble ROCm code object: " + art.filePath.string());
  }

  IsaExecutableReport exec;
  exec.name = !art.rocmSymbol.empty() ? art.rocmSymbol : art.filePath.filename().string();
  exec.description = "ROCm AMDGPU Disassembly (" + asic + ")";
  exec.subgroupSize = (asic.rfind("gfx9", 0) == 0) ? 64 : 32;
  analyzeIsaText(out, exec);
  inferRocmRegisterCounts(exec.fullIsaLines, exec.vgprCount, exec.sgprCount);
  exec.stats.push_back({"Instructions", std::to_string(exec.mix.total)});
  if (exec.vgprCount > 0) {
    exec.stats.push_back({"VGPRs (inferred)", std::to_string(exec.vgprCount)});
  }
  if (exec.sgprCount > 0) {
    exec.stats.push_back({"SGPRs (inferred)", std::to_string(exec.sgprCount)});
  }

  if (outReport) {
    outReport->benchName = art.benchmarkName;
    outReport->configLabel = art.configName;
    outReport->configIndex = art.configIndex;
    outReport->backend = art.backend;
    outReport->artifactFile = art.filePath.filename().string();
    outReport->executables.push_back(std::move(exec));
    return true;
  }

  if (!quiet) {
    std::cout << "━━━ [rocm] " << art.configName
              << " (" << art.filePath.filename().string()
              << (!art.rocmSymbol.empty() ? (" :: " + art.rocmSymbol) : "")
              << " — " << asic << ") ━━━\n\n";
    printIsaExecSummary(exec, fullIsa);
  }
  return true;
}

static std::set<std::string> resolveWantedBackends(const IsaDumperOptions &opts) {
  std::set<std::string> wantedBackends;
  for (const auto &b : opts.backends) {
    std::string low = normalizeKey(b);
    if (low == "all") {
      wantedBackends.insert("vulkan");
      wantedBackends.insert("rocm");
    } else if (low.empty() || low == "auto" || low == "vulkan") {
      wantedBackends.insert("vulkan");
    } else {
      wantedBackends.insert(low);
    }
  }
  if (wantedBackends.empty()) {
    wantedBackends = {"vulkan"};
  }
  return wantedBackends;
}

static bool matchesBenchmarkMapping(const BenchmarkMapping &m, const std::string &normTarget) {
  if ((normalizeKey(m.benchName) == normTarget) ||
      (normalizeKey(m.configLabel) == normTarget)) {
    return true;
  }
  for (const auto &al : m.aliases) {
    if (normalizeKey(al) == normTarget) {
      return true;
    }
  }
  return false;
}

static std::string buildArtifactVisitKey(const KernelArtifact &art, bool explicitConfigs) {
  std::string key = art.backend + ":";
  if (explicitConfigs || art.backend == "vulkan") {
    key += std::to_string(art.configIndex) + ":";
  }
  key += art.filePath.string() + ":" + art.rocmSymbol + ":" + art.configName;
  if (!art.specConstants.empty()) {
    key += ":spec(";
    for (size_t i = 0; i < art.specConstants.size(); ++i) {
      if (i > 0) key.push_back(',');
      key += std::to_string(art.specConstants[i].id) + "=" +
             std::to_string(art.specConstants[i].value);
    }
    key.push_back(')');
  }
  if (art.rasterConfig.custom) {
    key += ":raster(" +
           std::to_string(static_cast<int>(art.rasterConfig.colorFormat)) + "," +
           (art.rasterConfig.blendEnable ? "1" : "0") + "," +
           (art.rasterConfig.depthEnable ? "1" : "0") + ")";
  }
  return key;
}

static std::vector<KernelArtifact> resolveArtifactsForOptions(const IsaDumperOptions &opts,
                                                              const fs::path &kernelDir,
                                                              const std::set<std::string> &wantedBackends) {
  std::vector<KernelArtifact> artifacts;
  auto mappings = getBenchmarkMappings();

  for (const auto &target : opts.targets) {
    fs::path directPath(target);
    if (fs::exists(directPath) && fs::is_regular_file(directPath)) {
      std::string ext = directPath.extension().string();
      std::string backend = (ext == ".co" || ext == ".hsaco" || ext == ".o") ? "rocm" : "vulkan";
      artifacts.push_back(
          {directPath.filename().string(), directPath.filename().string(), 0, backend, directPath, "", "", {}, {}});
      continue;
    }

    std::string normTarget = normalizeKey(target);
    auto resolveMatchingConfigs = [&opts](const BenchmarkMapping &m) {
      std::vector<int> matched;
      if (m.configIndices.empty()) {
        return matched;
      }
      if (opts.configs.empty()) {
        matched.push_back(m.configIndices.front());
        return matched;
      }
      for (int cfg : opts.configs) {
        if (std::find(m.configIndices.begin(), m.configIndices.end(), cfg) !=
            m.configIndices.end()) {
          matched.push_back(cfg);
        }
      }
      return matched;
    };

    if (normTarget == "all") {
      for (const auto &m : mappings) {
        const std::vector<int> matchedConfigs = resolveMatchingConfigs(m);
        const std::string rocmLabel =
            !m.rocmConfigLabel.empty() ? m.rocmConfigLabel : m.configLabel;
        for (int cfgIdx : matchedConfigs) {
          if (wantedBackends.count("vulkan") && !m.vulkanSpv.empty()) {
            fs::path p = kernelDir / "vulkan" / m.vulkanSpv;
            if (fs::exists(p)) {
              artifacts.push_back({m.benchName, m.configLabel, cfgIdx, "vulkan", p, "", "",
                                   m.specConstants, m.rasterConfig});
            }
          }
          if (wantedBackends.count("rocm") && !m.rocmCo.empty()) {
            fs::path p = kernelDir / "rocm" / m.rocmCo;
            if (fs::exists(p)) {
              artifacts.push_back({m.benchName, rocmLabel, cfgIdx, "rocm", p, m.rocmSymbol, "", {}, {}});
            }
          }
        }
      }
      continue;
    }

    bool matchedBench = false;
    struct PendingSkip {
      KernelArtifact artifact;
      bool isFixedSkip = false;
    };
    std::vector<KernelArtifact> realTargetArtifacts;
    std::vector<PendingSkip> pendingSkips;
    std::set<std::string> realBackendKeys; // benchName|cfgIdx|backend
    std::set<std::string> realBackendsPresent;

    for (const auto &m : mappings) {
      if (!matchesBenchmarkMapping(m, normTarget)) {
        continue;
      }

      const std::vector<int> matchedConfigs = resolveMatchingConfigs(m);
      if (matchedConfigs.empty()) {
        continue;
      }

      matchedBench = true;
      const std::string rocmLabel =
          !m.rocmConfigLabel.empty() ? m.rocmConfigLabel : m.configLabel;
      for (int cfgIdx : matchedConfigs) {
        if (wantedBackends.count("vulkan")) {
          if (!m.vulkanSpv.empty()) {
            fs::path p = kernelDir / "vulkan" / m.vulkanSpv;
            if (fs::exists(p)) {
              realTargetArtifacts.push_back({m.benchName, m.configLabel, cfgIdx, "vulkan", p, "", "",
                                             m.specConstants, m.rasterConfig});
              realBackendKeys.insert(m.benchName + "|" + std::to_string(cfgIdx) + "|vulkan");
              realBackendsPresent.insert("vulkan");
            }
          } else if (wantedBackends.size() == 1) {
            std::string reason;
            if (!m.fixedSkipReason.empty()) {
              reason = m.fixedSkipReason;
            } else if (!m.rocmCo.empty()) {
              reason = m.configLabel + " has no Vulkan shader artifact (supported on ROCm via " + m.rocmCo + "; pass '-k rocm').";
            } else {
              reason = m.configLabel + " is a capability-probed benchmark with no compiled shader artifact on this toolchain.";
            }
            pendingSkips.push_back(
                {{m.benchName, m.configLabel, cfgIdx, "vulkan", {}, "", reason, {}, {}},
                 !m.fixedSkipReason.empty()});
          }
        }
        if (wantedBackends.count("rocm")) {
          if (!m.rocmCo.empty()) {
            fs::path p = kernelDir / "rocm" / m.rocmCo;
            if (fs::exists(p)) {
              realTargetArtifacts.push_back(
                  {m.benchName, rocmLabel, cfgIdx, "rocm", p, m.rocmSymbol, "", {}, {}});
              realBackendKeys.insert(m.benchName + "|" + std::to_string(cfgIdx) + "|rocm");
              realBackendsPresent.insert("rocm");
            } else if (wantedBackends.size() == 1) {
              std::string reason =
                  rocmLabel + " ROCm code object (" + m.rocmCo +
                  ") not found on disk; build with -DGPUBENCH_ENABLE_ROCM=ON.";
              pendingSkips.push_back(
                  {{m.benchName, rocmLabel, cfgIdx, "rocm", {}, "", reason, {}, {}}, false});
            }
          } else if (wantedBackends.size() == 1) {
            std::string reason;
            if (!m.fixedSkipReason.empty()) {
              reason = m.fixedSkipReason;
            } else if (!m.vulkanSpv.empty()) {
              reason = m.configLabel + " has no ROCm kernel artifact (supported on Vulkan via " + m.vulkanSpv + "; pass '-k vulkan').";
            } else {
              reason = m.configLabel + " is a capability-probed benchmark with no compiled kernel artifact on this toolchain.";
            }
            pendingSkips.push_back(
                {{m.benchName, m.configLabel, cfgIdx, "rocm", {}, "", reason, {}, {}},
                 !m.fixedSkipReason.empty()});
          }
        }
      }
    }

    for (auto &ra : realTargetArtifacts) {
      artifacts.push_back(std::move(ra));
    }
    std::set<std::string> emittedSkipKeys;
    for (auto &ps : pendingSkips) {
      const std::string cfgKey =
          ps.artifact.benchmarkName + "|" + std::to_string(ps.artifact.configIndex) + "|" +
          ps.artifact.backend;
      // Never emit a skip when this (bench, config, backend) already has real artifact(s),
      // and deduplicate multi-stage Vulkan-only skips to one entry per config.
      if (realBackendKeys.count(cfgKey) || !emittedSkipKeys.insert(cfgKey).second) {
        continue;
      }
      // When inspecting a whole benchmark without -c and the requested backend already has
      // supported configs (e.g. --dump-isa indirect -k rocm or --dump-isa async -k rocm),
      // cleanly omit backend-absent skips for other-backend-only configs.
      if (opts.configs.empty() && !ps.isFixedSkip &&
          realBackendsPresent.count(ps.artifact.backend) > 0) {
        continue;
      }
      artifacts.push_back(std::move(ps.artifact));
    }

    if (!matchedBench) {
      if (wantedBackends.count("vulkan") && fs::is_directory(kernelDir / "vulkan")) {
        for (const auto &entry : fs::directory_iterator(kernelDir / "vulkan")) {
          if (entry.path().extension() == ".spv") {
            std::string fname = entry.path().filename().string();
            if (normalizeKey(fname).find(normTarget) != std::string::npos) {
              artifacts.push_back({fname, fname, 0, "vulkan", entry.path(), "", "", {}, {}});
            }
          }
        }
      }
      if (wantedBackends.count("rocm") && fs::is_directory(kernelDir / "rocm")) {
        for (const auto &entry : fs::directory_iterator(kernelDir / "rocm")) {
          if (entry.path().extension() == ".co") {
            std::string fname = entry.path().filename().string();
            if (normalizeKey(fname).find(normTarget) != std::string::npos) {
              artifacts.push_back({fname, fname, 0, "rocm", entry.path(), "", "", {}, {}});
            }
          }
        }
      }
    }
  }

  return artifacts;
}

static IsaDumperOptions applyUnmappedConfigFallback(const IsaDumperOptions &opts) {
  IsaDumperOptions fb = opts;
  if (!fb.targets.empty() && normalizeKey(fb.targets[0]) == "rayscheduling") {
    fb.configs = {15};
  } else {
    fb.configs.clear();
  }
  return fb;
}

static std::vector<KernelArtifact> resolveEffectiveArtifacts(const IsaDumperOptions &opts,
                                                             const fs::path &kernelDir,
                                                             const std::set<std::string> &wantedBackends) {
  std::vector<KernelArtifact> artifacts = resolveArtifactsForOptions(opts, kernelDir, wantedBackends);
  if (artifacts.empty() && !opts.configs.empty()) {
    IsaDumperOptions fb = applyUnmappedConfigFallback(opts);
    artifacts = resolveArtifactsForOptions(fb, kernelDir, wantedBackends);
  }
  return artifacts;
}

struct FileStamp {
  std::string path;
  uint64_t size = 0;
  int64_t mtime = 0;
};

static void appendUniqueFileStamp(const fs::path &p, std::vector<FileStamp> &out, std::set<std::string> &seen) {
  if (p.empty()) return;
  std::error_code ec;
  if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) return;
  std::string key = p.lexically_normal().string();
  if (!seen.insert(key).second) return;
  uint64_t sz = fs::file_size(p, ec);
  if (ec) return;
  auto tp = fs::last_write_time(p, ec);
  if (ec) return;
  int64_t mt = static_cast<int64_t>(tp.time_since_epoch().count());
  out.push_back({key, sz, mt});
}

static std::vector<FileStamp> collectDependencyStamps(const IsaDumperOptions &opts) {
  std::vector<FileStamp> stamps;
  std::set<std::string> seen;
  fs::path kernelDir = KernelPath::find();
  std::set<std::string> wantedBackends = resolveWantedBackends(opts);
  std::vector<KernelArtifact> artifacts = resolveEffectiveArtifacts(opts, kernelDir, wantedBackends);

  for (const auto &art : artifacts) {
    if (!art.skipReason.empty() || art.filePath.empty()) continue;
    appendUniqueFileStamp(art.filePath, stamps, seen);
    if (art.backend != "vulkan") continue;

    fs::path dir = art.filePath.parent_path();
    std::string fname = art.filePath.filename().string();
    if (fname.find(".frag.spv") != std::string::npos || fname.find(".vert.spv") != std::string::npos) {
      const bool isAsyncGfx = (fname.rfind("async_gfx.", 0) == 0);
      fs::path vertPath = dir / (isAsyncGfx ? "async_gfx.vert.spv" : "pixel_fill.vert.spv");
      fs::path fragPath = dir / (isAsyncGfx ? "async_gfx.frag.spv" : "pixel_fill.frag.spv");
      appendUniqueFileStamp(vertPath, stamps, seen);
      appendUniqueFileStamp(fragPath, stamps, seen);
    } else if (fname.find(".rgen.spv") != std::string::npos) {
      std::string prefix = fname;
      size_t dotPos = prefix.find('.');
      if (dotPos != std::string::npos) prefix = prefix.substr(0, dotPos);
      std::string basePrefix = prefix;
      if (prefix == "rt_scheduling_ser") basePrefix = "rt_scheduling";
      else if (prefix == "raydiv_pipeline_ser" || prefix.rfind("raydiv_pipeline_", 0) == 0) basePrefix = "raydiv_pipeline";
      else if (prefix == "rayomm_omm") basePrefix = "rayomm";

      appendUniqueFileStamp(dir / (prefix + ".rgen.spv"), stamps, seen);
      appendUniqueFileStamp(dir / (basePrefix + ".rgen.spv"), stamps, seen);
      appendUniqueFileStamp(dir / (basePrefix + ".rmiss.spv"), stamps, seen);
      appendUniqueFileStamp(dir / (prefix + ".rchit.spv"), stamps, seen);
      appendUniqueFileStamp(dir / (basePrefix + ".rchit.spv"), stamps, seen);
      appendUniqueFileStamp(dir / (basePrefix + "_a.rchit.spv"), stamps, seen);
      appendUniqueFileStamp(dir / (basePrefix + ".rahit.spv"), stamps, seen);
      appendUniqueFileStamp(dir / (basePrefix + ".rint.spv"), stamps, seen);
    }
  }
  return stamps;
}

static fs::path getIsaCacheRoot() {
  fs::path base;
#if defined(_WIN32)
  if (const char *localApp = std::getenv("LOCALAPPDATA")) {
    if (*localApp) base = fs::path(localApp) / "gpubench" / "cache";
  }
  if (base.empty()) {
    const char *userProfile = std::getenv("USERPROFILE");
    base = (userProfile && *userProfile ? fs::path(userProfile) : fs::path(".")) / ".cache" / "gpubench";
  }
#else
  if (const char *xdg = std::getenv("XDG_CACHE_HOME")) {
    if (*xdg) base = fs::path(xdg) / "gpubench";
  }
  if (base.empty()) {
    const char *home = std::getenv("HOME");
    base = (home && *home ? fs::path(home) : fs::path(".")) / ".cache" / "gpubench";
  }
#endif
  return base / "isa_v2";
}

static std::string canonicalBackendForCache(const IsaDumperOptions &opts) {
  if (opts.backends.empty()) return "vulkan";
  std::string low = toLowerCopy(opts.backends[0]);
  if (low.empty() || low == "auto") return "vulkan";
  return low;
}

static bool hasValidCacheContext(const IsaCacheContext &ctx) {
  return ctx.vendorId != 0 || !ctx.driverVersion.empty() || !ctx.driverInfo.empty();
}

static uint64_t fnv1a64(const std::string &s) {
  uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

static std::string sanitizeFilenamePart(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (std::isalnum(c)) out.push_back(static_cast<char>(c));
    else if (out.empty() || out.back() != '_') out.push_back('_');
  }
  while (!out.empty() && out.back() == '_') out.pop_back();
  if (out.empty()) out = "target";
  if (out.size() > 36) out.resize(36);
  return out;
}

static fs::path buildIsaCachePath(const IsaDumperOptions &opts, const IsaCacheContext &ctx) {
  const std::string target = opts.targets.empty() ? "" : opts.targets[0];
  const int cfg = opts.configs.empty() ? -1 : opts.configs[0];
  const std::string backend = canonicalBackendForCache(opts);

  std::ostringstream keySs;
  keySs << "v4|" << ctx.vendorId << "|" << ctx.deviceId << "|"
        << ctx.driverVersion << "|" << ctx.driverInfo << "|"
        << opts.asic << "|" << backend << "|" << opts.deviceIndex << "|"
        << target << "|" << cfg << "|" << (opts.fullIsa ? 1 : 0);

  std::ostringstream nameSs;
  nameSs << sanitizeFilenamePart(target)
         << "_c" << (cfg >= 0 ? std::to_string(cfg) : std::string("all"))
         << "_" << sanitizeFilenamePart(backend)
         << "_" << std::hex << std::setw(16) << std::setfill('0') << fnv1a64(keySs.str())
         << ".json";
  return getIsaCacheRoot() / nameSs.str();
}

static std::string readJsonStr(yyjson_val *obj, const char *key) {
  if (!obj) return "";
  yyjson_val *v = yyjson_obj_get(obj, key);
  const char *s = v ? yyjson_get_str(v) : nullptr;
  return s ? std::string(s) : std::string();
}

static uint64_t readJsonU64(yyjson_val *obj, const char *key) {
  if (!obj) return 0;
  yyjson_val *v = yyjson_obj_get(obj, key);
  return v ? yyjson_get_uint(v) : 0;
}

static int64_t readJsonI64(yyjson_val *obj, const char *key) {
  if (!obj) return 0;
  yyjson_val *v = yyjson_obj_get(obj, key);
  return v ? yyjson_get_sint(v) : 0;
}

static bool readJsonBool(yyjson_val *obj, const char *key) {
  if (!obj) return false;
  yyjson_val *v = yyjson_obj_get(obj, key);
  return v ? yyjson_get_bool(v) : false;
}

static bool runInspectionCore(const IsaDumperOptions &opts, bool emitStdout,
                              IsaInspectionResult *outResult) {
  fs::path kernelDir = KernelPath::find();
  std::set<std::string> wantedBackends = resolveWantedBackends(opts);
  std::vector<KernelArtifact> artifacts = resolveArtifactsForOptions(opts, kernelDir, wantedBackends);

  if (artifacts.empty()) {
    std::string msg = "No compiled shaders or kernels matched for ISA inspection (searched in " +
                      kernelDir.string() + ").";
    if (outResult) {
      outResult->ok = false;
      outResult->errorSummary = msg;
    }
    if (emitStdout) {
      std::cerr << "Error: No compiled shaders or kernels matched for --dump-isa (searched in "
                << kernelDir.string() << ").\n";
    }
    return false;
  }

  std::string resolvedAsic = opts.asic;

#ifdef HAVE_VULKAN
  std::unique_ptr<NativeVulkanIsaInspector> vkInspector;
  bool needVulkan = false;
  bool needRocm = false;
  for (const auto &a : artifacts) {
    if (!a.skipReason.empty())
      continue;
    if (a.backend == "vulkan") {
      needVulkan = true;
    } else if (a.backend == "rocm") {
      needRocm = true;
    }
  }
  if (needVulkan || (needRocm && resolvedAsic.empty())) {
    vkInspector = std::make_unique<NativeVulkanIsaInspector>();
    if (vkInspector->init(opts.deviceIndex, !emitStdout)) {
      if (resolvedAsic.empty()) {
        const auto &prof = DeviceDatabase::lookup(vkInspector->vendorId, vkInspector->deviceId,
                                                  vkInspector->deviceName);
        resolvedAsic = extractGfxToken(prof.archName);
      }
      if (outResult) {
        outResult->deviceName = vkInspector->deviceName;
        outResult->driverInfo = vkInspector->driverInfo;
      }
      if (emitStdout && needVulkan) {
        std::cout << "╭─ GPUBench Native Shader ISA Inspector (VK_KHR_pipeline_executable_properties) ─╮\n";
        std::cout << "│ Target Device : " << std::left << std::setw(63) << vkInspector->deviceName << "│\n";
        std::cout << "│ Active Driver : " << std::left << std::setw(63) << vkInspector->driverInfo << "│\n";
        std::cout << "╰────────────────────────────────────────────────────────────────────────────────╯\n\n";
      }
    } else if (needVulkan) {
      if (outResult) {
        outResult->ok = false;
        outResult->errorSummary = !vkInspector->lastError.empty()
                                      ? vkInspector->lastError
                                      : "Vulkan ISA inspector initialization failed.";
      }
      return false;
    }
  }
#endif
  if (resolvedAsic.empty()) {
    resolvedAsic = "gfx1201";
  }
  if (outResult) {
    outResult->resolvedAsic = resolvedAsic;
  }

  std::set<std::string> visitedFiles;
  const bool explicitConfigs = !opts.configs.empty();
  for (const auto &art : artifacts) {
    const std::string visitKey = buildArtifactVisitKey(art, explicitConfigs);
    if (!visitedFiles.insert(visitKey).second) {
      continue;
    }

    if (!art.skipReason.empty()) {
      if (outResult) {
        IsaArtifactReport rep;
        rep.benchName = art.benchmarkName;
        rep.configLabel = art.configName;
        rep.configIndex = art.configIndex;
        rep.backend = art.backend;
        rep.artifactFile = art.filePath.filename().string();
        rep.skipReason = art.skipReason;
        outResult->artifacts.push_back(std::move(rep));
      } else if (emitStdout) {
        std::cout << "━━━ [" << art.backend << "] " << art.configName << " ━━━\n"
                  << "  [Skipped] " << art.skipReason << "\n\n";
      }
      continue;
    }

    if (art.backend == "vulkan") {
#ifdef HAVE_VULKAN
      if (vkInspector) {
        if (outResult) {
          IsaArtifactReport rep;
          vkInspector->inspectSpv(art, opts.fullIsa, &rep);
          outResult->artifacts.push_back(std::move(rep));
        } else {
          vkInspector->inspectSpv(art, opts.fullIsa, nullptr);
        }
      }
#endif
    } else if (art.backend == "rocm") {
      if (outResult) {
        IsaArtifactReport rep;
        inspectRocmCodeObject(art, resolvedAsic, opts.fullIsa, !emitStdout, &rep);
        outResult->artifacts.push_back(std::move(rep));
      } else {
        inspectRocmCodeObject(art, resolvedAsic, opts.fullIsa, !emitStdout, nullptr);
      }
    }
  }

  if (outResult) {
    outResult->ok = true;
  }
  return true;
}

} // namespace

int IsaDumper::dump(const IsaDumperOptions &opts) {
  return runInspectionCore(opts, true, nullptr) ? EXIT_SUCCESS : EXIT_FAILURE;
}

IsaInspectionResult IsaDumper::inspect(const IsaDumperOptions &opts) {
  IsaInspectionResult res;
  runInspectionCore(opts, false, &res);
  return res;
}

bool IsaDumper::tryLoadCached(const IsaDumperOptions &opts,
                              const IsaCacheContext &ctx,
                              IsaInspectionResult &outResult) {
  if (opts.targets.size() != 1 || !hasValidCacheContext(ctx)) {
    return false;
  }

  const fs::path cachePath = buildIsaCachePath(opts, ctx);
  std::error_code ec;
  if (!fs::exists(cachePath, ec) || !fs::is_regular_file(cachePath, ec)) {
    return false;
  }

  yyjson_doc *rawDoc = yyjson_read_file(cachePath.string().c_str(), 0, nullptr, nullptr);
  if (!rawDoc) {
    return false;
  }
  struct DocGuard {
    yyjson_doc *d = nullptr;
    ~DocGuard() { if (d) yyjson_doc_free(d); }
  } guard{rawDoc};

  yyjson_val *root = yyjson_doc_get_root(rawDoc);
  if (!root || !yyjson_is_obj(root)) {
    return false;
  }

  if (readJsonI64(root, "schemaVersion") != 2) return false;
  if (readJsonU64(root, "vendorId") != ctx.vendorId) return false;
  if (readJsonU64(root, "deviceId") != ctx.deviceId) return false;
  if (readJsonStr(root, "driverVersion") != ctx.driverVersion) return false;
  if (readJsonStr(root, "driverInfo") != ctx.driverInfo) return false;
  if (readJsonStr(root, "backend") != canonicalBackendForCache(opts)) return false;
  if (readJsonU64(root, "deviceIndex") != opts.deviceIndex) return false;
  if (readJsonStr(root, "target") != opts.targets[0]) return false;
  const int64_t wantCfg = opts.configs.empty() ? -1 : static_cast<int64_t>(opts.configs[0]);
  if (readJsonI64(root, "configIndex") != wantCfg) return false;
  if (readJsonBool(root, "fullIsa") != opts.fullIsa) return false;

  // Strictly verify every backing shader/kernel file size and mtime against disk.
  yyjson_val *depsArr = yyjson_obj_get(root, "deps");
  if (!depsArr || !yyjson_is_arr(depsArr)) {
    return false;
  }
  const std::vector<FileStamp> curDeps = collectDependencyStamps(opts);
  if (yyjson_arr_size(depsArr) != curDeps.size()) {
    return false;
  }
  std::map<std::string, std::pair<uint64_t, int64_t>> curMap;
  for (const auto &d : curDeps) {
    curMap[d.path] = {d.size, d.mtime};
  }
  size_t depIdx = 0;
  size_t depMax = 0;
  yyjson_val *depVal = nullptr;
  yyjson_arr_foreach(depsArr, depIdx, depMax, depVal) {
    if (!yyjson_is_obj(depVal)) return false;
    std::string p = readJsonStr(depVal, "path");
    auto it = curMap.find(p);
    if (it == curMap.end()) return false;
    if (readJsonU64(depVal, "size") != it->second.first) return false;
    if (readJsonI64(depVal, "mtime") != it->second.second) return false;
  }

  yyjson_val *resObj = yyjson_obj_get(root, "result");
  if (!resObj || !yyjson_is_obj(resObj)) {
    return false;
  }

  IsaInspectionResult loaded;
  loaded.ok = readJsonBool(resObj, "ok");
  loaded.deviceName = readJsonStr(resObj, "deviceName");
  loaded.driverInfo = readJsonStr(resObj, "driverInfo");
  loaded.resolvedAsic = readJsonStr(resObj, "resolvedAsic");
  loaded.errorSummary = readJsonStr(resObj, "errorSummary");

  yyjson_val *artsArr = yyjson_obj_get(resObj, "artifacts");
  if (!artsArr || !yyjson_is_arr(artsArr)) {
    return false;
  }
  size_t aIdx = 0, aMax = 0;
  yyjson_val *artVal = nullptr;
  yyjson_arr_foreach(artsArr, aIdx, aMax, artVal) {
    if (!yyjson_is_obj(artVal)) return false;
    IsaArtifactReport rep;
    rep.benchName = readJsonStr(artVal, "benchName");
    rep.configLabel = readJsonStr(artVal, "configLabel");
    rep.configIndex = static_cast<int>(readJsonI64(artVal, "configIndex"));
    rep.backend = readJsonStr(artVal, "backend");
    rep.artifactFile = readJsonStr(artVal, "artifactFile");
    rep.skipReason = readJsonStr(artVal, "skipReason");
    rep.errorMessage = readJsonStr(artVal, "errorMessage");

    yyjson_val *execArr = yyjson_obj_get(artVal, "executables");
    if (execArr && yyjson_is_arr(execArr)) {
      size_t eIdx = 0, eMax = 0;
      yyjson_val *execVal = nullptr;
      yyjson_arr_foreach(execArr, eIdx, eMax, execVal) {
        if (!yyjson_is_obj(execVal)) return false;
        IsaExecutableReport ex;
        ex.name = readJsonStr(execVal, "name");
        ex.description = readJsonStr(execVal, "description");
        ex.subgroupSize = static_cast<uint32_t>(readJsonU64(execVal, "subgroupSize"));
        ex.vgprCount = static_cast<uint32_t>(readJsonU64(execVal, "vgprCount"));
        ex.sgprCount = static_cast<uint32_t>(readJsonU64(execVal, "sgprCount"));
        ex.ldsBytes = static_cast<uint32_t>(readJsonU64(execVal, "ldsBytes"));
        ex.scratchBytes = static_cast<uint32_t>(readJsonU64(execVal, "scratchBytes"));
        ex.hotLoopStartLine = static_cast<int>(readJsonI64(execVal, "hotLoopStartLine"));
        ex.hotLoopEndLine = static_cast<int>(readJsonI64(execVal, "hotLoopEndLine"));
        ex.unavailableReason = readJsonStr(execVal, "unavailableReason");

        if (yyjson_val *mixObj = yyjson_obj_get(execVal, "mix"); mixObj && yyjson_is_obj(mixObj)) {
          ex.mix.wmma = static_cast<int>(readJsonI64(mixObj, "wmma"));
          ex.mix.valu = static_cast<int>(readJsonI64(mixObj, "valu"));
          ex.mix.vmem = static_cast<int>(readJsonI64(mixObj, "vmem"));
          ex.mix.lds = static_cast<int>(readJsonI64(mixObj, "lds"));
          ex.mix.bvh = static_cast<int>(readJsonI64(mixObj, "bvh"));
          ex.mix.salu = static_cast<int>(readJsonI64(mixObj, "salu"));
          ex.mix.other = static_cast<int>(readJsonI64(mixObj, "other"));
          ex.mix.total = static_cast<int>(readJsonI64(mixObj, "total"));
        }

        if (yyjson_val *statsArr = yyjson_obj_get(execVal, "stats"); statsArr && yyjson_is_arr(statsArr)) {
          size_t sIdx = 0, sMax = 0;
          yyjson_val *stVal = nullptr;
          yyjson_arr_foreach(statsArr, sIdx, sMax, stVal) {
            if (yyjson_is_obj(stVal)) {
              ex.stats.push_back({readJsonStr(stVal, "name"), readJsonStr(stVal, "value")});
            }
          }
        }
        if (yyjson_val *opArr = yyjson_obj_get(execVal, "topOpcodes"); opArr && yyjson_is_arr(opArr)) {
          size_t oIdx = 0, oMax = 0;
          yyjson_val *opVal = nullptr;
          yyjson_arr_foreach(opArr, oIdx, oMax, opVal) {
            if (yyjson_is_obj(opVal)) {
              ex.topOpcodes.emplace_back(readJsonStr(opVal, "op"),
                                         static_cast<int>(readJsonI64(opVal, "count")));
            }
          }
        }
        if (yyjson_val *hlArr = yyjson_obj_get(execVal, "hotLoopLines"); hlArr && yyjson_is_arr(hlArr)) {
          size_t lIdx = 0, lMax = 0;
          yyjson_val *lv = nullptr;
          yyjson_arr_foreach(hlArr, lIdx, lMax, lv) {
            const char *s = yyjson_get_str(lv);
            ex.hotLoopLines.emplace_back(s ? s : "");
          }
        }
        if (yyjson_val *flArr = yyjson_obj_get(execVal, "fullIsaLines"); flArr && yyjson_is_arr(flArr)) {
          size_t lIdx = 0, lMax = 0;
          yyjson_val *lv = nullptr;
          yyjson_arr_foreach(flArr, lIdx, lMax, lv) {
            const char *s = yyjson_get_str(lv);
            ex.fullIsaLines.emplace_back(s ? s : "");
          }
        }
        rep.executables.push_back(std::move(ex));
      }
    }
    loaded.artifacts.push_back(std::move(rep));
  }

  if (!loaded.ok || loaded.artifacts.empty()) {
    return false;
  }
  outResult = std::move(loaded);
  return true;
}

bool IsaDumper::saveCached(const IsaDumperOptions &opts,
                           const IsaCacheContext &ctx,
                           const IsaInspectionResult &result) {
  if (opts.targets.size() != 1 || !hasValidCacheContext(ctx) || !result.ok || result.artifacts.empty()) {
    return false;
  }
  for (const auto &art : result.artifacts) {
    if (!art.errorMessage.empty()) {
      return false;
    }
  }

  const fs::path cachePath = buildIsaCachePath(opts, ctx);
  std::error_code ec;
  fs::create_directories(cachePath.parent_path(), ec);
  if (ec) {
    return false;
  }

  yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
  if (!doc) return false;
  struct MutGuard {
    yyjson_mut_doc *d = nullptr;
    ~MutGuard() { if (d) yyjson_mut_doc_free(d); }
  } guard{doc};

  yyjson_mut_val *root = yyjson_mut_obj(doc);
  yyjson_mut_doc_set_root(doc, root);

  yyjson_mut_obj_add_int(doc, root, "schemaVersion", 2);
  yyjson_mut_obj_add_uint(doc, root, "vendorId", ctx.vendorId);
  yyjson_mut_obj_add_uint(doc, root, "deviceId", ctx.deviceId);
  yyjson_mut_obj_add_strcpy(doc, root, "driverVersion", ctx.driverVersion.c_str());
  yyjson_mut_obj_add_strcpy(doc, root, "driverInfo", ctx.driverInfo.c_str());
  const std::string backendStr = canonicalBackendForCache(opts);
  yyjson_mut_obj_add_strcpy(doc, root, "backend", backendStr.c_str());
  yyjson_mut_obj_add_uint(doc, root, "deviceIndex", opts.deviceIndex);
  yyjson_mut_obj_add_strcpy(doc, root, "target", opts.targets[0].c_str());
  yyjson_mut_obj_add_int(doc, root, "configIndex", opts.configs.empty() ? -1 : opts.configs[0]);
  yyjson_mut_obj_add_bool(doc, root, "fullIsa", opts.fullIsa);

  yyjson_mut_val *depsArr = yyjson_mut_obj_add_arr(doc, root, "deps");
  const std::vector<FileStamp> deps = collectDependencyStamps(opts);
  for (const auto &d : deps) {
    yyjson_mut_val *doObj = yyjson_mut_arr_add_obj(doc, depsArr);
    yyjson_mut_obj_add_strcpy(doc, doObj, "path", d.path.c_str());
    yyjson_mut_obj_add_uint(doc, doObj, "size", d.size);
    yyjson_mut_obj_add_int(doc, doObj, "mtime", d.mtime);
  }

  yyjson_mut_val *resObj = yyjson_mut_obj_add_obj(doc, root, "result");
  yyjson_mut_obj_add_bool(doc, resObj, "ok", result.ok);
  yyjson_mut_obj_add_strcpy(doc, resObj, "deviceName", result.deviceName.c_str());
  yyjson_mut_obj_add_strcpy(doc, resObj, "driverInfo", result.driverInfo.c_str());
  yyjson_mut_obj_add_strcpy(doc, resObj, "resolvedAsic", result.resolvedAsic.c_str());
  yyjson_mut_obj_add_strcpy(doc, resObj, "errorSummary", result.errorSummary.c_str());

  yyjson_mut_val *artsArr = yyjson_mut_obj_add_arr(doc, resObj, "artifacts");
  for (const auto &art : result.artifacts) {
    yyjson_mut_val *aObj = yyjson_mut_arr_add_obj(doc, artsArr);
    yyjson_mut_obj_add_strcpy(doc, aObj, "benchName", art.benchName.c_str());
    yyjson_mut_obj_add_strcpy(doc, aObj, "configLabel", art.configLabel.c_str());
    yyjson_mut_obj_add_int(doc, aObj, "configIndex", art.configIndex);
    yyjson_mut_obj_add_strcpy(doc, aObj, "backend", art.backend.c_str());
    yyjson_mut_obj_add_strcpy(doc, aObj, "artifactFile", art.artifactFile.c_str());
    yyjson_mut_obj_add_strcpy(doc, aObj, "skipReason", art.skipReason.c_str());
    yyjson_mut_obj_add_strcpy(doc, aObj, "errorMessage", art.errorMessage.c_str());

    yyjson_mut_val *execArr = yyjson_mut_obj_add_arr(doc, aObj, "executables");
    for (const auto &ex : art.executables) {
      yyjson_mut_val *eObj = yyjson_mut_arr_add_obj(doc, execArr);
      yyjson_mut_obj_add_strcpy(doc, eObj, "name", ex.name.c_str());
      yyjson_mut_obj_add_strcpy(doc, eObj, "description", ex.description.c_str());
      yyjson_mut_obj_add_uint(doc, eObj, "subgroupSize", ex.subgroupSize);
      yyjson_mut_obj_add_uint(doc, eObj, "vgprCount", ex.vgprCount);
      yyjson_mut_obj_add_uint(doc, eObj, "sgprCount", ex.sgprCount);
      yyjson_mut_obj_add_uint(doc, eObj, "ldsBytes", ex.ldsBytes);
      yyjson_mut_obj_add_uint(doc, eObj, "scratchBytes", ex.scratchBytes);
      yyjson_mut_obj_add_int(doc, eObj, "hotLoopStartLine", ex.hotLoopStartLine);
      yyjson_mut_obj_add_int(doc, eObj, "hotLoopEndLine", ex.hotLoopEndLine);
      yyjson_mut_obj_add_strcpy(doc, eObj, "unavailableReason", ex.unavailableReason.c_str());

      yyjson_mut_val *mixObj = yyjson_mut_obj_add_obj(doc, eObj, "mix");
      yyjson_mut_obj_add_int(doc, mixObj, "wmma", ex.mix.wmma);
      yyjson_mut_obj_add_int(doc, mixObj, "valu", ex.mix.valu);
      yyjson_mut_obj_add_int(doc, mixObj, "vmem", ex.mix.vmem);
      yyjson_mut_obj_add_int(doc, mixObj, "lds", ex.mix.lds);
      yyjson_mut_obj_add_int(doc, mixObj, "bvh", ex.mix.bvh);
      yyjson_mut_obj_add_int(doc, mixObj, "salu", ex.mix.salu);
      yyjson_mut_obj_add_int(doc, mixObj, "other", ex.mix.other);
      yyjson_mut_obj_add_int(doc, mixObj, "total", ex.mix.total);

      yyjson_mut_val *statsArr = yyjson_mut_obj_add_arr(doc, eObj, "stats");
      for (const auto &st : ex.stats) {
        yyjson_mut_val *sObj = yyjson_mut_arr_add_obj(doc, statsArr);
        yyjson_mut_obj_add_strcpy(doc, sObj, "name", st.name.c_str());
        yyjson_mut_obj_add_strcpy(doc, sObj, "value", st.value.c_str());
      }

      yyjson_mut_val *opArr = yyjson_mut_obj_add_arr(doc, eObj, "topOpcodes");
      for (const auto &op : ex.topOpcodes) {
        yyjson_mut_val *oObj = yyjson_mut_arr_add_obj(doc, opArr);
        yyjson_mut_obj_add_strcpy(doc, oObj, "op", op.first.c_str());
        yyjson_mut_obj_add_int(doc, oObj, "count", op.second);
      }

      yyjson_mut_val *hlArr = yyjson_mut_obj_add_arr(doc, eObj, "hotLoopLines");
      for (const auto &line : ex.hotLoopLines) {
        yyjson_mut_arr_add_strcpy(doc, hlArr, line.c_str());
      }

      yyjson_mut_val *flArr = yyjson_mut_obj_add_arr(doc, eObj, "fullIsaLines");
      for (const auto &line : ex.fullIsaLines) {
        yyjson_mut_arr_add_strcpy(doc, flArr, line.c_str());
      }
    }
  }

  const uint64_t tmpSeq = g_isaTmpCounter.fetch_add(1, std::memory_order_relaxed);
  const size_t tidHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
#ifdef _WIN32
  const unsigned long pid = static_cast<unsigned long>(GetCurrentProcessId());
#else
  const unsigned long pid = static_cast<unsigned long>(getpid());
#endif
  fs::path tmpPath = cachePath;
  tmpPath += ".tmp." + std::to_string(pid) + "." + std::to_string(tidHash) + "." + std::to_string(tmpSeq);
  if (!yyjson_mut_write_file(tmpPath.string().c_str(), doc, 0, nullptr, nullptr)) {
    return false;
  }
  fs::rename(tmpPath, cachePath, ec);
  if (ec) {
    fs::remove(tmpPath, ec);
    return false;
  }
  return true;
}

bool IsaDumper::isDeferredWarmupTarget(const std::string &target, int configIndex) {
  const std::string normTarget = normalizeKey(target);
  if (normTarget.empty()) {
    return false;
  }
  const auto mappings = getBenchmarkMappings();
  for (const auto &m : mappings) {
    if (!matchesBenchmarkMapping(m, normTarget)) {
      continue;
    }
    if (std::find(m.configIndices.begin(), m.configIndices.end(), configIndex) !=
        m.configIndices.end()) {
      return m.vulkanSpv == "rt_scheduling_traditional_megakernel.comp.spv";
    }
  }
  return false;
}

void IsaDumper::warmCacheBatch(
    const std::vector<IsaDumperOptions> &items,
    const IsaCacheContext &ctx,
    const std::atomic<bool> *cancelFlag,
    const std::function<void(const IsaDumperOptions &, IsaInspectionResult &&)> &onItemReady) {
  if (items.empty()) return;

  const fs::path kernelDir = KernelPath::find();
#ifdef HAVE_VULKAN
  std::unique_ptr<NativeVulkanIsaInspector> vkInspector;
  uint32_t vkInitDevice = 0xFFFFFFFFu;
  std::string vkResolvedAsic;
#endif

  for (const auto &opts : items) {
    if (cancelFlag && cancelFlag->load(std::memory_order_acquire)) {
      break;
    }

    IsaInspectionResult res;
    if (tryLoadCached(opts, ctx, res)) {
      if (onItemReady) onItemReady(opts, std::move(res));
      continue;
    }

    const std::set<std::string> wantedBackends = resolveWantedBackends(opts);
    const std::vector<KernelArtifact> artifacts = resolveEffectiveArtifacts(opts, kernelDir, wantedBackends);
    if (artifacts.empty()) {
      res.ok = false;
      res.errorSummary = "No compiled shaders or kernels matched for ISA inspection (searched in " +
                         kernelDir.string() + ").";
      if (onItemReady) onItemReady(opts, std::move(res));
      continue;
    }

    std::string resolvedAsic = opts.asic;
#ifdef HAVE_VULKAN
    bool needVulkan = false;
    bool needRocm = false;
    for (const auto &a : artifacts) {
      if (!a.skipReason.empty()) continue;
      if (a.backend == "vulkan") needVulkan = true;
      else if (a.backend == "rocm") needRocm = true;
    }

    if (needVulkan || (needRocm && resolvedAsic.empty())) {
      if (!vkInspector || vkInitDevice != opts.deviceIndex) {
        vkInspector = std::make_unique<NativeVulkanIsaInspector>();
        if (vkInspector->init(opts.deviceIndex, true)) {
          vkInitDevice = opts.deviceIndex;
          const auto &prof = DeviceDatabase::lookup(vkInspector->vendorId, vkInspector->deviceId,
                                                    vkInspector->deviceName);
          vkResolvedAsic = extractGfxToken(prof.archName);
        } else {
          const std::string err = !vkInspector->lastError.empty()
                                      ? vkInspector->lastError
                                      : "Vulkan ISA inspector initialization failed.";
          vkInspector.reset();
          vkInitDevice = 0xFFFFFFFFu;
          if (needVulkan) {
            res.ok = false;
            res.errorSummary = err;
            if (onItemReady) onItemReady(opts, std::move(res));
            continue;
          }
        }
      }
      if (vkInspector) {
        if (resolvedAsic.empty()) resolvedAsic = vkResolvedAsic;
        res.deviceName = vkInspector->deviceName;
        res.driverInfo = vkInspector->driverInfo;
      }
    }
#endif
    if (resolvedAsic.empty()) resolvedAsic = "gfx1201";
    res.resolvedAsic = resolvedAsic;

    std::set<std::string> visitedFiles;
    const bool explicitConfigs = !opts.configs.empty();
    for (const auto &art : artifacts) {
      if (cancelFlag && cancelFlag->load(std::memory_order_acquire)) {
        break;
      }
      const std::string visitKey = buildArtifactVisitKey(art, explicitConfigs);
      if (!visitedFiles.insert(visitKey).second) {
        continue;
      }
      if (!art.skipReason.empty()) {
        IsaArtifactReport rep;
        rep.benchName = art.benchmarkName;
        rep.configLabel = art.configName;
        rep.configIndex = art.configIndex;
        rep.backend = art.backend;
        rep.artifactFile = art.filePath.filename().string();
        rep.skipReason = art.skipReason;
        res.artifacts.push_back(std::move(rep));
        continue;
      }
      if (art.backend == "vulkan") {
#ifdef HAVE_VULKAN
        if (vkInspector) {
          IsaArtifactReport rep;
          vkInspector->inspectSpv(art, opts.fullIsa, &rep);
          res.artifacts.push_back(std::move(rep));
        }
#endif
      } else if (art.backend == "rocm") {
        IsaArtifactReport rep;
        inspectRocmCodeObject(art, resolvedAsic, opts.fullIsa, true, &rep);
        res.artifacts.push_back(std::move(rep));
      }
    }

    if (cancelFlag && cancelFlag->load(std::memory_order_acquire)) {
      break;
    }

    res.ok = true;
    saveCached(opts, ctx, res);
    if (onItemReady) {
      onItemReady(opts, std::move(res));
    }
  }
}
