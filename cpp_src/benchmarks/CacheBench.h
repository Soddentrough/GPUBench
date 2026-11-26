#pragma once

#include "IBenchmark.h"
#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

class CacheBench : public IBenchmark {
public:
  CacheBench(std::string name, std::string metric, uint64_t bufferSize,
             std::string kernelFile, std::vector<uint32_t> initData = {},
             std::vector<std::string> aliases = {}, int targetCacheLevel = -1);
  ~CacheBench();

  const char *GetName() const override;
  std::vector<std::string> GetAliases() const override;
  const char *GetMetric() const override;
  bool IsSupported(const DeviceInfo &info,
                   IComputeContext *context = nullptr) const override;
  std::string GetSupportNote(const DeviceInfo &info,
                             IComputeContext *context = nullptr) const override {
    (void)context;
    if (targetCacheLevel == 0) {
      return (metric == "GB/s")
                 ? "Measures nearest vector cache bandwidth: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1; Intel L1"
                 : "Measures nearest vector cache latency: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1 / Texture; Intel L1";
    }
    if (targetCacheLevel == 3 && info.l3CacheSize == 0) {
      return "Device does not have an L3 / Infinity Cache (l3CacheSize = 0)";
    }
    if (targetCacheLevel == 2 && info.l2CacheSize == 0) {
      return "Device does not have an L2 cache reported";
    }
    return "";
  }
  std::string GetSupportNote() const override {
    if (targetCacheLevel == 0) {
      return (metric == "GB/s")
                 ? "Measures nearest vector cache bandwidth: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1; Intel L1"
                 : "Measures nearest vector cache latency: AMD RDNA WGP Vector L0 (TCP, 16-32 KB); NVIDIA L1 / Texture; Intel L1";
    }
    if (targetCacheLevel == 3) {
      return "Device does not have an L3 / Infinity Cache (l3CacheSize = 0)";
    }
    return "";
  }
  SupportLimitation GetSupportLimitation() const override {
    return SupportLimitation::kHardware;
  }
  SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                         IComputeContext *context = nullptr) const override {
    (void)info;
    (void)context;
    return SupportLimitation::kHardware;
  }
  void Setup(IComputeContext &context, const std::string &kernel_dir) override;
  void Run(uint32_t config_idx = 0) override;
  void Teardown() override;
  BenchmarkResult GetResult(uint32_t config_idx = 0) const override;
  bool ValidateResults(uint32_t config_idx = 0) const override;
  const char *GetComponent(uint32_t config_idx = 0) const override;
  const char *GetSubCategory(uint32_t config_idx = 0) const override;
  using IBenchmark::GetDescription;
  const char *GetDescription() const override {
    if (metric == "GB/s") {
      switch (targetCacheLevel) {
      case 0:
        return "Multi-CU streaming GPU L0 vector cache bandwidth (16 KB working set targeting per-WGP/CU L0 TCP cache).";
      case 1:
        return "Multi-CU streaming GPU L1 cache bandwidth (128 KB working set targeting shader-array L1 cache).";
      case 2:
        return "Multi-CU streaming GPU L2 shared cache bandwidth (dynamically sized to half of reported L2 capacity).";
      case 3:
        return "Multi-CU streaming GPU L3 / Infinity Cache (MALL) bandwidth (dynamically sized to half of reported L3 capacity).";
      default:
        return "Multi-CU streaming GPU cache bandwidth targeting the selected cache hierarchy tier.";
      }
    }
    switch (targetCacheLevel) {
    case 0:
      return "Dependent pointer-chasing load latency isolating nearest GPU L0 vector cache (16 KB working set, 128 B cache-line stride).";
    case 1:
      return "Dependent pointer-chasing load latency targeting GPU L1 cache tier (128 B cache-line stride).";
    case 2:
      return "Dependent pointer-chasing load latency targeting GPU L2 shared cache tier (128 B cache-line stride).";
    case 3:
      return "Dependent pointer-chasing load latency targeting GPU L3 / Infinity Cache (MALL) tier (128 B cache-line stride).";
    default:
      return "Dependent pointer-chasing load latency isolating GPU cache round-trip time.";
    }
  }
  using IBenchmark::GetExplanation;
  BenchmarkExplanation GetExplanation(uint32_t config_idx = 0) const override {
    (void)config_idx;
    if (metric == "GB/s") {
      switch (targetCacheLevel) {
      case 0:
        return {
          "Peak bandwidth of each Compute Unit's nearest L0 vector data cache (16 KB working set).",
          "Feeds fast per-CU texture fetches, uniform/material tables, and inner shader loops without stalling on lower caches.",
          "Multi-CU compute shader streaming coalesced vec4 loads from a 16 KB buffer resident in L0 cache."
        };
      case 1:
        return {
          "Peak bandwidth of the Shader-Array L1 cache tier (128 KB working set) shared across nearby Compute Units.",
          "Serves spatially coherent screen tiles, neighboring pixel textures, and local BVH nodes across CUs in a shader engine.",
          "Multi-CU compute shader streaming coalesced vec4 loads from a 128 KB buffer sized for L1 cache."
        };
      case 2:
        return {
          "Peak bandwidth of the GPU-wide shared L2 cache (sized to half of reported L2 capacity).",
          "Critical hub for cross-CU sharing: shadow maps, G-buffer tiles, ray-tracing BVH trees, and atomic/indirect buffers.",
          "Multi-CU compute shader streaming coalesced vec4 loads from a working set sized to half the GPU's L2 cache."
        };
      case 3:
        return {
          "Peak bandwidth of the GPU L3 / Infinity Cache (MALL) tier (sized to half of reported L3 capacity).",
          "Keeps active render targets, depth buffers, and BVH working sets out of slow external GDDR VRAM—boosting effective 1440p/4K memory speed.",
          "Multi-CU compute shader streaming coalesced vec4 loads from a working set sized to half the GPU's L3 cache."
        };
      default:
        return {
          GetDescription(),
          "Measures sustained streaming throughput of the targeted GPU cache tier.",
          "Multi-CU compute shader streaming coalesced vec4 loads."
        };
      }
    }
    return {
      "True round-trip latency (in nanoseconds) of a dependent memory load hitting the GPU's nearest L0 vector cache.",
      "Determines how fast a single wave recovers from irregular/uncoalesced reads such as indirect texture lookups, pointer tables, and BVH node traversal.",
      "Single-thread dependent pointer-chasing chain over a 16 KB buffer at 128-byte cache-line stride."
    };
  }
  double GetOperationalIntensity(uint32_t config_idx = 0) const override {
    (void)config_idx;
    // cache_bw_robust.comp executes 8 vec4 FP32 additions (32 FLOPs) per 8 vec4 loads (128 bytes) = 0.25 OP/Byte
    return (metric == "GB/s") ? 0.25 : 0.0;
  }
  int GetSortWeight() const override;

private:
  template <typename Tag> friend struct ::gpubench::test::TestBody;
  std::string name;
  std::vector<std::string> aliases;
  std::string metric;
  uint64_t bufferSize;
  std::string kernelFile;
  IComputeContext *context = nullptr;
  ComputeKernel kernel = nullptr;
  ComputeBuffer buffer = nullptr;
  ComputeBuffer outBuffer = nullptr;
  ComputeBuffer pcBuffer = nullptr;
  std::vector<uint32_t> initData;
  void *hostMem = nullptr;
  int targetCacheLevel = -1;
  uint32_t numWorkgroups = 1;
  bool debug = false;

public:
  void setDebug(bool d) { debug = d; }
};
