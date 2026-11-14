#pragma once

#include "core/IComputeContext.h"
#include <cstdint>
#include <string>
#include <vector>

namespace gpubench {
namespace test {
template <typename Tag> struct TestBody;
} // namespace test
} // namespace gpubench

struct BenchmarkResult {
  uint64_t operations;
  double elapsedTime; // in milliseconds
};

class IBenchmark {
public:
  virtual ~IBenchmark() = default;
  virtual const char *GetName() const = 0;
  virtual std::vector<std::string> GetAliases() const { return {}; }
  virtual const char *GetMetric() const { return "TFLOPS"; }
  virtual const char *GetMetric(uint32_t /*config_idx*/) const { return GetMetric(); }
  virtual bool IsSupported(const DeviceInfo &info,
                           IComputeContext *context = nullptr) const = 0;
  virtual void Setup(IComputeContext &context,
                     const std::string &kernel_dir) = 0;
  virtual void PrepareConfig(uint32_t /*config_idx*/) {}
  virtual void ReleaseConfigResources(uint32_t /*config_idx*/) {}
  virtual void SetResolution(uint32_t /*w*/, uint32_t /*h*/) {}
  virtual void Run(uint32_t config_idx = 0) = 0;
  virtual void Teardown() = 0;
  virtual bool RequiresInterIterationBarrier(uint32_t /*config_idx*/ = 0) const {
    return false;
  }
  virtual BenchmarkResult GetResult(uint32_t config_idx = 0) const = 0;
  virtual void RecordRunResult(uint32_t /*config_idx*/, uint64_t /*total_invocations*/, double /*total_time_ms*/) {}
  virtual double FilterDuration(uint32_t /*config_idx*/, uint64_t /*total_invocations*/, double duration_ms) { return duration_ms; }
  virtual bool UsesCustomTiming(uint32_t /*config_idx*/ = 0) const { return false; }
  virtual double GetLastRunDurationMs(uint32_t /*config_idx*/ = 0) const { return 0.0; }
  virtual void RebuildAccelerationStructures() {}

  // --- Adaptive work sizing -------------------------------------------------
  // Benchmarks are measured over a fixed time window, not a fixed amount of
  // work. A benchmark whose single Run() has a grid that can be shrunk
  // linearly (per-workgroup work is constant) opts in by returning true here.
  // The runner then probes with a tiny scale and grows it until one Run()
  // takes a target duration, so slow GPUs never submit a dispatch long enough
  // to trigger a driver watchdog (TDR) and fast GPUs are fed via batching.
  // GetResult() must report the operations of the *currently scaled* Run().
  virtual bool SupportsWorkScaling(uint32_t /*config_idx*/ = 0) const { return false; }
  // scale is in (0, 1]; 1.0 is the full-size reference workload.
  virtual void SetWorkScale(uint32_t /*config_idx*/, double scale) { m_workScale = scale; }

  virtual bool ValidateResults(uint32_t /*config_idx*/ = 0) const { return true; }
  // Optional 0-indexed baseline config index for speedup comparison within this benchmark.
  // Returns -1 if this config is itself a baseline or has no comparable baseline.
  virtual int32_t GetBaselineConfigIndex(uint32_t /*config_idx*/) const { return -1; }
  virtual bool IsEmulated(uint32_t /*config_idx*/ = 0) const { return false; }
  virtual void SetVerifyParity(bool /*verify*/) {}
  virtual bool HasParityFailure() const { return false; }
  virtual bool HasVisualVerification() const { return false; }
  virtual void RunVisualVerification(bool isInteractive = false) { (void)isInteractive; }
  // Why a benchmark is unsupported (only meaningful when IsSupported()
  // returns false). Used to clearly distinguish hardware limitations from
  // API/toolchain limitations in reports.
  enum class SupportLimitation {
    kNone,      // supported, or reason unknown
    kHardware,  // the GPU physically lacks the capability
    kApi,       // the GPU has it, but the compute API cannot express it
    kToolchain, // GPU + API support it, but no shader compiler can emit it
  };
  virtual SupportLimitation GetSupportLimitation() const {
    return SupportLimitation::kNone;
  }
  virtual SupportLimitation GetSupportLimitation(const DeviceInfo &info,
                                                IComputeContext *context = nullptr) const {
    (void)info;
    (void)context;
    return GetSupportLimitation();
  }
  // Optional human-readable note explaining a capability limitation (shown
  // by --list-benchmarks and useful when IsSupported() returns false).
  // Empty string means no note.
  virtual std::string GetSupportNote() const { return ""; }
  virtual std::string GetSupportNote(const DeviceInfo &info,
                                     IComputeContext *context = nullptr) const {
    (void)info;
    (void)context;
    return GetSupportNote();
  }
  virtual uint32_t GetNumConfigs() const { return 1; }
  virtual std::string GetConfigName(uint32_t /*config_idx*/) const { return ""; }
  virtual const char *GetDescription() const { return ""; }
  virtual const char *GetDescription(uint32_t /*config_idx*/) const { return GetDescription(); }
  // Empirical Roofline Operational Intensity (FLOPs or OPs per byte of memory traffic).
  // Returns > 0.0 for compute/matrix/vector workloads plotted on the Roofline chart.
  virtual double GetOperationalIntensity(uint32_t /*config_idx*/ = 0) const { return 0.0; }
  virtual bool IsConfigSupported(uint32_t /*config_idx*/) const { return true; }
  virtual bool IsConfigSupported(uint32_t config_idx, const DeviceInfo &info,
                                 IComputeContext *context = nullptr) const {
    (void)info;
    (void)context;
    return IsConfigSupported(config_idx);
  }
  virtual std::string GetConfigSupportNote(uint32_t /*config_idx*/) const { return ""; }
  virtual std::string GetConfigSupportNote(uint32_t config_idx,
                                           const DeviceInfo &info,
                                           IComputeContext *context = nullptr) const {
    (void)info;
    (void)context;
    return GetConfigSupportNote(config_idx);
  }
  // Optional human-readable note explaining a performance or API caveat when
  // a benchmark is supported and completes, but utilized a fallback or emulation path.
  virtual std::string GetConfigCaveat(uint32_t /*config_idx*/ = 0) const { return ""; }
  virtual std::string GetConfigCaveat(uint32_t config_idx,
                                      const DeviceInfo &info,
                                      IComputeContext *context = nullptr) const {
    (void)info;
    (void)context;
    return GetConfigCaveat(config_idx);
  }
  virtual SupportLimitation GetConfigSupportLimitation(uint32_t /*config_idx*/) const {
    return GetSupportLimitation();
  }
  virtual SupportLimitation GetConfigSupportLimitation(uint32_t config_idx,
                                                       const DeviceInfo &info,
                                                       IComputeContext *context = nullptr) const {
    (void)info;
    (void)context;
    return GetConfigSupportLimitation(config_idx);
  }
  virtual uint32_t GetExpectedKernelCount() const { return 1; }

  // Returns true if this benchmark depends on the selected GPU device context.
  // Returns false if it is a system-wide or host-only benchmark (runs once).
  virtual bool IsDeviceDependent() const { return true; }

  virtual const char *GetComponent(uint32_t /*config_idx*/ = 0) const {
    return "Other";
  }
  virtual const char *GetSubCategory(uint32_t /*config_idx*/ = 0) const {
    return "";
  }
  virtual int GetSortWeight() const { return 999; }
  virtual int GetSortWeight(uint32_t /*config_idx*/) const {
    return GetSortWeight();
  }

  // Exports the scene geometry to an external file (e.g. OBJ) for
  // visualization.
  virtual void DumpGeometry() const {}

protected:
  double m_workScale = 1.0;

  // Workgroup count for the current work scale (never below 1).
  uint32_t ScaledGroups(uint32_t baseGroups) const {
    double g = static_cast<double>(baseGroups) * m_workScale;
    if (g < 1.0) g = 1.0;
    if (g > static_cast<double>(baseGroups)) g = static_cast<double>(baseGroups);
    return static_cast<uint32_t>(g);
  }
  // Operation count of a Run() that dispatched ScaledGroups(baseGroups).
  uint64_t ScaledOps(uint64_t fullOps, uint32_t baseGroups) const {
    return static_cast<uint64_t>(static_cast<double>(fullOps) *
                                     static_cast<double>(ScaledGroups(baseGroups)) /
                                     static_cast<double>(baseGroups) + 0.5);
  }
};
