#include "benchmarks/MemBandwidthBench.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

bool MemBandwidthBench::IsSupported(const DeviceInfo &info,
                                    IComputeContext *context) const {
  return true;
}

void MemBandwidthBench::createKernel(BandwidthConfig &config,
                                     const std::string &kernel_dir) {
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path
      kernel_file_path; // Renamed to avoid conflict with string kernel_file

  // The order of backend checks in the edit is different from original,
  // following the edit's logic for path construction.
  if (this->context->getBackend() == ComputeBackend::ROCm) {
    kernel_file_path = kdir / "rocm" / (config.kernelFile + ".hip");
  } else if (this->context->getBackend() == ComputeBackend::OpenCL) {
    kernel_file_path = kdir / "opencl" / (config.kernelFile + ".cl");
  } else { // Default to Vulkan if not ROCm or OpenCL
    kernel_file_path = kdir / "vulkan" / (config.kernelFile + ".comp");
  }

  std::string kernel_name;
  if (this->context->getBackend() == ComputeBackend::Vulkan) {
    kernel_name = "main";
  } else {
    kernel_name = "run_benchmark";
  }
  config.kernel =
      this->context->createKernel(kernel_file_path.string(), kernel_name, 2);
  this->context->setKernelArg(config.kernel, 0, inputBuffer);
  this->context->setKernelArg(config.kernel, 1, outputBuffer);
  uint32_t mode = static_cast<uint32_t>(config.mode);
  this->context->setKernelArg(config.kernel, 2, sizeof(mode), &mode);
  // Pass buffer size in bytes (all kernels now expect bytes)
  uint32_t bufferSizeInBytes = static_cast<uint32_t>(bufferSize);
  this->context->setKernelArg(config.kernel, 3, sizeof(bufferSizeInBytes),
                              &bufferSizeInBytes);

  // Debug logging for Vulkan (only if debug flag is enabled)
  if (debug && this->context->getBackend() == ComputeBackend::Vulkan) {
    std::cout << "  [DEBUG] Vulkan kernel '" << config.name
              << "': bufferSize=" << bufferSizeInBytes << " bytes ("
              << (bufferSizeInBytes / (1024 * 1024 * 1024))
              << "GB), mode=" << mode << std::endl;
  }
}

#include "core/DeviceDatabase.h"

void MemBandwidthBench::Setup(IComputeContext &context,
                              const std::string &kernel_dir) {
  this->context = &context;

  // Get device info first
  DeviceInfo deviceInfo = context.getCurrentDeviceInfo();
  uint32_t maxWorkgroupSize = deviceInfo.maxWorkGroupSize;

  // Fixed 512 MB buffer per stream (1 GB total across input + output) across all devices
  // so every GPU executes an identical per-dispatch memory workload (exceeding GPU L3/Infinity
  // Cache capacities while fitting comfortably on all modern GPUs).
  this->bufferSize = 512ULL * 1024ULL * 1024ULL;

  // Log buffer size for debugging
  if (this->context->getBackend() == ComputeBackend::Vulkan && debug) {
    std::cout << "Allocating memory buffers: "
              << (bufferSize / (1024 * 1024)) << " MB per buffer ("
              << ((bufferSize * 2) / (1024 * 1024)) << " MB total)"
              << std::endl;
  }

  inputBuffer = this->context->createBuffer(bufferSize);
  outputBuffer = this->context->createBuffer(bufferSize);

  // Initialize input buffer with 2.0f and output buffer with 0.0f in safe chunks.
  // This ensures Mode 0 (Read) accumulates to 2048.0f, Mode 1 (Write) writes 1.0f,
  // and Mode 2 (ReadWrite) writes 2.0f, guaranteeing positive and negative validation.
  constexpr size_t initChunkSize = 64 * 1024 * 1024; // 64 MB max host allocation
  size_t testDataSize = std::min<size_t>(bufferSize, initChunkSize);
  std::vector<float> inputInitData(testDataSize / sizeof(float), 2.0f);
  std::vector<float> outputInitData(testDataSize / sizeof(float), 0.0f);
  for (size_t off = 0; off < bufferSize; off += initChunkSize) {
    size_t cur = std::min<size_t>(bufferSize - off, initChunkSize);
    this->context->writeBuffer(inputBuffer, off, cur, inputInitData.data());
    this->context->writeBuffer(outputBuffer, off, cur, outputInitData.data());
  }

  this->context->waitIdle();

  // Calculate safe number of workgroups to avoid buffer aliasing
  // Each thread needs: 32 iterations × 32 vec4s = 1024 vec4s of unique buffer
  // space With stride = total_threads × 32, we need: total_threads × 32 × 32 ≤
  // buffer_size_in_vec4s
  uint32_t bufferSizeInVec4s = bufferSize / 16;
  uint32_t maxTotalThreads =
      bufferSizeInVec4s / (32 * 32); // = bufferSize / 16384

  // Create configurations based on device capabilities
  // We scale workgroups based on maxTotalThreads, with higher caps for modern
  // GPUs
  uint32_t numWorkgroups128 = std::min(16384u, maxTotalThreads / 128);
  configs.push_back({"Read 128 threads/group", "membw_128", 128,
                     numWorkgroups128, TestMode::Read, nullptr});
  configs.push_back({"Write 128 threads/group", "membw_128", 128,
                     numWorkgroups128, TestMode::Write, nullptr});
  configs.push_back({"R/W 128 threads/group", "membw_128", 128,
                     numWorkgroups128, TestMode::ReadWrite, nullptr});

  uint32_t workgroupSize256 = std::min(256u, maxWorkgroupSize);
  uint32_t numWorkgroups256 =
      std::min(8192u, maxTotalThreads / workgroupSize256);
  configs.push_back({"Read 256 threads/group", "membw_256", workgroupSize256,
                     numWorkgroups256, TestMode::Read, nullptr});
  configs.push_back({"Write 256 threads/group", "membw_256", workgroupSize256,
                     numWorkgroups256, TestMode::Write, nullptr});
  configs.push_back({"R/W 256 threads/group", "membw_256", workgroupSize256,
                     numWorkgroups256, TestMode::ReadWrite, nullptr});

  // Only add 1024 config if device supports it
  if (maxWorkgroupSize >= 1024) {
    uint32_t numWorkgroups1024 = std::min(2048u, maxTotalThreads / 1024);
    configs.push_back({"Read 1024 threads/group", "membw_1024", 1024,
                       numWorkgroups1024, TestMode::Read, nullptr});
    configs.push_back({"Write 1024 threads/group", "membw_1024", 1024,
                       numWorkgroups1024, TestMode::Write, nullptr});
    configs.push_back({"R/W 1024 threads/group", "membw_1024", 1024,
                       numWorkgroups1024, TestMode::ReadWrite, nullptr});
  }

  if (debug) {
    std::cout << "Max safe threads for " << (bufferSize / (1024 * 1024 * 1024))
              << "GB buffer: " << maxTotalThreads
              << " (128tpg: " << numWorkgroups128
              << " wg, 256tpg: " << numWorkgroups256 << " wg)" << std::endl;
  }

  for (auto &config : configs) {
    createKernel(config, kernel_dir);
  }
}

void MemBandwidthBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size()) {
    throw std::runtime_error("Invalid config index in MemBandwidthBench::Run");
  }

  auto &config = configs[config_idx];
  context->dispatch(config.kernel, config.numWorkgroups, 1, 1,
                    config.workgroupSize, 1, 1);
}

void MemBandwidthBench::Teardown() {
  for (auto &config : configs) {
    if (config.kernel) {
      context->releaseKernel(config.kernel);
    }
  }
  configs.clear();

  if (inputBuffer) {
    context->releaseBuffer(inputBuffer);
    inputBuffer = nullptr;
  }
  if (outputBuffer) {
    context->releaseBuffer(outputBuffer);
    outputBuffer = nullptr;
  }
}

const char *MemBandwidthBench::GetMetric() const { return "GB/s"; }

BenchmarkResult MemBandwidthBench::GetResult(uint32_t config_idx) const {
  if (config_idx >= configs.size()) {
    return {0, 0.0};
  }

  const auto &config = configs[config_idx];
  // Byte accounting: each thread moves 32 vec4s (32*16 = 512 bytes) per loop
  // iteration, for 32 iterations (16 KiB per thread per direction). The
  // membw_* kernels select the access pattern at runtime via the mode
  // argument, so the reported bytes are mode-aware:
  //   Read:      reads 512 B/iter only              -> 1x base
  //   Write:     writes 512 B/iter only             -> 1x base
  //   ReadWrite: reads AND writes 512 B/iter each   -> 2x base
  // All three backends (Vulkan, ROCm, OpenCL) hoist the mode branch outside
  // the streaming loop so Read issues only loads, Write issues only stores,
  // and ReadWrite issues both loads and stores.
  uint64_t base_bytes =
      (uint64_t)config.workgroupSize * config.numWorkgroups * 512 * 32;
  uint64_t bytes_transferred =
      (config.mode == TestMode::ReadWrite) ? base_bytes * 2 : base_bytes;
  return {bytes_transferred, 0.0};
}

uint32_t MemBandwidthBench::GetNumConfigs() const { return configs.size(); }

std::string MemBandwidthBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx >= configs.size()) {
    return "Invalid Config";
  }
  return configs[config_idx].name;
}

bool MemBandwidthBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= configs.size() || !context || !outputBuffer || bufferSize == 0) {
    return false;
  }

  const auto &config = configs[config_idx];
  const uint32_t numThreads = config.workgroupSize * config.numWorkgroups;
  if (numThreads == 0) {
    return false;
  }

  // Sample up to 16,384 threads (or all active threads if fewer)
  const size_t checkThreads = std::min<size_t>(numThreads, 16384);
  const size_t checkBytes = checkThreads * sizeof(float) * 4; // vec4 per thread

  std::vector<float> results(checkThreads * 4, 0.0f);
  try {
    context->readBuffer(outputBuffer, 0, checkBytes, results.data());
  } catch (const std::exception &e) {
    std::cerr << "MemBandwidthBench::ValidateResults readBuffer failed (config "
              << config_idx << "): " << e.what() << "\n";
    return false;
  }

  float expectedVal = 0.0f;
  switch (config.mode) {
    case TestMode::Read:
      // 32 outer loops * 32 vector loads = 1024 loads of vec4(2.0f) = 2048.0f
      expectedVal = 2048.0f;
      break;
    case TestMode::Write:
      // Kernel stores constant vec4(1.0f)
      expectedVal = 1.0f;
      break;
    case TestMode::ReadWrite:
      // Kernel loads from inputBuffer (2.0f) and writes directly to outputBuffer (2.0f)
      expectedVal = 2.0f;
      break;
  }

  // Verify non-zero data movement matching expected value across sampled lanes
  for (size_t i = 0; i < checkThreads; ++i) {
    for (int comp = 0; comp < 4; ++comp) {
      float val = results[i * 4 + comp];
      if (std::abs(val - expectedVal) > 1e-3f) {
        std::cerr << "MemBandwidthBench validation failed (config " << config_idx
                  << " '" << config.name << "'): results[" << i << "][" << comp
                  << "] = " << val << ", expected " << expectedVal << "\n";
        return false;
      }
    }
  }

  return true;
}
