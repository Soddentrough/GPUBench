#include "benchmarks/MemBandwidthBench.h"
#include <iostream>
#include <stdexcept>

bool MemBandwidthBench::IsSupported(const DeviceInfo &info,
                                    IComputeContext *context) const {
  return true;
}

void MemBandwidthBench::createKernel(BandwidthConfig &config,
                                     const std::string &kernel_dir) {
  std::string kernel_file;
  if (context->getBackend() == ComputeBackend::Vulkan) {
    kernel_file = kernel_dir + "/vulkan/" + config.kernelFile + ".comp";
  } else if (context->getBackend() == ComputeBackend::ROCm) {
    kernel_file = kernel_dir + "/rocm/" + config.kernelFile + ".hip";
  } else {
    kernel_file = kernel_dir + "/opencl/" + config.kernelFile + ".cl";
  }

  std::string kernel_name;
  if (context->getBackend() == ComputeBackend::Vulkan) {
    kernel_name = "main";
  } else if (context->getBackend() == ComputeBackend::ROCm) {
    kernel_name = "run_benchmark";
  } else {
    kernel_name = "run_benchmark";
  }
  config.kernel = this->context->createKernel(kernel_file, kernel_name, 4);
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

void MemBandwidthBench::Setup(IComputeContext &context,
                              const std::string &kernel_dir) {
  this->context = &context;

  // Get device info first
  DeviceInfo deviceInfo = context.getCurrentDeviceInfo();

  // Calculate buffer size based on available VRAM and workload requirements
  uint64_t availableVRAM = deviceInfo.memorySize;
  uint32_t maxWorkgroupSize = deviceInfo.maxWorkGroupSize;

  // Determine maximum number of threads across all configurations
  // Each config has: workgroupSize * numWorkgroups threads
  uint32_t maxThreads = 0;

  // 128 threads/group configs
  maxThreads = std::max(maxThreads, 128u * 4096u);

  // 256 threads/group configs (if supported)
  if (maxWorkgroupSize >= 256) {
    maxThreads = std::max(maxThreads, std::min(256u, maxWorkgroupSize) * 2048u);
  }

  // 1024 threads/group configs (if supported)
  if (maxWorkgroupSize >= 1024) {
    maxThreads = std::max(maxThreads, 1024u * 512u);
  }

  // Calculate max safe size: 50% of VRAM or 2GB, whichever is smaller.
  // We increase this to 2GB to better saturate modern high-bandwidth GPUs
  // (H100, MI300).
  uint64_t maxSafeSize =
      std::min<uint64_t>(availableVRAM / 2, 2048ULL * 1024ULL * 1024ULL);

  // Find largest power of 2 that fits in maxSafeSize
  this->bufferSize = 16ULL * 1024ULL * 1024ULL; // Start at 16MB min
  while (this->bufferSize * 2 <= maxSafeSize) {
    this->bufferSize *= 2;
  }

  // Log buffer size for debugging
  if (this->context->getBackend() == ComputeBackend::Vulkan && debug) {
    std::cout << "Allocating memory buffers: "
              << (bufferSize / (1024 * 1024 * 1024)) << " GB per buffer ("
              << ((bufferSize * 2) / (1024 * 1024 * 1024)) << " GB total)"
              << std::endl;
  }

  inputBuffer = this->context->createBuffer(bufferSize);
  outputBuffer = this->context->createBuffer(bufferSize);

  // Initialize input buffer with test data to prevent reading uninitialized
  // memory
  std::vector<float> testData(bufferSize / sizeof(float), 1.0f);
  this->context->writeBuffer(inputBuffer, 0, bufferSize, testData.data());

  // Initialize output buffer as well to ensure pages are mapped/resident
  // (prevents page faults on unified memory)
  this->context->writeBuffer(outputBuffer, 0, bufferSize, testData.data());

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
  }
  if (outputBuffer) {
    context->releaseBuffer(outputBuffer);
  }
}

const char *MemBandwidthBench::GetMetric() const { return "GB/s"; }

BenchmarkResult MemBandwidthBench::GetResult(uint32_t config_idx) const {
  if (config_idx >= configs.size()) {
    return {0, 0.0};
  }

  const auto &config = configs[config_idx];
  // Each thread transfers 32 vec4s (32*16=512 bytes) per iteration, for 32
  // iterations
  uint64_t bytes_transferred =
      (uint64_t)config.workgroupSize * config.numWorkgroups * 512 * 32;
  return {bytes_transferred, 0.0};
}

uint32_t MemBandwidthBench::GetNumConfigs() const { return configs.size(); }

std::string MemBandwidthBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx >= configs.size()) {
    return "Invalid Config";
  }
  return configs[config_idx].name;
}
