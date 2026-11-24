#include "benchmarks/MemBandwidthBench.h"
#include "core/DeviceDatabase.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
uint32_t nextPowerOfTwoU32(uint32_t v) {
  if (v <= 1u) return 1u;
  --v;
  v |= v >> 1u;
  v |= v >> 2u;
  v |= v >> 4u;
  v |= v >> 8u;
  v |= v >> 16u;
  return v + 1u;
}
} // namespace

MemBandwidthBench::MemBandwidthBench() {
  initDefaultConfigs(1024);
}

bool MemBandwidthBench::IsSupported(const DeviceInfo &info,
                                    IComputeContext *context) const {
  (void)info;
  (void)context;
  return true;
}

bool MemBandwidthBench::IsConfigSupported(uint32_t config_idx,
                                          const DeviceInfo &info,
                                          IComputeContext *context) const {
  (void)context;
  if (config_idx >= configs.size()) {
    return false;
  }
  if (config_idx >= 6 && info.maxWorkGroupSize > 0 &&
      info.maxWorkGroupSize < 1024) {
    return false;
  }
  return true;
}

void MemBandwidthBench::initDefaultConfigs(uint32_t maxWorkgroupSize,
                                           uint32_t computeUnits,
                                           size_t targetBufferSize,
                                           bool scaleByCu) {
  if (maxWorkgroupSize == 0) {
    maxWorkgroupSize = 1024;
  }
  this->bufferSize = targetBufferSize;
  const uint32_t bufferSizeInVec4s = static_cast<uint32_t>(bufferSize / 16ULL);
  uint32_t targetThreads = bufferSizeInVec4s / (32u * 32u);
  uint32_t outerIters = 32u;

  if (scaleByCu) {
    const uint32_t cus = std::max(16u, computeUnits);
    const uint32_t minThreads = 65536u;
    const uint32_t maxThreads = std::max(minThreads, bufferSizeInVec4s / 128u);
    targetThreads = std::clamp(nextPowerOfTwoU32(cus * 4096u), minThreads, maxThreads);
    outerIters = std::max(1u, bufferSizeInVec4s / (targetThreads * 32u));
  }

  configs.clear();

  uint32_t numWorkgroups128 = std::max(1u, targetThreads / 128u);
  configs.push_back({"Read 128 threads/group", "membw_128", 128,
                     numWorkgroups128, outerIters, TestMode::Read, nullptr});
  configs.push_back({"Write 128 threads/group", "membw_128", 128,
                     numWorkgroups128, outerIters, TestMode::Write, nullptr});
  configs.push_back({"R/W 128 threads/group", "membw_128", 128,
                     numWorkgroups128, outerIters, TestMode::ReadWrite, nullptr});

  uint32_t workgroupSize256 = std::min(256u, maxWorkgroupSize);
  uint32_t numWorkgroups256 = std::max(1u, targetThreads / workgroupSize256);
  configs.push_back({"Read 256 threads/group", "membw_256", workgroupSize256,
                     numWorkgroups256, outerIters, TestMode::Read, nullptr});
  configs.push_back({"Write 256 threads/group", "membw_256", workgroupSize256,
                     numWorkgroups256, outerIters, TestMode::Write, nullptr});
  configs.push_back({"R/W 256 threads/group", "membw_256", workgroupSize256,
                     numWorkgroups256, outerIters, TestMode::ReadWrite, nullptr});

  uint32_t numWorkgroups1024 = std::max(1u, targetThreads / 1024u);
  configs.push_back({"Read 1024 threads/group", "membw_1024", 1024,
                     numWorkgroups1024, outerIters, TestMode::Read, nullptr});
  configs.push_back({"Write 1024 threads/group", "membw_1024", 1024,
                     numWorkgroups1024, outerIters, TestMode::Write, nullptr});
  configs.push_back({"R/W 1024 threads/group", "membw_1024", 1024,
                     numWorkgroups1024, outerIters, TestMode::ReadWrite, nullptr});
}

void MemBandwidthBench::createKernel(BandwidthConfig &config,
                                     const std::string &kernel_dir) {
  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path kernel_file_path;

  if (this->context->getBackend() == ComputeBackend::ROCm) {
    kernel_file_path = kdir / "rocm" / (config.kernelFile + ".hip");
  } else if (this->context->getBackend() == ComputeBackend::OpenCL) {
    kernel_file_path = kdir / "opencl" / (config.kernelFile + ".cl");
  } else {
    kernel_file_path = kdir / "vulkan" / (config.kernelFile + ".comp");
  }

  std::string kernel_name =
      (this->context->getBackend() == ComputeBackend::Vulkan) ? "main" : "run_benchmark";
  if (this->context->getBackend() == ComputeBackend::Vulkan) {
    config.kernel = this->context->createKernelWithSpecConstants(
        kernel_file_path.string(), kernel_name, 2,
        {static_cast<uint32_t>(config.mode)});
  } else {
    config.kernel =
        this->context->createKernel(kernel_file_path.string(), kernel_name, 2);
  }
  this->context->setKernelArg(config.kernel, 0, inputBuffer);
  this->context->setKernelArg(config.kernel, 1, outputBuffer);
  uint32_t mode = static_cast<uint32_t>(config.mode);
  this->context->setKernelArg(config.kernel, 2, sizeof(mode), &mode);
  uint32_t bufferSizeInBytes = static_cast<uint32_t>(bufferSize);
  this->context->setKernelArg(config.kernel, 3, sizeof(bufferSizeInBytes),
                              &bufferSizeInBytes);
  if (this->context->getBackend() == ComputeBackend::Vulkan ||
      this->context->getBackend() == ComputeBackend::ROCm) {
    uint32_t outerIters = config.outerIters;
    this->context->setKernelArg(config.kernel, 4, sizeof(outerIters), &outerIters);
  }

  if (debug && (this->context->getBackend() == ComputeBackend::Vulkan ||
                this->context->getBackend() == ComputeBackend::ROCm)) {
    std::cout << "  [DEBUG] Kernel '" << config.name
              << "': bufferSize=" << (bufferSizeInBytes / (1024 * 1024))
              << " MB, wgs=" << config.numWorkgroups
              << " x " << config.workgroupSize
              << ", outerIters=" << config.outerIters
              << ", mode=" << mode << std::endl;
  }
}

void MemBandwidthBench::Setup(IComputeContext &context,
                              const std::string &kernel_dir) {
  this->context = &context;

  DeviceInfo deviceInfo = context.getCurrentDeviceInfo();
  uint32_t maxWorkgroupSize = deviceInfo.maxWorkGroupSize > 0
                                  ? deviceInfo.maxWorkGroupSize
                                  : 1024u;

  const uint64_t availVram = (deviceInfo.dedicatedVramBytes > 0)
                                 ? deviceInfo.dedicatedVramBytes
                                 : deviceInfo.memorySize;
  size_t chosenBytes = 512ULL * 1024ULL * 1024ULL;
  if (availVram >= 8ULL * 1024ULL * 1024ULL * 1024ULL) {
    chosenBytes = 1024ULL * 1024ULL * 1024ULL;
  } else if (availVram > 0 && availVram < 4ULL * 1024ULL * 1024ULL * 1024ULL) {
    chosenBytes = 256ULL * 1024ULL * 1024ULL;
  }

#ifdef HAVE_VULKAN
  if (context.getBackend() == ComputeBackend::Vulkan) {
    if (auto *vc = dynamic_cast<VulkanContext *>(&context)) {
      VkPhysicalDeviceProperties props{};
      vkGetPhysicalDeviceProperties(vc->getVulkanPhysicalDevice(), &props);
      while (chosenBytes > (256ULL * 1024ULL * 1024ULL) &&
             chosenBytes > static_cast<size_t>(props.limits.maxStorageBufferRange)) {
        chosenBytes /= 2ULL;
      }
    }
  }
#endif

  const bool useDynamicCuScaling =
      (context.getBackend() == ComputeBackend::Vulkan ||
       context.getBackend() == ComputeBackend::ROCm);
  initDefaultConfigs(maxWorkgroupSize, deviceInfo.computeUnits, chosenBytes,
                     useDynamicCuScaling);

  if ((this->context->getBackend() == ComputeBackend::Vulkan ||
       this->context->getBackend() == ComputeBackend::ROCm) &&
      debug) {
    std::cout << "Allocating memory buffers: "
              << (bufferSize / (1024 * 1024)) << " MB per buffer ("
              << ((bufferSize * 2) / (1024 * 1024)) << " MB total)"
              << std::endl;
  }

  inputBuffer = this->context->createBuffer(bufferSize);
  outputBuffer = this->context->createBuffer(bufferSize);

  constexpr size_t initChunkSize = 64 * 1024 * 1024;
  size_t testDataSize = std::min<size_t>(bufferSize, initChunkSize);
  std::vector<float> inputInitData(testDataSize / sizeof(float), 2.0f);
  std::vector<float> outputInitData(testDataSize / sizeof(float), 0.0f);
  for (size_t off = 0; off < bufferSize; off += initChunkSize) {
    size_t cur = std::min<size_t>(bufferSize - off, initChunkSize);
    this->context->writeBuffer(inputBuffer, off, cur, inputInitData.data());
    this->context->writeBuffer(outputBuffer, off, cur, outputInitData.data());
  }

  this->context->waitIdle();

  for (auto &config : configs) {
    if (config.workgroupSize <= maxWorkgroupSize) {
      createKernel(config, kernel_dir);
    }
  }
}

void MemBandwidthBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size() || !configs[config_idx].kernel) {
    throw std::runtime_error("Invalid config index in MemBandwidthBench::Run");
  }

  auto &config = configs[config_idx];
  if (context->getBackend() == ComputeBackend::Vulkan ||
      context->getBackend() == ComputeBackend::ROCm) {
    uint32_t mode = static_cast<uint32_t>(config.mode);
    uint32_t bufBytes = static_cast<uint32_t>(bufferSize);
    uint32_t outer = config.outerIters;
    context->setKernelArg(config.kernel, 2, sizeof(mode), &mode);
    context->setKernelArg(config.kernel, 3, sizeof(bufBytes), &bufBytes);
    context->setKernelArg(config.kernel, 4, sizeof(outer), &outer);
  }
  context->dispatch(config.kernel, config.numWorkgroups, 1, 1,
                    config.workgroupSize, 1, 1);
}

void MemBandwidthBench::Teardown() {
  for (auto &config : configs) {
    if (config.kernel) {
      if (context) {
        context->releaseKernel(config.kernel);
      }
      config.kernel = nullptr;
    }
  }

  if (inputBuffer) {
    if (context) {
      context->releaseBuffer(inputBuffer);
    }
    inputBuffer = nullptr;
  }
  if (outputBuffer) {
    if (context) {
      context->releaseBuffer(outputBuffer);
    }
    outputBuffer = nullptr;
  }
}

const char *MemBandwidthBench::GetMetric() const { return "GB/s"; }

BenchmarkResult MemBandwidthBench::GetResult(uint32_t config_idx) const {
  if (config_idx >= configs.size()) {
    return {0, 0.0};
  }

  const auto &config = configs[config_idx];
  uint64_t base_bytes =
      static_cast<uint64_t>(config.workgroupSize) * config.numWorkgroups *
      config.outerIters * 32ULL * 16ULL;
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
  if (numThreads == 0 || config.outerIters == 0) {
    return false;
  }

  float expectedVal = 0.0f;
  switch (config.mode) {
    case TestMode::Read:
      expectedVal = static_cast<float>(config.outerIters * 32u) * 2.0f;
      break;
    case TestMode::Write:
      expectedVal = 1.0f;
      break;
    case TestMode::ReadWrite:
      expectedVal = 2.0f;
      break;
  }

  const size_t totalBufferVec4s = bufferSize / 16ULL;
  const size_t activeVec4s = (config.mode == TestMode::Read)
                                 ? std::min<size_t>(numThreads, totalBufferVec4s)
                                 : totalBufferVec4s;
  if (activeVec4s == 0) return false;

  auto checkSlice = [&](size_t startVec4, size_t countVec4) -> bool {
    if (countVec4 == 0 || startVec4 >= activeVec4s) return true;
    countVec4 = std::min(countVec4, activeVec4s - startVec4);
    std::vector<float> results(countVec4 * 4, 0.0f);
    try {
      context->readBuffer(outputBuffer, startVec4 * 16ULL, countVec4 * 16ULL,
                          results.data());
    } catch (const std::exception &e) {
      std::cerr << "MemBandwidthBench::ValidateResults readBuffer failed (config "
                << config_idx << "): " << e.what() << "\n";
      return false;
    }
    for (size_t i = 0; i < countVec4; ++i) {
      for (int comp = 0; comp < 4; ++comp) {
        float val = results[i * 4 + comp];
        if (std::abs(val - expectedVal) > 1e-3f) {
          std::cerr << "MemBandwidthBench validation failed (config " << config_idx
                    << " '" << config.name << "'): results[" << (startVec4 + i)
                    << "][" << comp << "] = " << val << ", expected "
                    << expectedVal << "\n";
          return false;
        }
      }
    }
    return true;
  };

  // Validate start (covers offset 0 & lane 100 checks), middle, and tail of active output
  if (!checkSlice(0, std::min<size_t>(activeVec4s, 16384u))) return false;
  if (activeVec4s > 16384u) {
    const size_t midStart = activeVec4s / 2u;
    if (!checkSlice(midStart, 1024u)) return false;
    const size_t tailStart = (activeVec4s >= 1024u) ? (activeVec4s - 1024u) : 0u;
    if (!checkSlice(tailStart, 1024u)) return false;
  }

  return true;
}
