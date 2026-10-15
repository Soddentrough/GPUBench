#include "benchmarks/InShaderIndirectBench.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

struct GpuIndirectCommand {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t activeCount;
  uint32_t workgroupsFinished;
};

InShaderIndirectBench::InShaderIndirectBench() {
  configs = {
    {"CPU Direct Fixed Grid (100% Active Baseline)", IndirectBenchMode::CPU_Direct, 100,
     "Host fixed grid dispatch with 100% item activity; baseline compute throughput"},
    {"In-Shader Indirect Synthesis (100% Active)", IndirectBenchMode::GPU_Indirect, 100,
     "Classifier synthesizes VkDispatchIndirectCommand on GPU; measures indirect dispatch overhead"},
    {"In-Shader Indirect Synthesis (50% Active)", IndirectBenchMode::GPU_Indirect, 50,
     "Stream compaction bins 50% active work; launches half the workgroups dynamically"},
    {"In-Shader Indirect Synthesis (10% Active)", IndirectBenchMode::GPU_Indirect, 10,
     "Sparse workload; GPU skips 90% of workgroups via synthesized grid sizing"},
    {"In-Shader Indirect Synthesis (1% Active)", IndirectBenchMode::GPU_Indirect, 1,
     "Highly sparse workload; dynamic indirect dispatch processes only 1% work"},
    {"In-Shader Zero-Dispatch Pruning (0% Active)", IndirectBenchMode::GPU_Indirect, 0,
     "Hardware Command Processor instantaneously dismisses zero-workgroup (0,0,0) dispatches"},
    {"CPU Direct Fixed Grid (10% Active)", IndirectBenchMode::CPU_Direct, 10,
     "Host fixed grid dispatch on sparse 10% active workload; shows wasted wave occupancy"}
  };
}

InShaderIndirectBench::~InShaderIndirectBench() {
  Teardown();
}

std::string InShaderIndirectBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].name;
  }
  return "";
}

std::string InShaderIndirectBench::GetConfigSupportNote(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].hint;
  }
  return "";
}

void InShaderIndirectBench::Setup(IComputeContext &context, const std::string &kernel_dir) {
  this->context = &context;

  size_t dataBufferSize = kTotalItems * sizeof(uint32_t);
  inputBuffer = context.createBuffer(dataBufferSize);
  workListBuffer = context.createBuffer(dataBufferSize);
  outputBuffer = context.createBuffer(dataBufferSize);
  indirectBuffer = context.createBuffer(256); // 256 bytes for indirect command
  GpuIndirectCommand initCmd{0, 0, 0, 0, 0};
  context.writeBuffer(indirectBuffer, 0, sizeof(initCmd), &initCmd);

  std::vector<uint32_t> initData(kTotalItems);
  for (uint32_t i = 0; i < kTotalItems; ++i) {
    initData[i] = i;
  }
  context.writeBuffer(inputBuffer, 0, dataBufferSize, initData.data());

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path reset_path = kdir / "vulkan" / "indirect_reset.comp";
  std::filesystem::path classify_path = kdir / "vulkan" / "indirect_classify.comp";
  std::filesystem::path worker_path = kdir / "vulkan" / "indirect_worker.comp";
  std::filesystem::path direct_path = kdir / "vulkan" / "direct_worker.comp";

  kernelReset = context.createKernel(reset_path.string(), "main", 1);
  context.setKernelArg(kernelReset, 0, indirectBuffer);

  kernelClassify = context.createKernel(classify_path.string(), "main", 3);
  context.setKernelArg(kernelClassify, 0, inputBuffer);
  context.setKernelArg(kernelClassify, 1, workListBuffer);
  context.setKernelArg(kernelClassify, 2, indirectBuffer);

  kernelIndirectWorker = context.createKernel(worker_path.string(), "main", 3);
  context.setKernelArg(kernelIndirectWorker, 0, workListBuffer);
  context.setKernelArg(kernelIndirectWorker, 1, outputBuffer);
  context.setKernelArg(kernelIndirectWorker, 2, indirectBuffer);

  kernelDirectWorker = context.createKernel(direct_path.string(), "main", 2);
  context.setKernelArg(kernelDirectWorker, 0, inputBuffer);
  context.setKernelArg(kernelDirectWorker, 1, outputBuffer);
}

void InShaderIndirectBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size() || !context) {
    throw std::runtime_error("InShaderIndirectBench: invalid config or uninitialized state");
  }

  const auto &cfg = configs[config_idx];

  if (cfg.mode == IndirectBenchMode::GPU_Indirect) {
    context->dispatch(kernelReset, 1, 1, 1, 1, 1, 1);

    struct ClassifyPushConstants {
      uint32_t selectivity;
      uint32_t totalItems;
      uint32_t totalWorkgroups;
    } pcClassify = { cfg.selectivity, kTotalItems, kTotalWorkgroups };

    context->setKernelArg(kernelClassify, 3, sizeof(pcClassify), &pcClassify);
    context->dispatch(kernelClassify, kTotalWorkgroups, 1, 1, kWorkgroupSize, 1, 1);

#ifdef HAVE_VULKAN
    auto *vContext = dynamic_cast<VulkanContext *>(context);
    if (vContext) {
      vContext->dispatchIndirect(kernelIndirectWorker, indirectBuffer, 0);
    }
#endif
  } else {
    struct DirectPushConstants {
      uint32_t selectivity;
      uint32_t totalItems;
    } pcDirect = { cfg.selectivity, kTotalItems };

    context->setKernelArg(kernelDirectWorker, 2, sizeof(pcDirect), &pcDirect);
    context->dispatch(kernelDirectWorker, kTotalWorkgroups, 1, 1, kWorkgroupSize, 1, 1);
  }
}

void InShaderIndirectBench::Teardown() {
  if (context) {
    if (kernelReset) {
      context->releaseKernel(kernelReset);
      kernelReset = nullptr;
    }
    if (kernelClassify) {
      context->releaseKernel(kernelClassify);
      kernelClassify = nullptr;
    }
    if (kernelIndirectWorker) {
      context->releaseKernel(kernelIndirectWorker);
      kernelIndirectWorker = nullptr;
    }
    if (kernelDirectWorker) {
      context->releaseKernel(kernelDirectWorker);
      kernelDirectWorker = nullptr;
    }
    if (inputBuffer) {
      context->releaseBuffer(inputBuffer);
      inputBuffer = nullptr;
    }
    if (workListBuffer) {
      context->releaseBuffer(workListBuffer);
      workListBuffer = nullptr;
    }
    if (indirectBuffer) {
      context->releaseBuffer(indirectBuffer);
      indirectBuffer = nullptr;
    }
    if (outputBuffer) {
      context->releaseBuffer(outputBuffer);
      outputBuffer = nullptr;
    }
    context = nullptr;
  }
}

BenchmarkResult InShaderIndirectBench::GetResult(uint32_t /*config_idx*/) const {
  return { 1, 0.0 };
}

static float computeIndirectExpectedWorkerVal(uint32_t itemId) {
  float val = static_cast<float>(itemId) * 0.001f;
  for (int i = 0; i < 32; ++i) {
    val = std::sin(val) * 1.05f + std::cos(val * 0.5f);
  }
  return val;
}

bool InShaderIndirectBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= configs.size() || !context) {
    return false;
  }
  if (!outputBuffer) {
    return false;
  }

  const auto &cfg = configs[config_idx];
  try {
    if (cfg.mode == IndirectBenchMode::GPU_Indirect) {
      if (!indirectBuffer || !workListBuffer) {
        return false;
      }
      GpuIndirectCommand cmd{};
      context->readBuffer(indirectBuffer, 0, sizeof(cmd), &cmd);

      if (cmd.workgroupsFinished != kTotalWorkgroups) {
        std::cerr << "InShaderIndirectBench validation failed (config "
                  << config_idx << "): workgroupsFinished="
                  << cmd.workgroupsFinished << ", expected " << kTotalWorkgroups
                  << "\n";
        return false;
      }

      uint32_t expectedX = (cmd.activeCount + 63u) / 64u;
      if (cmd.x != expectedX) {
        std::cerr << "InShaderIndirectBench validation failed (config "
                  << config_idx << "): cmd.x=" << cmd.x
                  << ", expected " << expectedX << "\n";
        return false;
      }

      if (cfg.selectivity == 100) {
        if (cmd.activeCount != kTotalItems ||
            cmd.x != (kTotalItems + 63u) / 64u) {
          std::cerr << "InShaderIndirectBench validation failed (100% active): "
                    << "activeCount=" << cmd.activeCount << ", cmd.x=" << cmd.x
                    << "\n";
          return false;
        }
      } else if (cfg.selectivity == 0) {
        if (cmd.activeCount != 0u || cmd.x != 0u || cmd.y != 0u || cmd.z != 0u) {
          std::cerr << "InShaderIndirectBench validation failed (0% active): "
                    << "activeCount=" << cmd.activeCount << ", cmd=(" << cmd.x
                    << "," << cmd.y << "," << cmd.z << ")\n";
          return false;
        }
      }

      // Numerical output buffer validation for indirect worker pass
      if (cmd.activeCount > 0) {
        uint32_t checkCount = std::min(cmd.activeCount, 1024u);
        std::vector<float> results(checkCount);
        std::vector<uint32_t> worklist(checkCount);
        context->readBuffer(outputBuffer, 0, checkCount * sizeof(float), results.data());
        context->readBuffer(workListBuffer, 0, checkCount * sizeof(uint32_t), worklist.data());

        for (uint32_t i = 0; i < checkCount; ++i) {
          float val = results[i];
          if (std::isnan(val) || std::isinf(val) || val == 0.0f) {
            std::cerr << "InShaderIndirectBench validation failed (config "
                      << config_idx << "): outputBuffer[" << i
                      << "] is invalid (val=" << val << ")\n";
            return false;
          }

          uint32_t itemId = worklist[i];
          if (itemId >= kTotalItems) {
            std::cerr << "InShaderIndirectBench validation failed (config "
                      << config_idx << "): workListBuffer[" << i
                      << "] itemId out of bounds (" << itemId << ")\n";
            return false;
          }

          uint32_t hash = itemId * 2654435761u;
          if ((hash % 100u) >= cfg.selectivity) {
            std::cerr << "InShaderIndirectBench validation failed (config "
                      << config_idx << "): workListBuffer[" << i
                      << "] itemId=" << itemId << " does not match selectivity\n";
            return false;
          }

          float expectedVal = computeIndirectExpectedWorkerVal(itemId);
          if (std::abs(val - expectedVal) > 0.05f) {
            std::cerr << "InShaderIndirectBench validation failed (config "
                      << config_idx << "): outputBuffer[" << i << "] = "
                      << val << ", expected " << expectedVal << "\n";
            return false;
          }
        }
      }
    } else {
      // CPU Direct mode (configs 0 and 6)
      uint32_t checkCount = std::min(kTotalItems, 1024u);
      std::vector<float> results(checkCount);
      context->readBuffer(outputBuffer, 0, checkCount * sizeof(float), results.data());

      uint32_t activeFound = 0;
      for (uint32_t gid = 0; gid < checkCount; ++gid) {
        uint32_t hash = gid * 2654435761u;
        bool activeItem = (hash % 100u) < cfg.selectivity;
        if (activeItem) {
          activeFound++;
          float val = results[gid];
          if (std::isnan(val) || std::isinf(val) || val == 0.0f) {
            std::cerr << "InShaderIndirectBench validation failed (CPU Direct config "
                      << config_idx << "): outputBuffer[" << gid
                      << "] is invalid (val=" << val << ")\n";
            return false;
          }

          float expectedVal = computeIndirectExpectedWorkerVal(gid);
          if (std::abs(val - expectedVal) > 0.05f) {
            std::cerr << "InShaderIndirectBench validation failed (CPU Direct config "
                      << config_idx << "): outputBuffer[" << gid << "] = "
                      << val << ", expected " << expectedVal << "\n";
            return false;
          }
        }
      }

      if (cfg.selectivity > 0 && activeFound == 0) {
        std::cerr << "InShaderIndirectBench validation failed (CPU Direct config "
                  << config_idx << "): no active items validated\n";
        return false;
      }
    }
  } catch (...) {
    return false;
  }

  return true;
}
