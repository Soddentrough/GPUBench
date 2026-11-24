#include "benchmarks/InShaderIndirectBench.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
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

namespace {

bool isMbActiveBucket(uint32_t bucket) {
  return bucket == 3u || bucket == 36u || bucket == 73u || bucket == 118u;
}

float computeIndirectExpectedWorkerVal(uint32_t itemId) {
  float val = static_cast<float>(itemId) * 0.001f;
  for (int i = 0; i < 32; ++i) {
    val = std::sin(val) * 1.05f + std::cos(val * 0.5f);
  }
  return val;
}

float computeMbExpectedWorkerVal(uint32_t itemId, uint32_t specializedPipe) {
  float val = static_cast<float>(itemId) * 0.001f +
              (1.0f + 0.05f * static_cast<float>(specializedPipe + 1u));
  for (int i = 0; i < 16; ++i) {
    val = std::sin(val) * 1.05f + std::cos(val * 0.5f);
  }
  return val;
}

} // namespace

InShaderIndirectBench::InShaderIndirectBench() {
  configs = {
    {"CPU Direct Fixed Grid (100% Active)", IndirectBenchMode::CPU_Direct, 100,
     "Host fixed grid dispatch with 100% item activity; baseline compute throughput"},
    {"CPU Direct Fixed Grid (10% Active)", IndirectBenchMode::CPU_Direct, 10,
     "Host fixed grid dispatch on sparse 10% active workload; shows wasted wave occupancy"},
    {"Legacy Indirect Dispatch (100% Active)", IndirectBenchMode::GPU_Indirect, 100,
     "Classifier synthesizes VkDispatchIndirectCommand on GPU; measures indirect dispatch overhead"},
    {"Legacy Indirect Dispatch (10% Active)", IndirectBenchMode::GPU_Indirect, 10,
     "Sparse workload; GPU skips 90% of workgroups via synthesized grid sizing vs 10% CPU Direct"},
    {"Legacy Zero-Dispatch Pruning (0% Active)", IndirectBenchMode::GPU_Indirect, 0,
     "Hardware Command Processor instantaneously dismisses zero-workgroup (0,0,0) dispatches"},
    {"Multi-Bucket 128x Sparse (Legacy vkCmdDispatchIndirect)", IndirectBenchMode::GPU_MultiBucket_Legacy, 0,
     "128 specialized buckets (only 4 active) recorded via 128 CPU pipeline-bind + vkCmdDispatchIndirect calls"},
    {"Multi-Bucket 128x Sparse (DGC ExecutionSet + Pruning)", IndirectBenchMode::GPU_MultiBucket_DGC, 0,
     "Single vkCmdExecuteGeneratedCommandsEXT + VkIndirectExecutionSetEXT with on-GPU empty-bucket pruning"}
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

std::string InShaderIndirectBench::GetConfigName(uint32_t config_idx,
                                                 const DeviceInfo & /*info*/,
                                                 IComputeContext *ctx) const {
  if (ctx && ctx->getBackend() == ComputeBackend::ROCm) {
    switch (config_idx) {
    case 2u:
      return "Persistent Work-Queue Compaction (100% Active)";
    case 3u:
      return "Persistent Work-Queue Compaction (10% Active)";
    case 4u:
      return "Persistent Work-Queue Early-Exit (0% Active)";
    default:
      break;
    }
  }
  return GetConfigName(config_idx);
}

const char *InShaderIndirectBench::GetDescription(uint32_t config_idx,
                                                  const DeviceInfo & /*info*/,
                                                  IComputeContext *ctx) const {
  if (ctx && ctx->getBackend() == ComputeBackend::ROCm) {
    switch (config_idx) {
    case 0u:
      return "Host fixed-grid HIP launch with 100% item activity; baseline compute throughput";
    case 1u:
      return "Host fixed-grid HIP launch on sparse 10% active workload; shows wasted wave occupancy";
    case 2u:
      return "HIP classifier compacts active items into VRAM work queue; fixed-grid persistent consumer reads compacted queue directly (100% active)";
    case 3u:
      return "Sparse 10% workload; HIP stream compaction eliminates inactive work via fixed-grid persistent consumer queue vs 10% CPU direct launch";
    case 4u:
      return "Empty compacted work queue (activeCount = 0); HIP persistent consumer exits immediately on atomic counter check";
    default:
      break;
    }
  }
  return GetDescription(config_idx);
}

bool InShaderIndirectBench::IsConfigSupported(uint32_t config_idx,
                                              const DeviceInfo &info,
                                              IComputeContext *ctx) const {
  if (config_idx >= configs.size()) return false;
  if (!IsSupported(info, ctx)) return false;
  if (ctx && ctx->getBackend() == ComputeBackend::ROCm) {
    return config_idx <= 4u;
  }
  if (config_idx == 6u) {
#ifdef HAVE_VULKAN
    if (const auto *vc = dynamic_cast<const VulkanContext *>(ctx)) {
      if (!vc->isDGCSupported() || !vc->isDGCExecutionSetSupported()) {
        return false;
      }
      return !mbSetupComplete || mbDgcAvailable;
    }
#endif
    return info.dgcSupported && info.dgcExecutionSetSupported &&
           (!mbSetupComplete || mbDgcAvailable);
  }
  return true;
}

IBenchmark::SupportLimitation InShaderIndirectBench::GetConfigSupportLimitation(
    uint32_t config_idx, const DeviceInfo &info, IComputeContext *ctx) const {
  if (ctx && ctx->getBackend() == ComputeBackend::ROCm) {
    return (config_idx <= 4u) ? SupportLimitation::kNone : SupportLimitation::kApi;
  }
  if (ctx && ctx->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if (config_idx == 6u) {
    bool basicDgc = info.dgcSupported;
    bool execSet = info.dgcExecutionSetSupported;
#ifdef HAVE_VULKAN
    if (const auto *vc = dynamic_cast<const VulkanContext *>(ctx)) {
      basicDgc = vc->isDGCSupported();
      execSet = vc->isDGCExecutionSetSupported();
    }
#endif
    if (basicDgc && !execSet) {
      return SupportLimitation::kApi;
    }
    return SupportLimitation::kHardware;
  }
  return SupportLimitation::kNone;
}

std::string InShaderIndirectBench::GetSupportNote(const DeviceInfo & /*info*/,
                                                  IComputeContext *ctx) const {
  if (ctx && ctx->getBackend() != ComputeBackend::Vulkan &&
      ctx->getBackend() != ComputeBackend::ROCm) {
    return "Requires Vulkan 1.3+ indirect dispatch or ROCm persistent GPU work queue";
  }
  return "";
}

std::string InShaderIndirectBench::GetConfigSupportNote(uint32_t config_idx) const {
  if (config_idx == 6u && !mbDgcUnsupportedReason.empty()) {
    return mbDgcUnsupportedReason;
  }
  if (config_idx < configs.size()) {
    return configs[config_idx].hint;
  }
  return "";
}

std::string InShaderIndirectBench::GetConfigSupportNote(uint32_t config_idx,
                                                        const DeviceInfo &info,
                                                        IComputeContext *ctx) const {
  if (ctx && ctx->getBackend() == ComputeBackend::ROCm && config_idx >= 5u) {
    return "128-bucket multi-pipeline indirect/DGC ExecutionSet requires Vulkan 1.4";
  }
  if (config_idx == 6u) {
    if (!mbDgcUnsupportedReason.empty()) {
      return mbDgcUnsupportedReason;
    }
    bool basicDgc = info.dgcSupported;
    bool execSet = info.dgcExecutionSetSupported;
#ifdef HAVE_VULKAN
    if (const auto *vc = dynamic_cast<const VulkanContext *>(ctx)) {
      basicDgc = vc->isDGCSupported();
      execSet = vc->isDGCExecutionSetSupported();
    }
#endif
    if (!basicDgc) {
      return "VK_EXT_device_generated_commands unsupported by device/driver";
    }
    if (!execSet) {
      return "VK_EXT_device_generated_commands IndirectExecutionSet pipeline binding unsupported for compute";
    }
  }
  return GetConfigSupportNote(config_idx);
}

std::string InShaderIndirectBench::GetConfigCaveat(uint32_t config_idx,
                                                   const DeviceInfo & /*info*/,
                                                   IComputeContext *ctx) const {
  if (ctx && ctx->getBackend() == ComputeBackend::ROCm &&
      config_idx >= 2u && config_idx <= 4u) {
    return "ROCm GPU-compacted persistent queue (HIP lacks CP vkCmdDispatchIndirect; consumer reads compacted queue directly in VRAM)";
  }
  return "";
}

void InShaderIndirectBench::Setup(IComputeContext &context_ref, const std::string &kernel_dir) {
  this->context = &context_ref;
  mbSetupComplete = false;
  mbDgcAvailable = false;
  mbDgcUnsupportedReason.clear();

  size_t dataBufferSize = kTotalItems * sizeof(uint32_t);
  inputBuffer = context_ref.createBuffer(dataBufferSize);
  workListBuffer = context_ref.createBuffer(dataBufferSize);
  outputBuffer = context_ref.createBuffer(dataBufferSize);
  indirectBuffer = context_ref.createBuffer(256); // 256 bytes for indirect command
  GpuIndirectCommand initCmd{0, 0, 0, 0, 0};
  context_ref.writeBuffer(indirectBuffer, 0, sizeof(initCmd), &initCmd);

  std::vector<uint32_t> initData(kTotalItems);
  for (uint32_t i = 0; i < kTotalItems; ++i) {
    initData[i] = i;
  }
  context_ref.writeBuffer(inputBuffer, 0, dataBufferSize, initData.data());

  const bool isRocm = (context_ref.getBackend() == ComputeBackend::ROCm);
  const DeviceInfo devInfo = context_ref.getCurrentDeviceInfo();
  m_rocmPersistentWorkgroups = std::clamp(devInfo.computeUnits * 16u, 256u, 4096u);

  std::filesystem::path kdir(kernel_dir);
  std::filesystem::path reset_path =
      isRocm ? (kdir / "rocm" / "indirect_reset.hip")
             : (kdir / "vulkan" / "indirect_reset.comp");
  std::filesystem::path classify_path =
      isRocm ? (kdir / "rocm" / "indirect_classify.hip")
             : (kdir / "vulkan" / "indirect_classify.comp");
  std::filesystem::path worker_path =
      isRocm ? (kdir / "rocm" / "indirect_worker.hip")
             : (kdir / "vulkan" / "indirect_worker.comp");
  std::filesystem::path direct_path =
      isRocm ? (kdir / "rocm" / "direct_worker.hip")
             : (kdir / "vulkan" / "direct_worker.comp");
  const std::string entry = isRocm ? "run_benchmark" : "main";

  kernelReset = context_ref.createKernel(reset_path.string(), entry, 1);
  context_ref.setKernelArg(kernelReset, 0, indirectBuffer);

  kernelClassify = context_ref.createKernel(classify_path.string(), entry, 3);
  context_ref.setKernelArg(kernelClassify, 0, inputBuffer);
  context_ref.setKernelArg(kernelClassify, 1, workListBuffer);
  context_ref.setKernelArg(kernelClassify, 2, indirectBuffer);

  kernelIndirectWorker = context_ref.createKernel(worker_path.string(), entry, 3);
  context_ref.setKernelArg(kernelIndirectWorker, 0, workListBuffer);
  context_ref.setKernelArg(kernelIndirectWorker, 1, outputBuffer);
  context_ref.setKernelArg(kernelIndirectWorker, 2, indirectBuffer);

  kernelDirectWorker = context_ref.createKernel(direct_path.string(), entry, 2);
  context_ref.setKernelArg(kernelDirectWorker, 0, inputBuffer);
  context_ref.setKernelArg(kernelDirectWorker, 1, outputBuffer);

  if (isRocm) {
    mbSetupComplete = true;
    return;
  }

  // Allocate Multi-Bucket 128x buffers (Vulkan only)
  mbWorkListBuffer = context_ref.createBuffer(
      static_cast<size_t>(kMbNumBuckets) * kMbBucketCapacity * sizeof(uint32_t));
  std::vector<uint32_t> zeroCounts(kMbNumBuckets * 2u, 0u);
  mbBucketCountsBuffer = context_ref.createBuffer(
      zeroCounts.size() * sizeof(uint32_t), zeroCounts.data());
  std::vector<uint32_t> zeroCmds(kMbNumBuckets * 3u, 0u);
  mbIndirectBuffer = context_ref.createBuffer(
      zeroCmds.size() * sizeof(uint32_t), zeroCmds.data());
  std::vector<uint32_t> zeroSeq(kMbNumBuckets * 8u, 0u);
  mbDgcSeqBuffer = context_ref.createBuffer(
      zeroSeq.size() * sizeof(uint32_t), zeroSeq.data());
  std::vector<uint32_t> zeroDgcCount(4u, 0u);
  mbDgcCountBuffer = context_ref.createBuffer(
      zeroDgcCount.size() * sizeof(uint32_t), zeroDgcCount.data());

  std::filesystem::path mb_classify_path = kdir / "vulkan" / "indirect_multibucket_classify.comp";
  std::filesystem::path mb_resolve_path = kdir / "vulkan" / "indirect_multibucket_resolve.comp";
  std::filesystem::path mb_worker_path = kdir / "vulkan" / "indirect_multibucket_worker.comp";

  kernelMbClassify = context_ref.createKernel(mb_classify_path.string(), "main", 3);
  context_ref.setKernelArg(kernelMbClassify, 0, inputBuffer);
  context_ref.setKernelArg(kernelMbClassify, 1, mbWorkListBuffer);
  context_ref.setKernelArg(kernelMbClassify, 2, mbBucketCountsBuffer);

  kernelMbResolve = context_ref.createKernel(mb_resolve_path.string(), "main", 4);
  context_ref.setKernelArg(kernelMbResolve, 0, mbBucketCountsBuffer);
  context_ref.setKernelArg(kernelMbResolve, 1, mbIndirectBuffer);
  context_ref.setKernelArg(kernelMbResolve, 2, mbDgcCountBuffer);
  context_ref.setKernelArg(kernelMbResolve, 3, mbDgcSeqBuffer);

#ifdef HAVE_VULKAN
  auto *vContext = dynamic_cast<VulkanContext *>(&context_ref);
  if (vContext) {
    for (uint32_t p = 0; p < 8u; ++p) {
      kernelMbWorker[p] = vContext->createKernelWithSpec(
          mb_worker_path.string(), "main", 3, 0, p);
      context_ref.setKernelArg(kernelMbWorker[p], 0, mbWorkListBuffer);
      context_ref.setKernelArg(kernelMbWorker[p], 1, outputBuffer);
      context_ref.setKernelArg(kernelMbWorker[p], 2, mbBucketCountsBuffer);
    }

    mbLegacyEntries.clear();
    mbLegacyEntries.reserve(kMbNumBuckets);
    for (uint32_t b = 0; b < kMbNumBuckets; ++b) {
      uint32_t pc[2] = {b, 0u};
      std::vector<uint8_t> pcBytes(sizeof(pc));
      std::memcpy(pcBytes.data(), pc, sizeof(pc));
      mbLegacyEntries.push_back(
          {b * sizeof(uint32_t) * 3u, std::move(pcBytes), kernelMbWorker[b & 7u]});
    }

    if (vContext->isDGCSupported() && vContext->isDGCExecutionSetSupported()) {
      try {
        VkIndirectCommandsExecutionSetTokenEXT execToken{};
        execToken.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
        execToken.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;

        VkIndirectCommandsPushConstantTokenEXT pcToken{};
        pcToken.updateRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcToken.updateRange.offset = 0;
        pcToken.updateRange.size = sizeof(uint32_t) * 2u;

        VkIndirectCommandsLayoutTokenEXT tokens[3]{};
        tokens[0].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        tokens[0].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT;
        tokens[0].data.pExecutionSet = &execToken;
        tokens[0].offset = 0;

        tokens[1].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        tokens[1].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT;
        tokens[1].data.pPushConstant = &pcToken;
        tokens[1].offset = sizeof(uint32_t); // 4

        tokens[2].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        tokens[2].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
        tokens[2].offset = sizeof(uint32_t) * 3u; // 12

        VkIndirectCommandsLayoutCreateInfoEXT layoutCi{
            VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT};
        layoutCi.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutCi.indirectStride = sizeof(uint32_t) * 8u; // 32 bytes
        layoutCi.pipelineLayout = vContext->getVkPipelineLayout(kernelMbWorker[0]);
        layoutCi.tokenCount = 3;
        layoutCi.pTokens = tokens;
        mbDgcLayout = vContext->createIndirectCommandsLayout(layoutCi);

        VkIndirectExecutionSetPipelineInfoEXT iesPipeInfo{
            VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT};
        iesPipeInfo.initialPipeline = vContext->getVkPipeline(kernelMbWorker[0]);
        iesPipeInfo.maxPipelineCount = 8;

        VkIndirectExecutionSetCreateInfoEXT iesCi{
            VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT};
        iesCi.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
        iesCi.info.pPipelineInfo = &iesPipeInfo;
        mbDgcExecSet = vContext->createIndirectExecutionSet(iesCi);
        for (uint32_t p = 0; p < 8u; ++p) {
          vContext->updateIndirectExecutionSetPipeline(
              mbDgcExecSet, p, kernelMbWorker[p]);
        }

        VkDeviceSize preReq = vContext->getGeneratedCommandsMemoryRequirements(
            mbDgcLayout, mbDgcExecSet, kMbNumBuckets, nullptr);
        VkDeviceSize preSize = std::max<VkDeviceSize>(preReq, 4096);
        mbDgcPreprocessBuffer =
            vContext->createPreprocessBuffer(static_cast<size_t>(preSize));

        mbDgcInfo = VulkanContext::DGCExecutionInfo{};
        mbDgcInfo.layout = mbDgcLayout;
        mbDgcInfo.executionSet = mbDgcExecSet;
        mbDgcInfo.sequenceBuffer = mbDgcSeqBuffer;
        mbDgcInfo.sequenceBufferOffset = 0;
        mbDgcInfo.sequenceBufferSize = sizeof(uint32_t) * 8u * kMbNumBuckets;
        mbDgcInfo.sequenceCountBuffer = mbDgcCountBuffer;
        mbDgcInfo.sequenceCountBufferOffset = 0;
        mbDgcInfo.preprocessBuffer = mbDgcPreprocessBuffer;
        mbDgcInfo.preprocessBufferSize = preSize;
        mbDgcInfo.maxSequenceCount = kMbNumBuckets;
        mbDgcAvailable = true;
      } catch (const std::exception &e) {
        mbDgcUnsupportedReason =
            std::string("Multi-bucket DGC ExecutionSet init failed: ") + e.what();
        if (mbDgcLayout != VK_NULL_HANDLE) {
          vContext->destroyIndirectCommandsLayout(mbDgcLayout);
          mbDgcLayout = VK_NULL_HANDLE;
        }
        if (mbDgcExecSet != VK_NULL_HANDLE) {
          vContext->destroyIndirectExecutionSet(mbDgcExecSet);
          mbDgcExecSet = VK_NULL_HANDLE;
        }
        if (mbDgcPreprocessBuffer) {
          context_ref.releaseBuffer(mbDgcPreprocessBuffer);
          mbDgcPreprocessBuffer = nullptr;
        }
        mbDgcAvailable = false;
      }
    } else if (!vContext->isDGCSupported()) {
      mbDgcUnsupportedReason =
          "VK_EXT_device_generated_commands unsupported by device/driver";
    } else {
      mbDgcUnsupportedReason =
          "VK_EXT_device_generated_commands IndirectExecutionSet pipeline binding unsupported for compute";
    }
  }
#endif
  mbSetupComplete = true;
}

void InShaderIndirectBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size() || !context) {
    throw std::runtime_error("InShaderIndirectBench: invalid config or uninitialized state");
  }

  const auto &cfg = configs[config_idx];

  if (cfg.mode == IndirectBenchMode::GPU_Indirect) {
    if (context->getBackend() == ComputeBackend::ROCm) {
      context->dispatch(kernelReset, 1, 1, 1, 1, 1, 1);
      uint32_t selectivity = cfg.selectivity;
      uint32_t totalItems = kTotalItems;
      uint32_t totalWorkgroups = kTotalWorkgroups;
      context->setKernelArg(kernelClassify, 3, sizeof(uint32_t), &selectivity);
      context->setKernelArg(kernelClassify, 4, sizeof(uint32_t), &totalItems);
      context->setKernelArg(kernelClassify, 5, sizeof(uint32_t), &totalWorkgroups);
      context->dispatch(kernelClassify, kTotalWorkgroups, 1, 1, kWorkgroupSize, 1, 1);
      context->dispatch(kernelIndirectWorker, m_rocmPersistentWorkgroups, 1, 1,
                        kWorkgroupSize, 1, 1);
      return;
    }

    struct ClassifyPushConstants {
      uint32_t selectivity;
      uint32_t totalItems;
      uint32_t totalWorkgroups;
    } pcClassify = { cfg.selectivity, kTotalItems, kTotalWorkgroups };

    context->setKernelArg(kernelClassify, 3, sizeof(pcClassify), &pcClassify);

#ifdef HAVE_VULKAN
    auto *vContext = dynamic_cast<VulkanContext *>(context);
    if (vContext) {
      std::vector<VulkanContext::IndirectBatchEntry> entries = {{0, {}, nullptr}};
      vContext->dispatchWorkListSequence(
          kernelReset, kernelClassify, kTotalWorkgroups, 1, 1,
          nullptr, kernelIndirectWorker, indirectBuffer, entries);
      return;
    }
#endif
    context->dispatch(kernelReset, 1, 1, 1, 1, 1, 1);
    context->dispatch(kernelClassify, kTotalWorkgroups, 1, 1, kWorkgroupSize, 1, 1);
  } else if (cfg.mode == IndirectBenchMode::GPU_MultiBucket_Legacy) {
#ifdef HAVE_VULKAN
    auto *vContext = dynamic_cast<VulkanContext *>(context);
    if (!vContext) return;
    struct MbClassifyPC {
      uint32_t totalItems;
      uint32_t numBuckets;
      uint32_t bucketCapacity;
      uint32_t pad;
    } pcMb{kMbTotalItems, kMbNumBuckets, kMbBucketCapacity, 0u};
    context->setKernelArg(kernelMbClassify, 3, sizeof(pcMb), &pcMb);
    vContext->dispatchWorkListSequence(
        nullptr, kernelMbClassify, kMbTotalItems / kWorkgroupSize, 1, 1,
        kernelMbResolve, kernelMbWorker[0], mbIndirectBuffer, mbLegacyEntries,
        false, nullptr, 0u);
#endif
  } else if (cfg.mode == IndirectBenchMode::GPU_MultiBucket_DGC) {
#ifdef HAVE_VULKAN
    if (!mbDgcAvailable) return;
    auto *vContext = dynamic_cast<VulkanContext *>(context);
    if (!vContext) return;
    struct MbClassifyPC {
      uint32_t totalItems;
      uint32_t numBuckets;
      uint32_t bucketCapacity;
      uint32_t pad;
    } pcMb{kMbTotalItems, kMbNumBuckets, kMbBucketCapacity, 0u};
    context->setKernelArg(kernelMbClassify, 3, sizeof(pcMb), &pcMb);
    struct MbResolvePC {
      uint32_t arg0;
      uint32_t dgcMode;
      uint32_t pad;
    } pcRes{kMbNumBuckets, 1u, 0u};
    vContext->dispatchDGCWorkListSequence(
        nullptr, kernelMbClassify, kMbTotalItems / kWorkgroupSize, 1, 1,
        kernelMbResolve, kernelMbWorker[0], mbDgcInfo, &pcRes, sizeof(pcRes));
#endif
  } else {
    if (context->getBackend() == ComputeBackend::ROCm) {
      uint32_t selectivity = cfg.selectivity;
      uint32_t totalItems = kTotalItems;
      context->setKernelArg(kernelDirectWorker, 2, sizeof(uint32_t), &selectivity);
      context->setKernelArg(kernelDirectWorker, 3, sizeof(uint32_t), &totalItems);
      context->dispatch(kernelDirectWorker, kTotalWorkgroups, 1, 1, kWorkgroupSize, 1, 1);
      return;
    }
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
#ifdef HAVE_VULKAN
    if (auto *vContext = dynamic_cast<VulkanContext *>(context)) {
      if (mbDgcLayout != VK_NULL_HANDLE) {
        vContext->destroyIndirectCommandsLayout(mbDgcLayout);
        mbDgcLayout = VK_NULL_HANDLE;
      }
      if (mbDgcExecSet != VK_NULL_HANDLE) {
        vContext->destroyIndirectExecutionSet(mbDgcExecSet);
        mbDgcExecSet = VK_NULL_HANDLE;
      }
    }
    mbLegacyEntries.clear();
    mbDgcInfo = VulkanContext::DGCExecutionInfo{};
#endif
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
    if (kernelMbClassify) {
      context->releaseKernel(kernelMbClassify);
      kernelMbClassify = nullptr;
    }
    if (kernelMbResolve) {
      context->releaseKernel(kernelMbResolve);
      kernelMbResolve = nullptr;
    }
    for (auto &k : kernelMbWorker) {
      if (k) {
        context->releaseKernel(k);
        k = nullptr;
      }
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
    if (mbWorkListBuffer) {
      context->releaseBuffer(mbWorkListBuffer);
      mbWorkListBuffer = nullptr;
    }
    if (mbBucketCountsBuffer) {
      context->releaseBuffer(mbBucketCountsBuffer);
      mbBucketCountsBuffer = nullptr;
    }
    if (mbIndirectBuffer) {
      context->releaseBuffer(mbIndirectBuffer);
      mbIndirectBuffer = nullptr;
    }
    if (mbDgcSeqBuffer) {
      context->releaseBuffer(mbDgcSeqBuffer);
      mbDgcSeqBuffer = nullptr;
    }
    if (mbDgcCountBuffer) {
      context->releaseBuffer(mbDgcCountBuffer);
      mbDgcCountBuffer = nullptr;
    }
    if (mbDgcPreprocessBuffer) {
      context->releaseBuffer(mbDgcPreprocessBuffer);
      mbDgcPreprocessBuffer = nullptr;
    }
    mbSetupComplete = false;
    mbDgcAvailable = false;
    mbDgcUnsupportedReason.clear();
    context = nullptr;
  }
}

BenchmarkResult InShaderIndirectBench::GetResult(uint32_t /*config_idx*/) const {
  return { 1, 0.0 };
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
    } else if (cfg.mode == IndirectBenchMode::GPU_MultiBucket_Legacy ||
               cfg.mode == IndirectBenchMode::GPU_MultiBucket_DGC) {
      if (cfg.mode == IndirectBenchMode::GPU_MultiBucket_DGC && !mbDgcAvailable) {
        return true;
      }
      if (!mbIndirectBuffer || !mbBucketCountsBuffer) {
        return false;
      }
      std::vector<uint32_t> cmds(kMbNumBuckets * 3u, 0u);
      context->readBuffer(mbIndirectBuffer, 0, cmds.size() * sizeof(uint32_t), cmds.data());
      std::vector<uint32_t> snapshots(kMbNumBuckets, 0u);
      context->readBuffer(mbBucketCountsBuffer, kMbNumBuckets * sizeof(uint32_t),
                          snapshots.size() * sizeof(uint32_t), snapshots.data());

      for (uint32_t b = 0; b < kMbNumBuckets; ++b) {
        const uint32_t x = cmds[b * 3u + 0u];
        const uint32_t y = cmds[b * 3u + 1u];
        const uint32_t z = cmds[b * 3u + 2u];
        if (isMbActiveBucket(b)) {
          if (snapshots[b] != 1024u || x != 16u || y != 1u || z != 1u) {
            std::cerr << "InShaderIndirectBench MultiBucket validation failed (bucket "
                      << b << "): snapshot=" << snapshots[b]
                      << ", cmd=(" << x << "," << y << "," << z << ")\n";
            return false;
          }
        } else {
          if (snapshots[b] != 0u || x != 0u || y != 0u || z != 0u) {
            std::cerr << "InShaderIndirectBench MultiBucket validation failed (empty bucket "
                      << b << "): snapshot=" << snapshots[b]
                      << ", cmd=(" << x << "," << y << "," << z << ")\n";
            return false;
          }
        }
      }

      if (cfg.mode == IndirectBenchMode::GPU_MultiBucket_DGC) {
        if (!mbDgcCountBuffer) return false;
        uint32_t seqCount = 0u;
        context->readBuffer(mbDgcCountBuffer, 0, sizeof(seqCount), &seqCount);
        if (seqCount != 4u) {
          std::cerr << "InShaderIndirectBench MultiBucket DGC validation failed: seqCount="
                    << seqCount << ", expected 4 pruned active sequences\n";
          return false;
        }
      }

      constexpr uint32_t kCheckCount = 4096u;
      std::vector<float> results(kCheckCount, 0.0f);
      context->readBuffer(outputBuffer, 0, results.size() * sizeof(float), results.data());
      uint32_t activeChecked = 0u;
      for (uint32_t gid = 0; gid < kCheckCount; ++gid) {
        uint32_t bucket = (gid * 2654435761u) & 127u;
        if (isMbActiveBucket(bucket)) {
          float val = results[gid];
          float expectedVal = computeMbExpectedWorkerVal(gid, bucket & 7u);
          if (!std::isfinite(val) || std::abs(val - expectedVal) > 0.05f) {
            std::cerr << "InShaderIndirectBench MultiBucket validation failed (config "
                      << config_idx << "): outputBuffer[" << gid << "] = "
                      << val << ", expected " << expectedVal << "\n";
            return false;
          }
          activeChecked++;
        }
      }
      if (activeChecked == 0u) {
        return false;
      }
    } else {
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
