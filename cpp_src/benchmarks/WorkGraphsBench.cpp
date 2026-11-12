#include "WorkGraphsBench.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {

// Validated SPV_AMDX_shader_enqueue module (see shaders/workgraphs_amdx.spvasm)
const uint32_t kWorkGraphAmdxSpv[] = {
    0x07230203u, 0x00010600u, 0x00070000u, 0x00000064u, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
    0x000013CBu, 0x0007000Au, 0x5F565053u, 0x58444D41u, 0x6168735Fu, 0x5F726564u, 0x75716E65u, 0x00657565u,
    0x0006000Bu, 0x00000001u, 0x4C534C47u, 0x6474732Eu, 0x3035342Eu, 0x00000000u, 0x0003000Eu, 0x00000000u,
    0x00000001u, 0x0009000Fu, 0x00000005u, 0x00000002u, 0x646F7270u, 0x72656375u, 0x65646F4Eu, 0x00000000u,
    0x00000003u, 0x00000004u, 0x000A000Fu, 0x00000005u, 0x00000005u, 0x736E6F63u, 0x72656D75u, 0x65646F4Eu,
    0x00000000u, 0x00000006u, 0x00000007u, 0x00000008u, 0x00060010u, 0x00000002u, 0x00000011u, 0x00000100u,
    0x00000001u, 0x00000001u, 0x0004014Bu, 0x00000002u, 0x000013CEu, 0x00000009u, 0x0004014Bu, 0x00000002u,
    0x000013D1u, 0x0000000Au, 0x0006014Bu, 0x00000002u, 0x000013D0u, 0x0000000Bu, 0x0000000Cu, 0x0000000Cu,
    0x00060010u, 0x00000005u, 0x00000011u, 0x00000040u, 0x00000001u, 0x00000001u, 0x00030010u, 0x00000005u,
    0x000013CDu, 0x0004014Bu, 0x00000005u, 0x000013D1u, 0x0000000Cu, 0x00040047u, 0x00000003u, 0x0000000Bu,
    0x0000001Cu, 0x00040047u, 0x00000006u, 0x0000000Bu, 0x0000001Du, 0x00030047u, 0x0000000Du, 0x00000002u,
    0x00050048u, 0x0000000Du, 0x00000000u, 0x00000023u, 0x00000000u, 0x00050048u, 0x0000000Du, 0x00000001u,
    0x00000023u, 0x00000004u, 0x00050048u, 0x0000000Du, 0x00000002u, 0x00000023u, 0x00000008u, 0x0004014Cu,
    0x00000004u, 0x000013EAu, 0x0000000Cu, 0x0004014Cu, 0x00000004u, 0x000013ECu, 0x0000000Cu, 0x0004014Cu,
    0x00000007u, 0x0000139Cu, 0x0000000Eu, 0x00030047u, 0x00000004u, 0x000013D6u, 0x00040047u, 0x0000000Fu,
    0x00000006u, 0x00000004u, 0x00030047u, 0x00000010u, 0x00000002u, 0x00050048u, 0x00000010u, 0x00000000u,
    0x00000023u, 0x00000000u, 0x00040047u, 0x00000008u, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000008u,
    0x00000021u, 0x00000000u, 0x00020013u, 0x00000011u, 0x00030021u, 0x00000012u, 0x00000011u, 0x00040015u,
    0x00000013u, 0x00000020u, 0x00000000u, 0x00040017u, 0x00000014u, 0x00000013u, 0x00000003u, 0x00030016u,
    0x00000015u, 0x00000020u, 0x00020014u, 0x00000016u, 0x00030029u, 0x00000016u, 0x00000009u, 0x0004002Bu,
    0x00000013u, 0x0000000Au, 0x00000000u, 0x0004002Bu, 0x00000013u, 0x0000000Cu, 0x00000001u, 0x0004002Bu,
    0x00000013u, 0x00000017u, 0x00000002u, 0x0004002Bu, 0x00000013u, 0x00000018u, 0x00000004u, 0x0004002Bu,
    0x00000013u, 0x00000019u, 0x00000016u, 0x0004002Bu, 0x00000013u, 0x0000001Au, 0x0000001Cu, 0x0004002Bu,
    0x00000013u, 0x0000001Bu, 0x00000020u, 0x0004002Bu, 0x00000013u, 0x0000000Eu, 0x00000040u, 0x0004002Bu,
    0x00000013u, 0x0000001Cu, 0x000000FFu, 0x0004002Bu, 0x00000013u, 0x0000000Bu, 0x00000400u, 0x0004002Bu,
    0x00000013u, 0x0000001Du, 0x0000FFFFu, 0x0004002Bu, 0x00000013u, 0x0000001Eu, 0x2C9277B5u, 0x0004002Bu,
    0x00000013u, 0x0000001Fu, 0xAC564B05u, 0x0004002Bu, 0x00000013u, 0x00000020u, 0x108EF2D9u, 0x0004002Bu,
    0x00000015u, 0x00000021u, 0x39000000u, 0x0004002Bu, 0x00000015u, 0x00000022u, 0x3E000000u, 0x0004002Bu,
    0x00000015u, 0x00000023u, 0x3F7FE000u, 0x0004002Bu, 0x00000015u, 0x00000024u, 0x3A800000u, 0x0005001Eu,
    0x0000000Du, 0x00000013u, 0x00000013u, 0x00000013u, 0x000313D4u, 0x00000025u, 0x0000000Du, 0x00040020u,
    0x00000026u, 0x000013CCu, 0x00000025u, 0x00040020u, 0x00000027u, 0x000013CCu, 0x00000013u, 0x00040020u,
    0x00000028u, 0x00000001u, 0x00000014u, 0x00040020u, 0x00000029u, 0x00000001u, 0x00000013u, 0x0003001Du,
    0x0000000Fu, 0x00000015u, 0x0003001Eu, 0x00000010u, 0x0000000Fu, 0x00040020u, 0x0000002Au, 0x0000000Cu,
    0x00000010u, 0x00040020u, 0x0000002Bu, 0x0000000Cu, 0x00000015u, 0x0004003Bu, 0x00000028u, 0x00000003u,
    0x00000001u, 0x0004003Bu, 0x00000029u, 0x00000006u, 0x00000001u, 0x0004003Bu, 0x00000026u, 0x00000004u,
    0x000013CCu, 0x0004003Bu, 0x00000026u, 0x00000007u, 0x000013CCu, 0x0004003Bu, 0x0000002Au, 0x00000008u,
    0x0000000Cu, 0x00050036u, 0x00000011u, 0x00000002u, 0x00000000u, 0x00000012u, 0x000200F8u, 0x0000002Cu,
    0x00050041u, 0x00000029u, 0x0000002Du, 0x00000003u, 0x0000000Au, 0x0004003Du, 0x00000013u, 0x0000002Eu,
    0x0000002Du, 0x00050084u, 0x00000013u, 0x0000002Fu, 0x0000002Eu, 0x0000001Eu, 0x00050080u, 0x00000013u,
    0x00000030u, 0x0000002Fu, 0x0000001Fu, 0x000500C2u, 0x00000013u, 0x00000031u, 0x00000030u, 0x0000001Au,
    0x00050080u, 0x00000013u, 0x00000032u, 0x00000031u, 0x00000018u, 0x000500C2u, 0x00000013u, 0x00000033u,
    0x00000030u, 0x00000032u, 0x000500C6u, 0x00000013u, 0x00000034u, 0x00000033u, 0x00000030u, 0x00050084u,
    0x00000013u, 0x00000035u, 0x00000034u, 0x00000020u, 0x000500C2u, 0x00000013u, 0x00000036u, 0x00000035u,
    0x00000019u, 0x000500C6u, 0x00000013u, 0x00000037u, 0x00000036u, 0x00000035u, 0x000500C7u, 0x00000013u,
    0x00000038u, 0x00000037u, 0x0000000Cu, 0x000500AAu, 0x00000016u, 0x00000039u, 0x00000038u, 0x0000000Au,
    0x000600A9u, 0x00000013u, 0x0000003Au, 0x00000039u, 0x0000000Cu, 0x0000000Au, 0x000613D2u, 0x00000026u,
    0x0000003Bu, 0x00000017u, 0x0000003Au, 0x0000000Cu, 0x000300F7u, 0x0000003Cu, 0x00000000u, 0x000400FAu,
    0x00000039u, 0x0000003Du, 0x0000003Eu, 0x000200F8u, 0x0000003Du, 0x00060041u, 0x00000027u, 0x0000003Fu,
    0x0000003Bu, 0x0000000Au, 0x0000000Au, 0x00060041u, 0x00000027u, 0x00000040u, 0x0000003Bu, 0x0000000Au,
    0x0000000Cu, 0x00060041u, 0x00000027u, 0x00000041u, 0x0000003Bu, 0x0000000Au, 0x00000017u, 0x000500C5u,
    0x00000013u, 0x00000042u, 0x00000037u, 0x0000000Cu, 0x000500C7u, 0x00000013u, 0x00000043u, 0x0000002Eu,
    0x0000001Cu, 0x00050080u, 0x00000013u, 0x00000044u, 0x00000043u, 0x0000000Cu, 0x0003003Eu, 0x0000003Fu,
    0x0000002Eu, 0x0003003Eu, 0x00000040u, 0x00000042u, 0x0003003Eu, 0x00000041u, 0x00000044u, 0x000200F9u,
    0x0000003Eu, 0x000200F8u, 0x0000003Eu, 0x000413D6u, 0x00000016u, 0x00000045u, 0x0000003Bu, 0x000300F7u,
    0x00000046u, 0x00000000u, 0x000400FAu, 0x00000045u, 0x00000047u, 0x00000046u, 0x000200F8u, 0x00000047u,
    0x000213D3u, 0x0000003Bu, 0x000200F9u, 0x00000046u, 0x000200F8u, 0x00000046u, 0x000200F9u, 0x0000003Cu,
    0x000200F8u, 0x0000003Cu, 0x000100FDu, 0x00010038u, 0x00050036u, 0x00000011u, 0x00000005u, 0x00000000u,
    0x00000012u, 0x000200F8u, 0x00000048u, 0x0004003Du, 0x00000013u, 0x00000049u, 0x00000006u, 0x000413E2u,
    0x00000013u, 0x0000004Au, 0x00000007u, 0x000500B0u, 0x00000016u, 0x0000004Bu, 0x00000049u, 0x0000004Au,
    0x000300F7u, 0x0000004Cu, 0x00000000u, 0x000400FAu, 0x0000004Bu, 0x0000004Du, 0x0000004Cu, 0x000200F8u,
    0x0000004Du, 0x00060041u, 0x00000027u, 0x0000004Eu, 0x00000007u, 0x00000049u, 0x0000000Au, 0x00060041u,
    0x00000027u, 0x0000004Fu, 0x00000007u, 0x00000049u, 0x0000000Cu, 0x00060041u, 0x00000027u, 0x00000050u,
    0x00000007u, 0x00000049u, 0x00000017u, 0x0004003Du, 0x00000013u, 0x00000051u, 0x0000004Eu, 0x0004003Du,
    0x00000013u, 0x00000052u, 0x0000004Fu, 0x0004003Du, 0x00000013u, 0x00000053u, 0x00000050u, 0x000500C7u,
    0x00000013u, 0x00000054u, 0x00000052u, 0x0000001Du, 0x00040070u, 0x00000015u, 0x00000055u, 0x00000054u,
    0x00050085u, 0x00000015u, 0x00000056u, 0x00000055u, 0x00000021u, 0x00040070u, 0x00000015u, 0x00000057u,
    0x00000053u, 0x00050085u, 0x00000015u, 0x00000058u, 0x00000057u, 0x00000022u, 0x00050081u, 0x00000015u,
    0x00000059u, 0x00000056u, 0x00000058u, 0x000200F9u, 0x0000005Au, 0x000200F8u, 0x0000005Au, 0x000700F5u,
    0x00000013u, 0x0000005Bu, 0x0000000Au, 0x0000004Du, 0x0000005Cu, 0x0000005Du, 0x000700F5u, 0x00000015u,
    0x0000005Eu, 0x00000059u, 0x0000004Du, 0x0000005Fu, 0x0000005Du, 0x000500B0u, 0x00000016u, 0x00000060u,
    0x0000005Bu, 0x0000001Bu, 0x000400F6u, 0x00000061u, 0x0000005Du, 0x00000000u, 0x000400FAu, 0x00000060u,
    0x00000062u, 0x00000061u, 0x000200F8u, 0x00000062u, 0x0008000Cu, 0x00000015u, 0x0000005Fu, 0x00000001u,
    0x00000032u, 0x0000005Eu, 0x00000023u, 0x00000024u, 0x000200F9u, 0x0000005Du, 0x000200F8u, 0x0000005Du,
    0x00050080u, 0x00000013u, 0x0000005Cu, 0x0000005Bu, 0x0000000Cu, 0x000200F9u, 0x0000005Au, 0x000200F8u,
    0x00000061u, 0x00060041u, 0x0000002Bu, 0x00000063u, 0x00000008u, 0x0000000Au, 0x00000051u, 0x0003003Eu,
    0x00000063u, 0x0000005Eu, 0x000200F9u, 0x0000004Cu, 0x000200F8u, 0x0000004Cu, 0x000100FDu, 0x00010038u};

uint32_t pcgClassifyHash(uint32_t gid) {
  uint32_t hash = (gid * 747796405u + 2891336453u);
  hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
  hash = (hash >> 22u) ^ hash;
  return hash;
}

float expectedIndirectConsumerVal(uint32_t gid, uint32_t hash) {
  float v = static_cast<float>((hash | 1u) & 0xFFFFu) * 0.0001220703125f +
            static_cast<float>((gid & 255u) + 1u) * 0.125f;
  for (uint32_t i = 0; i < 32u; ++i) {
    v = std::fma(v, 0.99951171875f, 0.0009765625f);
  }
  return v;
}

} // namespace

std::string WorkGraphsBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Indirect Dispatch Producer-Consumer (Baseline)";
  case 1:
    return "Device-Generated Commands Producer-Consumer (DGC)";
  case 2:
    return "Hardware GPU Work Graph (VK_AMDX_shader_enqueue)";
  default:
    return "";
  }
}

const char *WorkGraphsBench::GetDescription(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Two-pass producer-consumer pipeline using atomic work-queue compaction + vkCmdDispatchIndirect.";
  case 1:
    return "Two-pass producer-consumer pipeline using VK_EXT_device_generated_commands GPU command preprocessing + execution.";
  case 2:
    return "Single-submission hardware GPU Work Graph (VK_AMDX_shader_enqueue) with on-chip producer-to-coalescing-consumer payload enqueue.";
  default:
    return "GPU producer-consumer work graph scheduling benchmark.";
  }
}

bool WorkGraphsBench::IsSupported(const DeviceInfo & /*info*/,
                                  IComputeContext *context) const {
  if (!context) return true;
  return context->getBackend() == ComputeBackend::Vulkan;
}

bool WorkGraphsBench::IsConfigSupported(uint32_t config_idx,
                                        const DeviceInfo &info,
                                        IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) return false;
  const auto *vc = dynamic_cast<const VulkanContext *>(context);
  if (config_idx == 0) return true;
  if (config_idx == 1) {
    return vc ? vc->isDGCSupported() : info.dgcSupported;
  }
  if (config_idx == 2) {
    return vc ? vc->isWorkGraphsSupported() : info.workGraphsSupported;
  }
  return false;
}

IBenchmark::SupportLimitation
WorkGraphsBench::GetSupportLimitation(const DeviceInfo & /*info*/,
                                      IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  return SupportLimitation::kNone;
}

IBenchmark::SupportLimitation WorkGraphsBench::GetConfigSupportLimitation(
    uint32_t config_idx, const DeviceInfo & /*info*/,
    IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if (config_idx == 1 || config_idx == 2) {
    return SupportLimitation::kApi;
  }
  return SupportLimitation::kNone;
}

std::string WorkGraphsBench::GetSupportNote(const DeviceInfo & /*info*/,
                                            IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4 compute pipeline";
  }
  return "";
}

std::string WorkGraphsBench::GetConfigSupportNote(
    uint32_t config_idx, const DeviceInfo & /*info*/,
    IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4 backend";
  }
  if (config_idx == 1) {
    return "Requires VK_EXT_device_generated_commands";
  }
  if (config_idx == 2) {
    return "Requires VK_AMDX_shader_enqueue (GPU Work Graphs)";
  }
  return "";
}

void WorkGraphsBench::Setup(IComputeContext &context,
                            const std::string &kernel_dir) {
  m_vk = dynamic_cast<VulkanContext *>(&context);
  if (!m_vk) {
    throw std::runtime_error("WorkGraphsBench requires VulkanContext");
  }
  m_kernelDir = kernel_dir;

  std::filesystem::path vkDir = std::filesystem::path(kernel_dir) / "vulkan";
  m_producerKernel = m_vk->createKernel(
      (vkDir / "workgraphs_producer.comp").string(), "main", 2);
  m_consumerKernel = m_vk->createKernel(
      (vkDir / "workgraphs_consumer.comp").string(), "main", 3);
}

void WorkGraphsBench::PrepareConfig(uint32_t config_idx) {
  ReleaseConfigResources(config_idx);
  if (!m_vk) return;

  uint32_t initHeader[4] = {0u, 1u, 1u, 0u};
  m_indirectBuf = m_vk->createBuffer(sizeof(initHeader), initHeader);
  m_payloadQueueBuf =
      m_vk->createBuffer(static_cast<size_t>(kTotalItems) * sizeof(uint32_t) * 4u);
  std::vector<float> zeroOut(kTotalItems, 0.0f);
  m_outBuf =
      m_vk->createBuffer(zeroOut.size() * sizeof(float), zeroOut.data());

  m_vk->setKernelArg(m_producerKernel, 0, m_indirectBuf);
  m_vk->setKernelArg(m_producerKernel, 1, m_payloadQueueBuf);

  m_vk->setKernelArg(m_consumerKernel, 0, m_indirectBuf);
  m_vk->setKernelArg(m_consumerKernel, 1, m_payloadQueueBuf);
  m_vk->setKernelArg(m_consumerKernel, 2, m_outBuf);

  if (config_idx == 1 && m_vk->isDGCSupported()) {
    VkIndirectCommandsLayoutTokenEXT token{
        VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT};
    token.type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
    token.offset = 0;

    VkIndirectCommandsLayoutCreateInfoEXT layoutCi{
        VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT};
    layoutCi.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutCi.indirectStride = sizeof(VkDispatchIndirectCommand);
    layoutCi.pipelineLayout = m_vk->getVkPipelineLayout(m_consumerKernel);
    layoutCi.tokenCount = 1;
    layoutCi.pTokens = &token;
    m_dgcLayout = m_vk->createIndirectCommandsLayout(layoutCi);

    VkDeviceSize preSize = m_vk->getGeneratedCommandsMemoryRequirements(
        m_dgcLayout, VK_NULL_HANDLE, 1, m_consumerKernel);
    m_dgcPreprocessSize = std::max<VkDeviceSize>(preSize, 4096);
    m_dgcPreprocessBuf =
        m_vk->createPreprocessBuffer(static_cast<size_t>(m_dgcPreprocessSize));
    uint32_t oneSeq = 1u;
    m_dgcCountBuf = m_vk->createBuffer(sizeof(uint32_t), &oneSeq);
  } else if (config_idx == 2 && m_vk->isWorkGraphsSupported()) {
    VkDevice dev = m_vk->getDevice();
    m_pfnCreateGraphPipelines =
        reinterpret_cast<PFN_vkCreateExecutionGraphPipelinesAMDX>(
            vkGetDeviceProcAddr(dev, "vkCreateExecutionGraphPipelinesAMDX"));
    m_pfnGetGraphScratchSize =
        reinterpret_cast<PFN_vkGetExecutionGraphPipelineScratchSizeAMDX>(
            vkGetDeviceProcAddr(dev, "vkGetExecutionGraphPipelineScratchSizeAMDX"));
    m_pfnGetGraphNodeIndex =
        reinterpret_cast<PFN_vkGetExecutionGraphPipelineNodeIndexAMDX>(
            vkGetDeviceProcAddr(dev, "vkGetExecutionGraphPipelineNodeIndexAMDX"));
    m_pfnCmdInitGraphScratch =
        reinterpret_cast<PFN_vkCmdInitializeGraphScratchMemoryAMDX>(
            vkGetDeviceProcAddr(dev, "vkCmdInitializeGraphScratchMemoryAMDX"));
    m_pfnCmdDispatchGraph =
        reinterpret_cast<PFN_vkCmdDispatchGraphAMDX>(
            vkGetDeviceProcAddr(dev, "vkCmdDispatchGraphAMDX"));
    if (!m_pfnCreateGraphPipelines || !m_pfnGetGraphScratchSize ||
        !m_pfnCmdInitGraphScratch || !m_pfnCmdDispatchGraph) {
      throw std::runtime_error(
          "WorkGraphsBench: VK_AMDX_shader_enqueue entry points unavailable");
    }

    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0;
    b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b0.descriptorCount = 1;
    b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslCi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslCi.bindingCount = 1;
    dslCi.pBindings = &b0;
    vkCreateDescriptorSetLayout(dev, &dslCi, nullptr, &m_wgSetLayout);

    VkPipelineLayoutCreateInfo plCi{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plCi.setLayoutCount = 1;
    plCi.pSetLayouts = &m_wgSetLayout;
    vkCreatePipelineLayout(dev, &plCi, nullptr, &m_wgPipeLayout);

    std::vector<uint32_t> spvWords;
    std::filesystem::path spvPath =
        std::filesystem::path(m_kernelDir) / "vulkan" / "workgraphs_amdx.comp.spv";
    std::ifstream spvFile(spvPath, std::ios::binary | std::ios::ate);
    if (spvFile.is_open()) {
      std::streamsize sz = spvFile.tellg();
      if (sz > 0 && (sz % 4) == 0) {
        spvWords.resize(static_cast<size_t>(sz) / 4u);
        spvFile.seekg(0, std::ios::beg);
        spvFile.read(reinterpret_cast<char *>(spvWords.data()), sz);
      }
    }

    VkShaderModuleCreateInfo smCi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    if (!spvWords.empty()) {
      smCi.codeSize = spvWords.size() * sizeof(uint32_t);
      smCi.pCode = spvWords.data();
    } else {
      smCi.codeSize = sizeof(kWorkGraphAmdxSpv);
      smCi.pCode = kWorkGraphAmdxSpv;
    }
    VkShaderModule wgMod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(dev, &smCi, nullptr, &wgMod) != VK_SUCCESS) {
      throw std::runtime_error("WorkGraphsBench: failed to create AMDX shader module");
    }

    VkPipelineShaderStageNodeCreateInfoAMDX nodeInfo0{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NODE_CREATE_INFO_AMDX};
    nodeInfo0.pName = "producerNode";
    nodeInfo0.index = 0;

    VkPipelineShaderStageNodeCreateInfoAMDX nodeInfo1{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NODE_CREATE_INFO_AMDX};
    nodeInfo1.pName = "consumerNode";
    nodeInfo1.index = 1;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].pNext = &nodeInfo0;
    stages[0].stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stages[0].module = wgMod;
    stages[0].pName = "producerNode";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].pNext = &nodeInfo1;
    stages[1].stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stages[1].module = wgMod;
    stages[1].pName = "consumerNode";

    VkExecutionGraphPipelineCreateInfoAMDX egCi{
        VK_STRUCTURE_TYPE_EXECUTION_GRAPH_PIPELINE_CREATE_INFO_AMDX};
    egCi.stageCount = 2;
    egCi.pStages = stages;
    egCi.layout = m_wgPipeLayout;

    VkResult egRes = m_pfnCreateGraphPipelines(
        dev, m_vk->getPipelineCache(), 1, &egCi, nullptr, &m_wgPipeline);
    vkDestroyShaderModule(dev, wgMod, nullptr);
    if (egRes != VK_SUCCESS) {
      throw std::runtime_error("WorkGraphsBench: vkCreateExecutionGraphPipelinesAMDX failed");
    }

    m_wgProducerNodeIndex = 0;
    if (m_pfnGetGraphNodeIndex) {
      VkPipelineShaderStageNodeCreateInfoAMDX queryNode{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NODE_CREATE_INFO_AMDX};
      queryNode.pName = "producerNode";
      queryNode.index = 0;
      m_pfnGetGraphNodeIndex(dev, m_wgPipeline, &queryNode,
                             &m_wgProducerNodeIndex);
    }

    VkExecutionGraphPipelineScratchSizeAMDX scratchSz{
        VK_STRUCTURE_TYPE_EXECUTION_GRAPH_PIPELINE_SCRATCH_SIZE_AMDX};
    m_pfnGetGraphScratchSize(dev, m_wgPipeline, &scratchSz);
    m_wgScratchSize = std::max<VkDeviceSize>(scratchSz.minSize, 4096);

    VkBufferUsageFlags2CreateInfoKHR usage2{
        VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR};
    usage2.usage = VK_BUFFER_USAGE_2_EXECUTION_GRAPH_SCRATCH_BIT_AMDX |
                   VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR;
    VkBufferCreateInfo sbCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sbCi.pNext = &usage2;
    sbCi.size = m_wgScratchSize;
    sbCi.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    sbCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo vmaCi{};
    vmaCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateBuffer(m_vk->getVmaAllocator(), &sbCi, &vmaCi,
                        &m_wgScratchBuf, &m_wgScratchAlloc,
                        nullptr) != VK_SUCCESS) {
      throw std::runtime_error("WorkGraphsBench: failed to allocate graph scratch");
    }
    VkBufferDeviceAddressInfo bdaInfo{
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bdaInfo.buffer = m_wgScratchBuf;
    m_wgScratchAddr = vkGetBufferDeviceAddress(dev, &bdaInfo);

    VkDescriptorPoolSize poolSz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpCi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpCi.maxSets = 1;
    dpCi.poolSizeCount = 1;
    dpCi.pPoolSizes = &poolSz;
    vkCreateDescriptorPool(dev, &dpCi, nullptr, &m_wgDescPool);

    VkDescriptorSetAllocateInfo dsAi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAi.descriptorPool = m_wgDescPool;
    dsAi.descriptorSetCount = 1;
    dsAi.pSetLayouts = &m_wgSetLayout;
    vkAllocateDescriptorSets(dev, &dsAi, &m_wgDescSet);

    VkDescriptorBufferInfo dbi{m_vk->getVkBuffer(m_outBuf), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = m_wgDescSet;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

    VkCommandPoolCreateInfo cpCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpCi.queueFamilyIndex = m_vk->getComputeQueueFamilyIndex();
    cpCi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(dev, &cpCi, nullptr, &m_wgCmdPool);

    VkCommandBufferAllocateInfo cbAi{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAi.commandPool = m_wgCmdPool;
    cbAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAi.commandBufferCount = 1;
    vkAllocateCommandBuffers(dev, &cbAi, &m_wgCmd);

    VkFenceCreateInfo fCi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(dev, &fCi, nullptr, &m_wgFence);

    VkQueryPoolCreateInfo qpCi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpCi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpCi.queryCount = 2;
    vkCreateQueryPool(dev, &qpCi, nullptr, &m_wgTsPool);

    // Initialize graph scratch memory once
    vkResetCommandBuffer(m_wgCmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_wgCmd, &bi);
    vkCmdBindPipeline(m_wgCmd, VK_PIPELINE_BIND_POINT_EXECUTION_GRAPH_AMDX,
                      m_wgPipeline);
    m_pfnCmdInitGraphScratch(m_wgCmd, m_wgPipeline, m_wgScratchAddr,
                             m_wgScratchSize);
    vkEndCommandBuffer(m_wgCmd);
    vkResetFences(dev, 1, &m_wgFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_wgCmd;
    {
      std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
      vkQueueSubmit(m_vk->getComputeQueue(), 1, &si, m_wgFence);
    }
    vkWaitForFences(dev, 1, &m_wgFence, VK_TRUE, UINT64_MAX);
  }
}

void WorkGraphsBench::ReleaseConfigResources(uint32_t /*config_idx*/) {
  if (!m_vk) return;
  m_vk->waitIdle();
  VkDevice dev = m_vk->getDevice();

  if (m_dgcLayout != VK_NULL_HANDLE) {
    m_vk->destroyIndirectCommandsLayout(m_dgcLayout);
    m_dgcLayout = VK_NULL_HANDLE;
  }
  if (m_dgcPreprocessBuf) {
    m_vk->releaseBuffer(m_dgcPreprocessBuf);
    m_dgcPreprocessBuf = nullptr;
  }
  if (m_dgcCountBuf) {
    m_vk->releaseBuffer(m_dgcCountBuf);
    m_dgcCountBuf = nullptr;
  }
  if (m_indirectBuf) {
    m_vk->releaseBuffer(m_indirectBuf);
    m_indirectBuf = nullptr;
  }
  if (m_payloadQueueBuf) {
    m_vk->releaseBuffer(m_payloadQueueBuf);
    m_payloadQueueBuf = nullptr;
  }
  if (m_outBuf) {
    m_vk->releaseBuffer(m_outBuf);
    m_outBuf = nullptr;
  }

  if (m_wgTsPool != VK_NULL_HANDLE) {
    vkDestroyQueryPool(dev, m_wgTsPool, nullptr);
    m_wgTsPool = VK_NULL_HANDLE;
  }
  if (m_wgFence != VK_NULL_HANDLE) {
    vkDestroyFence(dev, m_wgFence, nullptr);
    m_wgFence = VK_NULL_HANDLE;
  }
  if (m_wgCmdPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(dev, m_wgCmdPool, nullptr);
    m_wgCmdPool = VK_NULL_HANDLE;
    m_wgCmd = VK_NULL_HANDLE;
  }
  if (m_wgScratchBuf != VK_NULL_HANDLE &&
      m_vk->getVmaAllocator() != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_vk->getVmaAllocator(), m_wgScratchBuf, m_wgScratchAlloc);
    m_wgScratchBuf = VK_NULL_HANDLE;
    m_wgScratchAlloc = VK_NULL_HANDLE;
  }
  if (m_wgPipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(dev, m_wgPipeline, nullptr);
    m_wgPipeline = VK_NULL_HANDLE;
  }
  if (m_wgPipeLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(dev, m_wgPipeLayout, nullptr);
    m_wgPipeLayout = VK_NULL_HANDLE;
  }
  if (m_wgDescPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(dev, m_wgDescPool, nullptr);
    m_wgDescPool = VK_NULL_HANDLE;
  }
  if (m_wgSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(dev, m_wgSetLayout, nullptr);
    m_wgSetLayout = VK_NULL_HANDLE;
  }
}

void WorkGraphsBench::Run(uint32_t config_idx) {
  if (!m_vk) return;

  if (config_idx == 0) {
    uint32_t total = kTotalItems;
    uint32_t resetMode = 0u;
    m_vk->setKernelArg(m_producerKernel, 2, sizeof(uint32_t), &total);
    m_vk->setKernelArg(m_producerKernel, 3, sizeof(uint32_t), &resetMode);
    m_vk->dispatch(m_producerKernel, 1, 1, 1, 256, 1, 1);

    uint32_t classifyMode = 1u;
    m_vk->setKernelArg(m_producerKernel, 2, sizeof(uint32_t), &total);
    m_vk->setKernelArg(m_producerKernel, 3, sizeof(uint32_t), &classifyMode);
    m_vk->dispatch(m_producerKernel, kTotalItems / 256u, 1, 1, 256, 1, 1);

    m_vk->dispatchIndirect(m_consumerKernel, m_indirectBuf, 0);
  } else if (config_idx == 1) {
    uint32_t total = kTotalItems;
    uint32_t resetMode = 0u;
    m_vk->setKernelArg(m_producerKernel, 2, sizeof(uint32_t), &total);
    m_vk->setKernelArg(m_producerKernel, 3, sizeof(uint32_t), &resetMode);
    m_vk->dispatch(m_producerKernel, 1, 1, 1, 256, 1, 1);

    uint32_t classifyMode = 1u;
    m_vk->setKernelArg(m_producerKernel, 2, sizeof(uint32_t), &total);
    m_vk->setKernelArg(m_producerKernel, 3, sizeof(uint32_t), &classifyMode);
    m_vk->dispatch(m_producerKernel, kTotalItems / 256u, 1, 1, 256, 1, 1);

    VulkanContext::DGCExecutionInfo dgcInfo{};
    dgcInfo.layout = m_dgcLayout;
    dgcInfo.executionSet = VK_NULL_HANDLE;
    dgcInfo.sequenceBuffer = m_indirectBuf;
    dgcInfo.sequenceBufferOffset = 0;
    dgcInfo.sequenceBufferSize = sizeof(VkDispatchIndirectCommand);
    dgcInfo.sequenceCountBuffer = m_dgcCountBuf;
    dgcInfo.sequenceCountBufferOffset = 0;
    dgcInfo.preprocessBuffer = m_dgcPreprocessBuf;
    dgcInfo.preprocessBufferSize = m_dgcPreprocessSize;
    dgcInfo.maxSequenceCount = 1;
    m_vk->dispatchDGCSequence(m_consumerKernel, dgcInfo);
  } else if (config_idx == 2 && m_wgPipeline != VK_NULL_HANDLE) {
    VkDevice dev = m_vk->getDevice();
    double tsPeriodNs = static_cast<double>(
        m_vk->getPhysicalDeviceProperties().limits.timestampPeriod);

    vkResetCommandBuffer(m_wgCmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_wgCmd, &bi);
    vkCmdResetQueryPool(m_wgCmd, m_wgTsPool, 0, 2);
    vkCmdWriteTimestamp(m_wgCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_wgTsPool,
                        0);
    vkCmdBindPipeline(m_wgCmd, VK_PIPELINE_BIND_POINT_EXECUTION_GRAPH_AMDX,
                      m_wgPipeline);
    vkCmdBindDescriptorSets(m_wgCmd,
                            VK_PIPELINE_BIND_POINT_EXECUTION_GRAPH_AMDX,
                            m_wgPipeLayout, 0, 1, &m_wgDescSet, 0, nullptr);

    VkDispatchGraphInfoAMDX dgInfo{};
    dgInfo.nodeIndex = m_wgProducerNodeIndex;
    dgInfo.payloadCount = 1;
    dgInfo.payloads.deviceAddress = 0;
    dgInfo.payloadStride = 0;

    VkDispatchGraphCountInfoAMDX countInfo{};
    countInfo.count = 1;
    countInfo.infos.hostAddress = &dgInfo;
    countInfo.stride = sizeof(VkDispatchGraphInfoAMDX);

    m_pfnCmdDispatchGraph(m_wgCmd, m_wgScratchAddr, m_wgScratchSize,
                          &countInfo);
    vkCmdWriteTimestamp(m_wgCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        m_wgTsPool, 1);
    vkEndCommandBuffer(m_wgCmd);

    vkResetFences(dev, 1, &m_wgFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_wgCmd;
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
      vkQueueSubmit(m_vk->getComputeQueue(), 1, &si, m_wgFence);
    }
    vkWaitForFences(dev, 1, &m_wgFence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();

    uint64_t ts[2] = {0, 0};
    if (tsPeriodNs > 0.0 &&
        vkGetQueryPoolResults(dev, m_wgTsPool, 0, 2, sizeof(ts), ts,
                              sizeof(uint64_t),
                              VK_QUERY_RESULT_64_BIT |
                                  VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS &&
        ts[1] > ts[0]) {
      m_lastGraphDurationMs =
          static_cast<double>(ts[1] - ts[0]) * tsPeriodNs * 1e-6;
    } else {
      m_lastGraphDurationMs =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
  }
}

BenchmarkResult WorkGraphsBench::GetResult(uint32_t config_idx) const {
  return {kTotalItems, GetLastRunDurationMs(config_idx)};
}

bool WorkGraphsBench::ValidateResults(uint32_t config_idx) const {
  if (!m_vk || !m_outBuf) return false;
  constexpr uint32_t kCheckCount = 4096u;
  std::vector<float> out(kCheckCount, -1.0f);
  m_vk->readBuffer(m_outBuf, 0, kCheckCount * sizeof(float), out.data());

  if (config_idx <= 2u) {
    uint32_t activeChecked = 0;
    for (uint32_t gid = 0; gid < kCheckCount; ++gid) {
      if (!std::isfinite(out[gid])) return false;
      uint32_t h = pcgClassifyHash(gid);
      if ((h & 1u) == 0u) {
        float expVal = expectedIndirectConsumerVal(gid, h);
        if (std::fabs(out[gid] - expVal) > 1e-3f) return false;
        activeChecked++;
      } else {
        if (std::fabs(out[gid]) > 1e-6f) return false;
      }
    }
    return activeChecked > 1000u;
  }
  return false;
}

void WorkGraphsBench::Teardown() {
  ReleaseConfigResources(0);
  if (m_vk) {
    if (m_producerKernel) m_vk->releaseKernel(m_producerKernel);
    if (m_consumerKernel) m_vk->releaseKernel(m_consumerKernel);
  }
  m_producerKernel = nullptr;
  m_consumerKernel = nullptr;
  m_vk = nullptr;
}
