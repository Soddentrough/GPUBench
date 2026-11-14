#include "AsyncOverlapBench.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

std::string AsyncOverlapBench::GetConfigName(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Serial Graphics + Compute Baseline";
  case 1:
    return "Concurrent Graphics + Async Compute (ACE)";
  case 2:
    return "Serial Compute + DMA Copy Baseline";
  case 3:
    return "Concurrent Compute + Dedicated Transfer (SDMA)";
  default:
    return "";
  }
}

const char *AsyncOverlapBench::GetDescription(uint32_t config_idx) const {
  switch (config_idx) {
  case 0:
    return "Executes a multi-layer 4K Early-Z/Hi-Z rasterization + color pass (Vulkan 1.4 Dynamic Rendering) and a register-resident VALU compute pass serially on the primary queue.";
  case 1:
    return "Submits the fixed-function 4K Early-Z/Hi-Z rasterization pass on VK_QUEUE_GRAPHICS_BIT concurrently with register-resident VALU compute on a dedicated Async Compute Engine (ACE) queue.";
  case 2:
    return "Executes a heavy ALU compute pass followed serially by a 128 MB DMA transfer on the primary queue.";
  case 3:
    return "Overlaps heavy ALU compute on the compute queue with a concurrent 128 MB PCIe DMA copy on a dedicated SDMA transfer queue.";
  default:
    return "Measures hardware multi-engine concurrency across Graphics, Async Compute (ACE), and Dedicated Copy (SDMA) queues.";
  }
}

bool AsyncOverlapBench::IsSupported(const DeviceInfo & /*info*/,
                                    IComputeContext *context) const {
  if (!context) return true;
  if (context->getBackend() != ComputeBackend::Vulkan) return false;
  const auto *vc = dynamic_cast<const VulkanContext *>(context);
  return vc && vc->isDynamicRenderingSupported();
}

bool AsyncOverlapBench::IsConfigSupported(uint32_t config_idx,
                                          const DeviceInfo &info,
                                          IComputeContext *context) const {
  if (!IsSupported(info, context)) return false;
  const auto *vc = dynamic_cast<const VulkanContext *>(context);
  if (config_idx == 1) {
    return vc ? vc->hasAsyncComputeQueue() : info.dedicatedAsyncComputeQueue;
  }
  if (config_idx == 3) {
    return vc ? vc->hasDedicatedTransferQueue() : info.dedicatedTransferQueue;
  }
  return (config_idx == 0 || config_idx == 2);
}

IBenchmark::SupportLimitation
AsyncOverlapBench::GetSupportLimitation(const DeviceInfo & /*info*/,
                                        IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  return SupportLimitation::kHardware;
}

IBenchmark::SupportLimitation AsyncOverlapBench::GetConfigSupportLimitation(
    uint32_t config_idx, const DeviceInfo & /*info*/,
    IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return SupportLimitation::kApi;
  }
  if (config_idx == 1 || config_idx == 3) {
    return SupportLimitation::kHardware;
  }
  return SupportLimitation::kNone;
}

std::string AsyncOverlapBench::GetSupportNote(const DeviceInfo & /*info*/,
                                              IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4 Dynamic Rendering & multi-queue submission";
  }
  return "";
}

std::string AsyncOverlapBench::GetConfigSupportNote(
    uint32_t config_idx, const DeviceInfo & /*info*/,
    IComputeContext *context) const {
  if (context && context->getBackend() != ComputeBackend::Vulkan) {
    return "Requires Vulkan 1.4 backend";
  }
  if (config_idx == 1) {
    return "GPU/driver exposes no dedicated Async Compute queue separate from primary graphics queue";
  }
  if (config_idx == 3) {
    return "GPU/driver exposes no dedicated DMA transfer queue separate from compute/graphics queues";
  }
  return "";
}

std::string AsyncOverlapBench::GetConfigCaveat(uint32_t config_idx,
                                               const DeviceInfo &info,
                                               IComputeContext *context) const {
  if (config_idx == 2 || config_idx == 3) {
    return gpubench::formatRebarCaveat(info, context);
  }
  return "";
}

VkShaderModule
AsyncOverlapBench::loadSpirvModule(const std::string &shaderStem) const {
  std::filesystem::path spvPath =
      std::filesystem::path(m_kernelDir) / "vulkan" / (shaderStem + ".spv");
  std::ifstream file(spvPath, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("AsyncOverlapBench: missing SPIR-V file " +
                             spvPath.string());
  }
  size_t sz = static_cast<size_t>(file.tellg());
  if (sz == 0 || (sz % 4) != 0) {
    throw std::runtime_error("AsyncOverlapBench: invalid SPIR-V file " +
                             spvPath.string());
  }
  std::vector<uint32_t> words(sz / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char *>(words.data()),
            static_cast<std::streamsize>(sz));

  VkShaderModuleCreateInfo smCi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smCi.codeSize = sz;
  smCi.pCode = words.data();
  VkShaderModule mod = VK_NULL_HANDLE;
  if (vkCreateShaderModule(m_vk->getDevice(), &smCi, nullptr, &mod) !=
      VK_SUCCESS) {
    throw std::runtime_error("AsyncOverlapBench: vkCreateShaderModule failed");
  }
  return mod;
}

void AsyncOverlapBench::Setup(IComputeContext &context,
                              const std::string &kernel_dir) {
  m_vk = dynamic_cast<VulkanContext *>(&context);
  if (!m_vk) {
    throw std::runtime_error("AsyncOverlapBench requires VulkanContext");
  }
  m_kernelDir = kernel_dir;
  m_lastDurationMs.fill(0.0);

  VkDevice dev = m_vk->getDevice();

  // 1. Build Dynamic Rendering Graphics Pipeline (async_gfx.vert + async_gfx.frag)
  VkShaderModule vertMod = loadSpirvModule("async_gfx.vert");
  VkShaderModule fragMod = loadSpirvModule("async_gfx.frag");

  VkPushConstantRange gfxPcRange{};
  gfxPcRange.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  gfxPcRange.offset = 0;
  gfxPcRange.size = 8; // uint layerCount + float seed

  VkPipelineLayoutCreateInfo gfxPlCi{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  gfxPlCi.pushConstantRangeCount = 1;
  gfxPlCi.pPushConstantRanges = &gfxPcRange;
  vkCreatePipelineLayout(dev, &gfxPlCi, nullptr, &m_gfxPipeLayout);

  VkPipelineShaderStageCreateInfo stages[2]{};
  stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vertMod;
  stages[0].pName = "main";
  stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = fragMod;
  stages[1].pName = "main";

  VkPipelineVertexInputStateCreateInfo viState{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo iaState{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  iaState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkViewport vp{0.0f, 0.0f, static_cast<float>(kGfxWidth),
                static_cast<float>(kGfxHeight), 0.0f, 1.0f};
  VkRect2D scissor{{0, 0}, {kGfxWidth, kGfxHeight}};
  VkPipelineViewportStateCreateInfo vpState{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vpState.viewportCount = 1;
  vpState.pViewports = &vp;
  vpState.scissorCount = 1;
  vpState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rsState{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rsState.polygonMode = VK_POLYGON_MODE_FILL;
  rsState.cullMode = VK_CULL_MODE_NONE;
  rsState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rsState.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo msState{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  msState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo dsState{
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  dsState.depthTestEnable = VK_TRUE;
  dsState.depthWriteEnable = VK_TRUE;
  dsState.depthCompareOp = VK_COMPARE_OP_LESS;

  VkPipelineColorBlendAttachmentState cbAttach{};
  cbAttach.blendEnable = VK_FALSE;
  cbAttach.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo cbState{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cbState.attachmentCount = 1;
  cbState.pAttachments = &cbAttach;

  VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkPipelineRenderingCreateInfo dynRenderCi{
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  dynRenderCi.colorAttachmentCount = 1;
  dynRenderCi.pColorAttachmentFormats = &colorFormat;
  dynRenderCi.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

  VkGraphicsPipelineCreateInfo gpCi{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gpCi.pNext = &dynRenderCi;
  gpCi.stageCount = 2;
  gpCi.pStages = stages;
  gpCi.pVertexInputState = &viState;
  gpCi.pInputAssemblyState = &iaState;
  gpCi.pViewportState = &vpState;
  gpCi.pRasterizationState = &rsState;
  gpCi.pMultisampleState = &msState;
  gpCi.pDepthStencilState = &dsState;
  gpCi.pColorBlendState = &cbState;
  gpCi.layout = m_gfxPipeLayout;

  VkResult gpRes = vkCreateGraphicsPipelines(
      dev, m_vk->getPipelineCache(), 1, &gpCi, nullptr, &m_gfxPipeline);
  vkDestroyShaderModule(dev, vertMod, nullptr);
  vkDestroyShaderModule(dev, fragMod, nullptr);
  if (gpRes != VK_SUCCESS) {
    throw std::runtime_error("AsyncOverlapBench: failed to create graphics pipeline");
  }
  m_vk->notifyKernelCreated("async_gfx");

  // 2. Build Compute Pipeline (async_compute_heavy.comp)
  VkShaderModule compMod = loadSpirvModule("async_compute_heavy.comp");
  VkDescriptorSetLayoutBinding b0{};
  b0.binding = 0;
  b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b0.descriptorCount = 1;
  b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dslCi{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslCi.bindingCount = 1;
  dslCi.pBindings = &b0;
  vkCreateDescriptorSetLayout(dev, &dslCi, nullptr, &m_compSetLayout);

  VkPushConstantRange compPcRange{};
  compPcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  compPcRange.offset = 0;
  compPcRange.size = 12; // uint count, uint iterations, float seed
  VkPipelineLayoutCreateInfo compPlCi{
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  compPlCi.setLayoutCount = 1;
  compPlCi.pSetLayouts = &m_compSetLayout;
  compPlCi.pushConstantRangeCount = 1;
  compPlCi.pPushConstantRanges = &compPcRange;
  vkCreatePipelineLayout(dev, &compPlCi, nullptr, &m_compPipeLayout);

  VkComputePipelineCreateInfo cpCi{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpCi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpCi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpCi.stage.module = compMod;
  cpCi.stage.pName = "main";
  cpCi.layout = m_compPipeLayout;
  VkResult cpRes = vkCreateComputePipelines(
      dev, m_vk->getPipelineCache(), 1, &cpCi, nullptr, &m_compPipeline);
  vkDestroyShaderModule(dev, compMod, nullptr);
  if (cpRes != VK_SUCCESS) {
    throw std::runtime_error("AsyncOverlapBench: failed to create compute pipeline");
  }
  m_vk->notifyKernelCreated("async_compute_heavy");

  VkDescriptorPoolSize poolSz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo dpCi{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpCi.maxSets = 1;
  dpCi.poolSizeCount = 1;
  dpCi.pPoolSizes = &poolSz;
  vkCreateDescriptorPool(dev, &dpCi, nullptr, &m_compDescPool);

  VkDescriptorSetAllocateInfo dsAi{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsAi.descriptorPool = m_compDescPool;
  dsAi.descriptorSetCount = 1;
  dsAi.pSetLayouts = &m_compSetLayout;
  vkAllocateDescriptorSets(dev, &dsAi, &m_compDescSet);

  // Command pools & fences
  VkFenceCreateInfo fCi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkCommandPoolCreateInfo poolCi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolCi.queueFamilyIndex = m_vk->getGraphicsQueueFamilyIndex();
  poolCi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkCreateCommandPool(dev, &poolCi, nullptr, &m_gfxPool);

  VkCommandBufferAllocateInfo cmdAi{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cmdAi.commandPool = m_gfxPool;
  cmdAi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAi.commandBufferCount = 1;
  vkAllocateCommandBuffers(dev, &cmdAi, &m_gfxCmd);
  vkCreateFence(dev, &fCi, nullptr, &m_gfxFence);

  if (m_vk->hasAsyncComputeQueue()) {
    poolCi.queueFamilyIndex = m_vk->getAsyncComputeQueueFamilyIndex();
    vkCreateCommandPool(dev, &poolCi, nullptr, &m_asyncPool);
    cmdAi.commandPool = m_asyncPool;
    vkAllocateCommandBuffers(dev, &cmdAi, &m_asyncCmd);
    vkCreateFence(dev, &fCi, nullptr, &m_asyncFence);
  }
  if (m_vk->hasDedicatedTransferQueue()) {
    poolCi.queueFamilyIndex = m_vk->getTransferQueueFamilyIndex();
    vkCreateCommandPool(dev, &poolCi, nullptr, &m_xferPool);
    cmdAi.commandPool = m_xferPool;
    vkAllocateCommandBuffers(dev, &cmdAi, &m_xferCmd);
    vkCreateFence(dev, &fCi, nullptr, &m_xferFence);
  }
}

void AsyncOverlapBench::PrepareConfig(uint32_t config_idx) {
  ReleaseConfigResources(config_idx);
  if (!m_vk || m_vk->getVmaAllocator() == VK_NULL_HANDLE) return;

  VkDevice dev = m_vk->getDevice();
  VmaAllocator vma = m_vk->getVmaAllocator();

  // Compute output buffer (shared across graphics/compute/async families if distinct)
  std::set<uint32_t> compFamilies = {m_vk->getGraphicsQueueFamilyIndex(),
                                     m_vk->getComputeQueueFamilyIndex()};
  if (m_vk->hasAsyncComputeQueue()) {
    compFamilies.insert(m_vk->getAsyncComputeQueueFamilyIndex());
  }
  std::vector<uint32_t> compQfis(compFamilies.begin(), compFamilies.end());

  VkBufferCreateInfo compBufCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  compBufCi.size = static_cast<VkDeviceSize>(kComputeElements) * sizeof(float) * 4u;
  compBufCi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (compQfis.size() > 1) {
    compBufCi.sharingMode = VK_SHARING_MODE_CONCURRENT;
    compBufCi.queueFamilyIndexCount = static_cast<uint32_t>(compQfis.size());
    compBufCi.pQueueFamilyIndices = compQfis.data();
  } else {
    compBufCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }
  VmaAllocationCreateInfo devAllocCi{};
  devAllocCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (vmaCreateBuffer(vma, &compBufCi, &devAllocCi, &m_compOutBuf,
                      &m_compOutAlloc, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("AsyncOverlapBench: failed to allocate compute buffer");
  }

  VkDescriptorBufferInfo dbi{m_compOutBuf, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wds.dstSet = m_compDescSet;
  wds.dstBinding = 0;
  wds.descriptorCount = 1;
  wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wds.pBufferInfo = &dbi;
  vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

  if (config_idx == 0 || config_idx == 1) {
    VkImageCreateInfo imgCi{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgCi.imageType = VK_IMAGE_TYPE_2D;
    imgCi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imgCi.extent = {kGfxWidth, kGfxHeight, 1};
    imgCi.mipLevels = 1;
    imgCi.arrayLayers = 1;
    imgCi.samples = VK_SAMPLE_COUNT_1_BIT;
    imgCi.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCi.usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgCi.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vmaCreateImage(vma, &imgCi, &devAllocCi, &m_colorImage,
                       &m_colorImageAlloc, nullptr) != VK_SUCCESS) {
      throw std::runtime_error("AsyncOverlapBench: failed to create color target");
    }
    m_colorImageInitialized = false;

    VkImageViewCreateInfo ivCi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivCi.image = m_colorImage;
    ivCi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivCi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ivCi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivCi.subresourceRange.levelCount = 1;
    ivCi.subresourceRange.layerCount = 1;
    vkCreateImageView(dev, &ivCi, nullptr, &m_colorView);

    VkImageCreateInfo depthCi = imgCi;
    depthCi.format = VK_FORMAT_D32_SFLOAT;
    depthCi.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (vmaCreateImage(vma, &depthCi, &devAllocCi, &m_depthImage,
                       &m_depthImageAlloc, nullptr) != VK_SUCCESS) {
      throw std::runtime_error("AsyncOverlapBench: failed to create depth target");
    }
    VkImageViewCreateInfo divCi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    divCi.image = m_depthImage;
    divCi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    divCi.format = VK_FORMAT_D32_SFLOAT;
    divCi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    divCi.subresourceRange.levelCount = 1;
    divCi.subresourceRange.layerCount = 1;
    vkCreateImageView(dev, &divCi, nullptr, &m_depthView);
  } else if (config_idx == 2 || config_idx == 3) {
    std::set<uint32_t> dmaFamilies = {m_vk->getGraphicsQueueFamilyIndex(),
                                      m_vk->getComputeQueueFamilyIndex()};
    if (m_vk->hasDedicatedTransferQueue()) {
      dmaFamilies.insert(m_vk->getTransferQueueFamilyIndex());
    }
    std::vector<uint32_t> dmaQfis(dmaFamilies.begin(), dmaFamilies.end());

    VkBufferCreateInfo dmaCi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    dmaCi.size = kCopyBytes;
    dmaCi.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (dmaQfis.size() > 1) {
      dmaCi.sharingMode = VK_SHARING_MODE_CONCURRENT;
      dmaCi.queueFamilyIndexCount = static_cast<uint32_t>(dmaQfis.size());
      dmaCi.pQueueFamilyIndices = dmaQfis.data();
    } else {
      dmaCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VmaAllocationCreateInfo hostUpCi{};
    hostUpCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    hostUpCi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo srcInfo{};
    if (vmaCreateBuffer(vma, &dmaCi, &hostUpCi, &m_dmaSrcBuf, &m_dmaSrcAlloc,
                        &srcInfo) != VK_SUCCESS) {
      throw std::runtime_error("AsyncOverlapBench: failed to allocate DMA src");
    }
    auto *srcU32 = static_cast<uint32_t *>(srcInfo.pMappedData);
    for (size_t i = 0; i < kCopyBytes / sizeof(uint32_t); ++i) {
      srcU32[i] = 0xCAFE0000u ^ static_cast<uint32_t>(i);
    }

    VmaAllocationCreateInfo hostDnCi{};
    hostDnCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    hostDnCi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo dstInfo{};
    if (vmaCreateBuffer(vma, &dmaCi, &hostDnCi, &m_dmaDstBuf, &m_dmaDstAlloc,
                        &dstInfo) != VK_SUCCESS) {
      throw std::runtime_error("AsyncOverlapBench: failed to allocate DMA dst");
    }
    m_dmaDstMapped = dstInfo.pMappedData;
    std::memset(m_dmaDstMapped, 0, kCopyBytes);
  }
}

void AsyncOverlapBench::ReleaseConfigResources(uint32_t /*config_idx*/) {
  if (!m_vk) return;
  m_vk->waitIdle();
  VkDevice dev = m_vk->getDevice();
  VmaAllocator vma = m_vk->getVmaAllocator();

  if (m_depthView != VK_NULL_HANDLE) {
    vkDestroyImageView(dev, m_depthView, nullptr);
    m_depthView = VK_NULL_HANDLE;
  }
  if (m_depthImage != VK_NULL_HANDLE && vma != VK_NULL_HANDLE) {
    vmaDestroyImage(vma, m_depthImage, m_depthImageAlloc);
    m_depthImage = VK_NULL_HANDLE;
    m_depthImageAlloc = VK_NULL_HANDLE;
  }
  if (m_colorView != VK_NULL_HANDLE) {
    vkDestroyImageView(dev, m_colorView, nullptr);
    m_colorView = VK_NULL_HANDLE;
  }
  if (m_colorImage != VK_NULL_HANDLE && vma != VK_NULL_HANDLE) {
    vmaDestroyImage(vma, m_colorImage, m_colorImageAlloc);
    m_colorImage = VK_NULL_HANDLE;
    m_colorImageAlloc = VK_NULL_HANDLE;
    m_colorImageInitialized = false;
  }
  if (m_compOutBuf != VK_NULL_HANDLE && vma != VK_NULL_HANDLE) {
    vmaDestroyBuffer(vma, m_compOutBuf, m_compOutAlloc);
    m_compOutBuf = VK_NULL_HANDLE;
    m_compOutAlloc = VK_NULL_HANDLE;
  }
  if (m_dmaSrcBuf != VK_NULL_HANDLE && vma != VK_NULL_HANDLE) {
    vmaDestroyBuffer(vma, m_dmaSrcBuf, m_dmaSrcAlloc);
    m_dmaSrcBuf = VK_NULL_HANDLE;
    m_dmaSrcAlloc = VK_NULL_HANDLE;
  }
  if (m_dmaDstBuf != VK_NULL_HANDLE && vma != VK_NULL_HANDLE) {
    vmaDestroyBuffer(vma, m_dmaDstBuf, m_dmaDstAlloc);
    m_dmaDstBuf = VK_NULL_HANDLE;
    m_dmaDstAlloc = VK_NULL_HANDLE;
    m_dmaDstMapped = nullptr;
  }
}

void AsyncOverlapBench::Run(uint32_t config_idx) {
  if (!m_vk) return;
  VkDevice dev = m_vk->getDevice();
  m_runSeed = (m_runSeed > 1.5f) ? 0.25f : (m_runSeed + 0.0625f);

  struct GfxPc {
    uint32_t layerCount;
    float seed;
  } gfxPc{kGfxDepthLayers, m_runSeed};

  struct CompPc {
    uint32_t count;
    uint32_t iterations;
    float seed;
  } compPc{(config_idx <= 1u) ? kAceComputeElements : kComputeElements,
           (config_idx <= 1u) ? 98304u : 12288u, m_runSeed};

  auto recordGfxPass = [&](VkCommandBuffer cmd) {
    VkImageMemoryBarrier imgBars[2]{};
    imgBars[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imgBars[0].srcAccessMask = m_colorImageInitialized
                                   ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                   : 0;
    imgBars[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imgBars[0].oldLayout = m_colorImageInitialized
                               ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                               : VK_IMAGE_LAYOUT_UNDEFINED;
    imgBars[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imgBars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBars[0].image = m_colorImage;
    imgBars[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    imgBars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imgBars[1].srcAccessMask =
        m_colorImageInitialized
            ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            : 0;
    imgBars[1].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    imgBars[1].oldLayout = m_colorImageInitialized
                               ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                               : VK_IMAGE_LAYOUT_UNDEFINED;
    imgBars[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    imgBars[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBars[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBars[1].image = m_depthImage;
    imgBars[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(
        cmd,
        m_colorImageInitialized
            ? (VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, nullptr, 0, nullptr, 2, imgBars);
    m_colorImageInitialized = true;

    VkRenderingAttachmentInfo colorAtt{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAtt.imageView = m_colorView;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingAttachmentInfo depthAtt{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAtt.imageView = m_depthView;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.clearValue.depthStencil = {1.0f, 0u};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {kGfxWidth, kGfxHeight}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;
    ri.pDepthAttachment = &depthAtt;

    auto pfnBegin = m_vk->getCmdBeginRendering();
    auto pfnEnd = m_vk->getCmdEndRendering();
    pfnBegin(cmd, &ri);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gfxPipeline);
    vkCmdPushConstants(cmd, m_gfxPipeLayout,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(gfxPc), &gfxPc);
    vkCmdDraw(cmd, 3, kGfxDepthLayers, 0, 0);
    pfnEnd(cmd);
  };

  auto recordCompPass = [&](VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_compPipeLayout, 0, 1, &m_compDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(compPc), &compPc);
    vkCmdDispatch(cmd, (compPc.count + 255u) / 256u, 1, 1);
  };

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  if (config_idx == 0) {
    vkResetCommandBuffer(m_gfxCmd, 0);
    vkBeginCommandBuffer(m_gfxCmd, &bi);
    recordGfxPass(m_gfxCmd);
    // Serialize graphics completion before starting compute in the baseline
    m_vk->cmdPipelineMemoryBarrier2(
        m_gfxCmd,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    recordCompPass(m_gfxCmd);
    vkEndCommandBuffer(m_gfxCmd);

    vkResetFences(dev, 1, &m_gfxFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_gfxCmd;
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
      vkQueueSubmit(m_vk->getGraphicsQueue(), 1, &si, m_gfxFence);
    }
    vkWaitForFences(dev, 1, &m_gfxFence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();
    m_lastDurationMs[0] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  } else if (config_idx == 1) {
    vkResetCommandBuffer(m_gfxCmd, 0);
    vkBeginCommandBuffer(m_gfxCmd, &bi);
    recordGfxPass(m_gfxCmd);
    vkEndCommandBuffer(m_gfxCmd);

    vkResetCommandBuffer(m_asyncCmd, 0);
    vkBeginCommandBuffer(m_asyncCmd, &bi);
    recordCompPass(m_asyncCmd);
    vkEndCommandBuffer(m_asyncCmd);

    VkFence fences[2] = {m_gfxFence, m_asyncFence};
    vkResetFences(dev, 2, fences);
    VkSubmitInfo siGfx{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    siGfx.commandBufferCount = 1;
    siGfx.pCommandBuffers = &m_gfxCmd;
    VkSubmitInfo siAsync{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    siAsync.commandBufferCount = 1;
    siAsync.pCommandBuffers = &m_asyncCmd;

    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
      vkQueueSubmit(m_vk->getGraphicsQueue(), 1, &siGfx, m_gfxFence);
      vkQueueSubmit(m_vk->getAsyncComputeQueue(), 1, &siAsync, m_asyncFence);
    }
    vkWaitForFences(dev, 2, fences, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();
    m_lastDurationMs[1] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  } else if (config_idx == 2) {
    vkResetCommandBuffer(m_gfxCmd, 0);
    vkBeginCommandBuffer(m_gfxCmd, &bi);
    recordCompPass(m_gfxCmd);
    m_vk->cmdPipelineMemoryBarrier2(
        m_gfxCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferCopy region{0, 0, kCopyBytes};
    vkCmdCopyBuffer(m_gfxCmd, m_dmaSrcBuf, m_dmaDstBuf, 1, &region);
    vkEndCommandBuffer(m_gfxCmd);

    vkResetFences(dev, 1, &m_gfxFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_gfxCmd;
    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
      vkQueueSubmit(m_vk->getComputeQueue(), 1, &si, m_gfxFence);
    }
    vkWaitForFences(dev, 1, &m_gfxFence, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();
    m_lastDurationMs[2] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  } else if (config_idx == 3) {
    vkResetCommandBuffer(m_gfxCmd, 0);
    vkBeginCommandBuffer(m_gfxCmd, &bi);
    recordCompPass(m_gfxCmd);
    vkEndCommandBuffer(m_gfxCmd);

    vkResetCommandBuffer(m_xferCmd, 0);
    vkBeginCommandBuffer(m_xferCmd, &bi);
    VkBufferCopy region{0, 0, kCopyBytes};
    vkCmdCopyBuffer(m_xferCmd, m_dmaSrcBuf, m_dmaDstBuf, 1, &region);
    vkEndCommandBuffer(m_xferCmd);

    VkFence fences[2] = {m_gfxFence, m_xferFence};
    vkResetFences(dev, 2, fences);
    VkSubmitInfo siComp{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    siComp.commandBufferCount = 1;
    siComp.pCommandBuffers = &m_gfxCmd;
    VkSubmitInfo siXfer{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    siXfer.commandBufferCount = 1;
    siXfer.pCommandBuffers = &m_xferCmd;

    auto t0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
      vkQueueSubmit(m_vk->getComputeQueue(), 1, &siComp, m_gfxFence);
      vkQueueSubmit(m_vk->getTransferQueue(), 1, &siXfer, m_xferFence);
    }
    vkWaitForFences(dev, 2, fences, VK_TRUE, UINT64_MAX);
    auto t1 = std::chrono::steady_clock::now();
    m_lastDurationMs[3] =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
}

BenchmarkResult AsyncOverlapBench::GetResult(uint32_t config_idx) const {
  return {1ULL, GetLastRunDurationMs(config_idx)};
}

bool AsyncOverlapBench::ValidateResults(uint32_t config_idx) const {
  if (!m_vk || m_compOutBuf == VK_NULL_HANDLE) return false;
  VkDevice dev = m_vk->getDevice();

  // Read back 64 vec4s from compute buffer + (for Config 0/1) 16 pixels from color image
  constexpr size_t kCheckFloats = 256;
  void *mappedRead = nullptr;
  ComputeBuffer readback =
      m_vk->createPinnedHostBuffer(kCheckFloats * sizeof(float) * 2, false,
                                   &mappedRead);
  if (!readback || !mappedRead) return false;
  std::memset(mappedRead, 0, kCheckFloats * sizeof(float) * 2);

  vkResetCommandBuffer(m_gfxCmd, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_gfxCmd, &bi);

  VkBufferCopy bufCopy{0, 0, kCheckFloats * sizeof(float)};
  vkCmdCopyBuffer(m_gfxCmd, m_compOutBuf, m_vk->getVkBuffer(readback), 1,
                  &bufCopy);

  if ((config_idx == 0 || config_idx == 1) && m_colorImage != VK_NULL_HANDLE &&
      m_colorImageInitialized) {
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bar.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = m_colorImage;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(m_gfxCmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &bar);

    VkBufferImageCopy imgCopy{};
    imgCopy.bufferOffset = kCheckFloats * sizeof(float);
    imgCopy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    imgCopy.imageOffset = {0, 0, 0};
    imgCopy.imageExtent = {16, 1, 1};
    vkCmdCopyImageToBuffer(m_gfxCmd, m_colorImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_vk->getVkBuffer(readback), 1, &imgCopy);

    bar.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bar.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    vkCmdPipelineBarrier(m_gfxCmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &bar);
  }

  m_vk->cmdPipelineMemoryBarrier2(
      m_gfxCmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
      VK_ACCESS_2_HOST_READ_BIT);
  vkEndCommandBuffer(m_gfxCmd);

  vkResetFences(dev, 1, &m_gfxFence);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &m_gfxCmd;
  {
    std::lock_guard<std::recursive_mutex> qlock(m_vk->getQueueMutex());
    vkQueueSubmit(m_vk->getGraphicsQueue(), 1, &si, m_gfxFence);
  }
  vkWaitForFences(dev, 1, &m_gfxFence, VK_TRUE, UINT64_MAX);

  const auto *floats = static_cast<const float *>(mappedRead);
  bool computeOk = false;
  for (size_t i = 0; i < kCheckFloats; ++i) {
    if (!std::isfinite(floats[i])) {
      m_vk->releaseBuffer(readback);
      return false;
    }
    if (std::fabs(floats[i]) > 1e-4f) {
      computeOk = true;
    }
  }

  bool secondOk = false;
  if (config_idx == 0 || config_idx == 1) {
    const auto *halfPix =
        reinterpret_cast<const uint16_t *>(floats + kCheckFloats);
    for (size_t i = 0; i < 64; ++i) {
      uint16_t h = halfPix[i];
      uint16_t exp = static_cast<uint16_t>((h >> 10) & 0x1Fu);
      if (exp == 0x1Fu) { // IEEE-754 binary16 Inf / NaN
        m_vk->releaseBuffer(readback);
        return false;
      }
      if ((i % 4 != 3) && (h & 0x8000u) == 0u && exp >= 10u) {
        secondOk = true;
      }
    }
  } else {
    if (m_dmaDstMapped) {
      const auto *dstU32 = static_cast<const uint32_t *>(m_dmaDstMapped);
      size_t tailOff = (kCopyBytes / sizeof(uint32_t)) - 1024;
      secondOk = true;
      for (size_t i = 0; i < 1024; ++i) {
        if (dstU32[tailOff + i] !=
            (0xCAFE0000u ^ static_cast<uint32_t>(tailOff + i))) {
          secondOk = false;
          break;
        }
      }
    }
  }

  m_vk->releaseBuffer(readback);
  return computeOk && secondOk;
}

void AsyncOverlapBench::Teardown() {
  ReleaseConfigResources(0);
  if (m_vk) {
    VkDevice dev = m_vk->getDevice();
    if (m_gfxPipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(dev, m_gfxPipeline, nullptr);
    if (m_gfxPipeLayout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(dev, m_gfxPipeLayout, nullptr);
    if (m_compPipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(dev, m_compPipeline, nullptr);
    if (m_compPipeLayout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(dev, m_compPipeLayout, nullptr);
    if (m_compDescPool != VK_NULL_HANDLE)
      vkDestroyDescriptorPool(dev, m_compDescPool, nullptr);
    if (m_compSetLayout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(dev, m_compSetLayout, nullptr);

    if (m_gfxFence != VK_NULL_HANDLE) vkDestroyFence(dev, m_gfxFence, nullptr);
    if (m_gfxPool != VK_NULL_HANDLE)
      vkDestroyCommandPool(dev, m_gfxPool, nullptr);
    if (m_asyncFence != VK_NULL_HANDLE)
      vkDestroyFence(dev, m_asyncFence, nullptr);
    if (m_asyncPool != VK_NULL_HANDLE)
      vkDestroyCommandPool(dev, m_asyncPool, nullptr);
    if (m_xferFence != VK_NULL_HANDLE)
      vkDestroyFence(dev, m_xferFence, nullptr);
    if (m_xferPool != VK_NULL_HANDLE)
      vkDestroyCommandPool(dev, m_xferPool, nullptr);
  }
  m_gfxPipeline = VK_NULL_HANDLE;
  m_gfxPipeLayout = VK_NULL_HANDLE;
  m_compPipeline = VK_NULL_HANDLE;
  m_compPipeLayout = VK_NULL_HANDLE;
  m_compDescPool = VK_NULL_HANDLE;
  m_compSetLayout = VK_NULL_HANDLE;
  m_gfxFence = VK_NULL_HANDLE;
  m_gfxPool = VK_NULL_HANDLE;
  m_asyncFence = VK_NULL_HANDLE;
  m_asyncPool = VK_NULL_HANDLE;
  m_xferFence = VK_NULL_HANDLE;
  m_xferPool = VK_NULL_HANDLE;
  m_vk = nullptr;
}
