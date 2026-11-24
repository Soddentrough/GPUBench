#include "showcase/ShowcaseRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace gpubench::showcase {

namespace {

std::string resolveShowcaseShaderPath(const std::string &kernelDir, const std::string &name) {
  std::vector<std::filesystem::path> candidates = {
      std::filesystem::path(kernelDir) / name,
      std::filesystem::path("build/kernels/vulkan") / name,
      std::filesystem::path("kernels/vulkan") / name,
      std::filesystem::path("../kernels/vulkan") / name,
  };
  std::error_code ec;
  for (const auto &p : candidates) {
    if (std::filesystem::exists(p, ec) && std::filesystem::is_regular_file(p, ec)) {
      return p.string();
    }
  }
  return (std::filesystem::path(kernelDir) / name).string();
}

} // namespace

ShowcaseRenderer::~ShowcaseRenderer() {
  shutdown();
}

bool ShowcaseRenderer::init(::VulkanContext *ctx, const std::string &kernelDir,
                            uint32_t initialW, uint32_t initialH) {
  shutdown();
  if (!ctx || !ctx->getDevice() || !ctx->getPhysicalDevice() || !ctx->getVmaAllocator()) {
    return false;
  }
  if (!ctx->isDynamicRenderingSupported()) {
    std::cerr << "[ShowcaseRenderer] VK_KHR_dynamic_rendering not supported.\n";
    return false;
  }

  m_ctx = ctx;
  m_device = ctx->getDevice();
  m_physDev = ctx->getPhysicalDevice();
  m_vma = ctx->getVmaAllocator();

  std::string sceneErr;
  if (!buildShowcaseScene(m_scene, sceneErr)) {
    std::cerr << "[ShowcaseRenderer] Scene build error: " << sceneErr << "\n";
    shutdown();
    return false;
  }

  if (!m_scene.presets.empty()) {
    m_settings.activeCam = m_scene.presets[0].camera;
    m_settings.frozenCam = m_scene.presets[0].camera;
  }

  // Transient command pool & fence for synchronous uploads and headless captures
  VkCommandPoolCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpInfo.queueFamilyIndex = m_ctx->getGraphicsQueueFamilyIndex();
  cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  if (vkCreateCommandPool(m_device, &cpInfo, nullptr, &m_uploadCmdPool) != VK_SUCCESS) {
    shutdown();
    return false;
  }
  VkCommandBufferAllocateInfo cbAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbAlloc.commandPool = m_uploadCmdPool;
  cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAlloc.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(m_device, &cbAlloc, &m_headlessCmd) != VK_SUCCESS) {
    shutdown();
    return false;
  }
  VkFenceCreateInfo fenceCi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(m_device, &fenceCi, nullptr, &m_headlessFence) != VK_SUCCESS) {
    shutdown();
    return false;
  }

  // Descriptor Pool + Layouts
  VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128},
  };
  VkDescriptorPoolCreateInfo dpCi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpCi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpCi.maxSets = 64;
  dpCi.poolSizeCount = static_cast<uint32_t>(sizeof(poolSizes) / sizeof(poolSizes[0]));
  dpCi.pPoolSizes = poolSizes;
  if (vkCreateDescriptorPool(m_device, &dpCi, nullptr, &m_descPool) != VK_SUCCESS) {
    shutdown();
    return false;
  }

  // Samplers
  {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.anisotropyEnable = VK_TRUE;
    sci.maxAnisotropy = 4.0f;
    sci.maxLod = 12.0f;
    vkCreateSampler(m_device, &sci, nullptr, &m_texSampler);

    VkSamplerCreateInfo shCi{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    shCi.magFilter = VK_FILTER_NEAREST;
    shCi.minFilter = VK_FILTER_NEAREST;
    shCi.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    shCi.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shCi.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    shCi.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(m_device, &shCi, nullptr, &m_shadowSampler);
    vkCreateSampler(m_device, &shCi, nullptr, &m_pointSampler);

    VkSamplerCreateInfo hzCi = shCi;
    hzCi.magFilter = VK_FILTER_LINEAR;
    hzCi.minFilter = VK_FILTER_LINEAR;
    hzCi.maxLod = 16.0f;
    VkSamplerReductionModeCreateInfo redCi{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO};
    redCi.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;
    hzCi.pNext = &redCi;
    vkCreateSampler(m_device, &hzCi, nullptr, &m_hizSampler);
  }

  // Descriptor Set Layouts
  {
    // Set 0: Static Scene SSBOs (0..8) + Bindless textures array (9)
    std::vector<VkDescriptorSetLayoutBinding> s0Bindings(10);
    std::vector<VkDescriptorBindingFlags> s0Flags(10, 0);
    const VkShaderStageFlags allStages =
        VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT |
        (m_ctx->isMeshShaderSupported() ? (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT) : 0u);
    for (uint32_t i = 0; i < 9u; ++i) {
      s0Bindings[i].binding = i;
      s0Bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      s0Bindings[i].descriptorCount = 1;
      s0Bindings[i].stageFlags = allStages;
    }
    s0Bindings[9].binding = 9;
    s0Bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    s0Bindings[9].descriptorCount = kMaxShowcaseTextures;
    s0Bindings[9].stageFlags = allStages;
    s0Flags[9] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo bfCi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfCi.bindingCount = static_cast<uint32_t>(s0Flags.size());
    bfCi.pBindingFlags = s0Flags.data();

    VkDescriptorSetLayoutCreateInfo dsl0{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl0.pNext = &bfCi;
    dsl0.bindingCount = static_cast<uint32_t>(s0Bindings.size());
    dsl0.pBindings = s0Bindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &dsl0, nullptr, &m_sceneSetLayout) != VK_SUCCESS) {
      shutdown();
      return false;
    }

    // Set 1: Frame dynamic bindings (0..18)
    VkDescriptorSetLayoutBinding s1Bindings[19]{};
    auto setB = [&](uint32_t idx, VkDescriptorType type) {
      s1Bindings[idx].binding = idx;
      s1Bindings[idx].descriptorType = type;
      s1Bindings[idx].descriptorCount = 1;
      s1Bindings[idx].stageFlags = allStages;
    };
    setB(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    setB(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    setB(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // uShadowMap
    setB(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // uHiZTex
    setB(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // uOverdrawImg
    setB(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // uTriAreaImg
    setB(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // uVisBufferImg
    setB(7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // uDepthTex
    setB(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // uGBufAlbedo
    setB(9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER); // uGBufNormal
    setB(10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);// uGBufWorldPos
    setB(11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);         // uSceneColorImg
    setB(12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);// uSceneColorTex
    setB(13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);         // uDisplayImg
    setB(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);// uAltColorTex
    setB(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);        // Worklists
    setB(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);        // DgcBuffer
    setB(17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);         // uDebugTagImg
    setB(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);        // FallbackBuf

    VkDescriptorSetLayoutCreateInfo dsl1{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl1.bindingCount = 19;
    dsl1.pBindings = s1Bindings;
    if (vkCreateDescriptorSetLayout(m_device, &dsl1, nullptr, &m_frameSetLayout) != VK_SUCCESS) {
      shutdown();
      return false;
    }

    // Set 2: Hi-Z Reduction (binding 0 = sampler2D src, binding 1 = r32f storage image dst)
    VkDescriptorSetLayoutBinding s2Bindings[2]{};
    s2Bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    s2Bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsl2{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl2.bindingCount = 2;
    dsl2.pBindings = s2Bindings;
    if (vkCreateDescriptorSetLayout(m_device, &dsl2, nullptr, &m_hizSetLayout) != VK_SUCCESS) {
      shutdown();
      return false;
    }
  }

  // Pipeline Layouts
  {
    VkDescriptorSetLayout sets01[2] = {m_sceneSetLayout, m_frameSetLayout};

    VkPipelineLayoutCreateInfo plGfx{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plGfx.setLayoutCount = 2;
    plGfx.pSetLayouts = sets01;
    vkCreatePipelineLayout(m_device, &plGfx, nullptr, &m_gfxLayout);

    if (m_ctx->isMeshShaderSupported()) {
      VkPushConstantRange pcTask{VK_SHADER_STAGE_TASK_BIT_EXT, 0, sizeof(uint32_t) * 2u};
      VkPipelineLayoutCreateInfo plTask = plGfx;
      plTask.pushConstantRangeCount = 1;
      plTask.pPushConstantRanges = &pcTask;
      vkCreatePipelineLayout(m_device, &plTask, nullptr, &m_taskLayout);
    }

    VkPushConstantRange pcComp{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 4u};
    VkPipelineLayoutCreateInfo plComp = plGfx;
    plComp.pushConstantRangeCount = 1;
    plComp.pPushConstantRanges = &pcComp;
    vkCreatePipelineLayout(m_device, &plComp, nullptr, &m_compLayout);

    VkPushConstantRange pcShade{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 8u};
    VkPipelineLayoutCreateInfo plShade = plGfx;
    plShade.pushConstantRangeCount = 1;
    plShade.pPushConstantRanges = &pcShade;
    vkCreatePipelineLayout(m_device, &plShade, nullptr, &m_visShadeSpecLayout);

    VkPushConstantRange pcHiz{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 5u};
    VkPipelineLayoutCreateInfo plHiz{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plHiz.setLayoutCount = 1;
    plHiz.pSetLayouts = &m_hizSetLayout;
    plHiz.pushConstantRangeCount = 1;
    plHiz.pPushConstantRanges = &pcHiz;
    vkCreatePipelineLayout(m_device, &plHiz, nullptr, &m_hizLayout);
  }

  if (!createSceneGpuResources() ||
      !createPipelines(kernelDir) ||
      !createFrameSlots() ||
      !createRenderTargets(std::max(320u, initialW), std::max(240u, initialH))) {
    shutdown();
    return false;
  }

  createDgcExecutionSet();
  renderShadowMapOnce();

  m_telemetry.meshShaderSupported = m_ctx->isMeshShaderSupported() && (m_pipeModernVis != VK_NULL_HANDLE);
  m_telemetry.dgcExecutionSetSupported = m_dgcExecSetReady;
  m_telemetry.barycentricSupported = m_ctx->isFragmentBarycentricSupported();
  m_telemetry.tessellationSupported = m_ctx->isTessellationSupported();

  m_telemetry.meshShaderActive = (m_settings.mode == RenderMode::Modern) && m_telemetry.meshShaderSupported;
  m_telemetry.dgcExecutionSetActive = (m_settings.mode == RenderMode::Modern) &&
                                      m_settings.opts.visBuffer &&
                                      m_settings.opts.dgcMaterialBinning &&
                                      m_telemetry.dgcExecutionSetSupported;
  m_telemetry.barycentricActive = ((m_settings.mode != RenderMode::Modern) || !m_settings.opts.visBuffer) &&
                                  m_telemetry.barycentricSupported;
  m_telemetry.tessellationActive = (m_settings.mode == RenderMode::Naive || m_settings.mode == RenderMode::Legacy) &&
                                   m_telemetry.tessellationSupported;

  m_initialized = true;
  return true;
}

void ShowcaseRenderer::shutdown() {
  if (m_device != VK_NULL_HANDLE) {
    std::lock_guard<std::recursive_mutex> qlock(m_ctx->getQueueMutex());
    vkDeviceWaitIdle(m_device);
  }

  destroyRenderTargets();
  destroyFrameSlots();

  if (m_dgcLayout != VK_NULL_HANDLE) {
    m_ctx->destroyIndirectCommandsLayout(m_dgcLayout);
    m_dgcLayout = VK_NULL_HANDLE;
  }
  if (m_dgcExecSet != VK_NULL_HANDLE) {
    m_ctx->destroyIndirectExecutionSet(m_dgcExecSet);
    m_dgcExecSet = VK_NULL_HANDLE;
  }
  if (m_dgcPreprocessBuf.buffer != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
    vkDestroyBuffer(m_device, m_dgcPreprocessBuf.buffer, nullptr);
  }
  if (m_dgcPreprocessMem != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
    vkFreeMemory(m_device, m_dgcPreprocessMem, nullptr);
  }
  m_dgcPreprocessBuf = {};
  m_dgcPreprocessMem = VK_NULL_HANDLE;
  m_dgcExecSetReady = false;

  auto destroyPipe = [&](VkPipeline &p) {
    if (p != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
      vkDestroyPipeline(m_device, p, nullptr);
      p = VK_NULL_HANDLE;
    }
  };
  destroyPipe(m_pipeShadow);
  destroyPipe(m_pipeSky);
  destroyPipe(m_pipeNaiveFlat);
  destroyPipe(m_pipeNaiveTess);
  destroyPipe(m_pipeLegacyFlat);
  destroyPipe(m_pipeLegacyTess);
  destroyPipe(m_pipeLegacyLight);
  destroyPipe(m_pipeModernVisOpaque);
  destroyPipe(m_pipeModernVis);
  destroyPipe(m_pipeModernFwd);
  destroyPipe(m_pipeFallbackCull);
  destroyPipe(m_pipeFallbackVis);
  destroyPipe(m_pipeHizReduce);
  destroyPipe(m_pipeVisClassify);
  destroyPipe(m_pipeVisResolve);
  destroyPipe(m_pipeVisShadeUnified);
  for (auto &p : m_pipeVisShadeSpec) destroyPipe(p);
  destroyPipe(m_pipeOverlayCompose);
  destroyPipe(m_pipeFrustumWire);

  auto destroyPL = [&](VkPipelineLayout &pl) {
    if (pl != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(m_device, pl, nullptr);
      pl = VK_NULL_HANDLE;
    }
  };
  destroyPL(m_gfxLayout);
  destroyPL(m_taskLayout);
  destroyPL(m_compLayout);
  destroyPL(m_hizLayout);
  destroyPL(m_visShadeSpecLayout);

  destroyImage(m_shadowMap);
  for (auto &img : m_sceneTextures) destroyImage(img);
  m_sceneTextures.clear();

  destroyBuffer(m_vb);
  destroyBuffer(m_ib);
  destroyBuffer(m_meshletBuf);
  destroyBuffer(m_meshletVertBuf);
  destroyBuffer(m_meshletTriBuf);
  destroyBuffer(m_instanceBuf);
  destroyBuffer(m_materialBuf);
  destroyBuffer(m_lightBuf);
  destroyBuffer(m_taskPkt0Buf);
  destroyBuffer(m_taskPkt1Buf);
  destroyBuffer(m_dgcBuf);
  destroyBuffer(m_fallbackBuf);

  auto destroySamp = [&](VkSampler &s) {
    if (s != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
      vkDestroySampler(m_device, s, nullptr);
      s = VK_NULL_HANDLE;
    }
  };
  destroySamp(m_texSampler);
  destroySamp(m_shadowSampler);
  destroySamp(m_hizSampler);
  destroySamp(m_pointSampler);

  auto destroyDSL = [&](VkDescriptorSetLayout &dsl) {
    if (dsl != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(m_device, dsl, nullptr);
      dsl = VK_NULL_HANDLE;
    }
  };
  destroyDSL(m_sceneSetLayout);
  destroyDSL(m_frameSetLayout);
  destroyDSL(m_hizSetLayout);

  if (m_descPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    m_descPool = VK_NULL_HANDLE;
  }
  if (m_headlessFence != VK_NULL_HANDLE) {
    vkDestroyFence(m_device, m_headlessFence, nullptr);
    m_headlessFence = VK_NULL_HANDLE;
  }
  if (m_uploadCmdPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(m_device, m_uploadCmdPool, nullptr);
    m_uploadCmdPool = VK_NULL_HANDLE;
    m_headlessCmd = VK_NULL_HANDLE;
  }

  m_initialized = false;
  m_device = VK_NULL_HANDLE;
  m_physDev = VK_NULL_HANDLE;
  m_vma = VK_NULL_HANDLE;
  m_ctx = nullptr;
}

ShowcaseRenderer::GpuBuffer ShowcaseRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                                            bool hostVisible) {
  GpuBuffer out{};
  out.size = std::max<VkDeviceSize>(size, 64u);

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = out.size;
  bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo aci{};
  aci.usage = hostVisible ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (hostVisible) {
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }

  VmaAllocationInfo allocInfo{};
  if (vmaCreateBuffer(m_vma, &bci, &aci, &out.buffer, &out.allocation, &allocInfo) != VK_SUCCESS) {
    throw std::runtime_error("ShowcaseRenderer failed to create VMA buffer");
  }
  out.mapped = allocInfo.pMappedData;

  VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  addrInfo.buffer = out.buffer;
  out.address = vkGetBufferDeviceAddress(m_device, &addrInfo);
  return out;
}

void ShowcaseRenderer::destroyBuffer(GpuBuffer &buf) {
  if (buf.buffer != VK_NULL_HANDLE && m_vma != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_vma, buf.buffer, buf.allocation);
  }
  buf = {};
}

void ShowcaseRenderer::uploadToDeviceBuffer(GpuBuffer &dst, const void *data, VkDeviceSize size) {
  if (!data || size == 0u) return;
  GpuBuffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
  std::memcpy(staging.mapped, data, static_cast<size_t>(size));
  vmaFlushAllocation(m_vma, staging.allocation, 0, size);

  std::lock_guard<std::recursive_mutex> qlock(m_ctx->getQueueMutex());
  vkResetCommandPool(m_device, m_uploadCmdPool, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_headlessCmd, &bi);
  VkBufferCopy copy{0, 0, size};
  vkCmdCopyBuffer(m_headlessCmd, staging.buffer, dst.buffer, 1, &copy);
  vkEndCommandBuffer(m_headlessCmd);

  vkResetFences(m_device, 1, &m_headlessFence);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &m_headlessCmd;
  vkQueueSubmit(m_ctx->getGraphicsQueue(), 1, &si, m_headlessFence);
  vkWaitForFences(m_device, 1, &m_headlessFence, VK_TRUE, UINT64_MAX);
  destroyBuffer(staging);
}

ShowcaseRenderer::GpuImage ShowcaseRenderer::createImage(uint32_t w, uint32_t h, uint32_t mipLevels,
                                                          VkFormat fmt, VkImageUsageFlags usage,
                                                          VkImageAspectFlags aspect) {
  GpuImage out{};
  out.width = w;
  out.height = h;
  out.mipLevels = std::max(1u, mipLevels);
  out.format = fmt;
  out.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.extent = {w, h, 1};
  ici.mipLevels = out.mipLevels;
  ici.arrayLayers = 1;
  ici.format = fmt;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  ici.usage = usage;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (vmaCreateImage(m_vma, &ici, &aci, &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
    throw std::runtime_error("ShowcaseRenderer failed to create VMA image");
  }

  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = out.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = fmt;
  vci.subresourceRange = {aspect, 0, out.mipLevels, 0, 1};
  if (vkCreateImageView(m_device, &vci, nullptr, &out.view) != VK_SUCCESS) {
    throw std::runtime_error("ShowcaseRenderer failed to create image view");
  }

  if (out.mipLevels > 1u) {
    out.mipViews.resize(out.mipLevels, VK_NULL_HANDLE);
    for (uint32_t m = 0; m < out.mipLevels; ++m) {
      vci.subresourceRange.baseMipLevel = m;
      vci.subresourceRange.levelCount = 1;
      vkCreateImageView(m_device, &vci, nullptr, &out.mipViews[m]);
    }
  }
  return out;
}

void ShowcaseRenderer::destroyImage(GpuImage &img) {
  if (m_device != VK_NULL_HANDLE) {
    for (VkImageView mv : img.mipViews) {
      if (mv != VK_NULL_HANDLE) vkDestroyImageView(m_device, mv, nullptr);
    }
    if (img.view != VK_NULL_HANDLE) vkDestroyImageView(m_device, img.view, nullptr);
    if (img.image != VK_NULL_HANDLE && m_vma != VK_NULL_HANDLE) {
      vmaDestroyImage(m_vma, img.image, img.allocation);
    }
  }
  img = {};
}

VkShaderModule ShowcaseRenderer::loadShaderModule(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open shader: " + path);
  }
  size_t sz = static_cast<size_t>(f.tellg());
  std::vector<uint32_t> code(sz / sizeof(uint32_t));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(sz));

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code.size() * sizeof(uint32_t);
  smci.pCode = code.data();
  VkShaderModule mod = VK_NULL_HANDLE;
  if (vkCreateShaderModule(m_device, &smci, nullptr, &mod) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module: " + path);
  }
  return mod;
}

void ShowcaseRenderer::transitionImage(VkCommandBuffer cmd, GpuImage &img, VkImageLayout newLayout,
                                       VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                       VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                                       VkImageAspectFlags aspect) {
  VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
  b.srcStageMask = srcStage;
  b.srcAccessMask = srcAccess;
  b.dstStageMask = dstStage;
  b.dstAccessMask = dstAccess;
  b.oldLayout = img.currentLayout;
  b.newLayout = newLayout;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img.image;
  b.subresourceRange = {aspect, 0, img.mipLevels, 0, 1};
  m_ctx->cmdImageMemoryBarrier2(cmd, &b, 1);
  img.currentLayout = newLayout;
}

bool ShowcaseRenderer::createSceneGpuResources() {
  const VkBufferUsageFlags ssboDev = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  auto uploadVec = [&](const auto &vec, VkBufferUsageFlags usage) -> GpuBuffer {
    VkDeviceSize bytes = std::max<VkDeviceSize>(64u, static_cast<VkDeviceSize>(vec.size()) * sizeof(vec[0]));
    GpuBuffer b = createBuffer(bytes, usage, false);
    if (!vec.empty()) {
      uploadToDeviceBuffer(b, vec.data(), static_cast<VkDeviceSize>(vec.size()) * sizeof(vec[0]));
    }
    return b;
  };

  m_vb = uploadVec(m_scene.vertices, ssboDev);
  m_ib = uploadVec(m_scene.indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | ssboDev);
  m_meshletBuf = uploadVec(m_scene.meshlets, ssboDev);
  m_meshletVertBuf = uploadVec(m_scene.meshletVertices, ssboDev);
  m_meshletTriBuf = uploadVec(m_scene.meshletTriangles, ssboDev);
  m_instanceBuf = uploadVec(m_scene.instances, ssboDev);
  m_materialBuf = uploadVec(m_scene.materials, ssboDev);
  m_lightBuf = uploadVec(m_scene.lights, ssboDev);
  m_taskPkt0Buf = uploadVec(m_scene.phase0Packets, ssboDev);
  m_taskPkt1Buf = uploadVec(m_scene.phase1Packets, ssboDev);

  // DGC Buffer: [0..15] sequenceCount, [16..] 8 x 48B ShowcaseDgcSeq
  m_dgcBuf = createBuffer(sizeof(uint32_t) * 4u + 48u * 8u,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          false);
  m_fallbackBuf = createBuffer(16u + kMaxFallbackDraws * 32u,
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               false);

  // Upload scene textures with mipmaps
  m_sceneTextures.resize(m_scene.textures.size());
  for (size_t i = 0; i < m_scene.textures.size(); ++i) {
    const auto &src = m_scene.textures[i];
    uint32_t w = static_cast<uint32_t>(std::max(1, src.width));
    uint32_t h = static_cast<uint32_t>(std::max(1, src.height));
    uint32_t mips = static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1u;
    GpuImage img = createImage(w, h, mips, VK_FORMAT_R8G8B8A8_UNORM,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT);

    VkDeviceSize pxBytes = static_cast<VkDeviceSize>(w) * h * 4u;
    GpuBuffer stg = createBuffer(pxBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    if (!src.pixels.empty()) {
      std::memcpy(stg.mapped, src.pixels.data(), static_cast<size_t>(pxBytes));
    } else {
      std::memset(stg.mapped, 255, static_cast<size_t>(pxBytes));
    }
    vmaFlushAllocation(m_vma, stg.allocation, 0, pxBytes);

    {
      std::lock_guard<std::recursive_mutex> qlock(m_ctx->getQueueMutex());
      vkResetCommandPool(m_device, m_uploadCmdPool, 0);
      VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(m_headlessCmd, &bi);

      VkImageMemoryBarrier2 bInit{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      bInit.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
      bInit.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      bInit.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      bInit.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      bInit.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      bInit.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bInit.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bInit.image = img.image;
      bInit.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
      m_ctx->cmdImageMemoryBarrier2(m_headlessCmd, &bInit, 1);

      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {w, h, 1};
      vkCmdCopyBufferToImage(m_headlessCmd, stg.buffer, img.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

      int32_t mipW = static_cast<int32_t>(w), mipH = static_cast<int32_t>(h);
      for (uint32_t m = 1u; m < mips; ++m) {
        VkImageMemoryBarrier2 bSrc{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        bSrc.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        bSrc.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bSrc.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        bSrc.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        bSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        bSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bSrc.image = img.image;
        bSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1u, 1, 0, 1};
        m_ctx->cmdImageMemoryBarrier2(m_headlessCmd, &bSrc, 1);

        int32_t nextW = std::max(1, mipW / 2), nextH = std::max(1, mipH / 2);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1u, 0, 1};
        blit.srcOffsets[1] = {mipW, mipH, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
        blit.dstOffsets[1] = {nextW, nextH, 1};
        vkCmdBlitImage(m_headlessCmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        VkImageMemoryBarrier2 bDone = bSrc;
        bDone.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        bDone.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        bDone.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        bDone.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        bDone.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        bDone.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        m_ctx->cmdImageMemoryBarrier2(m_headlessCmd, &bDone, 1);

        mipW = nextW;
        mipH = nextH;
      }

      VkImageMemoryBarrier2 bLast{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
      bLast.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
      bLast.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      bLast.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      bLast.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
      bLast.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      bLast.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      bLast.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bLast.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bLast.image = img.image;
      bLast.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mips - 1u, 1, 0, 1};
      m_ctx->cmdImageMemoryBarrier2(m_headlessCmd, &bLast, 1);

      vkEndCommandBuffer(m_headlessCmd);
      vkResetFences(m_device, 1, &m_headlessFence);
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.commandBufferCount = 1;
      si.pCommandBuffers = &m_headlessCmd;
      vkQueueSubmit(m_ctx->getGraphicsQueue(), 1, &si, m_headlessFence);
      vkWaitForFences(m_device, 1, &m_headlessFence, VK_TRUE, UINT64_MAX);
    }
    destroyBuffer(stg);
    img.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    m_sceneTextures[i] = std::move(img);
  }

  // Create Directional Shadow Map (2048 x 2048 D32_SFLOAT)
  m_shadowMap = createImage(kShadowSize, kShadowSize, 1u, VK_FORMAT_D32_SFLOAT,
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            VK_IMAGE_ASPECT_DEPTH_BIT);

  // Allocate and populate Set 0 (Static Scene)
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = m_descPool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &m_sceneSetLayout;
  if (vkAllocateDescriptorSets(m_device, &dsai, &m_sceneSet) != VK_SUCCESS) {
    return false;
  }

  GpuBuffer *bufs[9] = {&m_vb, &m_meshletBuf, &m_meshletVertBuf, &m_meshletTriBuf,
                        &m_instanceBuf, &m_materialBuf, &m_lightBuf, &m_taskPkt0Buf, &m_taskPkt1Buf};
  VkDescriptorBufferInfo bufInfos[9]{};
  VkWriteDescriptorSet writes[10]{};
  for (uint32_t i = 0; i < 9u; ++i) {
    bufInfos[i] = {bufs[i]->buffer, 0, VK_WHOLE_SIZE};
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = m_sceneSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufInfos[i];
  }

  std::vector<VkDescriptorImageInfo> texInfos(std::min<size_t>(kMaxShowcaseTextures, m_sceneTextures.size()));
  for (size_t i = 0; i < texInfos.size(); ++i) {
    texInfos[i] = {m_texSampler, m_sceneTextures[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
  }
  writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[9].dstSet = m_sceneSet;
  writes[9].dstBinding = 9;
  writes[9].descriptorCount = static_cast<uint32_t>(texInfos.size());
  writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writes[9].pImageInfo = texInfos.data();
  vkUpdateDescriptorSets(m_device, texInfos.empty() ? 9u : 10u, writes, 0, nullptr);

  return true;
}

bool ShowcaseRenderer::createPipelines(const std::string &kernelDir) {
  auto loadMod = [&](const char *spv) {
    return loadShaderModule(resolveShowcaseShaderPath(kernelDir, spv));
  };

  struct StageSpec {
    VkShaderStageFlagBits stage;
    VkShaderModule mod;
  };
  auto buildGfxPipeline = [&](const std::vector<StageSpec> &stages,
                              const std::vector<VkFormat> &colorFormats,
                              bool depthTest, bool depthWrite, VkCompareOp depthOp,
                              bool tessellation, bool isLineList, bool noVertexInput,
                              VkPipelineLayout layout) -> VkPipeline {
    std::vector<VkPipelineShaderStageCreateInfo> sci(stages.size());
    for (size_t i = 0; i < stages.size(); ++i) {
      sci[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      sci[i].stage = stages[i].stage;
      sci[i].module = stages[i].mod;
      sci[i].pName = "main";
    }

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = tessellation ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST
                               : (isLineList ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

    VkPipelineTessellationStateCreateInfo ts{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
    ts.patchControlPoints = 3;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = isLineList ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = depthOp;

    std::vector<VkPipelineColorBlendAttachmentState> cba(colorFormats.size());
    for (auto &a : cba) {
      a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = static_cast<uint32_t>(cba.size());
    cb.pAttachments = cba.data();

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkPipelineRenderingCreateInfo dynRen{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    dynRen.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size());
    dynRen.pColorAttachmentFormats = colorFormats.empty() ? nullptr : colorFormats.data();
    dynRen.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.pNext = &dynRen;
    gpci.stageCount = static_cast<uint32_t>(sci.size());
    gpci.pStages = sci.data();
    gpci.pVertexInputState = noVertexInput ? nullptr : &vi;
    gpci.pInputAssemblyState = noVertexInput ? nullptr : &ia;
    gpci.pTessellationState = tessellation ? &ts : nullptr;
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pDepthStencilState = &ds;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dyn;
    gpci.layout = layout;

    VkPipeline outPipe = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(m_device, m_ctx->getPipelineCache(), 1, &gpci, nullptr, &outPipe) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create Showcase graphics pipeline");
    }
    return outPipe;
  };

  auto buildCompPipeline = [&](VkShaderModule mod, VkPipelineLayout layout,
                               const VkSpecializationInfo *specInfo = nullptr,
                               bool indirectBindable = false) -> VkPipeline {
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = specInfo;
    cpci.layout = layout;

    VkPipelineCreateFlags2CreateInfoKHR pf2{VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR};
    if (indirectBindable && m_ctx->isDGCSupported() && m_ctx->isMaintenance5Supported()) {
      pf2.flags = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT;
      cpci.pNext = &pf2;
    }

    VkPipeline outPipe = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(m_device, m_ctx->getPipelineCache(), 1, &cpci, nullptr, &outPipe) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create Showcase compute pipeline");
    }
    return outPipe;
  };

  std::vector<VkShaderModule> mods;
  auto track = [&](const char *spv) {
    VkShaderModule m = loadMod(spv);
    mods.push_back(m);
    return m;
  };

  try {
    VkShaderModule smShadowV = track("showcase_shadow.vert.spv");
    VkShaderModule smShadowF = track("showcase_shadow.frag.spv");
    VkShaderModule smSkyV = track("showcase_sky.vert.spv");
    VkShaderModule smSkyF = track("showcase_sky.frag.spv");
    VkShaderModule smNaiveV = track("showcase_naive.vert.spv");
    VkShaderModule smNaiveTessV = track("showcase_naive_tess.vert.spv");
    VkShaderModule smNaiveTesc = track("showcase_naive.tesc.spv");
    VkShaderModule smNaiveTese = track("showcase_naive.tese.spv");
    VkShaderModule smNaiveF = track("showcase_naive.frag.spv");
    VkShaderModule smLegGbufF = track("showcase_legacy_gbuf.frag.spv");
    VkShaderModule smLegLightC = track("showcase_legacy_lighting.comp.spv");
    VkShaderModule smHizC = track("showcase_hiz_reduce.comp.spv");
    VkShaderModule smVisClsC = track("showcase_vis_classify.comp.spv");
    VkShaderModule smVisResC = track("showcase_vis_resolve_dgc.comp.spv");
    VkShaderModule smVisShadeC = track("showcase_vis_shade.comp.spv");
    VkShaderModule smOverlayC = track("showcase_overlay_compose.comp.spv");
    VkShaderModule smWireV = track("showcase_frustum_wire.vert.spv");
    VkShaderModule smWireF = track("showcase_frustum_wire.frag.spv");

    m_pipeShadow = buildGfxPipeline({{VK_SHADER_STAGE_VERTEX_BIT, smShadowV}, {VK_SHADER_STAGE_FRAGMENT_BIT, smShadowF}},
                                    {}, true, true, VK_COMPARE_OP_LESS_OR_EQUAL, false, false, false, m_gfxLayout);

    m_pipeSky = buildGfxPipeline({{VK_SHADER_STAGE_VERTEX_BIT, smSkyV}, {VK_SHADER_STAGE_FRAGMENT_BIT, smSkyF}},
                                 {VK_FORMAT_R8G8B8A8_UNORM}, false, false, VK_COMPARE_OP_ALWAYS,
                                 false, true, false, m_gfxLayout);

    m_pipeNaiveFlat = buildGfxPipeline({{VK_SHADER_STAGE_VERTEX_BIT, smNaiveV}, {VK_SHADER_STAGE_FRAGMENT_BIT, smNaiveF}},
                                       {VK_FORMAT_R8G8B8A8_UNORM}, true, true, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                       false, false, false, m_gfxLayout);
    if (m_ctx->isTessellationSupported()) {
      m_pipeNaiveTess = buildGfxPipeline({{VK_SHADER_STAGE_VERTEX_BIT, smNaiveTessV},
                                          {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, smNaiveTesc},
                                          {VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, smNaiveTese},
                                          {VK_SHADER_STAGE_FRAGMENT_BIT, smNaiveF}},
                                         {VK_FORMAT_R8G8B8A8_UNORM}, true, true, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                         true, false, false, m_gfxLayout);
    }

    const std::vector<VkFormat> gbufFormats = {
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};
    m_pipeLegacyFlat = buildGfxPipeline({{VK_SHADER_STAGE_VERTEX_BIT, smNaiveV}, {VK_SHADER_STAGE_FRAGMENT_BIT, smLegGbufF}},
                                        gbufFormats, true, true, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                        false, false, false, m_gfxLayout);
    if (m_ctx->isTessellationSupported()) {
      m_pipeLegacyTess = buildGfxPipeline({{VK_SHADER_STAGE_VERTEX_BIT, smNaiveTessV},
                                           {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, smNaiveTesc},
                                           {VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, smNaiveTese},
                                           {VK_SHADER_STAGE_FRAGMENT_BIT, smLegGbufF}},
                                          gbufFormats, true, true, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                          true, false, false, m_gfxLayout);
    }

    m_pipeLegacyLight = buildCompPipeline(smLegLightC, m_gfxLayout);
    m_pipeHizReduce = buildCompPipeline(smHizC, m_hizLayout);
    m_pipeVisClassify = buildCompPipeline(smVisClsC, m_gfxLayout);
    m_pipeVisResolve = buildCompPipeline(smVisResC, m_gfxLayout);

    if (m_ctx->isMeshShaderSupported()) {
      VkShaderModule smTask = track("showcase_modern.task.spv");
      VkShaderModule smMesh = track("showcase_modern.mesh.spv");
      VkShaderModule smVisOpaqueF = track("showcase_visbuf_opaque.frag.spv");
      VkShaderModule smVisF = track("showcase_visbuf.frag.spv");
      VkShaderModule smModFwdF = track("showcase_modern_fwd.frag.spv");
      m_pipeModernVisOpaque = buildGfxPipeline({{VK_SHADER_STAGE_TASK_BIT_EXT, smTask},
                                                {VK_SHADER_STAGE_MESH_BIT_EXT, smMesh},
                                                {VK_SHADER_STAGE_FRAGMENT_BIT, smVisOpaqueF}},
                                               {VK_FORMAT_R32G32_UINT}, true, true, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                               false, false, true, m_taskLayout);
      m_pipeModernVis = buildGfxPipeline({{VK_SHADER_STAGE_TASK_BIT_EXT, smTask},
                                          {VK_SHADER_STAGE_MESH_BIT_EXT, smMesh},
                                          {VK_SHADER_STAGE_FRAGMENT_BIT, smVisF}},
                                         {VK_FORMAT_R32G32_UINT}, true, true, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                         false, false, true, m_taskLayout);
      m_pipeModernFwd = buildGfxPipeline({{VK_SHADER_STAGE_TASK_BIT_EXT, smTask},
                                          {VK_SHADER_STAGE_MESH_BIT_EXT, smMesh},
                                          {VK_SHADER_STAGE_FRAGMENT_BIT, smModFwdF}},
                                         {VK_FORMAT_R8G8B8A8_UNORM}, true, true, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                         false, false, true, m_taskLayout);
    }

    VkShaderModule smFbCullC = track("showcase_fallback_cull.comp.spv");
    VkShaderModule smFbV = track("showcase_fallback.vert.spv");
    VkShaderModule smFbF = track("showcase_fallback_vis.frag.spv");
    m_pipeFallbackCull = buildCompPipeline(smFbCullC, m_compLayout);
    m_pipeFallbackVis = buildGfxPipeline({{VK_SHADER_STAGE_VERTEX_BIT, smFbV}, {VK_SHADER_STAGE_FRAGMENT_BIT, smFbF}},
                                         {VK_FORMAT_R32G32_UINT}, true, true, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                         false, false, false, m_gfxLayout);

    // 8 specialized VisBuffer shading pipelines + unified (id = 8)
    for (uint32_t a = 0; a < 8u; ++a) {
      VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
      VkSpecializationInfo si{1, &entry, sizeof(uint32_t), &a};
      m_pipeVisShadeSpec[a] = buildCompPipeline(smVisShadeC, m_visShadeSpecLayout, &si, true);
    }
    {
      uint32_t unifiedVal = 8u;
      VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
      VkSpecializationInfo si{1, &entry, sizeof(uint32_t), &unifiedVal};
      m_pipeVisShadeUnified = buildCompPipeline(smVisShadeC, m_visShadeSpecLayout, &si, false);
    }

    m_pipeOverlayCompose = buildCompPipeline(smOverlayC, m_gfxLayout);
    m_pipeFrustumWire = buildGfxPipeline({{VK_SHADER_STAGE_VERTEX_BIT, smWireV}, {VK_SHADER_STAGE_FRAGMENT_BIT, smWireF}},
                                         {VK_FORMAT_R8G8B8A8_UNORM}, true, false, VK_COMPARE_OP_GREATER_OR_EQUAL,
                                         false, true, false, m_gfxLayout);
  } catch (const std::exception &e) {
    std::cerr << "[ShowcaseRenderer] Pipeline build error: " << e.what() << "\n";
    for (VkShaderModule m : mods) vkDestroyShaderModule(m_device, m, nullptr);
    return false;
  }

  for (VkShaderModule m : mods) vkDestroyShaderModule(m_device, m, nullptr);
  return true;
}

bool ShowcaseRenderer::createDgcExecutionSet() {
  if (!m_ctx->isDGCSupported() || !m_ctx->isDGCExecutionSetSupported()) {
    return false;
  }
  auto pfnGetMemReq = (PFN_vkGetGeneratedCommandsMemoryRequirementsEXT)vkGetDeviceProcAddr(
      m_device, "vkGetGeneratedCommandsMemoryRequirementsEXT");
  auto pfnUpdateSet = (PFN_vkUpdateIndirectExecutionSetPipelineEXT)vkGetDeviceProcAddr(
      m_device, "vkUpdateIndirectExecutionSetPipelineEXT");
  if (!pfnGetMemReq || !pfnUpdateSet) {
    return false;
  }

  try {
    VkIndirectCommandsPushConstantTokenEXT pcTok{};
    pcTok.updateRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 8u};

    VkIndirectCommandsLayoutTokenEXT tokens[3]{};
    tokens[0].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
    tokens[0].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT;
    tokens[0].data.pExecutionSet = nullptr;
    VkIndirectCommandsExecutionSetTokenEXT esTok{VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT,
                                                 VK_SHADER_STAGE_COMPUTE_BIT};
    tokens[0].data.pExecutionSet = &esTok;
    tokens[0].offset = 0;

    tokens[1].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
    tokens[1].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT;
    tokens[1].data.pPushConstant = &pcTok;
    tokens[1].offset = 4;

    tokens[2].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
    tokens[2].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
    tokens[2].offset = 36;

    VkIndirectCommandsLayoutCreateInfoEXT layoutCi{VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT};
    layoutCi.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutCi.indirectStride = 48u;
    layoutCi.pipelineLayout = m_visShadeSpecLayout;
    layoutCi.tokenCount = 3;
    layoutCi.pTokens = tokens;
    m_dgcLayout = m_ctx->createIndirectCommandsLayout(layoutCi);

    VkIndirectExecutionSetPipelineInfoEXT pipeInfo{VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT};
    pipeInfo.initialPipeline = m_pipeVisShadeSpec[0];
    pipeInfo.maxPipelineCount = 8;

    VkIndirectExecutionSetCreateInfoEXT iesCi{VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT};
    iesCi.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
    iesCi.info.pPipelineInfo = &pipeInfo;
    m_dgcExecSet = m_ctx->createIndirectExecutionSet(iesCi);

    std::vector<VkWriteIndirectExecutionSetPipelineEXT> writes(8);
    for (uint32_t m = 0; m < 8u; ++m) {
      writes[m].sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT;
      writes[m].index = m;
      writes[m].pipeline = m_pipeVisShadeSpec[m];
    }
    pfnUpdateSet(m_device, m_dgcExecSet, 8u, writes.data());

    VkGeneratedCommandsMemoryRequirementsInfoEXT reqInfo{
        VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT};
    reqInfo.indirectCommandsLayout = m_dgcLayout;
    reqInfo.indirectExecutionSet = m_dgcExecSet;
    reqInfo.maxSequenceCount = 8u;
    VkMemoryRequirements2 memReqs{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    pfnGetMemReq(m_device, &reqInfo, &memReqs);

    VkDeviceSize prepSz = std::max<VkDeviceSize>(4096u, memReqs.memoryRequirements.size);
    VkBufferUsageFlags2CreateInfoKHR u2{VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR};
    u2.usage = VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT |
               VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR |
               VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT_KHR;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &u2;
    bci.size = prepSz;
    bci.usage = 0;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_device, &bci, nullptr, &m_dgcPreprocessBuf.buffer) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateBuffer failed for DGC preprocess buffer");
    }

    VkMemoryRequirements bufMemReqs{};
    vkGetBufferMemoryRequirements(m_device, m_dgcPreprocessBuf.buffer, &bufMemReqs);
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(m_physDev, &memProps);
    uint32_t memTypeIdx = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
      if ((bufMemReqs.memoryTypeBits & (1u << i)) &&
          (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        memTypeIdx = i;
        break;
      }
    }
    if (memTypeIdx == UINT32_MAX) {
      throw std::runtime_error("No DEVICE_LOCAL memory type for DGC preprocess buffer");
    }

    VkMemoryAllocateFlagsInfo allocFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &allocFlags;
    mai.allocationSize = bufMemReqs.size;
    mai.memoryTypeIndex = memTypeIdx;
    if (vkAllocateMemory(m_device, &mai, nullptr, &m_dgcPreprocessMem) != VK_SUCCESS) {
      throw std::runtime_error("vkAllocateMemory failed for DGC preprocess buffer");
    }
    if (vkBindBufferMemory(m_device, m_dgcPreprocessBuf.buffer, m_dgcPreprocessMem, 0) != VK_SUCCESS) {
      throw std::runtime_error("vkBindBufferMemory failed for DGC preprocess buffer");
    }

    m_dgcPreprocessBuf.size = prepSz;
    VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_dgcPreprocessBuf.buffer};
    m_dgcPreprocessBuf.address = vkGetBufferDeviceAddress(m_device, &ai);
    m_dgcExecSetReady = true;
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[ShowcaseRenderer] DGC ExecutionSet init failed: " << e.what() << "\n";
    return false;
  }
}

bool ShowcaseRenderer::createFrameSlots() {
  for (uint32_t i = 0; i < kMaxSlots; ++i) {
    auto &s = m_slots[i];
    s.uboBuf = createBuffer(sizeof(GpuUniforms),
                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    s.statsBuf = createBuffer(sizeof(GpuStatsReadback), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    VkDeviceSize indSz = std::max<VkDeviceSize>(4096u, m_scene.instances.size() * sizeof(VkDrawIndexedIndirectCommand));
    s.legacyIndirectBuf = createBuffer(indSz, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, true);

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = m_descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &m_frameSetLayout;
    if (vkAllocateDescriptorSets(m_device, &dsai, &s.frameSet) != VK_SUCCESS) {
      return false;
    }

    VkQueryPoolCreateInfo tsCi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    tsCi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    tsCi.queryCount = 6;
    vkCreateQueryPool(m_device, &tsCi, nullptr, &s.tsPool);

    VkQueryPoolCreateInfo psCi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    psCi.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
    psCi.queryCount = 1;
    psCi.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                              VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
    vkCreateQueryPool(m_device, &psCi, nullptr, &s.pipeStatsPool);
  }
  return true;
}

void ShowcaseRenderer::destroyFrameSlots() {
  for (auto &s : m_slots) {
    destroyBuffer(s.uboBuf);
    destroyBuffer(s.statsBuf);
    destroyBuffer(s.legacyIndirectBuf);
    if (m_device != VK_NULL_HANDLE) {
      if (s.tsPool != VK_NULL_HANDLE) vkDestroyQueryPool(m_device, s.tsPool, nullptr);
      if (s.pipeStatsPool != VK_NULL_HANDLE) vkDestroyQueryPool(m_device, s.pipeStatsPool, nullptr);
    }
    s = {};
  }
}

bool ShowcaseRenderer::createRenderTargets(uint32_t w, uint32_t h) {
  m_width = w;
  m_height = h;
  const uint32_t hizMips = static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1u;
  const VkImageUsageFlags colorRtUse =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  m_depthImg = createImage(w, h, 1u, VK_FORMAT_D32_SFLOAT,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT);
  m_hizImg = createImage(w, h, hizMips, VK_FORMAT_R32_SFLOAT,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT);
  m_gbuf0 = createImage(w, h, 1u, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
  m_gbuf1 = createImage(w, h, 1u, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
  m_gbuf2 = createImage(w, h, 1u, VK_FORMAT_R32G32B32A32_SFLOAT,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_IMAGE_ASPECT_COLOR_BIT);
  m_visBufImg = createImage(w, h, 1u, VK_FORMAT_R32G32_UINT,
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            VK_IMAGE_ASPECT_COLOR_BIT);
  m_overdrawImg = createImage(w, h, 1u, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT);
  m_triAreaImg = createImage(w, h, 1u, VK_FORMAT_R32_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT);
  m_debugTagImg = createImage(w, h, 1u, VK_FORMAT_R32_UINT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_IMAGE_ASPECT_COLOR_BIT);
  m_sceneColorImg = createImage(w, h, 1u, VK_FORMAT_R8G8B8A8_UNORM, colorRtUse, VK_IMAGE_ASPECT_COLOR_BIT);
  m_altColorImg = createImage(w, h, 1u, VK_FORMAT_R8G8B8A8_UNORM, colorRtUse, VK_IMAGE_ASPECT_COLOR_BIT);
  m_displayColor = createImage(w, h, 1u, VK_FORMAT_R8G8B8A8_UNORM, colorRtUse, VK_IMAGE_ASPECT_COLOR_BIT);

  // Per-pixel compacted archetype worklist buffer: [0..15] counters, [16..] 8 x w*h uint32 coords
  const VkDeviceSize qBytes = 64u + static_cast<VkDeviceSize>(8u) * w * h * sizeof(uint32_t);
  m_worklistBuf = createBuffer(qBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);

  // Transition initial image layouts once cleanly & zero worklist counters
  {
    std::lock_guard<std::recursive_mutex> qlock(m_ctx->getQueueMutex());
    vkResetCommandPool(m_device, m_uploadCmdPool, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_headlessCmd, &bi);

    vkCmdFillBuffer(m_headlessCmd, m_worklistBuf.buffer, 0, 64, 0);
    transitionImage(m_headlessCmd, m_hizImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_depthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
    transitionImage(m_headlessCmd, m_gbuf0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_gbuf1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_gbuf2, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_visBufImg, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_overdrawImg, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_triAreaImg, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_debugTagImg, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_sceneColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_altColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(m_headlessCmd, m_displayColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    vkEndCommandBuffer(m_headlessCmd);
    vkResetFences(m_device, 1, &m_headlessFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_headlessCmd;
    vkQueueSubmit(m_ctx->getGraphicsQueue(), 1, &si, m_headlessFence);
    vkWaitForFences(m_device, 1, &m_headlessFence, VK_TRUE, UINT64_MAX);
  }

  // Rebind Frame Descriptor Sets for all slots
  for (uint32_t i = 0; i < kMaxSlots; ++i) {
    auto &s = m_slots[i];
    VkDescriptorBufferInfo uboI{s.uboBuf.buffer, 0, sizeof(GpuUniforms)};
    VkDescriptorBufferInfo stI{s.statsBuf.buffer, 0, sizeof(GpuStatsReadback)};
    VkDescriptorBufferInfo wlI{m_worklistBuf.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dgcI{m_dgcBuf.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo fbI{m_fallbackBuf.buffer, 0, VK_WHOLE_SIZE};

    VkDescriptorImageInfo shI{m_shadowSampler, m_shadowMap.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo hzI{m_hizSampler, m_hizImg.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo ovI{VK_NULL_HANDLE, m_overdrawImg.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo taI{VK_NULL_HANDLE, m_triAreaImg.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo vbI{VK_NULL_HANDLE, m_visBufImg.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo dpI{m_pointSampler, m_depthImg.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo g0I{m_pointSampler, m_gbuf0.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo g1I{m_pointSampler, m_gbuf1.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo g2I{m_pointSampler, m_gbuf2.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo scImgI{VK_NULL_HANDLE, m_sceneColorImg.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo scTexI{m_pointSampler, m_sceneColorImg.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo dsI{VK_NULL_HANDLE, m_displayColor.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo altI{m_pointSampler, m_altColorImg.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo dtI{VK_NULL_HANDLE, m_debugTagImg.view, VK_IMAGE_LAYOUT_GENERAL};

    VkWriteDescriptorSet wds[19]{};
    auto wb = [&](uint32_t b, VkDescriptorType t, const VkDescriptorBufferInfo *bi) {
      wds[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, s.frameSet, b, 0, 1, t, nullptr, bi, nullptr};
    };
    auto wi = [&](uint32_t b, VkDescriptorType t, const VkDescriptorImageInfo *ii) {
      wds[b] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, s.frameSet, b, 0, 1, t, ii, nullptr, nullptr};
    };
    wb(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &uboI);
    wb(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &stI);
    wi(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shI);
    wi(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hzI);
    wi(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ovI);
    wi(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &taI);
    wi(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &vbI);
    wi(7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dpI);
    wi(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &g0I);
    wi(9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &g1I);
    wi(10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &g2I);
    wi(11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &scImgI);
    wi(12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &scTexI);
    wi(13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dsI);
    wi(14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &altI);
    wb(15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &wlI);
    wb(16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &dgcI);
    wi(17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dtI);
    wb(18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &fbI);
    vkUpdateDescriptorSets(m_device, 19, wds, 0, nullptr);
  }

  // Allocate & update Hi-Z per-mip descriptor sets
  if (!m_hizMipSets.empty()) {
    vkFreeDescriptorSets(m_device, m_descPool, static_cast<uint32_t>(m_hizMipSets.size()), m_hizMipSets.data());
    m_hizMipSets.clear();
  }
  m_hizMipSets.resize(hizMips, VK_NULL_HANDLE);
  std::vector<VkDescriptorSetLayout> hizLayouts(hizMips, m_hizSetLayout);
  VkDescriptorSetAllocateInfo hizDsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  hizDsai.descriptorPool = m_descPool;
  hizDsai.descriptorSetCount = hizMips;
  hizDsai.pSetLayouts = hizLayouts.data();
  if (vkAllocateDescriptorSets(m_device, &hizDsai, m_hizMipSets.data()) != VK_SUCCESS) {
    return false;
  }
  for (uint32_t m = 0; m < hizMips; ++m) {
    VkImageView srcV = (m == 0u) ? m_depthImg.view : m_hizImg.mipViews[m - 1u];
    VkImageView dstV = (hizMips > 1u) ? m_hizImg.mipViews[m] : m_hizImg.view;
    VkDescriptorImageInfo sI{m_pointSampler, srcV, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo dI{VK_NULL_HANDLE, dstV, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet hw[2]{};
    hw[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_hizMipSets[m], 0, 0, 1,
             VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &sI, nullptr, nullptr};
    hw[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_hizMipSets[m], 1, 0, 1,
             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dI, nullptr, nullptr};
    vkUpdateDescriptorSets(m_device, 2, hw, 0, nullptr);
  }

  m_imageGeneration++;
  return true;
}

void ShowcaseRenderer::destroyRenderTargets() {
  destroyBuffer(m_worklistBuf);
  destroyImage(m_depthImg);
  destroyImage(m_hizImg);
  destroyImage(m_gbuf0);
  destroyImage(m_gbuf1);
  destroyImage(m_gbuf2);
  destroyImage(m_visBufImg);
  destroyImage(m_overdrawImg);
  destroyImage(m_triAreaImg);
  destroyImage(m_debugTagImg);
  destroyImage(m_sceneColorImg);
  destroyImage(m_altColorImg);
  destroyImage(m_displayColor);
  if (m_device != VK_NULL_HANDLE && m_descPool != VK_NULL_HANDLE && !m_hizMipSets.empty()) {
    vkFreeDescriptorSets(m_device, m_descPool, static_cast<uint32_t>(m_hizMipSets.size()), m_hizMipSets.data());
    m_hizMipSets.clear();
  }
}

void ShowcaseRenderer::resize(uint32_t width, uint32_t height) {
  width = std::clamp(width, 320u, 3840u);
  height = std::clamp(height, 240u, 2160u);
  if (!m_initialized || (width == m_width && height == m_height)) {
    return;
  }
  {
    std::lock_guard<std::recursive_mutex> qlock(m_ctx->getQueueMutex());
    vkDeviceWaitIdle(m_device);
  }
  for (auto &s : m_slots) s.submitted = false;
  destroyRenderTargets();
  createRenderTargets(width, height);
}

void ShowcaseRenderer::renderShadowMapOnce() {
  // Orthographic sun shadow map enclosing the Sponza courtyard
  Vec3 target{0.0f, 180.0f, 0.0f};
  Vec3 sunEye = target + normalize(m_scene.sunDir) * 3200.0f;
  Mat4 lightView = matLookAt(sunEye, target, {0.0f, 1.0f, 0.0f});
  Mat4 lightProj = matOrthoVulkan(-2100.0f, 2100.0f, -1500.0f, 1500.0f, 200.0f, 6500.0f);
  m_shadowViewProj = matMul(lightProj, lightView);

  GpuUniforms u{};
  std::memcpy(u.shadowViewProj, m_shadowViewProj.m, sizeof(u.shadowViewProj));
  std::memcpy(m_slots[0].uboBuf.mapped, &u, sizeof(u));
  vmaFlushAllocation(m_vma, m_slots[0].uboBuf.allocation, 0, sizeof(u));

  std::lock_guard<std::recursive_mutex> qlock(m_ctx->getQueueMutex());
  vkResetCommandPool(m_device, m_uploadCmdPool, 0);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(m_headlessCmd, &bi);

  transitionImage(m_headlessCmd, m_shadowMap, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT);

  VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depthAtt.imageView = m_shadowMap.view;
  depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depthAtt.clearValue.depthStencil = {1.0f, 0};

  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, {kShadowSize, kShadowSize}};
  ri.layerCount = 1;
  ri.pDepthAttachment = &depthAtt;

  m_ctx->getCmdBeginRendering()(m_headlessCmd, &ri);
  VkViewport vp{0.0f, 0.0f, static_cast<float>(kShadowSize), static_cast<float>(kShadowSize), 0.0f, 1.0f};
  VkRect2D sc{{0, 0}, {kShadowSize, kShadowSize}};
  vkCmdSetViewport(m_headlessCmd, 0, 1, &vp);
  vkCmdSetScissor(m_headlessCmd, 0, 1, &sc);

  vkCmdBindPipeline(m_headlessCmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeShadow);
  VkDescriptorSet sets[2] = {m_sceneSet, m_slots[0].frameSet};
  vkCmdBindDescriptorSets(m_headlessCmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gfxLayout, 0, 2, sets, 0, nullptr);
  vkCmdBindIndexBuffer(m_headlessCmd, m_ib.buffer, 0, VK_INDEX_TYPE_UINT32);

  for (uint32_t i = 0; i < static_cast<uint32_t>(m_scene.instances.size()); ++i) {
    const auto &inst = m_scene.instances[i];
    if (inst.baseIndexCount == 0u) continue;
    vkCmdDrawIndexed(m_headlessCmd, inst.baseIndexCount, 1, inst.baseFirstIndex, inst.vertexOffset, i);
  }
  m_ctx->getCmdEndRendering()(m_headlessCmd);

  transitionImage(m_headlessCmd, m_shadowMap, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT);

  vkEndCommandBuffer(m_headlessCmd);
  vkResetFences(m_device, 1, &m_headlessFence);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &m_headlessCmd;
  vkQueueSubmit(m_ctx->getGraphicsQueue(), 1, &si, m_headlessFence);
  vkWaitForFences(m_device, 1, &m_headlessFence, VK_TRUE, UINT64_MAX);
}

void ShowcaseRenderer::readbackCompletedSlot(uint32_t slotIdx, FrameTelemetry &outTel, bool updateHistory) {
  const auto &slot = m_slots[slotIdx];
  if (!slot.submitted) return;

  uint64_t timestamps[6]{};
  double periodNs = m_ctx->getPhysicalDeviceProperties().limits.timestampPeriod;
  if (periodNs > 0.0 && slot.tsPool != VK_NULL_HANDLE) {
    VkResult tr = vkGetQueryPoolResults(m_device, slot.tsPool, 0, 6, sizeof(timestamps), timestamps,
                                        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (tr == VK_SUCCESS && timestamps[5] > timestamps[0]) {
      auto deltaMs = [&](int a, int b) -> double {
        return (timestamps[b] > timestamps[a])
                   ? (static_cast<double>(timestamps[b] - timestamps[a]) * periodNs * 1e-6)
                   : 0.0;
      };
      outTel.totalMs = deltaMs(0, 5);
      outTel.fps = (outTel.totalMs > 0.001) ? (1000.0 / outTel.totalMs) : 0.0;
      if (slot.recordedMode == RenderMode::Modern) {
        outTel.cullHizMs = deltaMs(1, 2);
        outTel.geometryMs = deltaMs(0, 1) + deltaMs(2, 3);
        outTel.shadingMs = deltaMs(3, 4);
      } else if (slot.recordedMode == RenderMode::Legacy) {
        outTel.cullHizMs = 0.0;
        outTel.geometryMs = deltaMs(0, 2);
        outTel.shadingMs = deltaMs(2, 4);
      } else {
        outTel.cullHizMs = 0.0;
        outTel.geometryMs = deltaMs(0, 2);
        outTel.shadingMs = deltaMs(2, 4);
      }
    }
  }

  if (slot.usedPipeStats && slot.pipeStatsPool != VK_NULL_HANDLE) {
    uint64_t pipeStats[2]{};
    if (vkGetQueryPoolResults(m_device, slot.pipeStatsPool, 0, 1, sizeof(pipeStats), pipeStats,
                              sizeof(pipeStats), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
      outTel.clippedPrimitives = pipeStats[0];
      outTel.fragmentInvocations = pipeStats[1];
    }
  }

  vmaInvalidateAllocation(m_vma, slot.statsBuf.allocation, 0, sizeof(GpuStatsReadback));
  GpuStatsReadback st{};
  std::memcpy(&st, slot.statsBuf.mapped, sizeof(st));

  outTel.drawCalls = slot.recordedDrawCalls;
  outTel.meshShaderSupported = m_telemetry.meshShaderSupported;
  outTel.dgcExecutionSetSupported = m_telemetry.dgcExecutionSetSupported;
  outTel.barycentricSupported = m_telemetry.barycentricSupported;
  outTel.tessellationSupported = m_telemetry.tessellationSupported;
  outTel.meshShaderActive = (slot.recordedMode == RenderMode::Modern) && m_telemetry.meshShaderSupported;
  outTel.dgcExecutionSetActive = (slot.recordedMode == RenderMode::Modern) &&
                                 slot.recordedOpts.visBuffer &&
                                 slot.recordedOpts.dgcMaterialBinning &&
                                 m_telemetry.dgcExecutionSetSupported;
  outTel.barycentricActive = ((slot.recordedMode != RenderMode::Modern) || !slot.recordedOpts.visBuffer) &&
                             m_telemetry.barycentricSupported;
  outTel.tessellationActive = (slot.recordedMode == RenderMode::Naive || slot.recordedMode == RenderMode::Legacy) &&
                              m_telemetry.tessellationSupported;
  outTel.totalMeshlets = st.totalMeshlets;
  outTel.renderedMeshlets = st.renderedMeshlets;
  outTel.lodCulled = st.lodCulled;
  outTel.frustumCulled = st.frustumCulled;
  outTel.coneCulled = st.coneCulled;
  outTel.hizCulled = st.hizCulled;
  uint32_t culledCount = st.lodCulled + st.frustumCulled + st.coneCulled + st.hizCulled;
  outTel.culledPct = (st.totalMeshlets > 0u)
                         ? (static_cast<double>(culledCount) / static_cast<double>(st.totalMeshlets) * 100.0)
                         : 0.0;
  outTel.renderedTriangles = (st.renderedTriangles > 0u)
                                 ? st.renderedTriangles
                                 : static_cast<uint32_t>(std::min<uint64_t>(outTel.clippedPrimitives, 0xFFFFFFFFull));
  if (slot.recordedMode == RenderMode::Modern && st.renderedTriangles > 0u) {
    outTel.clippedPrimitives = st.renderedTriangles;
  }
  outTel.activeDgcBins = st.activeDgcBins;

  outTel.avgOverdraw = (st.totalShadedPixels > 0u)
                           ? std::max(1.0, static_cast<double>(st.overdrawSum) / static_cast<double>(st.totalShadedPixels))
                           : 1.0;
  outTel.subPixelTriPct = (st.totalShadedPixels > 0u)
                              ? (static_cast<double>(st.subPixelTriCount) / static_cast<double>(st.totalShadedPixels) * 100.0)
                              : 0.0;

  outTel.hasParitySample = slot.splitWipeActive && (st.parityTotalPixels > 0u);
  if (outTel.hasParitySample) {
    double mse = (static_cast<double>(st.paritySqErrQuant) / 65536.0) / static_cast<double>(st.parityTotalPixels);
    outTel.parityPsnr = (mse > 1e-10) ? (-10.0 * std::log10(mse)) : 99.9;
    outTel.parityExactPct = static_cast<double>(st.parityExactPixels) / static_cast<double>(st.parityTotalPixels) * 100.0;
    outTel.parityNearPct = static_cast<double>(st.parityNearPixels) / static_cast<double>(st.parityTotalPixels) * 100.0;
  }

  if (updateHistory && outTel.totalMs > 0.0) {
    if (outTel.smoothedMs <= 0.0) {
      outTel.smoothedMs = outTel.totalMs;
      outTel.smoothedFps = outTel.fps;
    } else {
      outTel.smoothedMs = outTel.smoothedMs * 0.85 + outTel.totalMs * 0.15;
      outTel.smoothedFps = (outTel.smoothedMs > 0.001) ? (1000.0 / outTel.smoothedMs) : 0.0;
    }
    outTel.fpsHistory[outTel.historyCursor] = static_cast<float>(std::min(outTel.fps, 600.0));
    outTel.msHistory[outTel.historyCursor] = static_cast<float>(std::min(outTel.totalMs, 100.0));
    outTel.historyCursor = (outTel.historyCursor + 1u) % FrameTelemetry::kHistoryLen;
    outTel.historyCount = std::min(outTel.historyCount + 1u, FrameTelemetry::kHistoryLen);
  }
}

void ShowcaseRenderer::recordModePass(VkCommandBuffer cmd, uint32_t slotIdx, RenderMode mode,
                                      const ShowcaseSettings &cfg, bool writeCpuUbo, bool isMeasuredPass) {
  auto &slot = m_slots[slotIdx];
  const float aspect = static_cast<float>(m_width) / static_cast<float>(std::max(1u, m_height));
  Mat4 drawView = cfg.activeCam.viewMatrix();
  Mat4 drawProj = cfg.activeCam.projMatrix(aspect);
  Mat4 drawVP = matMul(drawProj, drawView);

  const CameraState &cullCam = cfg.freezeCulling ? cfg.frozenCam : cfg.activeCam;
  Mat4 cullView = cullCam.viewMatrix();
  Mat4 cullProj = cullCam.projMatrix(aspect);
  Mat4 cullVP = matMul(cullProj, cullView);
  Vec4 cullPlanes[6];
  extractFrustumPlanesReversedZ(cullVP, cullPlanes);

  if (writeCpuUbo) {
    GpuUniforms u{};
    std::memcpy(u.viewProj, drawVP.m, sizeof(u.viewProj));
    std::memcpy(u.prevViewProj, m_prevViewProj.m, sizeof(u.prevViewProj));
    std::memcpy(u.cullViewProj, cullVP.m, sizeof(u.cullViewProj));
    std::memcpy(u.shadowViewProj, m_shadowViewProj.m, sizeof(u.shadowViewProj));
    for (int p = 0; p < 6; ++p) {
      u.cullFrustumPlanes[p][0] = cullPlanes[p].x;
      u.cullFrustumPlanes[p][1] = cullPlanes[p].y;
      u.cullFrustumPlanes[p][2] = cullPlanes[p].z;
      u.cullFrustumPlanes[p][3] = cullPlanes[p].w;
    }
    u.camPosFov[0] = cfg.activeCam.pos.x;
    u.camPosFov[1] = cfg.activeCam.pos.y;
    u.camPosFov[2] = cfg.activeCam.pos.z;
    u.camPosFov[3] = std::tan(cfg.activeCam.fovY * 0.5f);
    u.cullCamPos[0] = cullCam.pos.x;
    u.cullCamPos[1] = cullCam.pos.y;
    u.cullCamPos[2] = cullCam.pos.z;
    u.cullCamPos[3] = std::tan(cullCam.fovY * 0.5f);
    u.sunDirIntensity[0] = m_scene.sunDir.x;
    u.sunDirIntensity[1] = m_scene.sunDir.y;
    u.sunDirIntensity[2] = m_scene.sunDir.z;
    u.sunDirIntensity[3] = 4.6f;
    u.screenInfo[0] = m_width;
    u.screenInfo[1] = m_height;
    u.screenInfo[2] = static_cast<uint32_t>(mode);
    u.screenInfo[3] = static_cast<uint32_t>(cfg.overlay);
    u.countsInfo[0] = static_cast<uint32_t>(m_scene.phase0Packets.size());
    u.countsInfo[1] = static_cast<uint32_t>(m_scene.phase1Packets.size());
    u.countsInfo[2] = static_cast<uint32_t>(m_scene.lights.size());
    u.countsInfo[3] = cfg.opts.toBitmask();
    u.miscParams[0] = cfg.splitWipeX;
    u.miscParams[1] = cfg.splitWipeCompare ? 1.0f : 0.0f;
    u.miscParams[2] = std::clamp(cfg.tessLevelMax, 1.0f, 64.0f);
    u.miscParams[3] = static_cast<float>(m_frameCounter) * 0.016f;
    u.frameMeta[0] = m_frameCounter;
    u.frameMeta[1] = m_hizImg.mipLevels;
    u.frameMeta[2] = m_width * m_height;
    u.frameMeta[3] = cfg.freezeCulling ? 1u : 0u;
    std::memcpy(slot.uboBuf.mapped, &u, sizeof(u));
    vmaFlushAllocation(m_vma, slot.uboBuf.allocation, 0, sizeof(u));

    std::memset(slot.statsBuf.mapped, 0, sizeof(GpuStatsReadback));
    vmaFlushAllocation(m_vma, slot.statsBuf.allocation, 0, sizeof(GpuStatsReadback));
    m_prevViewProj = drawVP;
  }

  // Clear overdraw, triArea, debugTag images cleanly before geometry pass
  auto clearUintImg = [&](GpuImage &img, uint32_t val) {
    transitionImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue c{};
    c.uint32[0] = val; c.uint32[1] = val; c.uint32[2] = val; c.uint32[3] = val;
    VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &rng);
    transitionImage(cmd, img, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
  };
  clearUintImg(m_overdrawImg, 0u);
  clearUintImg(m_debugTagImg, 0u);
  {
    transitionImage(cmd, m_triAreaImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue c{};
    c.float32[0] = 0.0f;
    VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_triAreaImg.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &rng);
    transitionImage(cmd, m_triAreaImg, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
  }

  VkViewport vp{0.0f, 0.0f, static_cast<float>(m_width), static_cast<float>(m_height), 0.0f, 1.0f};
  VkRect2D sc{{0, 0}, {m_width, m_height}};
  VkDescriptorSet sets[2] = {m_sceneSet, slot.frameSet};

  auto tsWrite = [&](uint32_t qIdx) {
    if (isMeasuredPass && slot.tsPool != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot.tsPool, qIdx);
    }
  };

  const bool usePipeStats = isMeasuredPass && (slot.pipeStatsPool != VK_NULL_HANDLE) &&
                            !(mode == RenderMode::Modern && m_telemetry.meshShaderActive);
  slot.usedPipeStats = usePipeStats;
  if (isMeasuredPass) {
    if (slot.tsPool != VK_NULL_HANDLE) vkCmdResetQueryPool(cmd, slot.tsPool, 0, 6);
    if (usePipeStats) vkCmdResetQueryPool(cmd, slot.pipeStatsPool, 0, 1);
    tsWrite(0);
    if (usePipeStats) vkCmdBeginQuery(cmd, slot.pipeStatsPool, 0, 0);
  }

  uint32_t drawCallCounter = 0u;

  if (mode == RenderMode::Naive) {
    tsWrite(1);
    transitionImage(cmd, m_sceneColorImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transitionImage(cmd, m_depthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    VkRenderingAttachmentInfo colorAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAtt.imageView = m_sceneColorImg.view;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAtt.imageView = m_depthImg.view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {0.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {m_width, m_height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;
    ri.pDepthAttachment = &depthAtt;

    m_ctx->getCmdBeginRendering()(cmd, &ri);
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gfxLayout, 0, 2, sets, 0, nullptr);

    // Paint physical courtyard skylight gradient matching Legacy & Modern sky pass
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeSky);
    vkCmdDraw(cmd, 3u, 1u, 0u, 0u);

    vkCmdBindIndexBuffer(cmd, m_ib.buffer, 0, VK_INDEX_TYPE_UINT32);

    VkPipeline activePipe = VK_NULL_HANDLE;
    // Unculled back-to-front per-instance draw-call loop (CPU bottleneck + 64x tess + forward overdraw)
    for (uint32_t instIdx : m_scene.naiveDrawOrder) {
      const auto &inst = m_scene.instances[instIdx];
      if (inst.baseIndexCount == 0u) continue;
      bool isTess = ((inst.flags & SHOWCASE_FLAG_TESSELLATED) != 0u) && (m_pipeNaiveTess != VK_NULL_HANDLE);
      VkPipeline targetPipe = isTess ? m_pipeNaiveTess : m_pipeNaiveFlat;
      if (targetPipe != activePipe) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, targetPipe);
        activePipe = targetPipe;
      }
      vkCmdDrawIndexed(cmd, inst.baseIndexCount, 1, inst.baseFirstIndex, inst.vertexOffset, instIdx);
      drawCallCounter++;
    }
    m_ctx->getCmdEndRendering()(cmd);

    tsWrite(2);
    tsWrite(3);
  } else if (mode == RenderMode::Legacy) {
    tsWrite(1);
    // CPU frustum cull only (no occlusion cull, no cone cull, distance-clamped tess, 3-MRT G-Buffer + deferred lighting)
    auto *cmdList = reinterpret_cast<VkDrawIndexedIndirectCommand *>(slot.legacyIndirectBuf.mapped);
    uint32_t flatOpaqueStart = 0, flatOpaqueCount = 0;
    auto pushList = [&](const std::vector<uint32_t> &src, bool wantTess, uint32_t &outStart, uint32_t &outCount) {
      outStart = flatOpaqueStart + flatOpaqueCount;
      uint32_t cur = 0;
      for (uint32_t instIdx : src) {
        const auto &inst = m_scene.instances[instIdx];
        if (inst.baseIndexCount == 0u) continue;
        bool isTess = (inst.flags & SHOWCASE_FLAG_TESSELLATED) != 0u;
        if (isTess != wantTess) continue;
        bool inside = true;
        for (int p = 0; p < 6; ++p) {
          float dist = cullPlanes[p].x * inst.boundSphereWorld[0] +
                       cullPlanes[p].y * inst.boundSphereWorld[1] +
                       cullPlanes[p].z * inst.boundSphereWorld[2] +
                       cullPlanes[p].w;
          if (dist < -inst.boundSphereWorld[3]) {
            inside = false;
            break;
          }
        }
        if (!inside) continue;
        cmdList[outStart + cur] = {inst.baseIndexCount, 1u, inst.baseFirstIndex, inst.vertexOffset, instIdx};
        cur++;
      }
      outCount = cur;
      flatOpaqueStart = outStart;
      flatOpaqueCount = outCount;
    };

    uint32_t sFlat = 0, cFlat = 0, sTess = 0, cTess = 0, sCut = 0, cCut = 0;
    pushList(m_scene.legacyOpaqueDrawList, false, sFlat, cFlat);
    pushList(m_scene.legacyOpaqueDrawList, true, sTess, cTess);
    pushList(m_scene.legacyCutoutDrawList, false, sCut, cCut);
    vmaFlushAllocation(m_vma, slot.legacyIndirectBuf.allocation, 0,
                       static_cast<VkDeviceSize>(sCut + cCut) * sizeof(VkDrawIndexedIndirectCommand));

    auto transAtt = [&](GpuImage &img) {
      transitionImage(cmd, img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    };
    transAtt(m_gbuf0);
    transAtt(m_gbuf1);
    transAtt(m_gbuf2);
    transitionImage(cmd, m_depthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    VkRenderingAttachmentInfo mrts[3]{};
    VkImageView views[3] = {m_gbuf0.view, m_gbuf1.view, m_gbuf2.view};
    for (int k = 0; k < 3; ++k) {
      mrts[k].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      mrts[k].imageView = views[k];
      mrts[k].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      mrts[k].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      mrts[k].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAtt.imageView = m_depthImg.view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {0.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {m_width, m_height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 3;
    ri.pColorAttachments = mrts;
    ri.pDepthAttachment = &depthAtt;

    m_ctx->getCmdBeginRendering()(cmd, &ri);
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gfxLayout, 0, 2, sets, 0, nullptr);
    vkCmdBindIndexBuffer(cmd, m_ib.buffer, 0, VK_INDEX_TYPE_UINT32);

    if (cFlat > 0u) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeLegacyFlat);
      vkCmdDrawIndexedIndirect(cmd, slot.legacyIndirectBuf.buffer,
                               sFlat * sizeof(VkDrawIndexedIndirectCommand), cFlat, sizeof(VkDrawIndexedIndirectCommand));
      drawCallCounter++;
    }
    if (cTess > 0u && m_pipeLegacyTess != VK_NULL_HANDLE) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeLegacyTess);
      vkCmdDrawIndexedIndirect(cmd, slot.legacyIndirectBuf.buffer,
                               sTess * sizeof(VkDrawIndexedIndirectCommand), cTess, sizeof(VkDrawIndexedIndirectCommand));
      drawCallCounter++;
    }
    if (cCut > 0u) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeLegacyFlat);
      vkCmdDrawIndexedIndirect(cmd, slot.legacyIndirectBuf.buffer,
                               sCut * sizeof(VkDrawIndexedIndirectCommand), cCut, sizeof(VkDrawIndexedIndirectCommand));
      drawCallCounter++;
    }
    m_ctx->getCmdEndRendering()(cmd);
    tsWrite(2);

    // Transition G-Buffer + Depth -> SHADER_READ_ONLY_OPTIMAL, sceneColor -> GENERAL
    auto transRead = [&](GpuImage &img, VkImageAspectFlags asp) {
      transitionImage(cmd, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT, asp);
    };
    transRead(m_gbuf0, VK_IMAGE_ASPECT_COLOR_BIT);
    transRead(m_gbuf1, VK_IMAGE_ASPECT_COLOR_BIT);
    transRead(m_gbuf2, VK_IMAGE_ASPECT_COLOR_BIT);
    transRead(m_depthImg, VK_IMAGE_ASPECT_DEPTH_BIT);
    transitionImage(cmd, m_sceneColorImg, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeLegacyLight);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gfxLayout, 0, 2, sets, 0, nullptr);
    vkCmdDispatch(cmd, (m_width + 7u) / 8u, (m_height + 7u) / 8u, 1u);
    tsWrite(3);
  } else {
    // MODERN VULKAN 1.4 PIPELINE (Two-Phase Hi-Z Task+Mesh + Visibility Buffer + DGC ExecutionSet)
    const bool useVisBuf = cfg.opts.visBuffer;
    const bool useMesh = m_telemetry.meshShaderSupported && (m_ctx->getCmdDrawMeshTasks() != nullptr);

    if (useVisBuf) {
      transitionImage(cmd, m_visBufImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    } else {
      transitionImage(cmd, m_sceneColorImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    }
    transitionImage(cmd, m_depthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    VkRenderingAttachmentInfo colorAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAtt.imageView = useVisBuf ? m_visBufImg.view : m_sceneColorImg.view;
    colorAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAtt.loadOp = useVisBuf ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (useVisBuf) {
      colorAtt.clearValue.color.uint32[0] = 0xFFFFFFFFu;
      colorAtt.clearValue.color.uint32[1] = 0xFFFFFFFFu;
    }

    VkRenderingAttachmentInfo depthAtt{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAtt.imageView = m_depthImg.view;
    depthAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.clearValue.depthStencil = {0.0f, 0};

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {m_width, m_height}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &colorAtt;
    ri.pDepthAttachment = &depthAtt;

    auto dispatchMeshPhase = [&](uint32_t phaseIdx, uint32_t pktCount) {
      if (pktCount == 0u) return;
      if (useMesh) {
        m_ctx->getCmdBeginRendering()(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        if (phaseIdx == 0u && !useVisBuf) {
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gfxLayout, 0, 2, sets, 0, nullptr);
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeSky);
          vkCmdDraw(cmd, 3u, 1u, 0u, 0u);
        }
        VkPipeline mp = useVisBuf
                           ? ((phaseIdx == 0u && m_pipeModernVisOpaque != VK_NULL_HANDLE)
                                  ? m_pipeModernVisOpaque
                                  : m_pipeModernVis)
                           : m_pipeModernFwd;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mp);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_taskLayout, 0, 2, sets, 0, nullptr);
        uint32_t pc[2] = {phaseIdx, pktCount};
        vkCmdPushConstants(cmd, m_taskLayout, VK_SHADER_STAGE_TASK_BIT_EXT, 0, sizeof(pc), pc);
        m_ctx->getCmdDrawMeshTasks()(cmd, pktCount, 1u, 1u);
        m_ctx->getCmdEndRendering()(cmd);
        drawCallCounter++;
      } else {
        // Compute cull + vkCmdDrawIndexedIndirectCount fallback when mesh shaders are unavailable
        vkCmdFillBuffer(cmd, m_fallbackBuf.buffer, 0, 16, 0);
        m_ctx->cmdPipelineMemoryBarrier2(cmd,
                                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeFallbackCull);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_compLayout, 0, 2, sets, 0, nullptr);
        uint32_t pc[4] = {phaseIdx, pktCount, kMaxFallbackDraws, 0u};
        vkCmdPushConstants(cmd, m_compLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cmd, pktCount, 1u, 1u);
        m_ctx->cmdPipelineMemoryBarrier2(cmd,
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                         VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                         VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);

        m_ctx->getCmdBeginRendering()(cmd, &ri);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeFallbackVis);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gfxLayout, 0, 2, sets, 0, nullptr);
        vkCmdBindIndexBuffer(cmd, m_ib.buffer, 0, VK_INDEX_TYPE_UINT32);
        if (m_ctx->getCmdDrawIndexedIndirectCount() != nullptr) {
          m_ctx->getCmdDrawIndexedIndirectCount()(cmd, m_fallbackBuf.buffer, 16, m_fallbackBuf.buffer, 0, kMaxFallbackDraws, 32);
        }
        m_ctx->getCmdEndRendering()(cmd);
        drawCallCounter++;
      }
    };

    // Phase 0: Structural Sponza Occluders
    dispatchMeshPhase(0u, static_cast<uint32_t>(m_scene.phase0Packets.size()));
    tsWrite(1);

    // Build Same-Frame Hi-Z Pyramid from Phase 0 Occluders (unless freezeCulling holds frozen Hi-Z!)
    if (!cfg.freezeCulling && cfg.opts.hizOcclusion && m_hizImg.mipLevels > 0u) {
      transitionImage(cmd, m_depthImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                      VK_IMAGE_ASPECT_DEPTH_BIT);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeHizReduce);
      uint32_t srcW = m_width, srcH = m_height;
      for (uint32_t m = 0; m < m_hizImg.mipLevels; ++m) {
        uint32_t dstW = std::max(1u, m_width >> m);
        uint32_t dstH = std::max(1u, m_height >> m);

        VkImageMemoryBarrier2 bDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        bDst.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        bDst.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        bDst.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bDst.dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        bDst.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bDst.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bDst.image = m_hizImg.image;
        bDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1};
        m_ctx->cmdImageMemoryBarrier2(cmd, &bDst, 1);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_hizLayout, 0, 1, &m_hizMipSets[m], 0, nullptr);
        uint32_t hpc[5] = {srcW, srcH, dstW, dstH, (m == 0u) ? 1u : 0u};
        vkCmdPushConstants(cmd, m_hizLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hpc), hpc);
        vkCmdDispatch(cmd, (dstW + 7u) / 8u, (dstH + 7u) / 8u, 1u);

        VkImageMemoryBarrier2 bDone{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        bDone.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bDone.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        bDone.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                              (m_ctx->isMeshShaderSupported() ? VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT : 0u);
        bDone.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        bDone.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        bDone.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bDone.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bDone.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bDone.image = m_hizImg.image;
        bDone.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1};
        m_ctx->cmdImageMemoryBarrier2(cmd, &bDone, 1);

        srcW = dstW;
        srcH = dstH;
      }
      m_hizImg.currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      transitionImage(cmd, m_depthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                      VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    tsWrite(2);

    // Phase 1: Sculpted Relief Props + Dense Foliage + Hidden Cloisters (LOAD existing depth & visBuf)
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    dispatchMeshPhase(1u, static_cast<uint32_t>(m_scene.phase1Packets.size()));
    tsWrite(3);

    if (useVisBuf) {
      transitionImage(cmd, m_visBufImg, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
      transitionImage(cmd, m_sceneColorImg, VK_IMAGE_LAYOUT_GENERAL,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      VK_ACCESS_2_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeVisClassify);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gfxLayout, 0, 2, sets, 0, nullptr);
      vkCmdDispatch(cmd, (m_width + 7u) / 8u, (m_height + 7u) / 8u, 1u);

      m_ctx->cmdPipelineMemoryBarrier2(cmd,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

      if (cfg.opts.dgcMaterialBinning) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeVisResolve);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gfxLayout, 0, 2, sets, 0, nullptr);
        vkCmdDispatch(cmd, 1u, 1u, 1u);

        m_ctx->cmdPipelineMemoryBarrier2(
            cmd,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);

        if (m_dgcExecSetReady) {
          auto pfnExec = (PFN_vkCmdExecuteGeneratedCommandsEXT)vkGetDeviceProcAddr(
              m_device, "vkCmdExecuteGeneratedCommandsEXT");
          if (pfnExec) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeVisShadeSpec[0]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_visShadeSpecLayout, 0, 2, sets, 0, nullptr);

            VkGeneratedCommandsInfoEXT genCi{VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT};
            genCi.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
            genCi.indirectExecutionSet = m_dgcExecSet;
            genCi.indirectCommandsLayout = m_dgcLayout;
            genCi.indirectAddress = m_dgcBuf.address + 16u;
            genCi.indirectAddressSize = 48u * 8u;
            genCi.preprocessAddress = m_dgcPreprocessBuf.address;
            genCi.preprocessSize = m_dgcPreprocessBuf.size;
            genCi.maxSequenceCount = 8u;
            genCi.sequenceCountAddress = m_dgcBuf.address;
            pfnExec(cmd, VK_FALSE, &genCi);
            drawCallCounter++;
          }
        } else {
          for (uint32_t a = 0; a < 8u; ++a) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeVisShadeSpec[a]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_visShadeSpecLayout, 0, 2, sets, 0, nullptr);
            uint32_t pc[8] = {a, m_width * m_height, m_width, m_height, cfg.opts.toBitmask(),
                              static_cast<uint32_t>(cfg.overlay), static_cast<uint32_t>(mode), 0u};
            vkCmdPushConstants(cmd, m_visShadeSpecLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatchIndirect(cmd, m_fallbackBuf.buffer, 16u + a * 16u);
            drawCallCounter++;
          }
        }
      } else {
        // Unified shading pass when DGC Material Binning is toggled OFF
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeVisShadeUnified);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_visShadeSpecLayout, 0, 2, sets, 0, nullptr);
        uint32_t pc[8] = {8u, m_width * m_height, m_width, m_height, cfg.opts.toBitmask(),
                          static_cast<uint32_t>(cfg.overlay), static_cast<uint32_t>(mode), 0u};
        vkCmdPushConstants(cmd, m_visShadeSpecLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        uint32_t totalPx = m_width * m_height;
        vkCmdDispatch(cmd, (totalPx + 63u) / 64u, 1u, 1u);
        drawCallCounter++;
      }
    }
  }

  if (slot.usedPipeStats && slot.pipeStatsPool != VK_NULL_HANDLE) {
    vkCmdEndQuery(cmd, slot.pipeStatsPool, 0);
  }
  tsWrite(4);

  // Ensure m_sceneColorImg is in SHADER_READ_ONLY_OPTIMAL before overlay compose
  if (m_sceneColorImg.currentLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    transitionImage(cmd, m_sceneColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  } else {
    transitionImage(cmd, m_sceneColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  }

  slot.recordedDrawCalls = drawCallCounter;
  slot.recordedMode = mode;
  slot.recordedOpts = cfg.opts;
}

void ShowcaseRenderer::recordFrame(VkCommandBuffer cmd, uint32_t frameSlot) {
  if (!m_initialized || m_width == 0u || m_height == 0u) return;
  uint32_t slotIdx = frameSlot % kMaxSlots;

  // Harvest finished GPU metrics for this fence-synchronized swapchain slot
  readbackCompletedSlot(slotIdx, m_telemetry, true);

  m_frameCounter++;
  auto &slot = m_slots[slotIdx];
  slot.splitWipeActive = m_settings.splitWipeCompare;

  if (m_settings.splitWipeCompare) {
    // 1. Render left comparison mode (e.g. Naive) and copy exact pixels sceneColor -> altColor
    recordModePass(cmd, slotIdx, m_settings.splitLeftMode, m_settings, true, false);

    transitionImage(cmd, m_sceneColorImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    transitionImage(cmd, m_altColorImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy cp{};
    cp.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    cp.extent = {m_width, m_height, 1};
    vkCmdCopyImage(cmd, m_sceneColorImg.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   m_altColorImg.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    transitionImage(cmd, m_altColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    transitionImage(cmd, m_sceneColorImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // Update mode in UBO & reset stats buffer on GPU timeline between Left and Right passes
    uint32_t rightScreenInfo[4] = {m_width, m_height, static_cast<uint32_t>(m_settings.mode),
                                   static_cast<uint32_t>(m_settings.overlay)};
    vkCmdUpdateBuffer(cmd, slot.uboBuf.buffer, offsetof(GpuUniforms, screenInfo),
                      sizeof(rightScreenInfo), rightScreenInfo);
    vkCmdFillBuffer(cmd, slot.statsBuf.buffer, 0, sizeof(GpuStatsReadback), 0);
    m_ctx->cmdPipelineMemoryBarrier2(
        cmd,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
            (m_ctx->isMeshShaderSupported() ? VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT : 0u),
        VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    // 2. Render primary active mode (right side) & capture timestamps/stats
    recordModePass(cmd, slotIdx, m_settings.mode, m_settings, false, true);
  } else {
    recordModePass(cmd, slotIdx, m_settings.mode, m_settings, true, true);
  }

  // Composite X-Ray overlay / Split-Wipe into displayColorImg
  transitionImage(cmd, m_displayColor, VK_IMAGE_LAYOUT_GENERAL,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);

  VkDescriptorSet sets[2] = {m_sceneSet, slot.frameSet};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeOverlayCompose);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gfxLayout, 0, 2, sets, 0, nullptr);
  vkCmdDispatch(cmd, (m_width + 15u) / 16u, (m_height + 15u) / 16u, 1u);

  // Optional 3D Frozen Frustum Wireframe Overlay
  if (m_settings.freezeCulling) {
    transitionImage(cmd, m_displayColor, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transitionImage(cmd, m_depthImg, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);

    VkRenderingAttachmentInfo wCol{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    wCol.imageView = m_displayColor.view;
    wCol.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    wCol.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    wCol.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo wDep{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    wDep.imageView = m_depthImg.view;
    wDep.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    wDep.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    wDep.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

    VkRenderingInfo wRi{VK_STRUCTURE_TYPE_RENDERING_INFO};
    wRi.renderArea = {{0, 0}, {m_width, m_height}};
    wRi.layerCount = 1;
    wRi.colorAttachmentCount = 1;
    wRi.pColorAttachments = &wCol;
    wRi.pDepthAttachment = &wDep;

    m_ctx->getCmdBeginRendering()(cmd, &wRi);
    VkViewport vp{0.0f, 0.0f, static_cast<float>(m_width), static_cast<float>(m_height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, {m_width, m_height}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeFrustumWire);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gfxLayout, 0, 2, sets, 0, nullptr);
    vkCmdDraw(cmd, 24u, 1u, 0u, 0u);
    m_ctx->getCmdEndRendering()(cmd);

    transitionImage(cmd, m_displayColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  } else {
    transitionImage(cmd, m_displayColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  }

  if (slot.tsPool != VK_NULL_HANDLE) {
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot.tsPool, 5);
  }
  slot.submitted = true;
}

bool ShowcaseRenderer::renderHeadlessFrame(const ShowcaseSettings &settings, uint32_t width, uint32_t height,
                                           FrameTelemetry &outTelemetry, std::vector<uint8_t> *outRgba) {
  if (!m_initialized) return false;
  if (width != m_width || height != m_height) {
    resize(width, height);
  }
  m_settings = settings;

  {
    std::lock_guard<std::recursive_mutex> qlock(m_ctx->getQueueMutex());
    vkDeviceWaitIdle(m_device);
    vkResetCommandPool(m_device, m_uploadCmdPool, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_headlessCmd, &bi);

    m_slots[0].submitted = false;
    recordFrame(m_headlessCmd, 0u);

    GpuBuffer readbackBuf;
    VkDeviceSize pxBytes = static_cast<VkDeviceSize>(m_width) * m_height * 4u;
    if (outRgba) {
      readbackBuf = createBuffer(pxBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
      transitionImage(m_headlessCmd, m_displayColor, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
      VkBufferImageCopy reg{};
      reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      reg.imageExtent = {m_width, m_height, 1};
      vkCmdCopyImageToBuffer(m_headlessCmd, m_displayColor.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readbackBuf.buffer, 1, &reg);
      transitionImage(m_headlessCmd, m_displayColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    }

    vkEndCommandBuffer(m_headlessCmd);
    vkResetFences(m_device, 1, &m_headlessFence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &m_headlessCmd;
    if (vkQueueSubmit(m_ctx->getGraphicsQueue(), 1, &si, m_headlessFence) != VK_SUCCESS) {
      if (outRgba) destroyBuffer(readbackBuf);
      return false;
    }
    vkWaitForFences(m_device, 1, &m_headlessFence, VK_TRUE, UINT64_MAX);

    if (outRgba) {
      vmaInvalidateAllocation(m_vma, readbackBuf.allocation, 0, pxBytes);
      outRgba->resize(static_cast<size_t>(pxBytes));
      std::memcpy(outRgba->data(), readbackBuf.mapped, static_cast<size_t>(pxBytes));
      destroyBuffer(readbackBuf);
    }
  }

  outTelemetry = FrameTelemetry{};
  readbackCompletedSlot(0u, outTelemetry, false);
  return true;
}

ThreeWayResult ShowcaseRenderer::runThreeWayBenchmark(const CameraState &cam, uint32_t width, uint32_t height) {
  ThreeWayResult res{};
  if (!m_initialized) return res;

  ShowcaseSettings base{};
  base.activeCam = cam;
  base.frozenCam = cam;
  base.overlay = OverlayMode::Beauty;
  base.opts = OptimizationFlags{};

  auto measureMode = [&](RenderMode mode, double &outMs, double &outFps, uint32_t &outDraws,
                         uint64_t &outPrims, double &outOv, double &outSub, double &outCull) {
    ShowcaseSettings s = base;
    s.mode = mode;
    std::vector<double> samples;
    FrameTelemetry lastTel{};
    // 2 warmups + 5 measured frames -> median
    for (int it = 0; it < 7; ++it) {
      if (renderHeadlessFrame(s, width, height, lastTel, nullptr) && it >= 2 && lastTel.totalMs > 0.0) {
        samples.push_back(lastTel.totalMs);
      }
    }
    std::sort(samples.begin(), samples.end());
    outMs = samples.empty() ? lastTel.totalMs : samples[samples.size() / 2];
    outFps = (outMs > 0.001) ? (1000.0 / outMs) : 0.0;
    outDraws = lastTel.drawCalls;
    outPrims = (mode == RenderMode::Modern && lastTel.renderedTriangles > 0u)
                   ? lastTel.renderedTriangles
                   : lastTel.clippedPrimitives;
    outOv = lastTel.avgOverdraw;
    outSub = lastTel.subPixelTriPct;
    outCull = lastTel.culledPct;
  };

  double dummyCull = 0.0;
  measureMode(RenderMode::Naive, res.naiveMs, res.naiveFps, res.naiveDrawCalls,
              res.naivePrimitives, res.naiveOverdraw, res.naiveSubPxPct, dummyCull);
  measureMode(RenderMode::Legacy, res.legacyMs, res.legacyFps, res.legacyDrawCalls,
              res.legacyPrimitives, res.legacyOverdraw, res.legacySubPxPct, dummyCull);
  measureMode(RenderMode::Modern, res.modernMs, res.modernFps, res.modernDrawCalls,
              res.modernPrimitives, res.modernOverdraw, res.modernSubPxPct, res.modernCulledPct);

  // One split-wipe pass to measure Naive vs Modern Beauty PSNR parity cleanly
  ShowcaseSettings paritySet = base;
  paritySet.mode = RenderMode::Modern;
  paritySet.splitWipeCompare = true;
  paritySet.splitLeftMode = RenderMode::Naive;
  paritySet.splitWipeX = 0.5f;
  FrameTelemetry parityTel{};
  if (renderHeadlessFrame(paritySet, width, height, parityTel, nullptr)) {
    res.parityPsnr = parityTel.parityPsnr;
    res.parityExactPct = parityTel.parityExactPct;
  }

  if (res.modernMs > 0.0) {
    res.modernVsNaiveSpeedup = res.naiveMs / res.modernMs;
    res.modernVsLegacySpeedup = res.legacyMs / res.modernMs;
  }
  res.valid = true;
  return res;
}

} // namespace gpubench::showcase
