#include "PixelFillRateBench.h"
#include "core/VulkanContext.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

PixelFillRateBench::PixelFillRateBench() {
  configs.push_back({"RGBA8 Color Fill", VK_FORMAT_R8G8B8A8_UNORM, false});
  configs.push_back({"RGBA16F HDR Fill", VK_FORMAT_R16G16B16A16_SFLOAT, false});
  configs.push_back({"Alpha Blending Fill", VK_FORMAT_R8G8B8A8_UNORM, true});
}

PixelFillRateBench::~PixelFillRateBench() {
  Teardown();
}

bool PixelFillRateBench::IsSupported(const DeviceInfo &info,
                                     IComputeContext *ctx) const {
  (void)info;
  if (!ctx || ctx->getBackend() != ComputeBackend::Vulkan) {
    return false;
  }
  auto *vkCtx = dynamic_cast<VulkanContext *>(ctx);
  if (vkCtx && !vkCtx->isDynamicRenderingSupported()) {
    return false;
  }
  return true;
}

std::string PixelFillRateBench::GetSupportNote(const DeviceInfo &info,
                                               IComputeContext *ctx) const {
  (void)info;
  if (ctx && ctx->getBackend() == ComputeBackend::OpenCL) {
    return "No support for graphics rasterization pipeline (ROPs) in OpenCL API";
  }
  if (ctx && ctx->getBackend() == ComputeBackend::ROCm) {
    return "No support for graphics rasterization pipeline (ROPs) in ROCm API";
  }
  if (ctx && ctx->getBackend() == ComputeBackend::Vulkan) {
    auto *vkCtx = dynamic_cast<VulkanContext *>(ctx);
    if (vkCtx && !vkCtx->isDynamicRenderingSupported()) {
      return "Dynamic rendering (VK_KHR_dynamic_rendering) not supported on this device";
    }
  }
  return "Pixel Fill Rate benchmark requires Vulkan graphics rasterization pipeline with dynamic rendering (ROPs)";
}

std::string PixelFillRateBench::GetConfigName(uint32_t config_idx) const {
  if (config_idx < configs.size()) {
    return configs[config_idx].name;
  }
  return "";
}

uint32_t PixelFillRateBench::findMemoryType(uint32_t typeFilter,
                                            VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type for pixel fill rate image!");
}

VkShaderModule PixelFillRateBench::loadShaderModule(const std::string &path) {
  std::string spv_path = path + ".spv";
  std::ifstream file(spv_path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    file.open(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open shader file: " + path);
    }
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
  file.close();

  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = buffer.size() * sizeof(uint32_t);
  createInfo.pCode = buffer.data();

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module for " + path);
  }
  return shaderModule;
}

void PixelFillRateBench::createPipelineForConfig(Config &cfg) {
  // 1. Create Image and ImageView
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = cfg.format;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(device, &imageInfo, nullptr, &cfg.image) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image for pixel fill rate benchmark!");
  }

  VkMemoryRequirements memReqs;
  vkGetImageMemoryRequirements(device, cfg.image, &memReqs);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &cfg.memory) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate memory for pixel fill rate image!");
  }

  vkBindImageMemory(device, cfg.image, cfg.memory, 0);

  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = cfg.image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = cfg.format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  if (vkCreateImageView(device, &viewInfo, nullptr, &cfg.imageView) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create image view for pixel fill rate benchmark!");
  }

  // 2. Create Graphics Pipeline with Dynamic Rendering
  VkPipelineShaderStageCreateInfo vertStageInfo{};
  vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertStageInfo.module = vertShaderModule;
  vertStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragStageInfo{};
  fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragStageInfo.module = fragShaderModule;
  fragStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo stages[] = {vertStageInfo, fragStageInfo};

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(width);
  viewport.height = static_cast<float>(height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = {width, height};

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  if (cfg.enableBlend) {
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
  } else {
    colorBlendAttachment.blendEnable = VK_FALSE;
  }

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;

  VkPipelineRenderingCreateInfoKHR renderingInfo{};
  renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachmentFormats = &cfg.format;

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.pNext = &renderingInfo;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = stages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = VK_NULL_HANDLE;
  pipelineInfo.subpass = 0;

  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                &cfg.pipeline) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create graphics pipeline for pixel fill rate!");
  }
}

void PixelFillRateBench::Setup(IComputeContext &ctx, const std::string &kernel_dir) {
  this->context = &ctx;
  auto *vulkanContext = dynamic_cast<VulkanContext *>(&ctx);
  if (!vulkanContext) {
    throw std::runtime_error("PixelFillRateBench requires a Vulkan context!");
  }

  device = vulkanContext->getVulkanDevice();
  physicalDevice = vulkanContext->getVulkanPhysicalDevice();
  queue = vulkanContext->getGraphicsQueue();
  queueFamilyIndex = vulkanContext->getGraphicsQueueFamilyIndex();

  pfnCmdBeginRendering = vulkanContext->getCmdBeginRendering();
  pfnCmdEndRendering = vulkanContext->getCmdEndRendering();

  if (!pfnCmdBeginRendering || !pfnCmdEndRendering) {
    pfnCmdBeginRendering =
        (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdBeginRendering");
    if (!pfnCmdBeginRendering) {
      pfnCmdBeginRendering =
          (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR");
    }
    pfnCmdEndRendering =
        (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdEndRendering");
    if (!pfnCmdEndRendering) {
      pfnCmdEndRendering =
          (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR");
    }
  }

  if (!pfnCmdBeginRendering || !pfnCmdEndRendering) {
    throw std::runtime_error("Dynamic rendering (vkCmdBeginRendering / vkCmdEndRendering) is not supported or could not be loaded!");
  }

  // Load Shaders
  std::filesystem::path kdir(kernel_dir);
  std::string vertPath = (kdir / "vulkan" / "pixel_fill.vert").string();
  std::string fragPath = (kdir / "vulkan" / "pixel_fill.frag").string();

  vertShaderModule = loadShaderModule(vertPath);
  fragShaderModule = loadShaderModule(fragPath);

  // Create Pipeline Layout with Push Constant for color seed
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(float) * 4;

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create pipeline layout for pixel fill rate!");
  }

  // Create Command Pool & Buffers
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = queueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool for pixel fill rate!");
  }

  for (size_t i = 0; i < kMaxInFlight; ++i) {
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &frames[i].commandBuffer) != VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate command buffer for pixel fill rate!");
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device, &fenceInfo, nullptr, &frames[i].fence) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create fence for pixel fill rate!");
    }
    frames[i].inFlight = false;
  }
  frameIndex = 0;

  // Create pipelines for all configurations
  for (auto &cfg : configs) {
    createPipelineForConfig(cfg);
  }

  // Transition all configuration images from UNDEFINED to COLOR_ATTACHMENT_OPTIMAL layout
  VkCommandBufferBeginInfo cmdBeginInfo{};
  cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frames[0].commandBuffer, &cmdBeginInfo);

  std::vector<VkImageMemoryBarrier> barriers;
  for (auto &cfg : configs) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = cfg.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers.push_back(barrier);
  }

  vkCmdPipelineBarrier(frames[0].commandBuffer,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       0, 0, nullptr, 0, nullptr,
                       static_cast<uint32_t>(barriers.size()), barriers.data());

  vkEndCommandBuffer(frames[0].commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frames[0].commandBuffer;

  vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
}

void PixelFillRateBench::Run(uint32_t config_idx) {
  if (config_idx >= configs.size()) return;
  auto &cfg = configs[config_idx];
  auto &frame = frames[frameIndex];

  if (frame.inFlight) {
    constexpr uint64_t kTimeoutNs = 3'000'000'000ULL;
    VkResult waitResult = vkWaitForFences(device, 1, &frame.fence, VK_TRUE, kTimeoutNs);
    if (waitResult == VK_TIMEOUT) {
      throw std::runtime_error("Pixel fill rate dispatch timed out (>3 s)!");
    } else if (waitResult != VK_SUCCESS) {
      throw std::runtime_error("Pixel fill rate fence wait failed!");
    }
    vkResetFences(device, 1, &frame.fence);
    frame.inFlight = false;
  }

  vkResetCommandBuffer(frame.commandBuffer, 0);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(frame.commandBuffer, &beginInfo);

  VkRenderingAttachmentInfoKHR colorAttachment{};
  colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
  colorAttachment.imageView = cfg.imageView;
  colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

  VkRenderingInfoKHR renderingInfo{};
  renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
  renderingInfo.renderArea.offset = {0, 0};
  renderingInfo.renderArea.extent = {width, height};
  renderingInfo.layerCount = 1;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments = &colorAttachment;

  float colorSeed[4] = {0.2f, 0.4f, 0.6f, 1.0f};

  pfnCmdBeginRendering(frame.commandBuffer, &renderingInfo);
  vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cfg.pipeline);

  for (uint32_t p = 0; p < passesPerDispatch; ++p) {
    colorSeed[0] = static_cast<float>(p) * 0.1f;
    vkCmdPushConstants(frame.commandBuffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(colorSeed), colorSeed);

    vkCmdDraw(frame.commandBuffer, 3, 1, 0, 0); // Fullscreen triangle
  }

  pfnCmdEndRendering(frame.commandBuffer);

  vkEndCommandBuffer(frame.commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &frame.commandBuffer;

  vkResetFences(device, 1, &frame.fence);
  vkQueueSubmit(queue, 1, &submitInfo, frame.fence);
  frame.inFlight = true;

  frameIndex = (frameIndex + 1) % kMaxInFlight;
}

void PixelFillRateBench::Teardown() {
  if (device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device);

    for (auto &cfg : configs) {
      if (cfg.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, cfg.pipeline, nullptr);
        cfg.pipeline = VK_NULL_HANDLE;
      }
      if (cfg.imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, cfg.imageView, nullptr);
        cfg.imageView = VK_NULL_HANDLE;
      }
      if (cfg.image != VK_NULL_HANDLE) {
        vkDestroyImage(device, cfg.image, nullptr);
        cfg.image = VK_NULL_HANDLE;
      }
      if (cfg.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, cfg.memory, nullptr);
        cfg.memory = VK_NULL_HANDLE;
      }
    }

    if (pipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      pipelineLayout = VK_NULL_HANDLE;
    }
    if (vertShaderModule != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device, vertShaderModule, nullptr);
      vertShaderModule = VK_NULL_HANDLE;
    }
    if (fragShaderModule != VK_NULL_HANDLE) {
      vkDestroyShaderModule(device, fragShaderModule, nullptr);
      fragShaderModule = VK_NULL_HANDLE;
    }
    for (size_t i = 0; i < kMaxInFlight; ++i) {
      if (frames[i].inFlight && frames[i].fence != VK_NULL_HANDLE) {
        vkWaitForFences(device, 1, &frames[i].fence, VK_TRUE, 3'000'000'000ULL);
        frames[i].inFlight = false;
      }
      if (frames[i].fence != VK_NULL_HANDLE) {
        vkDestroyFence(device, frames[i].fence, nullptr);
        frames[i].fence = VK_NULL_HANDLE;
      }
    }
    if (commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, commandPool, nullptr);
      commandPool = VK_NULL_HANDLE;
    }

    pfnCmdBeginRendering = nullptr;
    pfnCmdEndRendering = nullptr;
    device = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
  }
}

BenchmarkResult PixelFillRateBench::GetResult(uint32_t config_idx) const {
  // Total pixels rendered = width * height * passesPerDispatch
  uint64_t totalPixels = static_cast<uint64_t>(width) * height * passesPerDispatch;
  return {totalPixels, 0.0};
}

bool PixelFillRateBench::ValidateResults(uint32_t config_idx) const {
  if (config_idx >= configs.size() || device == VK_NULL_HANDLE || queue == VK_NULL_HANDLE) {
    return false;
  }
  const auto &cfg = configs[config_idx];
  if (cfg.image == VK_NULL_HANDLE) {
    return false;
  }

  vkQueueWaitIdle(queue);

  // Allocate 64-byte host staging buffer for 1 pixel (supports up to RGBA32F)
  VkBuffer stagingBuffer = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;

  VkBufferCreateInfo bufInfo{};
  bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufInfo.size = 64;
  bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
    return false;
  }

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      memReqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    return false;
  }
  vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

  // Allocate command buffer from existing pool
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo cmdAllocInfo{};
  cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdAllocInfo.commandPool = commandPool;
  cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAllocInfo.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd) != VK_SUCCESS) {
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    return false;
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &beginInfo);

  // Barrier: COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL
  VkImageMemoryBarrier barrierToSrc{};
  barrierToSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrierToSrc.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barrierToSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrierToSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrierToSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrierToSrc.image = cfg.image;
  barrierToSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrierToSrc.subresourceRange.baseMipLevel = 0;
  barrierToSrc.subresourceRange.levelCount = 1;
  barrierToSrc.subresourceRange.baseArrayLayer = 0;
  barrierToSrc.subresourceRange.layerCount = 1;
  barrierToSrc.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  barrierToSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &barrierToSrc);

  // Copy center pixel (width / 2, height / 2)
  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {static_cast<int32_t>(width / 2), static_cast<int32_t>(height / 2), 0};
  region.imageExtent = {1, 1, 1};

  vkCmdCopyImageToBuffer(cmd, cfg.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         stagingBuffer, 1, &region);

  // Barrier: TRANSFER_SRC_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL
  VkImageMemoryBarrier barrierBack{};
  barrierBack.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrierBack.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrierBack.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrierBack.image = cfg.image;
  barrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrierBack.subresourceRange.baseMipLevel = 0;
  barrierBack.subresourceRange.levelCount = 1;
  barrierBack.subresourceRange.baseArrayLayer = 0;
  barrierBack.subresourceRange.layerCount = 1;
  barrierBack.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrierBack.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  vkCmdPipelineBarrier(cmd,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &barrierBack);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;

  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(device, &fenceInfo, nullptr, &fence);

  vkQueueSubmit(queue, 1, &submitInfo, fence);
  vkWaitForFences(device, 1, &fence, VK_TRUE, 3000000000ULL);
  vkDestroyFence(device, fence, nullptr);
  vkFreeCommandBuffers(device, commandPool, 1, &cmd);

  bool valid = false;
  void *mapped = nullptr;
  if (vkMapMemory(device, stagingMemory, 0, 64, 0, &mapped) == VK_SUCCESS) {
    if (cfg.format == VK_FORMAT_R8G8B8A8_UNORM) {
      auto *rgba = static_cast<const uint8_t *>(mapped);
      // Verify non-zero color channels and non-zero alpha
      valid = (rgba[0] > 0 || rgba[1] > 0 || rgba[2] > 0) && rgba[3] > 0;
    } else if (cfg.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
      auto *hdr = static_cast<const uint16_t *>(mapped);
      // In half float, 0 is 0x0000 or 0x8000 (-0.0)
      valid = (hdr[0] != 0 || hdr[1] != 0 || hdr[2] != 0) && hdr[3] != 0;
    } else {
      valid = true;
    }
    vkUnmapMemory(device, stagingMemory);
  }

  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingMemory, nullptr);
  return valid;
}

