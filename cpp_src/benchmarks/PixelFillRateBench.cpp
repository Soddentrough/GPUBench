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
  return ctx && ctx->getBackend() == ComputeBackend::Vulkan;
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
  // 1. Create Render Pass
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = cfg.format;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;

  if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &cfg.renderPass) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create render pass for pixel fill rate benchmark!");
  }

  // 2. Create Image and Framebuffer
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
  imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
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

  VkFramebufferCreateInfo fbInfo{};
  fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  fbInfo.renderPass = cfg.renderPass;
  fbInfo.attachmentCount = 1;
  fbInfo.pAttachments = &cfg.imageView;
  fbInfo.width = width;
  fbInfo.height = height;
  fbInfo.layers = 1;

  if (vkCreateFramebuffer(device, &fbInfo, nullptr, &cfg.framebuffer) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create framebuffer for pixel fill rate benchmark!");
  }

  // 3. Create Graphics Pipeline
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

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = stages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.renderPass = cfg.renderPass;
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
  queue = vulkanContext->getComputeQueue();
  queueFamilyIndex = vulkanContext->getComputeQueueFamilyIndex();

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

  VkRenderPassBeginInfo renderPassBegin{};
  renderPassBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassBegin.renderPass = cfg.renderPass;
  renderPassBegin.framebuffer = cfg.framebuffer;
  renderPassBegin.renderArea.offset = {0, 0};
  renderPassBegin.renderArea.extent = {width, height};

  float colorSeed[4] = {0.2f, 0.4f, 0.6f, 1.0f};

  for (uint32_t p = 0; p < passesPerDispatch; ++p) {
    vkCmdBeginRenderPass(frame.commandBuffer, &renderPassBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cfg.pipeline);

    colorSeed[0] = static_cast<float>(p) * 0.1f;
    vkCmdPushConstants(frame.commandBuffer, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(colorSeed), colorSeed);

    vkCmdDraw(frame.commandBuffer, 3, 1, 0, 0); // Fullscreen triangle
    vkCmdEndRenderPass(frame.commandBuffer);
  }

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
      if (cfg.framebuffer != VK_NULL_HANDLE) {
        vkDestroyFramebuffer(device, cfg.framebuffer, nullptr);
        cfg.framebuffer = VK_NULL_HANDLE;
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
      if (cfg.renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, cfg.renderPass, nullptr);
        cfg.renderPass = VK_NULL_HANDLE;
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
  }
}

BenchmarkResult PixelFillRateBench::GetResult(uint32_t config_idx) const {
  // Total pixels rendered = width * height * passesPerDispatch
  uint64_t totalPixels = static_cast<uint64_t>(width) * height * passesPerDispatch;
  return {totalPixels, 0.0};
}
