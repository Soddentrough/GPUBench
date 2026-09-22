#include "VulkanContext.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "third_party/stb_image.h"
#include <filesystem>
#include <cstring>

namespace gpubench::gui {

VulkanContext::VulkanContext() = default;

VulkanContext::~VulkanContext() {
    shutdown();
}

bool VulkanContext::init(const char* title, int width, int height, float scaleOverride) {
    if (m_initialized) return true;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "Failed to initialize SDL3: " << SDL_GetError() << std::endl;
        return false;
    }

    // Set Wayland app id if applicable
    SDL_SetHint(SDL_HINT_APP_NAME, "GPUBench");

    int winW = width;
    int winH = height;
    SDL_DisplayID displayID = SDL_GetPrimaryDisplay();
    float detectedScale = 1.0f;
    if (displayID != 0) {
        float contentScale = SDL_GetDisplayContentScale(displayID);
        if (contentScale > 0.0f) {
            detectedScale = contentScale;
        }
        SDL_Rect usableBounds{};
        if (SDL_GetDisplayUsableBounds(displayID, &usableBounds)) {
            float sFactor = (scaleOverride > 0.0f) ? scaleOverride : detectedScale;
            if (sFactor > 1.2f) {
                winW = static_cast<int>(winW * std::min(sFactor, 1.8f));
                winH = static_cast<int>(winH * std::min(sFactor, 1.6f));
            }
            if (usableBounds.h > 0 && winH > usableBounds.h - 80) {
                winH = std::max(600, usableBounds.h - 80);
            }
            if (usableBounds.w > 0 && winW > usableBounds.w - 60) {
                winW = std::max(800, usableBounds.w - 60);
            }
        }
    }

    m_window = SDL_CreateWindow(
        title,
        winW,
        winH,
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY
    );

    if (!m_window) {
        std::cerr << "Failed to create SDL3 window: " << SDL_GetError() << std::endl;
        return false;
    }

    float winScale = SDL_GetWindowDisplayScale(m_window);
    if (winScale > 0.0f) {
        detectedScale = winScale;
    }
    m_displayScale = (scaleOverride > 0.0f) ? scaleOverride : detectedScale;
    if (m_displayScale <= 0.0f) {
        m_displayScale = 1.0f;
    }

    if (!setupVulkan()) {
        std::cerr << "Failed to initialize Vulkan context" << std::endl;
        return false;
    }

    int fbWidth = 0, fbHeight = 0;
    SDL_GetWindowSizeInPixels(m_window, &fbWidth, &fbHeight);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Load high-resolution system TrueType font for workstation UI
    const char* fontCandidates[] = {
#if defined(_WIN32)
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
#elif defined(__APPLE__)
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
#endif
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/google-noto/NotoSans-Regular.ttf"
    };

    float baseFontSize = 16.0f;
    float scaledFontSize = std::round(baseFontSize * m_displayScale);

    bool fontLoaded = false;
    for (const char* fontPath : fontCandidates) {
        FILE* f = fopen(fontPath, "rb");
        if (f) {
            fclose(f);
            ImFontConfig fontCfg;
            fontCfg.OversampleH = 2;
            fontCfg.OversampleV = 2;
            fontCfg.PixelSnapH = true;
            io.Fonts->AddFontFromFileTTF(fontPath, scaledFontSize, &fontCfg);
            fontLoaded = true;
            break;
        }
    }
    if (!fontLoaded) {
        io.Fonts->AddFontDefault();
    }

    setupVulkanWindow(fbWidth, fbHeight);

    SDL_SetWindowPosition(m_window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(m_window);
    SDL_RaiseWindow(m_window);

#if defined(_WIN32)
    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(m_window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    if (hwnd) {
        ShowWindow(hwnd, SW_SHOWNORMAL);
        UpdateWindow(hwnd);
        SetForegroundWindow(hwnd);
    }
#endif

    m_initialized = true;
    return true;
}

bool VulkanContext::setupVulkan() {
    VkResult err;

    // 1. Instance extensions from SDL3
    uint32_t sdlExtCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
    if (!sdlExtensions) {
        std::cerr << "SDL_Vulkan_GetInstanceExtensions failed: " << SDL_GetError() << std::endl;
        return false;
    }

    std::vector<const char*> instanceExtensions(sdlExtensions, sdlExtensions + sdlExtCount);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "GPUBench";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 1);
    appInfo.pEngineName = "GPUBench";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 1);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

    err = vkCreateInstance(&createInfo, m_allocator, &m_instance);
    if (err != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed with code " << err << std::endl;
        return false;
    }

    // 2. Surface creation
    if (!SDL_Vulkan_CreateSurface(m_window, m_instance, m_allocator, &m_mainWindowData.Surface)) {
        std::cerr << "SDL_Vulkan_CreateSurface failed: " << SDL_GetError() << std::endl;
        return false;
    }

    // 3. Physical Device Selection
    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &gpuCount, nullptr);
    if (gpuCount == 0) {
        std::cerr << "No Vulkan physical devices found" << std::endl;
        return false;
    }
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(m_instance, &gpuCount, gpus.data());
    m_physicalDevice = gpus[0];

    // 4. Queue Family Selection
    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueProps(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueCount, queueProps.data());
    for (uint32_t i = 0; i < queueCount; ++i) {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(m_physicalDevice, i, m_mainWindowData.Surface, &presentSupport);
        if ((queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && presentSupport) {
            m_queueFamily = i;
            break;
        }
    }
    if (m_queueFamily == static_cast<uint32_t>(-1)) {
        for (uint32_t i = 0; i < queueCount; ++i) {
            if (queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                m_queueFamily = i;
                break;
            }
        }
    }

    // 5. Logical Device Creation
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = m_queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    err = vkCreateDevice(m_physicalDevice, &deviceCreateInfo, m_allocator, &m_device);
    if (err != VK_SUCCESS) {
        std::cerr << "vkCreateDevice failed with code " << err << std::endl;
        return false;
    }

    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);

    // 6. Descriptor Pool for Dear ImGui
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 100 },
        { VK_DESCRIPTOR_TYPE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 50 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 50 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 200;
    poolInfo.poolSizeCount = static_cast<uint32_t>(sizeof(poolSizes) / sizeof(poolSizes[0]));
    poolInfo.pPoolSizes = poolSizes;

    err = vkCreateDescriptorPool(m_device, &poolInfo, m_allocator, &m_descriptorPool);
    if (err != VK_SUCCESS) {
        std::cerr << "vkCreateDescriptorPool failed with code " << err << std::endl;
        return false;
    }

    // 7. Transient Command Pool for texture uploads & transfer operations
    VkCommandPoolCreateInfo cmdPoolInfo{};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = m_queueFamily;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    err = vkCreateCommandPool(m_device, &cmdPoolInfo, m_allocator, &m_transientCommandPool);
    if (err != VK_SUCCESS) {
        std::cerr << "vkCreateCommandPool (transient) failed with code " << err << std::endl;
        return false;
    }

    return true;
}

void VulkanContext::setupVulkanWindow(int width, int height) {
    const VkFormat requestFormats[] = {
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_R8G8B8A8_SRGB
    };
    const VkColorSpaceKHR requestColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
    m_mainWindowData.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        m_physicalDevice, m_mainWindowData.Surface,
        requestFormats, sizeof(requestFormats) / sizeof(requestFormats[0]),
        requestColorSpace
    );

    const VkPresentModeKHR presentModes[] = {
        VK_PRESENT_MODE_FIFO_KHR, // V-Sync enabled for UI stability
        VK_PRESENT_MODE_MAILBOX_KHR,
        VK_PRESENT_MODE_IMMEDIATE_KHR
    };
    m_mainWindowData.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        m_physicalDevice, m_mainWindowData.Surface,
        presentModes, sizeof(presentModes) / sizeof(presentModes[0])
    );

    ImGui_ImplVulkanH_CreateOrResizeWindow(
        m_instance, m_physicalDevice, m_device,
        &m_mainWindowData, m_queueFamily, m_allocator,
        width, height, m_minImageCount
    );

    // Initialize ImGui backends
    ImGui_ImplSDL3_InitForVulkan(m_window);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = m_instance;
    initInfo.PhysicalDevice = m_physicalDevice;
    initInfo.Device = m_device;
    initInfo.QueueFamily = m_queueFamily;
    initInfo.Queue = m_queue;
    initInfo.PipelineCache = m_pipelineCache;
    initInfo.DescriptorPool = m_descriptorPool;
    initInfo.RenderPass = m_mainWindowData.RenderPass;
    initInfo.Subpass = 0;
    initInfo.MinImageCount = m_minImageCount;
    initInfo.ImageCount = m_mainWindowData.ImageCount;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.Allocator = m_allocator;

    ImGui_ImplVulkan_Init(&initInfo);
}

void VulkanContext::cleanupVulkanWindow() {
    ImGui_ImplVulkanH_DestroyWindow(m_instance, m_device, &m_mainWindowData, m_allocator);
}

void VulkanContext::resize(int width, int height) {
    if (width <= 0 || height <= 0) return;
    ImGui_ImplVulkan_SetMinImageCount(m_minImageCount);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        m_instance, m_physicalDevice, m_device,
        &m_mainWindowData, m_queueFamily, m_allocator,
        width, height, m_minImageCount
    );
    m_mainWindowData.FrameIndex = 0;
    m_swapchainRebuild = false;
}

void VulkanContext::beginFrame() {
    if (m_swapchainRebuild) {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(m_window, &w, &h);
        if (w > 0 && h > 0) {
            resize(w, h);
        }
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void VulkanContext::endFrame() {
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    const bool isMinimized = (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f);

    if (!isMinimized) {
        frameRender(drawData);
        framePresent();
    }
}

void VulkanContext::frameRender(ImDrawData* drawData) {
    VkResult err;
    VkSemaphore imageAcquiredSemaphore = m_mainWindowData.FrameSemaphores[m_mainWindowData.SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore renderCompleteSemaphore = m_mainWindowData.FrameSemaphores[m_mainWindowData.SemaphoreIndex].RenderCompleteSemaphore;

    err = vkAcquireNextImageKHR(m_device, m_mainWindowData.Swapchain, UINT64_MAX, imageAcquiredSemaphore, VK_NULL_HANDLE, &m_mainWindowData.FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        m_swapchainRebuild = true;
        return;
    }

    ImGui_ImplVulkanH_Frame* fd = &m_mainWindowData.Frames[m_mainWindowData.FrameIndex];
    err = vkWaitForFences(m_device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);
    err = vkResetFences(m_device, 1, &fd->Fence);

    err = vkResetCommandPool(m_device, fd->CommandPool, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(fd->CommandBuffer, &beginInfo);

    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = m_mainWindowData.RenderPass;
    rpInfo.framebuffer = fd->Framebuffer;
    rpInfo.renderArea.extent.width = m_mainWindowData.Width;
    rpInfo.renderArea.extent.height = m_mainWindowData.Height;
    VkClearValue clearVal{};
    clearVal.color = m_clearColor;
    rpInfo.clearValueCount = 1;
    rpInfo.pClearValues = &clearVal;
    vkCmdBeginRenderPass(fd->CommandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    ImGui_ImplVulkan_RenderDrawData(drawData, fd->CommandBuffer);

    vkCmdEndRenderPass(fd->CommandBuffer);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &imageAcquiredSemaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &fd->CommandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderCompleteSemaphore;

    err = vkEndCommandBuffer(fd->CommandBuffer);
    err = vkQueueSubmit(m_queue, 1, &submitInfo, fd->Fence);
}

void VulkanContext::framePresent() {
    if (m_swapchainRebuild) return;

    VkSemaphore renderCompleteSemaphore = m_mainWindowData.FrameSemaphores[m_mainWindowData.SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderCompleteSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_mainWindowData.Swapchain;
    presentInfo.pImageIndices = &m_mainWindowData.FrameIndex;

    VkResult err = vkQueuePresentKHR(m_queue, &presentInfo);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        m_swapchainRebuild = true;
        return;
    }

    m_mainWindowData.SemaphoreIndex = (m_mainWindowData.SemaphoreIndex + 1) % m_mainWindowData.SemaphoreCount;
}

void VulkanContext::shutdown() {
    if (!m_initialized) return;

    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    cleanupVulkanWindow();

    if (m_transientCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_transientCommandPool, m_allocator);
        m_transientCommandPool = VK_NULL_HANDLE;
    }

    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, m_allocator);
        m_descriptorPool = VK_NULL_HANDLE;
    }

    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, m_allocator);
        m_device = VK_NULL_HANDLE;
    }

    if (m_mainWindowData.Surface != VK_NULL_HANDLE) {
        SDL_Vulkan_DestroySurface(m_instance, m_mainWindowData.Surface, m_allocator);
        m_mainWindowData.Surface = VK_NULL_HANDLE;
    }

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, m_allocator);
        m_instance = VK_NULL_HANDLE;
    }

    if (m_window) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }

    SDL_Quit();
    m_initialized = false;
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return 0;
}

VulkanContext::TextureResource VulkanContext::createTextureRgba(int width, int height, const uint8_t* rgbaPixels) {
    TextureResource tex{};
    if (!m_device || !rgbaPixels || width <= 0 || height <= 0) return tex;

    tex.width = width;
    tex.height = height;
    tex.channels = 4;
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;

    // 1. Create Staging Buffer
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_device, &bufferInfo, m_allocator, &stagingBuffer) != VK_SUCCESS) {
        return tex;
    }

    VkMemoryRequirements memReqs{};
    vkGetBufferMemoryRequirements(m_device, stagingBuffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(m_device, &allocInfo, m_allocator, &stagingMemory) != VK_SUCCESS) {
        vkDestroyBuffer(m_device, stagingBuffer, m_allocator);
        return tex;
    }
    vkBindBufferMemory(m_device, stagingBuffer, stagingMemory, 0);

    void* data = nullptr;
    if (vkMapMemory(m_device, stagingMemory, 0, imageSize, 0, &data) == VK_SUCCESS) {
        std::memcpy(data, rgbaPixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(m_device, stagingMemory);
    } else {
        vkDestroyBuffer(m_device, stagingBuffer, m_allocator);
        vkFreeMemory(m_device, stagingMemory, m_allocator);
        return tex;
    }

    // 2. Create Optimal Device Local Image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(width);
    imageInfo.extent.height = static_cast<uint32_t>(height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(m_device, &imageInfo, m_allocator, &tex.image) != VK_SUCCESS) {
        vkDestroyBuffer(m_device, stagingBuffer, m_allocator);
        vkFreeMemory(m_device, stagingMemory, m_allocator);
        return tex;
    }

    VkMemoryRequirements imgMemReqs{};
    vkGetImageMemoryRequirements(m_device, tex.image, &imgMemReqs);

    VkMemoryAllocateInfo imgAllocInfo{};
    imgAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imgAllocInfo.allocationSize = imgMemReqs.size;
    imgAllocInfo.memoryTypeIndex = findMemoryType(imgMemReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(m_device, &imgAllocInfo, m_allocator, &tex.memory) != VK_SUCCESS) {
        vkDestroyImage(m_device, tex.image, m_allocator);
        tex.image = VK_NULL_HANDLE;
        vkDestroyBuffer(m_device, stagingBuffer, m_allocator);
        vkFreeMemory(m_device, stagingMemory, m_allocator);
        return tex;
    }
    vkBindImageMemory(m_device, tex.image, tex.memory, 0);

    // 3. One-time copy command buffer
    VkCommandPool copyPool = m_transientCommandPool ? m_transientCommandPool : m_mainWindowData.Frames[0].CommandPool;
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = copyPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &cmd) == VK_SUCCESS) {
        VkCommandBufferBeginInfo cmdBeginInfo{};
        cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &cmdBeginInfo);

        // Transition Undefined -> Transfer Dst Optimal
        VkImageMemoryBarrier barrier1{};
        barrier1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier1.image = tex.image;
        barrier1.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier1.subresourceRange.baseMipLevel = 0;
        barrier1.subresourceRange.levelCount = 1;
        barrier1.subresourceRange.baseArrayLayer = 0;
        barrier1.subresourceRange.layerCount = 1;
        barrier1.srcAccessMask = 0;
        barrier1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier1);

        // Copy buffer to image
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };

        vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // Transition Transfer Dst Optimal -> Shader Read Only Optimal
        VkImageMemoryBarrier barrier2{};
        barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier2.image = tex.image;
        barrier2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier2.subresourceRange.baseMipLevel = 0;
        barrier2.subresourceRange.levelCount = 1;
        barrier2.subresourceRange.baseArrayLayer = 0;
        barrier2.subresourceRange.layerCount = 1;
        barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier2);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);

        vkFreeCommandBuffers(m_device, copyPool, 1, &cmd);
    }

    vkDestroyBuffer(m_device, stagingBuffer, m_allocator);
    vkFreeMemory(m_device, stagingMemory, m_allocator);

    // 4. Create Image View
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(m_device, &viewInfo, m_allocator, &tex.imageView) != VK_SUCCESS) {
        destroyTexture(tex);
        return tex;
    }

    // 5. Create Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;

    if (vkCreateSampler(m_device, &samplerInfo, m_allocator, &tex.sampler) != VK_SUCCESS) {
        destroyTexture(tex);
        return tex;
    }

    // 6. Register with ImGui Vulkan backend
    tex.descriptorSet = ImGui_ImplVulkan_AddTexture(tex.sampler, tex.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    return tex;
}

VulkanContext::TextureResource VulkanContext::loadTextureFromFile(const std::string& filepath) {
    TextureResource tex{};
    std::string candidatePaths[] = {
        filepath,
        "../" + filepath,
        "../../" + filepath
    };
    std::string resolvedPath = "";
    for (const auto& p : candidatePaths) {
        if (std::filesystem::exists(p)) {
            resolvedPath = p;
            break;
        }
    }
    if (resolvedPath.empty()) {
        return tex;
    }

    int w = 0, h = 0, ch = 0;
    uint8_t* pixels = stbi_load(resolvedPath.c_str(), &w, &h, &ch, 4);
    if (!pixels) {
        return tex;
    }

    tex = createTextureRgba(w, h, pixels);
    stbi_image_free(pixels);
    return tex;
}

void VulkanContext::destroyTexture(TextureResource& tex) {
    if (m_device == VK_NULL_HANDLE) return;

    if (tex.descriptorSet != VK_NULL_HANDLE) {
        ImGui_ImplVulkan_RemoveTexture(tex.descriptorSet);
        tex.descriptorSet = VK_NULL_HANDLE;
    }
    if (tex.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, tex.sampler, m_allocator);
        tex.sampler = VK_NULL_HANDLE;
    }
    if (tex.imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, tex.imageView, m_allocator);
        tex.imageView = VK_NULL_HANDLE;
    }
    if (tex.image != VK_NULL_HANDLE) {
        vkDestroyImage(m_device, tex.image, m_allocator);
        tex.image = VK_NULL_HANDLE;
    }
    if (tex.memory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, tex.memory, m_allocator);
        tex.memory = VK_NULL_HANDLE;
    }
    tex.width = 0;
    tex.height = 0;
    tex.channels = 0;
}

} // namespace gpubench::gui
