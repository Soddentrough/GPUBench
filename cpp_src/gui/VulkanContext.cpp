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

} // namespace gpubench::gui
