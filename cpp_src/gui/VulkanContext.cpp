#include "VulkanContext.h"
#include "core/VulkanContext.h"
#include "core/RunnerAPI.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

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
#include "utils/ImageExport.h"
#include "utils/KernelPath.h"

void ImGui_ImplVulkanH_CreateWindowSwapChain(VkPhysicalDevice physical_device, VkDevice device, ImGui_ImplVulkanH_Window* wd, const VkAllocationCallbacks* allocator, int w, int h, uint32_t min_image_count, VkImageUsageFlags image_usage);
void ImGui_ImplVulkanH_CreateWindowCommandBuffers(VkPhysicalDevice physical_device, VkDevice device, ImGui_ImplVulkanH_Window* wd, uint32_t queue_family, const VkAllocationCallbacks* allocator);

namespace gpubench::gui {

VulkanContext::VulkanContext() = default;

VulkanContext::~VulkanContext() {
    shutdown();
}

void VulkanContext::waitDeviceIdleSafe() {
    if (m_device == VK_NULL_HANDLE) return;
    if (m_sharedComputeCtx) {
        std::lock_guard<std::recursive_mutex> qlock(m_sharedComputeCtx->getQueueMutex());
        vkDeviceWaitIdle(m_device);
    } else {
        vkDeviceWaitIdle(m_device);
    }
}

bool VulkanContext::init(const char* title, int width, int height,
                         float scaleOverride, bool verbose, bool debug,
                         int preferredDeviceIndex) {
    if (m_initialized) return true;

#if defined(_WIN32)
    SDL_SetHint(SDL_HINT_WINDOWS_INTRESOURCE_ICON, "1");
    SDL_SetHint(SDL_HINT_WINDOWS_INTRESOURCE_ICON_SMALL, "1");
#endif

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::cerr << "Failed to initialize SDL3: " << SDL_GetError() << std::endl;
        return false;
    }

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

#if defined(_WIN32)
    {
        SDL_PropertiesID props = SDL_GetWindowProperties(m_window);
        HWND hwnd = static_cast<HWND>(SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
        if (hwnd) {
            HINSTANCE hInst = GetModuleHandleW(nullptr);
            HICON hIconBig = static_cast<HICON>(LoadImageW(
                hInst, MAKEINTRESOURCEW(1), IMAGE_ICON,
                GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR));
            HICON hIconSmall = static_cast<HICON>(LoadImageW(
                hInst, MAKEINTRESOURCEW(1), IMAGE_ICON,
                GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR));
            if (hIconBig) {
                SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hIconBig));
            }
            if (hIconSmall) {
                SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hIconSmall));
            }
        }
    }
#endif

    float winScale = SDL_GetWindowDisplayScale(m_window);
    if (winScale > 0.0f) {
        detectedScale = winScale;
    }
    m_displayScale = (scaleOverride > 0.0f) ? scaleOverride : detectedScale;
    if (m_displayScale <= 0.0f) {
        m_displayScale = 1.0f;
    }

    if (!setupVulkan(verbose, debug, preferredDeviceIndex)) {
        std::cerr << "Failed to initialize shared Vulkan 1.4 context" << std::endl;
        return false;
    }

    int fbWidth = 0, fbHeight = 0;
    SDL_GetWindowSizeInPixels(m_window, &fbWidth, &fbHeight);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Dear ImGui 1.93.0 dynamic font baking (ImGuiBackendFlags_RendererHasTextures):
    // Load unscaled base fonts once; FontScaleDpi scales rasterized glyphs dynamically without vkDeviceWaitIdle stalls.
    loadSystemFont();
    ImGui::GetStyle().FontScaleDpi = m_displayScale;
    m_builtFontScale = m_displayScale;

    setupVulkanWindow(fbWidth, fbHeight);

    SDL_SetWindowPosition(m_window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(m_window);
    SDL_RaiseWindow(m_window);

#if defined(_WIN32)
    HWND hwnd = static_cast<HWND>(SDL_GetPointerProperty(SDL_GetWindowProperties(m_window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    if (hwnd) {
        ShowWindow(hwnd, SW_SHOWNORMAL);
        UpdateWindow(hwnd);
        SetForegroundWindow(hwnd);
    }
#endif

    m_initialized = true;
    return true;
}

void VulkanContext::loadSystemFont() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();
    m_fontRegular = nullptr;
    m_fontBold = nullptr;
    m_fontMono = nullptr;

    // Symbol fallback Unicode ranges merged from platform symbol fonts
    static const ImWchar kSymbolMergeRanges[] = {
        0x2010, 0x206F, // General Punctuation (–, —, •, …, ‰)
        0x2190, 0x21FF, // Arrows (←, ↑, →, ↓, ↔, ⇒)
        0x2200, 0x22FF, // Mathematical Operators (≥, ≤, ≈, ≠, ∞, ∆, ∑)
        0x2300, 0x23FF, // Miscellaneous Technical (⏱, ⏳)
        0x2500, 0x259F, // Box Drawing & Block Elements (─, │, ┌, ┐, └, ┘, █, ▓)
        0x25A0, 0x25FF, // Geometric Shapes (▪, ▸, ▾, ▲, ▼, ●, ◆)
        0x2600, 0x26FF, // Miscellaneous Symbols (⚠, ⚡, ★)
        0x2700, 0x27BF, // Dingbats (✓, ✔, ✕, ✖, ✗, ✘)
        0,
    };

    const char* regularCandidates[] = {
#if defined(_WIN32)
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
#elif defined(__APPLE__)
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
#endif
        "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/rsms-inter-fonts/Inter-Regular.ttf",
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"
    };

    const char* boldCandidates[] = {
#if defined(_WIN32)
        "C:\\Windows\\Fonts\\consolab.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\calibrib.ttf",
#elif defined(__APPLE__)
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
#endif
        "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Bold.ttf",
        "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/rsms-inter-fonts/Inter-SemiBold.ttf",
        "/usr/share/fonts/rsms-inter-fonts/Inter-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
    };

    const char* monoCandidates[] = {
#if defined(_WIN32)
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\cour.ttf",
#elif defined(__APPLE__)
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
#endif
        "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    };

    const char* symbolFallbackCandidates[] = {
#if defined(_WIN32)
        "C:\\Windows\\Fonts\\seguisym.ttf",
#elif defined(__APPLE__)
        "/System/Library/Fonts/Apple Symbols.ttf",
#endif
        "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        "/usr/share/fonts/google-noto/NotoSansSymbols2-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    };

    auto fileExists = [](const char* path) -> bool {
        FILE* f = fopen(path, "rb");
        if (f) {
            fclose(f);
            return true;
        }
        return false;
    };

    auto mergeSymbolFallback = [&](float fontSize) {
        for (const char* symPath : symbolFallbackCandidates) {
            if (fileExists(symPath)) {
                ImFontConfig mergeCfg;
                mergeCfg.MergeMode = true;
                mergeCfg.OversampleH = 2;
                mergeCfg.OversampleV = 1;
                io.Fonts->AddFontFromFileTTF(symPath, fontSize, &mergeCfg, kSymbolMergeRanges);
                break;
            }
        }
    };

    constexpr float kBaseRegFontSize  = 14.0f;
    constexpr float kBaseBoldFontSize = 15.0f;
    constexpr float kBaseMonoFontSize = 13.5f;

    // 1. Primary Regular UI Font (unscaled base size; dynamic atlas bakes glyphs on demand)
    for (const char* fontPath : regularCandidates) {
        if (fileExists(fontPath)) {
            ImFontConfig fontCfg;
            fontCfg.OversampleH = 2;
            fontCfg.OversampleV = 1;
            m_fontRegular = io.Fonts->AddFontFromFileTTF(fontPath, kBaseRegFontSize, &fontCfg, nullptr);
            if (m_fontRegular) {
                mergeSymbolFallback(kBaseRegFontSize);
                break;
            }
        }
    }
    if (!m_fontRegular) {
        ImFontConfig fontCfg;
        fontCfg.SizePixels = kBaseRegFontSize;
        m_fontRegular = io.Fonts->AddFontDefaultVector(&fontCfg);
    }

    // 2. Bold Header / KPI Font
    for (const char* fontPath : boldCandidates) {
        if (fileExists(fontPath)) {
            ImFontConfig fontCfg;
            fontCfg.OversampleH = 2;
            fontCfg.OversampleV = 1;
            m_fontBold = io.Fonts->AddFontFromFileTTF(fontPath, kBaseBoldFontSize, &fontCfg, nullptr);
            if (m_fontBold) {
                mergeSymbolFallback(kBaseBoldFontSize);
                break;
            }
        }
    }
    if (!m_fontBold) {
        m_fontBold = m_fontRegular;
    }

    // 3. Tabular Monospace Font for Scores & Telemetry
    for (const char* fontPath : monoCandidates) {
        if (fileExists(fontPath)) {
            ImFontConfig fontCfg;
            fontCfg.OversampleH = 2;
            fontCfg.OversampleV = 1;
            m_fontMono = io.Fonts->AddFontFromFileTTF(fontPath, kBaseMonoFontSize, &fontCfg, nullptr);
            if (m_fontMono) {
                mergeSymbolFallback(kBaseMonoFontSize);
                break;
            }
        }
    }
    if (!m_fontMono) {
        m_fontMono = m_fontRegular;
    }
}

void VulkanContext::requestFontRebuild(float scale) {
    if (scale > 0.0f && std::abs(scale - m_builtFontScale) > 0.005f) {
        m_pendingFontScale = scale;
    }
}

bool VulkanContext::setupVulkan(bool verbose, bool debug, int preferredDeviceIndex) {
    uint32_t sdlExtCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
    if (!sdlExtensions) {
        std::cerr << "SDL_Vulkan_GetInstanceExtensions failed: " << SDL_GetError() << std::endl;
        return false;
    }

    std::vector<const char*> instanceExtensions(sdlExtensions, sdlExtensions + sdlExtCount);

    try {
        m_sharedComputeCtx = std::make_shared<::VulkanContext>(verbose, debug, instanceExtensions);
    } catch (const std::exception& e) {
        std::cerr << "Shared VulkanContext creation failed: " << e.what() << std::endl;
        return false;
    }

    if (!m_sharedComputeCtx || !m_sharedComputeCtx->isAvailable()) {
        std::cerr << "Vulkan 1.4 instance unavailable" << std::endl;
        return false;
    }

    m_instance = m_sharedComputeCtx->getInstance();

    if (!SDL_Vulkan_CreateSurface(m_window, m_instance, m_allocator, &m_mainWindowData.Surface)) {
        std::cerr << "SDL_Vulkan_CreateSurface failed: " << SDL_GetError() << std::endl;
        m_sharedComputeCtx.reset();
        m_instance = VK_NULL_HANDLE;
        return false;
    }

    m_sharedComputeCtx->setPresentationSurface(m_mainWindowData.Surface);

    const auto& devices = m_sharedComputeCtx->getDevices();
    if (devices.empty()) {
        std::cerr << "No Vulkan 1.4 physical devices enumerated" << std::endl;
        return false;
    }

    // Choose preferred device (or index 0, which ::VulkanContext sorts with discrete GPUs first)
    uint32_t chosenIdx = 0;
    if (preferredDeviceIndex >= 0 && static_cast<size_t>(preferredDeviceIndex) < devices.size()) {
        chosenIdx = static_cast<uint32_t>(preferredDeviceIndex);
    }

    try {
        m_sharedComputeCtx->pickDevice(chosenIdx);
    } catch (const std::exception& e) {
        std::cerr << "VulkanContext::pickDevice(" << chosenIdx << ") failed: " << e.what() << std::endl;
        return false;
    }

    m_activeDeviceIndex = chosenIdx;
    m_physicalDevice = m_sharedComputeCtx->getPhysicalDevice();
    m_device = m_sharedComputeCtx->getDevice();
    m_queueFamily = m_sharedComputeCtx->getPresentationQueueFamilyIndex();
    m_queue = m_sharedComputeCtx->getPresentationQueue();
    m_pipelineCache = m_sharedComputeCtx->getPipelineCache();

    // Verify presentation support on selected queue family
    VkBool32 presentSupported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(m_physicalDevice, m_queueFamily,
                                         m_mainWindowData.Surface, &presentSupported);
    if (presentSupported != VK_TRUE && devices.size() > 1) {
        // Fallback to another device if the headless compute GPU has no display outputs attached
        for (uint32_t d = 0; d < static_cast<uint32_t>(devices.size()); ++d) {
            if (d == chosenIdx) continue;
            try {
                m_sharedComputeCtx->pickDevice(d);
                VkPhysicalDevice candPhys = m_sharedComputeCtx->getPhysicalDevice();
                uint32_t candQf = m_sharedComputeCtx->getPresentationQueueFamilyIndex();
                VkBool32 candPres = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(candPhys, candQf, m_mainWindowData.Surface, &candPres);
                if (candPres == VK_TRUE) {
                    m_activeDeviceIndex = d;
                    m_physicalDevice = candPhys;
                    m_device = m_sharedComputeCtx->getDevice();
                    m_queueFamily = candQf;
                    m_queue = m_sharedComputeCtx->getPresentationQueue();
                    m_pipelineCache = m_sharedComputeCtx->getPipelineCache();
                    break;
                }
            } catch (...) {
            }
        }
    }

    // Register shared Vulkan 1.4 context globally so RunBenchmarksAPI & viewport share GPU memory directly
    SetSharedVulkanContextAPI(m_sharedComputeCtx);

    // Descriptor Pool for Dear ImGui 1.93.0 (dynamic font textures + user/shared preview textures)
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 512 },
        { VK_DESCRIPTOR_TYPE_SAMPLER, 64 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 600;
    poolInfo.poolSizeCount = static_cast<uint32_t>(sizeof(poolSizes) / sizeof(poolSizes[0]));
    poolInfo.pPoolSizes = poolSizes;

    VkResult err = vkCreateDescriptorPool(m_device, &poolInfo, m_allocator, &m_descriptorPool);
    if (err != VK_SUCCESS) {
        std::cerr << "vkCreateDescriptorPool failed with code " << err << std::endl;
        return false;
    }

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
        VK_PRESENT_MODE_MAILBOX_KHR,
        VK_PRESENT_MODE_FIFO_KHR,
        VK_PRESENT_MODE_IMMEDIATE_KHR
    };
    m_mainWindowData.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        m_physicalDevice, m_mainWindowData.Surface,
        presentModes, sizeof(presentModes) / sizeof(presentModes[0])
    );

    {
        const int presentMin = ImGui_ImplVulkanH_GetMinImageCountFromPresentMode(m_mainWindowData.PresentMode);
        m_minImageCount = presentMin > 2 ? static_cast<uint32_t>(presentMin) : 2u;
    }

    {
        std::unique_lock<std::recursive_mutex> qlock;
        if (m_sharedComputeCtx) {
            qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
        }
        ImGui_ImplVulkanH_CreateWindowSwapChain(
            m_physicalDevice, m_device, &m_mainWindowData, m_allocator,
            width, height, m_minImageCount, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        );
        ImGui_ImplVulkanH_CreateWindowCommandBuffers(
            m_physicalDevice, m_device, &m_mainWindowData, m_queueFamily, m_allocator
        );
    }

    if (m_mainWindowData.ImageCount > 0 && m_minImageCount > m_mainWindowData.ImageCount) {
        m_minImageCount = m_mainWindowData.ImageCount >= 2u ? m_mainWindowData.ImageCount : 2u;
    }

    ImGui_ImplSDL3_InitForVulkan(m_window);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_4;
    initInfo.Instance = m_instance;
    initInfo.PhysicalDevice = m_physicalDevice;
    initInfo.Device = m_device;
    initInfo.QueueFamily = m_queueFamily;
    initInfo.Queue = m_queue;
    initInfo.PipelineCache = m_pipelineCache;
    initInfo.DescriptorPool = m_descriptorPool;
    initInfo.MinImageCount = m_minImageCount;
    initInfo.ImageCount = m_mainWindowData.ImageCount;
    initInfo.PipelineInfoMain.RenderPass = m_mainWindowData.RenderPass;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.Allocator = m_allocator;

    ImGui_ImplVulkan_Init(&initInfo);
    createCrtPipeline();
}

void VulkanContext::setClearColor(float r, float g, float b, float a) {
    m_clearColor.float32[0] = r;
    m_clearColor.float32[1] = g;
    m_clearColor.float32[2] = b;
    m_clearColor.float32[3] = a;
}

void VulkanContext::setCrtPostFx(bool enabled, int paletteMode, float intensity) {
    m_crtEnabled = enabled;
    m_crtPaletteMode = paletteMode;
    m_crtIntensity = intensity;
}

bool VulkanContext::createCrtPipeline() {
    if (m_device == VK_NULL_HANDLE || m_mainWindowData.RenderPass == VK_NULL_HANDLE) {
        return false;
    }
    destroyCrtPipeline();

    std::vector<std::filesystem::path> searchDirs;
    std::filesystem::path kp = KernelPath::find();
    if (!kp.empty()) {
        searchDirs.push_back(kp / "vulkan");
    }
    if (const char* base = SDL_GetBasePath()) {
        std::filesystem::path bp(base);
        searchDirs.push_back(bp / "kernels" / "vulkan");
        searchDirs.push_back(bp / ".." / "share" / "gpubench" / "kernels" / "vulkan");
    }
    searchDirs.push_back(std::filesystem::current_path() / "kernels" / "vulkan");

    auto loadSpv = [&](const char* filename) -> VkShaderModule {
        std::error_code ec;
        for (const auto& dir : searchDirs) {
            std::filesystem::path p = dir / filename;
            if (!std::filesystem::exists(p, ec)) continue;
            std::ifstream ifs(p, std::ios::binary | std::ios::ate);
            if (!ifs) continue;
            std::streamsize sz = ifs.tellg();
            if (sz <= 0 || (sz % 4) != 0) continue;
            ifs.seekg(0, std::ios::beg);
            std::vector<uint32_t> code(static_cast<size_t>(sz) / sizeof(uint32_t));
            if (!ifs.read(reinterpret_cast<char*>(code.data()), sz)) continue;
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = static_cast<size_t>(sz);
            ci.pCode = code.data();
            VkShaderModule mod = VK_NULL_HANDLE;
            if (vkCreateShaderModule(m_device, &ci, m_allocator, &mod) == VK_SUCCESS) {
                return mod;
            }
        }
        return VK_NULL_HANDLE;
    };

    VkShaderModule vertMod = loadSpv("gui_crt_overlay.vert.spv");
    VkShaderModule fragMod = loadSpv("gui_crt_overlay.frag.spv");
    if (vertMod == VK_NULL_HANDLE || fragMod == VK_NULL_HANDLE) {
        if (vertMod != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, vertMod, m_allocator);
        if (fragMod != VK_NULL_HANDLE) vkDestroyShaderModule(m_device, fragMod, m_allocator);
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = 16;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(m_device, &layoutInfo, m_allocator, &m_crtPipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, vertMod, m_allocator);
        vkDestroyShaderModule(m_device, fragMod, m_allocator);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAsm{};
    inputAsm.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAsm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.blendEnable = VK_TRUE;
    blendAtt.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAtt.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAtt.colorBlendOp = VK_BLEND_OP_ADD;
    blendAtt.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAtt.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAtt.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAtt;

    VkDynamicState dynStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynInfo{};
    dynInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynInfo.dynamicStateCount = 2;
    dynInfo.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAsm;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &raster;
    pipeInfo.pMultisampleState = &ms;
    pipeInfo.pColorBlendState = &blend;
    pipeInfo.pDynamicState = &dynInfo;
    pipeInfo.layout = m_crtPipelineLayout;
    pipeInfo.renderPass = m_mainWindowData.RenderPass;
    pipeInfo.subpass = 0;

    VkResult res = vkCreateGraphicsPipelines(
        m_device, m_pipelineCache, 1, &pipeInfo, m_allocator, &m_crtPipeline);

    vkDestroyShaderModule(m_device, vertMod, m_allocator);
    vkDestroyShaderModule(m_device, fragMod, m_allocator);

    if (res != VK_SUCCESS) {
        destroyCrtPipeline();
        return false;
    }
    return true;
}

void VulkanContext::destroyCrtPipeline() {
    if (m_device == VK_NULL_HANDLE) return;
    if (m_crtPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_crtPipeline, m_allocator);
        m_crtPipeline = VK_NULL_HANDLE;
    }
    if (m_crtPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_crtPipelineLayout, m_allocator);
        m_crtPipelineLayout = VK_NULL_HANDLE;
    }
}

void VulkanContext::recordCrtOverlay(VkCommandBuffer cmd, uint32_t width, uint32_t height) {
    if (!m_crtEnabled || m_crtPipeline == VK_NULL_HANDLE ||
        m_crtPipelineLayout == VK_NULL_HANDLE || width == 0 || height == 0) {
        return;
    }

    VkViewport vp{};
    vp.x = 0.0f;
    vp.y = 0.0f;
    vp.width = static_cast<float>(width);
    vp.height = static_cast<float>(height);
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = {width, height};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_crtPipeline);
    struct {
        float width;
        float height;
        float intensity;
        int32_t paletteMode;
    } pc{vp.width, vp.height, m_crtIntensity, static_cast<int32_t>(m_crtPaletteMode)};
    vkCmdPushConstants(cmd, m_crtPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void VulkanContext::cleanupVulkanWindow() {
    std::unique_lock<std::recursive_mutex> qlock;
    if (m_sharedComputeCtx) {
        qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
    }
    ImGui_ImplVulkanH_DestroyWindow(m_instance, m_device, &m_mainWindowData, m_allocator);
}

void VulkanContext::requestSwapchainRebuild() {
    m_windowResizePending = true;
}

void VulkanContext::resize(int width, int height) {
    if (width <= 0 || height <= 0) return;
    std::unique_lock<std::recursive_mutex> qlock;
    if (m_sharedComputeCtx) {
        qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
    }
    ImGui_ImplVulkan_SetMinImageCount(m_minImageCount);
    ImGui_ImplVulkanH_CreateWindowSwapChain(
        m_physicalDevice, m_device, &m_mainWindowData, m_allocator,
        width, height, m_minImageCount, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    );
    if (m_mainWindowData.ImageCount > 0 && m_minImageCount > m_mainWindowData.ImageCount) {
        m_minImageCount = m_mainWindowData.ImageCount >= 2u ? m_mainWindowData.ImageCount : 2u;
        ImGui_ImplVulkan_SetMinImageCount(m_minImageCount);
    }
    ImGui_ImplVulkanH_CreateWindowCommandBuffers(
        m_physicalDevice, m_device, &m_mainWindowData, m_queueFamily, m_allocator
    );
    m_mainWindowData.FrameIndex = 0;
    m_mainWindowData.SemaphoreIndex = 0;
    m_swapchainRebuild = false;
    m_windowResizePending = false;
}

void VulkanContext::beginFrame() {
    if (m_swapchainRebuild || m_windowResizePending) {
        int w = 0, h = 0;
        SDL_GetWindowSizeInPixels(m_window, &w, &h);
        if (w > 0 && h > 0) {
            if (m_swapchainRebuild || w != m_mainWindowData.Width || h != m_mainWindowData.Height) {
                resize(w, h);
            } else {
                m_windowResizePending = false;
            }
        }
    }

    // Dynamic font baking in Dear ImGui 1.93.0: adjust FontScaleDpi without stalling GPU or rebuilding atlas!
    if (m_pendingFontScale > 0.0f) {
        ImGui::GetStyle().FontScaleDpi = m_pendingFontScale;
        m_builtFontScale = m_pendingFontScale;
        m_pendingFontScale = 0.0f;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void VulkanContext::flushDynamicImGuiTextures(ImDrawData* drawData) {
    if (!drawData || !drawData->Textures) return;
    bool needsUpload = false;
    for (ImTextureData* tex : *drawData->Textures) {
        if (tex && tex->Status != ImTextureStatus_OK) {
            needsUpload = true;
            break;
        }
    }
    if (!needsUpload) return;

    std::unique_lock<std::recursive_mutex> qlock;
    if (m_sharedComputeCtx) {
        qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
    }
    for (ImTextureData* tex : *drawData->Textures) {
        if (tex && tex->Status != ImTextureStatus_OK) {
            ImGui_ImplVulkan_UpdateTexture(tex);
        }
    }
}

void VulkanContext::endFrame() {
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    const bool isMinimized = (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f);

    if (!isMinimized) {
        if (frameRender(drawData)) {
            framePresent();
        }
    }
}

bool VulkanContext::frameRender(ImDrawData* drawData) {
    VkResult err;
    VkSemaphore imageAcquiredSemaphore = m_mainWindowData.FrameSemaphores[m_mainWindowData.SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore renderCompleteSemaphore = m_mainWindowData.FrameSemaphores[m_mainWindowData.SemaphoreIndex].RenderCompleteSemaphore;

    constexpr uint64_t kAcquireTimeoutNs = 100'000'000ULL;
    err = vkAcquireNextImageKHR(m_device, m_mainWindowData.Swapchain, kAcquireTimeoutNs, imageAcquiredSemaphore, VK_NULL_HANDLE, &m_mainWindowData.FrameIndex);
    if (err == VK_TIMEOUT || err == VK_NOT_READY) {
        SDL_PumpEvents();
        return false;
    }
    if (err == VK_ERROR_OUT_OF_DATE_KHR) {
        m_swapchainRebuild = true;
        return false;
    }
    if (err == VK_SUBOPTIMAL_KHR) {
        m_swapchainRebuild = true;
    } else if (err != VK_SUCCESS) {
        return false;
    }

    ImGui_ImplVulkanH_Frame* fd = &m_mainWindowData.Frames[m_mainWindowData.FrameIndex];

    while (true) {
        constexpr uint64_t kFenceTimeoutNs = 16'000'000ULL;
        err = vkWaitForFences(m_device, 1, &fd->Fence, VK_TRUE, kFenceTimeoutNs);
        if (err == VK_SUCCESS) {
            break;
        }
        if (err == VK_TIMEOUT) {
            SDL_PumpEvents();
            SDL_Event ev;
            if (SDL_PeepEvents(&ev, 1, SDL_PEEKEVENT, SDL_EVENT_QUIT, SDL_EVENT_QUIT) > 0) {
                err = vkWaitForFences(m_device, 1, &fd->Fence, VK_TRUE, UINT64_MAX);
                break;
            }
            continue;
        }
        break;
    }

    // Synchronize any dynamic font glyph uploads on the shared queue before command buffer recording
    flushDynamicImGuiTextures(drawData);

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
    if (m_crtEnabled) {
        recordCrtOverlay(fd->CommandBuffer,
                         static_cast<uint32_t>(m_mainWindowData.Width),
                         static_cast<uint32_t>(m_mainWindowData.Height));
    }

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
    {
        std::unique_lock<std::recursive_mutex> qlock;
        if (m_sharedComputeCtx) {
            qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
        }
        err = vkQueueSubmit(m_queue, 1, &submitInfo, fd->Fence);
    }
    (void)err;
    return true;
}

void VulkanContext::framePresent() {
    VkSemaphore renderCompleteSemaphore = m_mainWindowData.FrameSemaphores[m_mainWindowData.SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderCompleteSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_mainWindowData.Swapchain;
    presentInfo.pImageIndices = &m_mainWindowData.FrameIndex;

    VkResult err = VK_SUCCESS;
    {
        std::unique_lock<std::recursive_mutex> qlock;
        if (m_sharedComputeCtx) {
            qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
        }
        err = vkQueuePresentKHR(m_queue, &presentInfo);
    }
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        m_swapchainRebuild = true;
    }

    m_mainWindowData.SemaphoreIndex = (m_mainWindowData.SemaphoreIndex + 1) % m_mainWindowData.SemaphoreCount;
}

void VulkanContext::shutdown() {
    if (!m_initialized) return;

    waitDeviceIdleSafe();

    destroyCrtPipeline();

    {
        std::unique_lock<std::recursive_mutex> qlock;
        if (m_sharedComputeCtx) {
            qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
        }
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
    }

    cleanupVulkanWindow();

    if (m_transientCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_transientCommandPool, m_allocator);
        m_transientCommandPool = VK_NULL_HANDLE;
    }

    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, m_allocator);
        m_descriptorPool = VK_NULL_HANDLE;
    }

    // Destroy window surface after swapchain destruction and before shared VkInstance teardown
    if (m_mainWindowData.Surface != VK_NULL_HANDLE && m_instance != VK_NULL_HANDLE) {
        if (m_sharedComputeCtx) {
            m_sharedComputeCtx->setPresentationSurface(VK_NULL_HANDLE);
        }
        SDL_Vulkan_DestroySurface(m_instance, m_mainWindowData.Surface, m_allocator);
        m_mainWindowData.Surface = VK_NULL_HANDLE;
    }

    SetSharedVulkanContextAPI(nullptr);
    m_sharedComputeCtx.reset();
    m_device = VK_NULL_HANDLE;
    m_physicalDevice = VK_NULL_HANDLE;
    m_instance = VK_NULL_HANDLE;
    m_queue = VK_NULL_HANDLE;
    m_pipelineCache = VK_NULL_HANDLE;

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

VulkanContext::TextureResource VulkanContext::createTextureFromSharedView(
    VkImageView imageView, int width, int height, uint64_t generation) {
    TextureResource tex{};
    if (!m_device || imageView == VK_NULL_HANDLE || width <= 0 || height <= 0) {
        return tex;
    }
    tex.width = width;
    tex.height = height;
    tex.channels = 4;
    tex.imageView = imageView;
    tex.borrowsExternalImage = true;
    tex.generation = generation;
    tex.descriptorSet = ImGui_ImplVulkan_AddTexture(imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return tex;
}

VulkanContext::TextureResource VulkanContext::createTextureRgba(int width, int height, const uint8_t* rgbaPixels) {
    TextureResource tex{};
    if (!m_device || !rgbaPixels || width <= 0 || height <= 0) return tex;

    tex.width = width;
    tex.height = height;
    tex.channels = 4;
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * 4;

    VmaAllocator vma = m_sharedComputeCtx ? m_sharedComputeCtx->getVmaAllocator() : VK_NULL_HANDLE;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VmaAllocation stagingVmaAlloc = VK_NULL_HANDLE;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vma != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo stagingAllocCi{};
        stagingAllocCi.usage = VMA_MEMORY_USAGE_AUTO;
        stagingAllocCi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo mappedInfo{};
        if (vmaCreateBuffer(vma, &bufferInfo, &stagingAllocCi, &stagingBuffer,
                            &stagingVmaAlloc, &mappedInfo) != VK_SUCCESS ||
            !mappedInfo.pMappedData) {
            return tex;
        }
        std::memcpy(mappedInfo.pMappedData, rgbaPixels, static_cast<size_t>(imageSize));
        vmaFlushAllocation(vma, stagingVmaAlloc, 0, imageSize);
    } else {
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
    }

    auto destroyStaging = [&]() {
        if (vma != VK_NULL_HANDLE && stagingVmaAlloc != VK_NULL_HANDLE) {
            vmaDestroyBuffer(vma, stagingBuffer, stagingVmaAlloc);
        } else {
            if (stagingBuffer != VK_NULL_HANDLE) vkDestroyBuffer(m_device, stagingBuffer, m_allocator);
            if (stagingMemory != VK_NULL_HANDLE) vkFreeMemory(m_device, stagingMemory, m_allocator);
        }
    };

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

    if (vma != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo imgVmaCi{};
        imgVmaCi.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(vma, &imageInfo, &imgVmaCi, &tex.image, &tex.vmaAllocation, nullptr) != VK_SUCCESS) {
            destroyStaging();
            return tex;
        }
    } else {
        if (vkCreateImage(m_device, &imageInfo, m_allocator, &tex.image) != VK_SUCCESS) {
            destroyStaging();
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
            destroyStaging();
            return tex;
        }
        vkBindImageMemory(m_device, tex.image, tex.memory, 0);
    }

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

        VkImageMemoryBarrier barrier1{};
        barrier1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier1.image = tex.image;
        barrier1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier1.srcAccessMask = 0;
        barrier1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier1);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
        vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        VkImageMemoryBarrier barrier2{};
        barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier2.image = tex.image;
        barrier2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier2);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        {
            std::unique_lock<std::recursive_mutex> qlock;
            if (m_sharedComputeCtx) {
                qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
            }
            vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_queue);
        }

        vkFreeCommandBuffers(m_device, copyPool, 1, &cmd);
    }

    destroyStaging();

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (vkCreateImageView(m_device, &viewInfo, m_allocator, &tex.imageView) != VK_SUCCESS) {
        destroyTexture(tex);
        return tex;
    }

    tex.descriptorSet = ImGui_ImplVulkan_AddTexture(tex.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return tex;
}

VulkanContext::TextureResource VulkanContext::loadTextureFromFile(const std::string& filepath) {
    TextureResource tex{};
    std::filesystem::path exeDir;
    if (const char* basePath = SDL_GetBasePath()) {
        exeDir = std::filesystem::path(basePath);
    }

    std::vector<std::string> writablePaths = {
        filepath,
        "../" + filepath,
        "../../" + filepath
    };
    if (!exeDir.empty()) {
        writablePaths.push_back((exeDir / filepath).string());
        writablePaths.push_back((exeDir / ".." / filepath).string());
        writablePaths.push_back((exeDir / ".." / ".." / filepath).string());
    }
#ifdef _WIN32
    if (const char* localAppData = std::getenv("LOCALAPPDATA")) {
        writablePaths.push_back((std::filesystem::path(localAppData) / "gpubench" / filepath).string());
        writablePaths.push_back((std::filesystem::path(localAppData) / "GPUBench" / filepath).string());
    }
#else
    if (const char* xdgCache = std::getenv("XDG_CACHE_HOME")) {
        if (xdgCache[0] != '\0') {
            writablePaths.push_back((std::filesystem::path(xdgCache) / "gpubench" / filepath).string());
        }
    }
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            writablePaths.push_back((std::filesystem::path(home) / ".cache" / "gpubench" / filepath).string());
        }
    }
#endif

    std::string resolvedPath = "";
    std::error_code ec;
    std::filesystem::file_time_type newestTime{};
    for (const auto& p : writablePaths) {
        if (std::filesystem::exists(p, ec) && !ec) {
            auto ftime = std::filesystem::last_write_time(p, ec);
            if (!ec && (resolvedPath.empty() || ftime > newestTime)) {
                newestTime = ftime;
                resolvedPath = p;
            } else if (resolvedPath.empty()) {
                resolvedPath = p;
            }
        }
    }

    if (resolvedPath.empty()) {
        std::vector<std::string> candidatePaths = {
            "share/gpubench/" + filepath,
            "../share/gpubench/" + filepath,
            "/usr/share/gpubench/" + filepath,
            "/usr/local/share/gpubench/" + filepath
        };
        if (!exeDir.empty()) {
            candidatePaths.push_back((exeDir / ".." / "share" / "gpubench" / filepath).string());
            candidatePaths.push_back((exeDir / "share" / "gpubench" / filepath).string());
        }

        const std::string renderPrefix = "renders/render_";
        if (filepath.rfind(renderPrefix, 0) == 0) {
            std::string rest = filepath.substr(renderPrefix.size());
            auto replaceSuffix = [&](const std::string& fromSuf, const std::string& toSuf) {
                if (rest.size() >= fromSuf.size() &&
                    rest.compare(rest.size() - fromSuf.size(), fromSuf.size(), fromSuf) == 0) {
                    rest = rest.substr(0, rest.size() - fromSuf.size()) + toSuf;
                }
            };
            replaceSuffix("_traditional_megakernel.png", "_stage7_final.png");
            replaceSuffix("_worklist_dgc.png", "_stage7_final.png");
            replaceSuffix("_pathtracing_1spp_worklist.png", "_pathtracing_1spp.png");
            replaceSuffix("_pathtracing_1spp_traditional.png", "_pathtracing_1spp.png");
            replaceSuffix("_pathtracing_16spp_worklist.png", "_pathtracing_16spp.png");
            replaceSuffix("_pathtracing_16spp_traditional.png", "_pathtracing_16spp.png");
            replaceSuffix("_multilight_128_traditional.png", "_multilight_128_dgc.png");

            std::string thumbRel = "assets/thumbnails/thumb_" + rest;
            candidatePaths.push_back(thumbRel);
            candidatePaths.push_back("../" + thumbRel);
            candidatePaths.push_back("../../" + thumbRel);
            if (!exeDir.empty()) {
                candidatePaths.push_back((exeDir / thumbRel).string());
                candidatePaths.push_back((exeDir / ".." / thumbRel).string());
                candidatePaths.push_back((exeDir / ".." / "share" / "gpubench" / thumbRel).string());
                candidatePaths.push_back((exeDir / "share" / "gpubench" / thumbRel).string());
            }
            candidatePaths.push_back("/usr/share/gpubench/" + thumbRel);
            candidatePaths.push_back("/usr/local/share/gpubench/" + thumbRel);
        }

        for (const auto& p : candidatePaths) {
            if (std::filesystem::exists(p, ec) && !ec) {
                resolvedPath = p;
                break;
            }
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

    if (!tex.borrowsExternalImage) {
        if (tex.sampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, tex.sampler, m_allocator);
            tex.sampler = VK_NULL_HANDLE;
        }
        if (tex.imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, tex.imageView, m_allocator);
            tex.imageView = VK_NULL_HANDLE;
        }
        VmaAllocator vma = m_sharedComputeCtx ? m_sharedComputeCtx->getVmaAllocator() : VK_NULL_HANDLE;
        if (tex.vmaAllocation != VK_NULL_HANDLE && vma != VK_NULL_HANDLE) {
            vmaDestroyImage(vma, tex.image, tex.vmaAllocation);
            tex.image = VK_NULL_HANDLE;
            tex.vmaAllocation = VK_NULL_HANDLE;
        } else {
            if (tex.image != VK_NULL_HANDLE) {
                vkDestroyImage(m_device, tex.image, m_allocator);
                tex.image = VK_NULL_HANDLE;
            }
            if (tex.memory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, tex.memory, m_allocator);
                tex.memory = VK_NULL_HANDLE;
            }
        }
    } else {
        tex.imageView = VK_NULL_HANDLE;
        tex.image = VK_NULL_HANDLE;
    }

    tex.borrowsExternalImage = false;
    tex.generation = 0;
    tex.width = 0;
    tex.height = 0;
    tex.channels = 0;
}

bool VulkanContext::saveScreenshot(const std::string& filepath) {
    if (!m_initialized || m_device == VK_NULL_HANDLE || filepath.empty()) {
        return false;
    }
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData || m_mainWindowData.Width <= 0 || m_mainWindowData.Height <= 0) {
        return false;
    }

    waitDeviceIdleSafe();
    flushDynamicImGuiTextures(drawData);

    const uint32_t w = static_cast<uint32_t>(m_mainWindowData.Width);
    const uint32_t h = static_cast<uint32_t>(m_mainWindowData.Height);
    const VkFormat fmt = m_mainWindowData.SurfaceFormat.format;
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(w) * h * 4;

    VkImage offImage = VK_NULL_HANDLE;
    VkDeviceMemory offMem = VK_NULL_HANDLE;
    VkImageView offView = VK_NULL_HANDLE;
    VkRenderPass offRp = VK_NULL_HANDLE;
    VkFramebuffer offFb = VK_NULL_HANDLE;
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;

    auto cleanupAll = [&]() {
        if (offFb != VK_NULL_HANDLE) vkDestroyFramebuffer(m_device, offFb, m_allocator);
        if (offRp != VK_NULL_HANDLE) vkDestroyRenderPass(m_device, offRp, m_allocator);
        if (offView != VK_NULL_HANDLE) vkDestroyImageView(m_device, offView, m_allocator);
        if (offImage != VK_NULL_HANDLE) vkDestroyImage(m_device, offImage, m_allocator);
        if (offMem != VK_NULL_HANDLE) vkFreeMemory(m_device, offMem, m_allocator);
        if (stagingBuf != VK_NULL_HANDLE) vkDestroyBuffer(m_device, stagingBuf, m_allocator);
        if (stagingMem != VK_NULL_HANDLE) vkFreeMemory(m_device, stagingMem, m_allocator);
    };

    VkImageCreateInfo imgInfo{};
    imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.extent = {w, h, 1};
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.format = fmt;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    if (vkCreateImage(m_device, &imgInfo, m_allocator, &offImage) != VK_SUCCESS) {
        cleanupAll();
        return false;
    }

    VkMemoryRequirements imgReqs{};
    vkGetImageMemoryRequirements(m_device, offImage, &imgReqs);
    VkMemoryAllocateInfo imgAlloc{};
    imgAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    imgAlloc.allocationSize = imgReqs.size;
    imgAlloc.memoryTypeIndex = findMemoryType(imgReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(m_device, &imgAlloc, m_allocator, &offMem) != VK_SUCCESS) {
        cleanupAll();
        return false;
    }
    vkBindImageMemory(m_device, offImage, offMem, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = offImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = fmt;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(m_device, &viewInfo, m_allocator, &offView) != VK_SUCCESS) {
        cleanupAll();
        return false;
    }

    VkAttachmentDescription att{};
    att.format = fmt;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    // Match ImGui_ImplVulkanH_Window::RenderPass subpass dependency exactly for VUID-vkCmdDrawIndexed-renderPass-02684 compatibility
    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpCreate{};
    rpCreate.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCreate.attachmentCount = 1;
    rpCreate.pAttachments = &att;
    rpCreate.subpassCount = 1;
    rpCreate.pSubpasses = &subpass;
    rpCreate.dependencyCount = 1;
    rpCreate.pDependencies = &dep;
    if (vkCreateRenderPass(m_device, &rpCreate, m_allocator, &offRp) != VK_SUCCESS) {
        cleanupAll();
        return false;
    }

    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = offRp;
    fbInfo.attachmentCount = 1;
    fbInfo.pAttachments = &offView;
    fbInfo.width = w;
    fbInfo.height = h;
    fbInfo.layers = 1;
    if (vkCreateFramebuffer(m_device, &fbInfo, m_allocator, &offFb) != VK_SUCCESS) {
        cleanupAll();
        return false;
    }

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = byteSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_device, &bufInfo, m_allocator, &stagingBuf) != VK_SUCCESS) {
        cleanupAll();
        return false;
    }

    VkMemoryRequirements bufReqs{};
    vkGetBufferMemoryRequirements(m_device, stagingBuf, &bufReqs);
    VkMemoryAllocateInfo bufAlloc{};
    bufAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bufAlloc.allocationSize = bufReqs.size;
    bufAlloc.memoryTypeIndex = findMemoryType(
        bufReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(m_device, &bufAlloc, m_allocator, &stagingMem) != VK_SUCCESS) {
        cleanupAll();
        return false;
    }
    vkBindBufferMemory(m_device, stagingBuf, stagingMem, 0);

    VkCommandPool pool = m_transientCommandPool ? m_transientCommandPool : m_mainWindowData.Frames[0].CommandPool;
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = pool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &cmdAlloc, &cmd) != VK_SUCCESS) {
        cleanupAll();
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = offRp;
    rpBegin.framebuffer = offFb;
    rpBegin.renderArea.extent = {w, h};
    VkClearValue clearVal{};
    clearVal.color = m_clearColor;
    rpBegin.clearValueCount = 1;
    rpBegin.pClearValues = &clearVal;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
    if (m_crtEnabled) {
        recordCrtOverlay(cmd, w, h);
    }
    vkCmdEndRenderPass(cmd);

    VkMemoryBarrier copyMemBarrier{};
    copyMemBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    copyMemBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    copyMemBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &copyMemBarrier, 0, nullptr, 0, nullptr);

    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = {w, h, 1};
    vkCmdCopyImageToBuffer(cmd, offImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuf, 1, &copyRegion);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    {
        std::unique_lock<std::recursive_mutex> qlock;
        if (m_sharedComputeCtx) {
            qlock = std::unique_lock<std::recursive_mutex>(m_sharedComputeCtx->getQueueMutex());
        }
        vkQueueSubmit(m_queue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);
    }
    vkFreeCommandBuffers(m_device, pool, 1, &cmd);

    void* mapped = nullptr;
    if (vkMapMemory(m_device, stagingMem, 0, byteSize, 0, &mapped) != VK_SUCCESS || !mapped) {
        cleanupAll();
        return false;
    }

    const bool isBgra = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB);
    const uint8_t* src = static_cast<const uint8_t*>(mapped);
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    const size_t pixelCount = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < pixelCount; ++i) {
        if (isBgra) {
            rgb[i * 3 + 0] = src[i * 4 + 2];
            rgb[i * 3 + 1] = src[i * 4 + 1];
            rgb[i * 3 + 2] = src[i * 4 + 0];
        } else {
            rgb[i * 3 + 0] = src[i * 4 + 0];
            rgb[i * 3 + 1] = src[i * 4 + 1];
            rgb[i * 3 + 2] = src[i * 4 + 2];
        }
    }
    vkUnmapMemory(m_device, stagingMem);
    cleanupAll();

    std::error_code ec;
    std::filesystem::path outPath(filepath);
    if (outPath.has_parent_path()) {
        std::filesystem::create_directories(outPath.parent_path(), ec);
    }
    return gpubench::ImageExport::writePNG(filepath, w, h, rgb);
}

} // namespace gpubench::gui
