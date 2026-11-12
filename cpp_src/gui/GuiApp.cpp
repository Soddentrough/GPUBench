#include "GuiApp.h"
#include "GuiUiHelpers.h"
#include "core/ResultImporter.h"
#include "core/VulkanContext.h"
#include "utils/HardwareTelemetry.h"

#ifndef GPUBENCH_VERSION
#define GPUBENCH_VERSION "1.0.0"
#endif

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace gpubench::gui {

namespace {
struct HostCpuInfo {
    std::string modelName;
    int logicalCores = 0;
};

struct HostMemInfo {
    double totalRamGB = 0.0;
};

HostCpuInfo queryHostCpuInfo() {
    HostCpuInfo out;
    out.logicalCores = static_cast<int>(std::thread::hardware_concurrency());
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    auto cpuidLeaf = [](unsigned int leaf, unsigned int regs[4]) {
#ifdef _MSC_VER
        int msvcRegs[4] = {0, 0, 0, 0};
        __cpuid(msvcRegs, static_cast<int>(leaf));
        for (int i = 0; i < 4; ++i) regs[i] = static_cast<unsigned int>(msvcRegs[i]);
#else
        __cpuid(leaf, regs[0], regs[1], regs[2], regs[3]);
#endif
    };
    unsigned int ext[4] = {0, 0, 0, 0};
    cpuidLeaf(0x80000000u, ext);
    if (ext[0] >= 0x80000004u) {
        char brand[49];
        std::memset(brand, 0, sizeof(brand));
        for (unsigned int i = 0; i < 3; ++i) {
            unsigned int r[4] = {0, 0, 0, 0};
            cpuidLeaf(0x80000002u + i, r);
            std::memcpy(brand + i * 16, r, 16);
        }
        std::string s(brand);
        size_t first = s.find_first_not_of(' ');
        size_t last = s.find_last_not_of(' ');
        if (first != std::string::npos) {
            out.modelName = s.substr(first, last - first + 1);
        }
    }
#endif
#ifndef _WIN32
    if (out.modelName.empty()) {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("model name", 0) == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string val = line.substr(colon + 1);
                    size_t s0 = val.find_first_not_of(" \t");
                    if (s0 != std::string::npos) out.modelName = val.substr(s0);
                    break;
                }
            }
        }
    }
#endif
    return out;
}

HostMemInfo queryHostMemInfo() {
    HostMemInfo out;
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        out.totalRamGB = static_cast<double>(ms.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
    }
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
    long pages = sysconf(_SC_PHYS_PAGES);
    long pageSz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && pageSz > 0) {
        out.totalRamGB = (static_cast<double>(pages) * static_cast<double>(pageSz)) /
                         (1024.0 * 1024.0 * 1024.0);
    }
#endif
    return out;
}
} // namespace

GuiApp::GuiApp() = default;

GuiApp::~GuiApp() {
    abortBenchmarks();
    m_telemetryWorker.stop();
    if (m_execThread.joinable()) {
        m_execThread.join();
    }
    clearTextureCache();
}

void GuiApp::clearTextureCache() {
    if (!m_vulkanContext) {
        m_textureCache.clear();
        return;
    }
    if (!m_textureCache.empty()) {
        m_vulkanContext->waitDeviceIdleSafe();
        for (auto& kv : m_textureCache) {
            if (kv.second.isValid() || kv.second.sampler != VK_NULL_HANDLE ||
                kv.second.imageView != VK_NULL_HANDLE || kv.second.image != VK_NULL_HANDLE) {
                m_vulkanContext->destroyTexture(kv.second);
            }
        }
    }
    m_textureCache.clear();
}

VulkanContext::TextureResource GuiApp::getOrLoadTexture(const std::string& relPath) {
    if (!m_vulkanContext || !m_vulkanContext->isInitialized() || relPath.empty()) {
        return {};
    }

    std::string normKey = relPath;
    for (char& ch : normKey) {
        if (ch == '\\') ch = '/';
    }

    // 1. Zero-disk on-GPU shared preview image fast path (when shared VulkanContext has captured HDR -> RGBA8)
    if (auto sharedCtx = m_vulkanContext->getSharedComputeContext()) {
        ::VulkanContext::SharedPreviewImage sharedImg{};
        std::string bareKey = (normKey.rfind("renders/", 0) == 0) ? normKey.substr(8) : normKey;
        if ((sharedCtx->getSharedPreviewImage(normKey, sharedImg) ||
             sharedCtx->getSharedPreviewImage(bareKey, sharedImg)) &&
            sharedImg.view != VK_NULL_HANDLE) {
            auto it = m_textureCache.find(normKey);
            if (it != m_textureCache.end()) {
                if (it->second.borrowsExternalImage &&
                    it->second.generation == sharedImg.generation &&
                    it->second.isValid()) {
                    return it->second;
                }
                m_vulkanContext->waitDeviceIdleSafe();
                m_vulkanContext->destroyTexture(it->second);
                m_textureCache.erase(it);
            }
            VulkanContext::TextureResource borrowedTex =
                m_vulkanContext->createTextureFromSharedView(
                    sharedImg.view,
                    static_cast<int>(sharedImg.width),
                    static_cast<int>(sharedImg.height),
                    sharedImg.generation);
            m_textureCache[normKey] = borrowedTex;
            return borrowedTex;
        }
    }

    // 2. Disk-backed texture cache fallback
    auto it = m_textureCache.find(normKey);
    if (it != m_textureCache.end() && !it->second.borrowsExternalImage) {
        return it->second;
    }
    if (it != m_textureCache.end()) {
        m_vulkanContext->waitDeviceIdleSafe();
        m_vulkanContext->destroyTexture(it->second);
        m_textureCache.erase(it);
    }

    namespace fs = std::filesystem;
    std::vector<std::string> candidateRoots = {
        "",
        "../",
        "../../"
    };
    if (const char* base = SDL_GetBasePath()) {
        candidateRoots.push_back(std::string(base));
        candidateRoots.push_back(std::string(base) + "../");
        candidateRoots.push_back(std::string(base) + "../share/gpubench/");
    }
    candidateRoots.push_back("/usr/share/gpubench/");
    candidateRoots.push_back("/usr/local/share/gpubench/");

    for (const auto& root : candidateRoots) {
        std::string fullPath = root + normKey;
        std::error_code ec;
        if (fs::exists(fullPath, ec) && fs::is_regular_file(fullPath, ec)) {
            VulkanContext::TextureResource tex = m_vulkanContext->loadTextureFromFile(fullPath);
            m_textureCache[normKey] = tex;
            return tex;
        }
    }

    m_textureCache[normKey] = VulkanContext::TextureResource{};
    return {};
}

void GuiApp::init(VulkanContext* vulkanContext, float uiScale) {
    m_vulkanContext = vulkanContext;
    m_baseScale = std::clamp(uiScale, 0.5f, 3.0f);
    m_uiScale = m_baseScale;
    setupDarkTheme(m_uiScale);
    if (m_vulkanContext && m_vulkanContext->isInitialized()) {
        m_vulkanContext->requestFontRebuild(m_uiScale);
    }
    ImGui::GetStyle().FontScaleDpi = m_uiScale;
    ImGui::GetIO().FontGlobalScale = 1.0f;

    initializeBenchmarkCategories();
    hydrateMetadataFromCatalog();
    discoverHardware();
    if (m_vulkanContext && m_vulkanContext->isInitialized()) {
        setSelectedDevice(static_cast<int>(m_vulkanContext->getActiveDeviceIndex()));
    }
    m_telemetryWorker.start();
}

void GuiApp::setUiScale(float scale) {
    scale = std::clamp(scale, 0.5f, 3.0f);
    if (std::fabs(scale - m_uiScale) < 0.001f) return;
    m_uiScale = scale;
    setupDarkTheme(m_uiScale);
    if (m_vulkanContext && m_vulkanContext->isInitialized()) {
        m_vulkanContext->requestFontRebuild(m_uiScale);
    }
    ImGui::GetStyle().FontScaleDpi = m_uiScale;
    ImGui::GetIO().FontGlobalScale = 1.0f;
}

void GuiApp::updateZoomShortcuts() {
    ImGuiIO& io = ImGui::GetIO();
    if (m_zoomToastTimer > 0.0f) {
        m_zoomToastTimer = std::max(0.0f, m_zoomToastTimer - io.DeltaTime);
    }

    if (io.WantTextInput) return;

    const bool ctrlHeld = io.KeyCtrl || io.KeySuper;
    if (!ctrlHeld) return;

    const float step = 0.1f;
    bool changed = false;
    float target = m_uiScale;

    if (ImGui::IsKeyPressed(ImGuiKey_Equal, false) ||
        ImGui::IsKeyPressed(ImGuiKey_KeypadAdd, false)) {
        target = std::round((m_uiScale + step) * 10.0f) / 10.0f;
        changed = true;
    } else if (ImGui::IsKeyPressed(ImGuiKey_Minus, false) ||
               ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract, false)) {
        target = std::round((m_uiScale - step) * 10.0f) / 10.0f;
        changed = true;
    } else if (ImGui::IsKeyPressed(ImGuiKey_0, false) ||
               ImGui::IsKeyPressed(ImGuiKey_Keypad0, false)) {
        target = m_baseScale;
        changed = true;
    } else if (std::fabs(io.MouseWheel) > 0.01f) {
        float delta = (io.MouseWheel > 0.0f) ? step : -step;
        target = std::round((m_uiScale + delta) * 10.0f) / 10.0f;
        changed = true;
    }

    if (changed) {
        target = std::clamp(target, 0.5f, 3.0f);
        setUiScale(target);
        m_zoomToastTimer = 1.6f;
    }
}

void GuiApp::renderZoomToast() {
    if (m_zoomToastTimer <= 0.0f) return;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    float alpha = std::min(1.0f, m_zoomToastTimer / 0.35f);
    ImVec2 pos(vp->WorkPos.x + vp->WorkSize.x - s(20.0f),
               vp->WorkPos.y + s(20.0f));
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.88f * alpha);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, s(6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(s(12.0f), s(7.0f)));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.05f, 0.72f, 0.90f, 0.75f * alpha));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav |
                             ImGuiWindowFlags_NoInputs;
    if (ImGui::Begin("##ZoomToast", nullptr, flags)) {
        int pct = static_cast<int>(std::round((m_uiScale / std::max(0.1f, m_baseScale)) * 100.0f));
        ImGui::TextColored(ImVec4(0.92f, 0.96f, 1.0f, alpha),
                           "UI Scale: %d%% (%.2fx)  [Ctrl+0 reset]", pct, m_uiScale);
    }
    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void GuiApp::setupDarkTheme(float scale) {
    ImGuiStyle& style = ImGui::GetStyle();
    if (!m_baseStyleInitialized) {
        style = ImGuiStyle();
        ImGui::StyleColorsDark(&style);
        style.WindowRounding = 0.0f;
        style.ChildRounding = 8.0f;
        style.FrameRounding = 5.0f;
        style.PopupRounding = 6.0f;
        style.ScrollbarRounding = 6.0f;
        style.GrabRounding = 4.0f;
        style.TabRounding = 6.0f;
        style.WindowBorderSize = 1.0f;
        style.ChildBorderSize = 1.0f;
        style.FrameBorderSize = 1.0f;
        style.WindowPadding = ImVec2(12.0f, 12.0f);
        style.FramePadding = ImVec2(8.0f, 5.0f);
        style.ItemSpacing = ImVec2(8.0f, 6.0f);
        style.CellPadding = ImVec2(7.0f, 5.0f);
        style.ScrollbarSize = 12.0f;

        ImVec4* colors = style.Colors;
        colors[ImGuiCol_WindowBg]             = ImVec4(0.055f, 0.063f, 0.082f, 1.00f);
        colors[ImGuiCol_ChildBg]              = ImVec4(0.075f, 0.086f, 0.114f, 1.00f);
        colors[ImGuiCol_PopupBg]              = ImVec4(0.085f, 0.098f, 0.130f, 0.98f);
        colors[ImGuiCol_Border]               = ImVec4(0.165f, 0.195f, 0.255f, 0.85f);
        colors[ImGuiCol_BorderShadow]         = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_FrameBg]              = ImVec4(0.105f, 0.122f, 0.162f, 1.00f);
        colors[ImGuiCol_FrameBgHovered]       = ImVec4(0.145f, 0.172f, 0.230f, 1.00f);
        colors[ImGuiCol_FrameBgActive]        = ImVec4(0.180f, 0.220f, 0.295f, 1.00f);
        colors[ImGuiCol_TitleBg]              = ImVec4(0.055f, 0.063f, 0.082f, 1.00f);
        colors[ImGuiCol_TitleBgActive]        = ImVec4(0.075f, 0.090f, 0.125f, 1.00f);
        colors[ImGuiCol_MenuBarBg]            = ImVec4(0.070f, 0.080f, 0.105f, 1.00f);
        colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.055f, 0.063f, 0.082f, 0.60f);
        colors[ImGuiCol_ScrollbarGrab]        = ImVec4(0.200f, 0.235f, 0.305f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.270f, 0.320f, 0.410f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive]  = ImVec4(0.05f, 0.68f, 0.88f, 1.00f);
        colors[ImGuiCol_CheckMark]            = ImVec4(0.06f, 0.78f, 0.96f, 1.00f);
        colors[ImGuiCol_SliderGrab]           = ImVec4(0.05f, 0.68f, 0.88f, 1.00f);
        colors[ImGuiCol_SliderGrabActive]     = ImVec4(0.15f, 0.82f, 1.00f, 1.00f);
        colors[ImGuiCol_Button]               = ImVec4(0.120f, 0.145f, 0.200f, 1.00f);
        colors[ImGuiCol_ButtonHovered]        = ImVec4(0.175f, 0.220f, 0.305f, 1.00f);
        colors[ImGuiCol_ButtonActive]         = ImVec4(0.05f, 0.60f, 0.82f, 1.00f);
        colors[ImGuiCol_Header]               = ImVec4(0.125f, 0.155f, 0.220f, 0.90f);
        colors[ImGuiCol_HeaderHovered]        = ImVec4(0.175f, 0.225f, 0.315f, 0.95f);
        colors[ImGuiCol_HeaderActive]         = ImVec4(0.06f, 0.55f, 0.78f, 1.00f);
        colors[ImGuiCol_Separator]            = ImVec4(0.165f, 0.195f, 0.255f, 0.85f);
        colors[ImGuiCol_Tab]                  = ImVec4(0.090f, 0.108f, 0.145f, 1.00f);
        colors[ImGuiCol_TabHovered]           = ImVec4(0.170f, 0.240f, 0.350f, 1.00f);
        colors[ImGuiCol_TabSelected]          = ImVec4(0.045f, 0.460f, 0.680f, 1.00f);
        colors[ImGuiCol_TabDimmed]            = ImVec4(0.070f, 0.082f, 0.110f, 1.00f);
        colors[ImGuiCol_TabDimmedSelected]    = ImVec4(0.110f, 0.260f, 0.400f, 1.00f);
        colors[ImGuiCol_TableHeaderBg]        = ImVec4(0.095f, 0.115f, 0.158f, 1.00f);
        colors[ImGuiCol_TableBorderStrong]    = ImVec4(0.180f, 0.215f, 0.285f, 1.00f);
        colors[ImGuiCol_TableBorderLight]     = ImVec4(0.125f, 0.150f, 0.200f, 0.80f);
        colors[ImGuiCol_TableRowBg]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_TableRowBgAlt]        = ImVec4(0.12f, 0.15f, 0.22f, 0.22f);
        colors[ImGuiCol_Text]                 = ImVec4(0.93f, 0.95f, 0.98f, 1.00f);
        colors[ImGuiCol_TextDisabled]         = ImVec4(0.46f, 0.51f, 0.60f, 1.00f);

        m_baseStyle = style;
        m_baseStyleInitialized = true;
    } else {
        style = m_baseStyle;
    }

    if (std::fabs(scale - 1.0f) > 0.001f) {
        style.ScaleAllSizes(scale);
    }
    style.FontScaleDpi = scale;

    ImPlotStyle& plotStyle = ImPlot::GetStyle();
    if (!m_basePlotStyleInitialized) {
        ImPlot::StyleColorsDark(&plotStyle);
        plotStyle.PlotBorderSize = 1.0f;
        plotStyle.MajorGridSize = ImVec2(1.0f, 1.0f);
        plotStyle.MinorGridSize = ImVec2(0.5f, 0.5f);
        plotStyle.PlotPadding = ImVec2(10.0f, 8.0f);
        plotStyle.LabelPadding = ImVec2(5.0f, 4.0f);
        plotStyle.LegendPadding = ImVec2(8.0f, 6.0f);
        plotStyle.Colors[ImPlotCol_PlotBg] = ImVec4(0.055f, 0.065f, 0.090f, 1.00f);
        plotStyle.Colors[ImPlotCol_FrameBg] = ImVec4(0.075f, 0.086f, 0.114f, 0.00f);
        plotStyle.Colors[ImPlotCol_PlotBorder] = ImVec4(0.165f, 0.195f, 0.255f, 0.85f);
        m_basePlotStyle = plotStyle;
        m_basePlotStyleInitialized = true;
    } else {
        plotStyle = m_basePlotStyle;
    }
    plotStyle.PlotPadding = ImVec2(10.0f * scale, 8.0f * scale);
    plotStyle.LabelPadding = ImVec2(5.0f * scale, 4.0f * scale);
    plotStyle.LegendPadding = ImVec2(8.0f * scale, 6.0f * scale);
}

void GuiApp::discoverHardware() {
    m_devices.clear();
    m_apiSupportList = GetAllComputeApiSupportAPI();
    m_deviceProfiles = GetDeviceProfilesAPI();

    bool currentValid = false;
    for (const auto& api : m_apiSupportList) {
        if (api.name == m_selectedBackend && api.isSupported) {
            currentValid = true;
            break;
        }
    }
    if (!currentValid) {
        for (const auto& api : m_apiSupportList) {
            if (api.isSupported && api.name != "auto") {
                m_selectedBackend = api.name;
                break;
            }
        }
    }

    auto rawHw = GetAvailableHardwareAPI();
    auto cpuInfo = queryHostCpuInfo();
    auto memInfo = queryHostMemInfo();

    bool addedSystem = false;
    std::vector<uint32_t> seenGpuIndices;

    for (const auto& entry : rawHw) {
        std::stringstream ss(entry);
        std::string backend, idxStr, name;
        if (std::getline(ss, backend, '|') &&
            std::getline(ss, idxStr, '|') &&
            std::getline(ss, name)) {

            if (backend == "System") {
                if (!addedSystem) {
                    SelectableDevice sysDev;
                    sysDev.deviceIndex = 0xFFFFFFFF;
                    sysDev.name = cpuInfo.modelName.empty() ? name : cpuInfo.modelName;
                    sysDev.backend = "System";
                    char sysDesc[128];
                    std::snprintf(sysDesc, sizeof(sysDesc), "%d Threads | %.0f GB System RAM",
                                  cpuInfo.logicalCores, memInfo.totalRamGB);
                    sysDev.architecture = sysDesc;
                    sysDev.isSystem = true;
                    sysDev.selected = true;
                    m_devices.push_back(sysDev);
                    addedSystem = true;
                }
            } else {
                uint32_t idx = static_cast<uint32_t>(std::stoul(idxStr));
                if (std::find(seenGpuIndices.begin(), seenGpuIndices.end(), idx) == seenGpuIndices.end()) {
                    seenGpuIndices.push_back(idx);
                    SelectableDevice gpuDev;
                    gpuDev.deviceIndex = idx;
                    gpuDev.name = name;
                    gpuDev.backend = backend;
                    gpuDev.isSystem = false;
                    gpuDev.selected = (seenGpuIndices.size() == 1);

                    // Hydrate architecture, driver, and VRAM from DeviceProfile
                    for (const auto& prof : m_deviceProfiles) {
                        if (prof.deviceIndex == idx &&
                            (prof.backend == m_selectedBackend || gpuDev.architecture.empty())) {
                            gpuDev.vramTotalMb = prof.vramTotalMb;
                            gpuDev.driver = prof.driverName.empty() ? prof.driverVersion : prof.driverName;
                            std::ostringstream archSs;
                            if (prof.computeUnits > 0) {
                                archSs << prof.computeUnits << " CU/SM";
                            }
                            if (prof.vramTotalMb > 0) {
                                if (archSs.tellp() > 0) archSs << " | ";
                                archSs << (prof.vramTotalMb / 1024) << " GB VRAM";
                            }
                            if (prof.isApu) {
                                archSs << " (Unified)";
                            } else if (prof.rebarSupported) {
                                archSs << " (ReBAR)";
                            }
                            gpuDev.architecture = archSs.str();
                            if (prof.backend == m_selectedBackend) break;
                        }
                    }

                    m_devices.push_back(gpuDev);
                }
            }
        }
    }

    if (m_devices.empty()) {
        SelectableDevice sysDev;
        sysDev.deviceIndex = 0xFFFFFFFF;
        sysDev.name = cpuInfo.modelName.empty() ? "Host CPU / System Memory" : cpuInfo.modelName;
        sysDev.backend = "System";
        sysDev.isSystem = true;
        sysDev.selected = true;
        m_devices.push_back(sysDev);
    }

    m_telemetryWorker.refreshDevices();
    updateTelemetrySelection();
    applyAutoRenderResolution();
    updateBenchmarkSupport();
}

void GuiApp::setSelectedDevice(int deviceIndex) {
    if (deviceIndex < 0) {
        selectAllGpus();
        return;
    }
    for (auto& dev : m_devices) {
        if (!dev.isSystem) {
            dev.selected = (dev.deviceIndex == static_cast<uint32_t>(deviceIndex));
        }
    }
    updateTelemetrySelection();
    applyAutoRenderResolution();
    updateBenchmarkSupport();
}

void GuiApp::setSelectedDevices(const std::vector<int>& deviceIndices) {
    if (deviceIndices.empty()) return;
    for (int idx : deviceIndices) {
        if (idx < 0) {
            selectAllGpus();
            return;
        }
    }
    for (auto& dev : m_devices) {
        if (!dev.isSystem) {
            bool match = false;
            for (int idx : deviceIndices) {
                if (dev.deviceIndex == static_cast<uint32_t>(idx)) {
                    match = true;
                    break;
                }
            }
            dev.selected = match;
        }
    }
    updateTelemetrySelection();
    applyAutoRenderResolution();
    updateBenchmarkSupport();
}

void GuiApp::selectAllGpus() {
    for (auto& dev : m_devices) {
        if (!dev.isSystem) {
            dev.selected = true;
        }
    }
    updateTelemetrySelection();
    applyAutoRenderResolution();
    updateBenchmarkSupport();
}

void GuiApp::updateTelemetrySelection() {
    std::vector<uint32_t> selectedGpus;
    for (const auto& dev : m_devices) {
        if (!dev.isSystem && dev.selected) {
            selectedGpus.push_back(dev.deviceIndex);
        }
    }
    if (selectedGpus.size() >= 2) {
        m_telemetryDualGpuMode = true;
        m_telemetryGpuIndex = selectedGpus[0];
    } else if (selectedGpus.size() == 1) {
        m_telemetryDualGpuMode = false;
        m_telemetryGpuIndex = selectedGpus[0];
    } else {
        m_telemetryDualGpuMode = false;
        m_telemetryGpuIndex = 0;
    }
}

void GuiApp::updateBenchmarkSupport() {
    uint32_t probeDev = 0;
    bool foundSelectedGpu = false;
    for (const auto& dev : m_devices) {
        if (!dev.isSystem && dev.selected) {
            probeDev = dev.deviceIndex;
            foundSelectedGpu = true;
            break;
        }
    }
    if (!foundSelectedGpu) {
        for (const auto& dev : m_devices) {
            if (!dev.isSystem) {
                probeDev = dev.deviceIndex;
                break;
            }
        }
    }

    if (m_lastProbedBackend == m_selectedBackend && m_lastProbedDeviceIndex == probeDev) {
        return;
    }
    m_lastProbedBackend = m_selectedBackend;
    m_lastProbedDeviceIndex = probeDev;

    auto supportList = ProbeBenchmarkSupportAPI(m_selectedBackend, probeDev);
    std::unordered_map<std::string, BenchmarkSupportInfo> supportMap;
    for (const auto& sInfo : supportList) {
        supportMap[sInfo.id] = sInfo;
    }

    for (auto& cat : m_categories) {
        for (auto& sg : cat.subgroups) {
            for (auto& item : sg.items) {
                if (item.category == "System") {
                    item.isSupported = true;
                    item.supportReason.clear();
                    item.limitationCategory.clear();
                    continue;
                }

                bool found = false;
                if (item.configIndex >= 0) {
                    std::string cfgKey = item.id + "#" + std::to_string(item.configIndex);
                    auto itCfg = supportMap.find(cfgKey);
                    if (itCfg != supportMap.end()) {
                        item.isSupported = itCfg->second.isSupported;
                        item.supportReason = itCfg->second.reason;
                        item.limitationCategory = itCfg->second.limitationCategory;
                        found = true;
                    }
                }
                if (!found) {
                    auto it = supportMap.find(item.id);
                    if (it != supportMap.end()) {
                        item.isSupported = it->second.isSupported;
                        item.supportReason = it->second.reason;
                        item.limitationCategory = it->second.limitationCategory;
                    } else {
                        item.isSupported = true;
                        item.supportReason.clear();
                        item.limitationCategory.clear();
                    }
                }

                if (!item.isSupported) {
                    item.selected = false;
                }
            }
            bool anySel = false;
            for (const auto& item : sg.items) {
                if (item.selected) { anySel = true; break; }
            }
            sg.allSelected = anySel;
        }
        bool catAnySel = false;
        for (const auto& sg : cat.subgroups) {
            if (sg.allSelected) { catAnySel = true; break; }
        }
        cat.allSelected = catAnySel;
    }
}

void GuiApp::setSelectedBackend(const std::string& backend) {
    if (backend == "vulkan" || backend == "rocm" || backend == "opencl" || backend == "auto") {
        for (const auto& api : m_apiSupportList) {
            if (api.name == backend && !api.isSupported) {
                return;
            }
        }
        m_selectedBackend = backend;
        m_lastProbedBackend.clear();
        updateBenchmarkSupport();
    }
}

void GuiApp::setScene(const std::string& scene) {
    if (scene == "indoor" || scene == "sponza") {
        m_scene = "indoor";
        m_suiteActiveScene = "indoor";
    } else if (scene == "forest" || scene == "foliage") {
        m_scene = "forest";
        m_suiteActiveScene = "forest";
    } else if (scene == "landscape" || scene == "outdoor") {
        m_scene = "landscape";
        m_suiteActiveScene = "landscape";
    } else if (scene == "showroom" || scene == "studio" || scene == "car") {
        m_scene = "showroom";
        m_suiteActiveScene = "showroom";
    } else if (scene == "all") {
        m_scene = "all";
    }
}

void GuiApp::applyAutoRenderResolution() {
    bool isIntegratedGpu = false;
    for (const auto& dev : m_devices) {
        if (dev.isSystem || !dev.selected) continue;
        for (const auto& prof : m_deviceProfiles) {
            if (prof.deviceIndex == dev.deviceIndex && prof.isApu) {
                isIntegratedGpu = true;
                break;
            }
        }
        std::string lower = dev.name;
        for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (lower.find("8060s") != std::string::npos ||
            lower.find("8050s") != std::string::npos ||
            lower.find("radeon 890m") != std::string::npos ||
            lower.find("radeon 780m") != std::string::npos ||
            lower.find("integrated") != std::string::npos ||
            lower.find("apu") != std::string::npos) {
            isIntegratedGpu = true;
        }
    }

    if (isIntegratedGpu) {
        m_renderWidth = 2560;
        m_renderHeight = 1440;
    } else {
        m_renderWidth = 3840;
        m_renderHeight = 2160;
    }
}

void GuiApp::selectOnlyBenchmark(const std::string& benchmarkId) {
    if (benchmarkId.empty()) return;

    std::vector<std::string> rawTokens;
    std::stringstream ss(benchmarkId);
    std::string itemStr;
    while (std::getline(ss, itemStr, ',')) {
        size_t s0 = itemStr.find_first_not_of(" \t");
        size_t s1 = itemStr.find_last_not_of(" \t");
        if (s0 != std::string::npos) {
            rawTokens.push_back(itemStr.substr(s0, s1 - s0 + 1));
        }
    }
    if (rawTokens.empty()) return;

    std::unordered_map<std::string, std::string> aliasMap = {
        {"raster", "Pixel Fill Rate"}, {"graphics", "Pixel Fill Rate"},
        {"indirect", "In-Shader Indirect Synthesis"},
        {"lds", "LDS Bank Conflicts"},
        {"rayscheduling", "RayScheduling"}, {"rt", "RayScheduling"}
    };
    for (const auto& meta : GetBenchmarkMetadataCatalogAPI()) {
        std::string targetId = meta.benchmarkName;
        if (targetId.rfind("RayScheduling", 0) == 0) {
            targetId = "RayScheduling";
        }
        std::string lowName = targetId;
        for (char& ch : lowName) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        aliasMap[lowName] = targetId;
        for (const auto& al : meta.aliases) {
            std::string lowAl = al;
            for (char& ch : lowAl) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            aliasMap[lowAl] = targetId;
        }
    }

    for (auto& cat : m_categories) {
        cat.allSelected = false;
        for (auto& sg : cat.subgroups) {
            sg.allSelected = false;
            for (auto& item : sg.items) {
                item.selected = false;
            }
        }
    }

    bool anyGpuSelected = false;
    bool anySysSelected = false;

    for (const auto& tok : rawTokens) {
        std::string lowerTok = tok;
        for (char& ch : lowerTok) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

        if (lowerTok == "all") {
            for (auto& cat : m_categories) {
                for (auto& sg : cat.subgroups) {
                    for (auto& item : sg.items) {
                        if (item.isSupported) {
                            item.selected = true;
                            if (item.category == "System") anySysSelected = true;
                            else anyGpuSelected = true;
                        }
                    }
                }
            }
            continue;
        }

        // Category match (compute, memory, raytracing, scheduling, graphics, system)
        bool matchedCategory = false;
        for (auto& cat : m_categories) {
            std::string lowerCat = cat.name;
            lowerCat.erase(std::remove(lowerCat.begin(), lowerCat.end(), ' '), lowerCat.end());
            for (char& ch : lowerCat) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (lowerTok == lowerCat || (lowerTok == "system" && lowerCat == "hostsystem")) {
                for (auto& sg : cat.subgroups) {
                    for (auto& item : sg.items) {
                        if (item.isSupported) {
                            item.selected = true;
                            if (item.category == "System") anySysSelected = true;
                            else anyGpuSelected = true;
                        }
                    }
                }
                matchedCategory = true;
            }
        }
        if (matchedCategory) continue;

        std::string resolvedId = tok;
        auto aliasIt = aliasMap.find(lowerTok);
        if (aliasIt != aliasMap.end()) {
            resolvedId = aliasIt->second;
        }

        for (auto& cat : m_categories) {
            for (auto& sg : cat.subgroups) {
                for (auto& item : sg.items) {
                    if (equalsCaseInsensitive(item.id, resolvedId) ||
                        equalsCaseInsensitive(item.id, tok) ||
                        equalsCaseInsensitive(item.subcategory, tok) ||
                        containsCaseInsensitive(item.name, tok)) {
                        if (item.isSupported) {
                            item.selected = true;
                            if (item.category == "System") anySysSelected = true;
                            else anyGpuSelected = true;
                        }
                    }
                }
            }
        }
    }

    for (auto& cat : m_categories) {
        bool catAny = false;
        for (auto& sg : cat.subgroups) {
            bool sgAny = false;
            for (const auto& item : sg.items) {
                if (item.selected) { sgAny = true; break; }
            }
            sg.allSelected = sgAny;
            if (sgAny) catAny = true;
        }
        cat.allSelected = catAny;
    }

    for (auto& dev : m_devices) {
        if (dev.isSystem) {
            dev.selected = anySysSelected;
        } else if (!anyGpuSelected && anySysSelected) {
            dev.selected = false;
        }
    }
}

void GuiApp::startBenchmarks() {
    if (m_execState == ExecutionState::Running) {
        return;
    }

    if (m_execThread.joinable()) {
        m_execThread.join();
    }

    // Release any borrowed VkImageView descriptors on the main thread BEFORE
    // the background worker recreates SharedPreviewImages.
    clearTextureCache();

    std::vector<std::string> benchmarksToRun;
    std::vector<std::string> selectedWorkloads;
    size_t systemSelectedItemCount = 0;

    for (const auto& cat : m_categories) {
        for (const auto& sg : cat.subgroups) {
            for (const auto& item : sg.items) {
                if (item.selected && item.isSupported) {
                    if (std::find(benchmarksToRun.begin(), benchmarksToRun.end(), item.id) == benchmarksToRun.end()) {
                        benchmarksToRun.push_back(item.id);
                    }
                    if (item.configIndex >= 0) {
                        selectedWorkloads.push_back(item.id + "#" + std::to_string(item.configIndex));
                    } else {
                        selectedWorkloads.push_back(item.id + ":" + item.name);
                    }
                    if (item.category == "System") {
                        systemSelectedItemCount++;
                    }
                }
            }
        }
    }

    if (benchmarksToRun.empty()) {
        m_statusMessage = "No supported benchmarks selected!";
        return;
    }

    std::vector<uint32_t> deviceIndices;
    bool runSystemBenchmarks = false;
    for (const auto& dev : m_devices) {
        if (dev.selected) {
            if (dev.isSystem) {
                runSystemBenchmarks = true;
            } else {
                deviceIndices.push_back(dev.deviceIndex);
            }
        }
    }

    if (!runSystemBenchmarks) {
        benchmarksToRun.erase(
            std::remove_if(benchmarksToRun.begin(), benchmarksToRun.end(),
                [](const std::string& b) {
                    return b.find("System Memory") != std::string::npos;
                }),
            benchmarksToRun.end());
        systemSelectedItemCount = 0;
    }

    if (deviceIndices.empty()) {
        benchmarksToRun.erase(
            std::remove_if(benchmarksToRun.begin(), benchmarksToRun.end(),
                [](const std::string& b) {
                    return b.find("System Memory") == std::string::npos;
                }),
            benchmarksToRun.end());
    }

    if (benchmarksToRun.empty()) {
        m_statusMessage = "Please select at least one target device matching your selected benchmarks.";
        return;
    }

    m_cancelToken.store(false);
    m_execState = ExecutionState::Running;
    m_completedTasks = 0;

    size_t sceneMult = (m_scene == "all") ? 4 : 1;
    size_t perGpuTasks = 0;
    for (const auto& cat : m_categories) {
        for (const auto& sg : cat.subgroups) {
            for (const auto& item : sg.items) {
                if (item.selected && item.isSupported && item.category != "System") {
                    if (item.id == "RayScheduling") {
                        perGpuTasks += sceneMult;
                    } else if (item.id == "Cache Latency Curve") {
                        perGpuTasks += 15;
                    } else {
                        perGpuTasks += 1;
                    }
                }
            }
        }
    }

    m_totalTasks = (perGpuTasks * std::max<size_t>(1, deviceIndices.size())) + systemSelectedItemCount;
    if (m_totalTasks == 0) m_totalTasks = 1;

    m_benchmarkStartTime = std::chrono::steady_clock::now();
    m_elapsedSeconds = 0.0;
    m_statusMessage = "Initializing compute contexts...";
    m_currentBenchmarkName = "Starting...";
    m_currentlyRunningTestId.clear();
    m_hasCurrentlyRunningResult = false;
    m_telemetryNeedsFit = true;
    m_telemetryWorker.startRecording();

    {
        std::lock_guard<std::mutex> lock(m_resultsMutex);
        m_incomingResults.clear();
    }

    std::vector<std::string> backends = {m_selectedBackend};
    uint32_t rw = m_renderWidth;
    uint32_t rh = m_renderHeight;
    std::string scene = m_scene;
    uint32_t spp = m_samplesPerPixel;
    bool dumpRenders = m_dumpRenders;
    bool verbose = m_verbose;
    bool debug = m_debug;

    m_execThread = std::thread([this, benchmarksToRun, deviceIndices, backends,
                                rw, rh, scene, spp, dumpRenders, verbose, debug,
                                selectedWorkloads]() {
        try {
            auto callback = [this](const ResultData& res) {
                std::lock_guard<std::mutex> lock(m_resultsMutex);
                m_incomingResults.push_back(res);
            };

            RunBenchmarksAPI(
                benchmarksToRun,
                deviceIndices,
                backends,
                verbose,
                debug,
                false,
                dumpRenders,
                rw,
                rh,
                callback,
                scene,
                spp,
                &m_cancelToken,
                true,
                selectedWorkloads
            );

            m_telemetryWorker.stopRecording();
            m_textureCacheNeedsClear.store(true);

            std::lock_guard<std::mutex> lock(m_resultsMutex);
            if (m_cancelToken.load()) {
                m_execState = ExecutionState::Cancelled;
                m_statusMessage = "Benchmark execution aborted by user.";
            } else {
                m_execState = ExecutionState::Completed;
                m_statusMessage = "Suite execution complete.";
                m_switchToScorecard = true;
            }
        } catch (const std::exception& e) {
            m_telemetryWorker.stopRecording();
            std::lock_guard<std::mutex> lock(m_resultsMutex);
            m_execState = ExecutionState::Error;
            m_statusMessage = std::string("Error: ") + e.what();
        }
    });
}

void GuiApp::abortBenchmarks() {
    if (m_execState == ExecutionState::Running) {
        m_cancelToken.store(true);
        m_statusMessage = "Stopping after current dispatch...";
    }
}

void GuiApp::processIncomingResults() {
    if (m_textureCacheNeedsClear.exchange(false)) {
        clearTextureCache();
    }

    std::vector<ResultData> localBatch;
    {
        std::lock_guard<std::mutex> lock(m_resultsMutex);
        if (!m_incomingResults.empty()) {
            localBatch.swap(m_incomingResults);
        }
    }

    for (const auto& res : localBatch) {
        if (res.time_ms == -2.0) {
            m_currentBenchmarkName = res.benchmarkName;
            m_currentlyRunningTestId = res.benchmarkName;
            m_currentlyRunningResult = res;
            m_hasCurrentlyRunningResult = true;
            m_statusMessage = "Running: " + res.benchmarkName + " (" + res.deviceName + ")";
            continue;
        }

        auto existingIt = std::find_if(m_allResults.begin(), m_allResults.end(),
            [&res](const ResultData& existing) {
                return existing.deviceIndex == res.deviceIndex &&
                       existing.backendName == res.backendName &&
                       existing.benchmarkName == res.benchmarkName &&
                       existing.configIndex == res.configIndex;
            });
        if (existingIt != m_allResults.end()) {
            *existingIt = res;
        } else {
            m_allResults.push_back(res);
        }

        std::string key = std::to_string(res.deviceIndex) + "|" + res.benchmarkName +
                          "|" + std::to_string(res.configIndex);
        m_latestResults[key] = res;
        m_currentBenchmarkName = res.benchmarkName;
        if (m_currentlyRunningTestId == res.benchmarkName) {
            m_currentlyRunningTestId.clear();
            m_hasCurrentlyRunningResult = false;
        }
        m_completedTasks++;
        if (m_completedTasks > m_totalTasks) {
            m_totalTasks = m_completedTasks;
        }
        m_statusMessage = "Completed: " + res.benchmarkName;
    }

    if (m_execState == ExecutionState::Running) {
        auto now = std::chrono::steady_clock::now();
        m_elapsedSeconds = std::chrono::duration<double>(now - m_benchmarkStartTime).count();
    }
}

void GuiApp::processEvents() {
    processIncomingResults();
}

void GuiApp::exportResultsToJson(const std::string& filepath) {
    UnifiedSessionMetadata sessionMeta;
    sessionMeta.scene = m_scene;
    sessionMeta.samplesPerPixel = m_samplesPerPixel;
    sessionMeta.dumpRenders = m_dumpRenders;
    sessionMeta.uiScale = m_uiScale;

    if (m_deviceProfiles.empty()) {
        m_deviceProfiles = GetDeviceProfilesAPI();
    }

    std::string jsonStr = resultsToJson(m_allResults, m_deviceProfiles, &sessionMeta);
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        m_exportNotificationText = "Failed to open for writing: " + filepath;
        m_exportNotificationTimer = 5.0f;
        return;
    }
    out.write(jsonStr.data(), static_cast<std::streamsize>(jsonStr.size()));
    out.close();

    m_exportNotificationText = "Saved unified v2.0 JSON to " + filepath;
    m_exportNotificationTimer = 4.5f;
}

bool GuiApp::importResultsFromJson(const std::string& filepath) {
    ImportedRun run{};
    std::string err;
    if (!ResultImporter::loadFromFile(filepath, run, err)) {
        m_exportNotificationText = "JSON Import Error: " + err;
        m_exportNotificationTimer = 5.5f;
        return false;
    }

    m_allResults = run.results;
    m_latestResults.clear();
    for (const auto& r : m_allResults) {
        std::string key = std::to_string(r.deviceIndex) + "|" + r.benchmarkName +
                          "|" + std::to_string(r.configIndex);
        m_latestResults[key] = r;
    }

    if (run.hasGuiSession) {
        setScene(run.sessionScene);
        m_samplesPerPixel = std::clamp(run.sessionSamplesPerPixel, 1u, 256u);
        m_dumpRenders = run.sessionDumpRenders;
    }
    if (!run.resolution.empty()) {
        unsigned int w = 0, h = 0;
        if (std::sscanf(run.resolution.c_str(), "%ux%u", &w, &h) == 2 && w >= 64 && h >= 64) {
            m_renderWidth = w;
            m_renderHeight = h;
        }
    }

    m_execState = ExecutionState::Completed;
    m_completedTasks = m_allResults.size();
    m_totalTasks = std::max<size_t>(1, m_allResults.size());
    m_statusMessage = "Loaded session (" + std::to_string(m_allResults.size()) +
                      " results) from " + std::filesystem::path(filepath).filename().string();
    m_exportNotificationText = "Imported " + std::to_string(m_allResults.size()) +
                               " results from " + filepath;
    m_exportNotificationTimer = 4.5f;
    m_switchToScorecard = true;
    return true;
}

bool GuiApp::loadComparisonFromJson(const std::string& filepath) {
    ImportedRun run{};
    std::string err;
    if (!ResultImporter::loadFromFile(filepath, run, err)) {
        m_exportNotificationText = "Comparison Load Error: " + err;
        m_exportNotificationTimer = 5.5f;
        return false;
    }

    m_comparisonResults = run.results;
    std::string devTag = run.deviceProfile.deviceName;
    std::string fn = std::filesystem::path(filepath).filename().string();
    m_comparisonSourceLabel = devTag.empty() ? fn : (fn + " (" + devTag + ")");
    m_exportNotificationText = "Loaded comparison baseline: " + m_comparisonSourceLabel;
    m_exportNotificationTimer = 4.5f;
    m_switchToScorecard = true;
    return true;
}

void GuiApp::clearComparisonResults() {
    m_comparisonResults.clear();
    m_comparisonSourceLabel.clear();
}

const ResultData* GuiApp::findComparisonMatch(const ResultData& curRes) const {
    if (m_comparisonResults.empty()) return nullptr;

    // 1. Exact benchmarkName + configIndex match
    for (const auto& cmp : m_comparisonResults) {
        if (!cmp.isUnsupported && cmp.time_ms > 0.0 &&
            cmp.benchmarkName == curRes.benchmarkName &&
            cmp.configIndex == curRes.configIndex) {
            return &cmp;
        }
    }
    // 2. Exact benchmarkName match (ignoring configIndex for legacy v1 files)
    for (const auto& cmp : m_comparisonResults) {
        if (!cmp.isUnsupported && cmp.time_ms > 0.0 &&
            cmp.benchmarkName == curRes.benchmarkName) {
            return &cmp;
        }
    }
    return nullptr;
}

void GuiApp::updateAndRender() {
    processIncomingResults();
    updateZoomShortcuts();

    if (m_exportNotificationTimer > 0.0f) {
        m_exportNotificationTimer -= ImGui::GetIO().DeltaTime;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);

    ImGuiWindowFlags rootFlags = ImGuiWindowFlags_NoTitleBar |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoBringToFrontOnFocus |
                                 ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(s(10.0f), s(10.0f)));
    ImGui::Begin("GPUBenchRootWindow", nullptr, rootFlags);
    ImGui::PopStyleVar();

    float availWidth = ImGui::GetContentRegionAvail().x;
    float availHeight = ImGui::GetContentRegionAvail().y;

    float minSidebar = std::min(s(300.0f), availWidth * 0.34f);
    float maxSidebar = std::min(s(410.0f), availWidth * 0.38f);
    float leftSidebarWidth = std::clamp(availWidth * 0.25f, minSidebar, std::max(minSidebar, maxSidebar));
    float rightWorkspaceWidth = availWidth - leftSidebarWidth - s(10.0f);

    renderLeftSidebar(leftSidebarWidth, availHeight);
    ImGui::SameLine(0.0f, s(10.0f));
    renderRightWorkspace(rightWorkspaceWidth, availHeight);

    if (m_showSettingsModal) {
        renderSettingsModal();
    }
    if (m_showJsonSessionModal) {
        renderJsonSessionModal();
    }

    ImGui::End();

    renderZoomToast();
}

void GuiApp::renderLeftSidebar(float width, float height) {
    ImGui::BeginChild("LeftSidebar", ImVec2(width, height), ImGuiChildFlags_Borders);

    // Brand Header
    pushBoldFont();
    ImGui::TextColored(ImVec4(0.06f, 0.80f, 0.98f, 1.0f), "GPUBench");
    popBoldFont();
    ImGui::SameLine();
    ImGui::TextDisabled("v" GPUBENCH_VERSION " | Vulkan 1.4");
    ImGui::TextColored(ImVec4(0.55f, 0.62f, 0.72f, 1.0f), "Hardware Microarchitecture Suite");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Compute Backend Selector
    pushBoldFont();
    ImGui::TextColored(ImVec4(0.75f, 0.82f, 0.92f, 1.0f), "COMPUTE BACKEND");
    popBoldFont();
    const char* backends[] = {"vulkan", "rocm", "opencl", "auto"};
    const char* backendLabels[] = {"Vulkan", "ROCm", "OpenCL", "Auto"};
    float btnW = (ImGui::GetContentRegionAvail().x - s(18.0f)) / 4.0f;

    bool isRunning = (m_execState == ExecutionState::Running);
    if (isRunning) ImGui::BeginDisabled();

    for (int i = 0; i < 4; ++i) {
        if (i > 0) ImGui::SameLine(0.0f, s(6.0f));

        bool apiAvailable = true;
        const ComputeApiSupportInfo* apiInfo = nullptr;
        for (const auto& info : m_apiSupportList) {
            if (info.name == backends[i]) {
                apiInfo = &info;
                apiAvailable = info.isSupported;
                break;
            }
        }

        bool active = (m_selectedBackend == backends[i]);
        if (!apiAvailable) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.08f, 0.09f, 0.11f, 0.60f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.12f, 0.13f, 0.16f, 0.75f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.08f, 0.09f, 0.11f, 0.60f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.42f, 0.45f, 0.52f, 0.80f));
        } else if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.08f, 0.65f, 0.90f, 1.0f));
        }

        if (ImGui::Button(backendLabels[i], ImVec2(btnW, s(28.0f)))) {
            if (apiAvailable) {
                setSelectedBackend(backends[i]);
            }
        }

        if (!apiAvailable) {
            ImGui::PopStyleColor(4);
        } else if (active) {
            ImGui::PopStyleColor(2);
        }

        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && apiInfo) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 26.0f);
            if (apiInfo->isSupported) {
                ImGui::TextColored(ImVec4(0.25f, 0.92f, 0.55f, 1.0f), "[AVAILABLE] %s Backend", apiInfo->label.c_str());
                ImGui::Separator();
                ImGui::TextUnformatted(apiInfo->reason.c_str());
            } else {
                ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.35f, 1.0f), "[UNAVAILABLE] %s Backend", apiInfo->label.c_str());
                ImGui::Separator();
                ImGui::TextWrapped("Reason: %s", apiInfo->reason.c_str());
                if (!apiInfo->missingRequirement.empty()) {
                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(0.95f, 0.78f, 0.30f, 1.0f), "Requirement: %s", apiInfo->missingRequirement.c_str());
                }
            }
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    ImGui::Spacing();
    pushBoldFont();
    ImGui::TextColored(ImVec4(0.75f, 0.82f, 0.92f, 1.0f), "TARGET DEVICES");
    popBoldFont();

    size_t gpuCount = 0;
    for (const auto& d : m_devices) {
        if (!d.isSystem) gpuCount++;
    }
    if (gpuCount >= 2) {
        float quickW = (ImGui::GetContentRegionAvail().x - s(12.0f)) / 3.0f;
        if (ImGui::Button("GPU 0 Only", ImVec2(quickW, s(23.0f)))) setSelectedDevice(0);
        ImGui::SameLine(0.0f, s(6.0f));
        if (ImGui::Button("GPU 1 Only", ImVec2(quickW, s(23.0f)))) setSelectedDevice(1);
        ImGui::SameLine(0.0f, s(6.0f));
        if (ImGui::Button("All GPUs", ImVec2(quickW, s(23.0f)))) selectAllGpus();
    }

    ImGui::BeginChild("DeviceListRegion", ImVec2(0, s(128.0f)), ImGuiChildFlags_Borders);
    for (size_t i = 0; i < m_devices.size(); ++i) {
        auto& dev = m_devices[i];
        ImGui::PushID(static_cast<int>(i));

        std::string badge = dev.isSystem ? "[CPU] " : ("[GPU " + std::to_string(dev.deviceIndex) + "] ");
        std::string fullLabel = badge + dev.name;
        float maxLabelW = ImGui::GetContentRegionAvail().x - s(30.0f);
        std::string fittedLabel = fitTextToPixelWidth(fullLabel, maxLabelW);

        if (ImGui::Checkbox(fittedLabel.c_str(), &dev.selected)) {
            updateTelemetrySelection();
            applyAutoRenderResolution();
            updateBenchmarkSupport();
        }
        if (fittedLabel != fullLabel && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("%s", fullLabel.c_str());
        }
        if (!dev.architecture.empty()) {
            ImGui::Indent(s(26.0f));
            std::string fitArch = fitTextToPixelWidth(dev.architecture, ImGui::GetContentRegionAvail().x - s(4.0f));
            ImGui::TextDisabled("%s", fitArch.c_str());
            ImGui::Unindent(s(26.0f));
        }
        ImGui::PopID();
    }
    ImGui::EndChild();

    if (isRunning) ImGui::EndDisabled();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Live Hardware Telemetry Summary
    renderSidebarTelemetry();

    // Bottom Action Controls
    float currentY = ImGui::GetCursorPosY();
    float footerReserve = s(172.0f);
    if (height - currentY > footerReserve) {
        ImGui::SetCursorPosY(height - footerReserve);
    }

    ImGui::Separator();
    ImGui::Spacing();

    if (m_execState == ExecutionState::Running) {
        float progress = (m_totalTasks > 0)
            ? static_cast<float>(m_completedTasks) / static_cast<float>(m_totalTasks)
            : 0.0f;
        progress = std::clamp(progress, 0.0f, 1.0f);

        char overlay[64];
        std::snprintf(overlay, sizeof(overlay), "%zu / %zu  (%.1fs)",
                      m_completedTasks, m_totalTasks, m_elapsedSeconds);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.05f, 0.68f, 0.90f, 1.0f));
        ImGui::ProgressBar(progress, ImVec2(-1.0f, s(20.0f)), overlay);
        ImGui::PopStyleColor();

        std::string fittedStatus = fitTextToPixelWidth(m_statusMessage, ImGui::GetContentRegionAvail().x);
        ImGui::TextColored(ImVec4(0.65f, 0.80f, 0.95f, 1.0f), "%s", fittedStatus.c_str());

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.72f, 0.16f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.86f, 0.22f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.60f, 0.12f, 0.14f, 1.0f));
        if (ImGui::Button("STOP BENCHMARK", ImVec2(-1.0f, s(38.0f)))) {
            abortBenchmarks();
        }
        ImGui::PopStyleColor(3);
    } else {
        size_t selCount = 0;
        for (const auto& cat : m_categories) {
            for (const auto& sg : cat.subgroups) {
                for (const auto& itm : sg.items) {
                    if (itm.selected && itm.isSupported) selCount++;
                }
            }
        }

        DeviceTelemetrySnapshot preSnap{};
        if (m_telemetryWorker.getSnapshot(m_telemetryGpuIndex, preSnap) &&
            preSnap.hasGpuBusy && preSnap.gpuBusyPct >= 15.0f) {
            ImGui::TextColored(ImVec4(0.96f, 0.72f, 0.20f, 1.0f),
                               "Warning: GPU %u Busy %.0f%% (Contention Risk)",
                               m_telemetryGpuIndex, preSnap.gpuBusyPct);
        } else if (!m_statusMessage.empty()) {
            std::string fittedStatus = fitTextToPixelWidth(m_statusMessage, ImGui::GetContentRegionAvail().x);
            ImGui::TextDisabled("%s", fittedStatus.c_str());
        } else {
            ImGui::TextDisabled("Ready (%zu workloads selected)", selCount);
        }

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.56f, 0.34f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.08f, 0.68f, 0.42f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.04f, 0.46f, 0.28f, 1.0f));
        pushBoldFont();
        char runLabel[64];
        std::snprintf(runLabel, sizeof(runLabel), "RUN SELECTED SUITE (%zu)", selCount);
        if (ImGui::Button(runLabel, ImVec2(-1.0f, s(38.0f)))) {
            startBenchmarks();
        }
        popBoldFont();
        ImGui::PopStyleColor(3);
    }

    ImGui::Spacing();
    float thirdW = (ImGui::GetContentRegionAvail().x - s(12.0f)) / 3.0f;
    if (ImGui::Button("Settings", ImVec2(thirdW, s(28.0f)))) {
        m_showSettingsModal = true;
    }
    ImGui::SameLine(0.0f, s(6.0f));
    if (ImGui::Button("Import/Diff", ImVec2(thirdW, s(28.0f)))) {
        if (m_jsonPathBuffer[0] == '\0') {
            std::string defName = getDefaultJsonFilename();
            std::snprintf(m_jsonPathBuffer, sizeof(m_jsonPathBuffer), "%s", defName.c_str());
        }
        m_showJsonSessionModal = true;
    }
    ImGui::SameLine(0.0f, s(6.0f));
    bool canExport = !m_allResults.empty();
    if (!canExport) ImGui::BeginDisabled();
    if (ImGui::Button("Export JSON", ImVec2(thirdW, s(28.0f)))) {
        exportResultsToJson(getDefaultJsonFilename());
    }
    if (!canExport) ImGui::EndDisabled();

    if (m_exportNotificationTimer > 0.0f && !m_exportNotificationText.empty()) {
        ImGui::TextColored(ImVec4(0.20f, 0.90f, 0.55f, 1.0f), "%s",
                           fitTextToPixelWidth(m_exportNotificationText, ImGui::GetContentRegionAvail().x).c_str());
    }

    ImGui::EndChild();
}

void GuiApp::renderRightWorkspace(float width, float height) {
    ImGui::BeginChild("RightWorkspace", ImVec2(width, height), ImGuiChildFlags_None);

    if (ImGui::BeginTabBar("MainWorkspaceTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Benchmark Suite")) {
            renderBenchmarkSuitePanel();
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags scorecardFlags = ImGuiTabItemFlags_None;
        if (m_switchToScorecard) {
            scorecardFlags |= ImGuiTabItemFlags_SetSelected;
            m_switchToScorecard = false;
        }
        std::string scorecardLabel = "Results Scorecard";
        if (!m_allResults.empty()) {
            scorecardLabel += " (" + std::to_string(m_allResults.size()) + ")";
        }
        scorecardLabel += "###ResultsScorecardTab";

        if (ImGui::BeginTabItem(scorecardLabel.c_str(), nullptr, scorecardFlags)) {
            renderResultsScorecard();
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags rooflineFlags = ImGuiTabItemFlags_None;
        if (m_switchToRoofline) {
            rooflineFlags |= ImGuiTabItemFlags_SetSelected;
            m_switchToRoofline = false;
        }
        if (ImGui::BeginTabItem("Empirical Roofline###RooflineTab", nullptr, rooflineFlags)) {
            renderRooflinePanel();
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags rtFlags = ImGuiTabItemFlags_None;
        if (m_switchToRtViewport) {
            rtFlags |= ImGuiTabItemFlags_SetSelected;
            m_switchToRtViewport = false;
        }
        if (ImGui::BeginTabItem("Ray Tracing Viewport###RtViewportTab", nullptr, rtFlags)) {
            renderRayTracingViewport();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Hardware Telemetry")) {
            renderLiveTelemetryDock();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::EndChild();
}

void GuiApp::renderSettingsModal() {
    ImGui::OpenPopup("Advanced Benchmark Settings");
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(s(500.0f), 0.0f), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Advanced Benchmark Settings", &m_showSettingsModal, ImGuiWindowFlags_AlwaysAutoResize)) {
        pushBoldFont();
        ImGui::TextColored(ImVec4(0.06f, 0.80f, 0.98f, 1.0f), "Ray Tracing & Rasterization Settings");
        popBoldFont();
        ImGui::Separator();
        ImGui::Spacing();

        const char* scenes[] = {"all", "showroom", "indoor", "forest", "landscape"};
        const char* sceneLabels[] = {
            "All 4 Scenes (Comprehensive)",
            "Showroom Studio (Automotive Reflection)",
            "Indoor Atrium (Architectural GI)",
            "Open-World Forest (Alpha-Cutout Foliage)",
            "Outdoor Landscape (Heightfield Terrain)"
        };
        int currentSceneIdx = 0;
        for (int i = 0; i < 5; ++i) {
            if (m_scene == scenes[i]) { currentSceneIdx = i; break; }
        }
        if (ImGui::Combo("Target Scene", &currentSceneIdx, sceneLabels, 5)) {
            m_scene = scenes[currentSceneIdx];
            if (m_scene != "all") m_suiteActiveScene = m_scene;
        }

        const char* resLabels[] = {
            "1280 x 720 (HD)",
            "1920 x 1080 (FHD)",
            "2560 x 1440 (QHD)",
            "3840 x 2160 (4K UHD)"
        };
        int resIdx = 3;
        if (m_renderWidth == 1280) resIdx = 0;
        else if (m_renderWidth == 1920) resIdx = 1;
        else if (m_renderWidth == 2560) resIdx = 2;

        if (ImGui::Combo("Render Resolution", &resIdx, resLabels, 4)) {
            const uint32_t widths[]  = {1280, 1920, 2560, 3840};
            const uint32_t heights[] = {720,  1080, 1440, 2160};
            m_renderWidth = widths[resIdx];
            m_renderHeight = heights[resIdx];
        }

        int spp = static_cast<int>(m_samplesPerPixel);
        if (ImGui::SliderInt("Samples Per Pixel (SPP)", &spp, 1, 64)) {
            m_samplesPerPixel = static_cast<uint32_t>(std::max(1, spp));
        }

        ImGui::Checkbox("Dump Scene Renders to Disk (renders/*.png)", &m_dumpRenders);
        ImGui::TextDisabled("When unchecked, Ray Tracing Viewport still previews renders directly in GPU VRAM!");

        ImGui::Spacing();
        pushBoldFont();
        ImGui::TextColored(ImVec4(0.06f, 0.80f, 0.98f, 1.0f), "Diagnostics & Display");
        popBoldFont();
        ImGui::Separator();
        ImGui::Spacing();

        float uiScalePct = (m_uiScale / std::max(0.1f, m_baseScale)) * 100.0f;
        if (ImGui::SliderFloat("Interface Zoom (%)", &uiScalePct, 70.0f, 225.0f, "%.0f%%")) {
            setUiScale((uiScalePct / 100.0f) * m_baseScale);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset")) {
            setUiScale(m_baseScale);
        }
        ImGui::TextDisabled("Dynamic ImGui 1.93 glyph baking active (Ctrl+Scroll / Ctrl+/- / Ctrl+0)");

        ImGui::Spacing();
        ImGui::Checkbox("Verbose Console Logging", &m_verbose);
        ImGui::Checkbox("Enable Vulkan Validation Layers (--debug)", &m_debug);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Close", ImVec2(s(120.0f), s(30.0f)))) {
            m_showSettingsModal = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void GuiApp::renderJsonSessionModal() {
    ImGui::OpenPopup("Unified JSON Session & Comparison");
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(s(560.0f), 0.0f), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("Unified JSON Session & Comparison", &m_showJsonSessionModal, ImGuiWindowFlags_AlwaysAutoResize)) {
        pushBoldFont();
        ImGui::TextColored(ImVec4(0.06f, 0.80f, 0.98f, 1.0f), "Unified CLI / GUI JSON Schema (v2.0)");
        popBoldFont();
        ImGui::TextWrapped("Files exported by `gpubench -o <file.json>` and `gpubench-gui` share the same "
                           "yyjson v2.0 schema. Load a JSON file as the active session or as a baseline for side-by-side delta comparison.");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::InputText("JSON File Path", m_jsonPathBuffer, sizeof(m_jsonPathBuffer));

        // Quick-pick recent .json files in working directory
        namespace fs = std::filesystem;
        std::error_code ec;
        std::vector<std::string> localJsons;
        for (const auto& entry : fs::directory_iterator(".", ec)) {
            if (entry.is_regular_file(ec) && entry.path().extension() == ".json") {
                localJsons.push_back(entry.path().filename().string());
                if (localJsons.size() >= 8) break;
            }
        }
        if (!localJsons.empty()) {
            ImGui::TextDisabled("Workspace JSON files:");
            for (size_t i = 0; i < localJsons.size(); ++i) {
                if (i > 0) sameLineIfFits(s(140.0f), s(6.0f));
                if (ImGui::SmallButton(localJsons[i].c_str())) {
                    std::snprintf(m_jsonPathBuffer, sizeof(m_jsonPathBuffer), "%s", localJsons[i].c_str());
                }
            }
        }

        ImGui::Spacing();
        float btnW = (ImGui::GetContentRegionAvail().x - s(12.0f)) / 3.0f;

        if (ImGui::Button("Import as Session", ImVec2(btnW, s(32.0f)))) {
            if (importResultsFromJson(m_jsonPathBuffer)) {
                m_showJsonSessionModal = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::SameLine(0.0f, s(6.0f));
        if (ImGui::Button("Load as Comparison", ImVec2(btnW, s(32.0f)))) {
            if (loadComparisonFromJson(m_jsonPathBuffer)) {
                m_showJsonSessionModal = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::SameLine(0.0f, s(6.0f));
        bool canExport = !m_allResults.empty();
        if (!canExport) ImGui::BeginDisabled();
        if (ImGui::Button("Save Active Run", ImVec2(btnW, s(32.0f)))) {
            exportResultsToJson(m_jsonPathBuffer);
            m_showJsonSessionModal = false;
            ImGui::CloseCurrentPopup();
        }
        if (!canExport) ImGui::EndDisabled();

        if (!m_comparisonResults.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.35f, 0.88f, 0.98f, 1.0f),
                               "Active Comparison Baseline: %s (%zu items)",
                               m_comparisonSourceLabel.c_str(), m_comparisonResults.size());
            ImGui::SameLine();
            if (ImGui::SmallButton("Clear Comparison")) {
                clearComparisonResults();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        if (ImGui::Button("Close", ImVec2(s(110.0f), s(28.0f)))) {
            m_showJsonSessionModal = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

} // namespace gpubench::gui
