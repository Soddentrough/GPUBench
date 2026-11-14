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
#include <cstdlib>
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
    joinIsaThread();
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
#ifdef _WIN32
    if (const char* localApp = std::getenv("LOCALAPPDATA")) {
        if (*localApp) candidateRoots.push_back(std::string(localApp) + "/gpubench/");
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg) candidateRoots.push_back(std::string(xdg) + "/gpubench/");
    }
    if (const char* home = std::getenv("HOME")) {
        if (*home) candidateRoots.push_back(std::string(home) + "/.cache/gpubench/");
    }
#endif
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
    ensureDefaultInspectedBenchmark();
    if (const BenchmarkItem* defItem = getInspectedItem()) {
        requestIsaInspection(*defItem, false);
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

void GuiApp::syncBringUpTheme() {
    applyBringUpStyle(ImGui::GetStyle(), ImPlot::GetStyle(), m_palette, m_uiScale);
    if (m_vulkanContext && m_vulkanContext->isInitialized()) {
        const BringUpColors c = getBringUpColors(m_palette);
        m_vulkanContext->setClearColor(c.bgVoid.x, c.bgVoid.y, c.bgVoid.z, 1.0f);
        m_vulkanContext->setCrtPostFx(false, static_cast<int>(m_palette), 1.0f);
    }
}

void GuiApp::setupDarkTheme(float scale) {
    m_uiScale = std::clamp(scale, 0.5f, 3.0f);
    syncBringUpTheme();
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
                            } else if (prof.backend == "Vulkan" || prof.rebarSupported || prof.rebarSizeBytes > 0) {
                                archSs << " | ReBAR: "
                                       << gpubench::formatRebarStatus(prof.rebarSupported, prof.rebarSizeBytes, prof.isApu, true);
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
            bool anySupported = false;
            for (const auto& item : sg.items) {
                if (item.selected) anySel = true;
                if (item.isSupported) anySupported = true;
            }
            sg.allSelected = anySel;
            sg.collapsed = (!sg.items.empty() && !anySupported);
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
        m_rtSceneIndex = 1;
    } else if (scene == "forest" || scene == "foliage") {
        m_scene = "forest";
        m_suiteActiveScene = "forest";
        m_rtSceneIndex = 2;
    } else if (scene == "landscape" || scene == "outdoor") {
        m_scene = "landscape";
        m_suiteActiveScene = "landscape";
        m_rtSceneIndex = 3;
    } else if (scene == "showroom" || scene == "studio" || scene == "car") {
        m_scene = "showroom";
        m_suiteActiveScene = "showroom";
        m_rtSceneIndex = 0;
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
    int firstCi = -1;
    int firstSgi = -1;
    int firstIi = -1;

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

        for (size_t ci = 0; ci < m_categories.size(); ++ci) {
            auto& cat = m_categories[ci];
            for (size_t sgi = 0; sgi < cat.subgroups.size(); ++sgi) {
                auto& sg = cat.subgroups[sgi];
                for (size_t ii = 0; ii < sg.items.size(); ++ii) {
                    auto& item = sg.items[ii];
                    if (equalsCaseInsensitive(item.id, resolvedId) ||
                        equalsCaseInsensitive(item.id, tok) ||
                        equalsCaseInsensitive(item.subcategory, tok) ||
                        containsCaseInsensitive(item.name, tok)) {
                        if (firstCi < 0) {
                            firstCi = static_cast<int>(ci);
                            firstSgi = static_cast<int>(sgi);
                            firstIi = static_cast<int>(ii);
                        }
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

    if (firstCi >= 0) {
        selectInspectedBenchmark(firstCi, firstSgi, firstIi, false);
    }
}

void GuiApp::startBenchmarks() {
    if (m_execState == ExecutionState::Running) {
        return;
    }

    // Zero-Contention Interlock: ensure no background ISA compiler thread runs during timing.
    joinIsaThread();

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
                        // Whole-benchmark items (e.g. the Cache Latency Curve sweep)
                        // must match by bare id; "id:name" never matched any task.
                        selectedWorkloads.push_back(item.id);
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
    stampScopeTrigger(selectedWorkloads.size() == 1 ? selectedWorkloads.front() : "SUITE RUN");

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
    const bool keepRtViewport =
        m_stayOnRtViewportAfterRun ||
        (m_instrumentMode == 2 &&
         std::find(benchmarksToRun.begin(), benchmarksToRun.end(), "RayScheduling") !=
             benchmarksToRun.end());
    m_stayOnRtViewportAfterRun = false;

    struct IsaPreprobeTarget {
        std::string id;
        int configIndex{-1};
        std::string cacheKey;
    };
    std::vector<IsaPreprobeTarget> isaPreprobeTargets;
    for (const auto& cat : m_categories) {
        for (const auto& sg : cat.subgroups) {
            for (const auto& item : sg.items) {
                if (!item.selected || !item.isSupported || item.category == "System") {
                    continue;
                }
                const std::string key = makeIsaCacheKey(item);
                const bool dup = std::any_of(isaPreprobeTargets.begin(), isaPreprobeTargets.end(),
                                             [&key](const IsaPreprobeTarget& t) { return t.cacheKey == key; });
                if (!dup) {
                    isaPreprobeTargets.push_back(IsaPreprobeTarget{item.id, item.configIndex, key});
                }
            }
        }
    }
    const std::string preprobeBackend = m_selectedBackend;
    const uint32_t preprobeDeviceIdx = m_telemetryGpuIndex;

    m_execThread = std::thread([this, benchmarksToRun, deviceIndices, backends,
                                rw, rh, scene, spp, dumpRenders, verbose, debug,
                                selectedWorkloads, keepRtViewport,
                                isaPreprobeTargets, preprobeBackend, preprobeDeviceIdx]() {
        try {
            // Pre-populate missing ISA cache entries BEFORE timing starts (zero contention)
            // so each auto-selected test displays its hot loop & register file live as it runs.
            for (const auto& target : isaPreprobeTargets) {
                if (m_cancelToken.load()) {
                    break;
                }
                {
                    std::lock_guard<std::mutex> lock(m_isaMutex);
                    if (m_isaCache.find(target.cacheKey) != m_isaCache.end()) {
                        continue;
                    }
                }
                IsaDumperOptions opts;
                opts.targets = {target.id};
                if (target.configIndex >= 0) {
                    opts.configs = {target.configIndex};
                }
                opts.backends = {preprobeBackend};
                opts.deviceIndex = preprobeDeviceIdx;
                opts.fullIsa = true;

                IsaInspectionResult isaRes = IsaDumper::inspect(opts);
                if (isaRes.artifacts.empty() && !opts.configs.empty()) {
                    opts.configs.clear();
                    isaRes = IsaDumper::inspect(opts);
                }
                {
                    std::lock_guard<std::mutex> lock(m_isaMutex);
                    m_isaCache[target.cacheKey] = std::move(isaRes);
                }
            }

            if (m_cancelToken.load()) {
                m_telemetryWorker.stopRecording();
                m_textureCacheNeedsClear.store(true);
                std::lock_guard<std::mutex> lock(m_resultsMutex);
                m_execState = ExecutionState::Cancelled;
                m_statusMessage = "Benchmark execution aborted by user.";
                return;
            }

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
                if (keepRtViewport) {
                    m_switchToRtViewport = true;
                } else {
                    m_switchToScorecard = true;
                }
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
        if (res.time_ms == -1.0) {
            m_currentBenchmarkName = res.benchmarkName;
            m_currentlyRunningTestId = res.benchmarkName;
            m_currentlyRunningResult = res;
            m_hasCurrentlyRunningResult = true;
            m_statusMessage = "Running: " + res.benchmarkName + " (" + res.deviceName + ")";

            int ci = -1, sgi = -1, ii = -1;
            if (findBenchmarkItemIndices(res, ci, sgi, ii)) {
                m_categories[static_cast<size_t>(ci)].subgroups[static_cast<size_t>(sgi)].collapsed = false;
                selectInspectedBenchmark(ci, sgi, ii, false);
                m_suiteAutoScrollToInspected = true;
            }
            continue;
        }

        auto existingIt = std::find_if(m_allResults.begin(), m_allResults.end(),
            [&res](const ResultData& existing) {
                return existing.deviceIndex == res.deviceIndex &&
                       existing.backendName == res.backendName &&
                       existing.subcategory == res.subcategory &&
                       existing.benchmarkName == res.benchmarkName &&
                       existing.configIndex == res.configIndex;
            });
        if (existingIt != m_allResults.end()) {
            *existingIt = res;
        } else {
            m_allResults.push_back(res);
        }

        std::string key = std::to_string(res.deviceIndex) + "|" + res.subcategory + "|" +
                          res.benchmarkName + "|" + std::to_string(res.configIndex);
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
    } else {
        m_hasCurrentlyRunningResult = false;
        m_currentlyRunningTestId.clear();
        if (m_singleSlotSelectionRestore) {
            size_t idx = 0;
            for (auto& cat : m_categories) {
                bool catAll = true;
                for (auto& sg : cat.subgroups) {
                    bool sgAll = true;
                    for (auto& itm : sg.items) {
                        if (idx < m_savedSelectionSnapshot.size()) {
                            itm.selected = m_savedSelectionSnapshot[idx++];
                        }
                        if (!itm.selected && itm.isSupported) sgAll = false;
                    }
                    sg.allSelected = sgAll;
                    if (!sgAll) catAll = false;
                }
                cat.allSelected = catAll;
            }
            m_savedSelectionSnapshot.clear();
            m_singleSlotSelectionRestore = false;
            if (m_restoreSceneAfterRun) {
                m_scene = m_savedSceneBeforeCapture;
                m_restoreSceneAfterRun = false;
            }
        }
    }
}

void GuiApp::processEvents() {
    processIncomingResults();
}

namespace {

std::filesystem::path getDocumentsResultsDir() {
    namespace fs = std::filesystem;
#ifdef _WIN32
    if (const char* userProfile = std::getenv("USERPROFILE"); userProfile && *userProfile) {
        return fs::path(userProfile) / "Documents" / "GPUBench";
    }
    const char* homeDrive = std::getenv("HOMEDRIVE");
    const char* homePath = std::getenv("HOMEPATH");
    if (homeDrive && *homeDrive && homePath && *homePath) {
        return (fs::path(std::string(homeDrive) + std::string(homePath))) / "Documents" / "GPUBench";
    }
#else
    if (const char* xdgDocs = std::getenv("XDG_DOCUMENTS_DIR"); xdgDocs && *xdgDocs) {
        return fs::path(xdgDocs) / "GPUBench";
    }
#endif
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home) / "Documents" / "GPUBench";
    }
    return {};
}

} // namespace

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

    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path requested(filepath);
    fs::path fileName = requested.filename();
    if (fileName.empty()) fileName = "gpubench_results.json";

    const fs::path docsDir = getDocumentsResultsDir();
    const bool hasExplicitDir = requested.has_parent_path() && requested.parent_path() != fs::path(".");

    // When no explicit directory was given (the normal GUI export case),
    // save to ~/Documents/GPUBench/ first on both Windows and Linux, and only
    // fall back to the local working directory, home, or temp if that fails.
    std::vector<fs::path> candidates;
    if (hasExplicitDir) {
        candidates.push_back(fs::absolute(requested, ec));
        if (!docsDir.empty()) candidates.push_back(docsDir / fileName);
    } else {
        if (!docsDir.empty()) candidates.push_back(docsDir / fileName);
        candidates.push_back(fs::absolute(fileName, ec));
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        candidates.push_back(fs::path(home) / fileName);
    }
    candidates.push_back(fs::temp_directory_path(ec) / fileName);

    std::string failures;
    for (const fs::path& cand : candidates) {
        if (cand.empty()) continue;
        if (!cand.parent_path().empty()) {
            fs::create_directories(cand.parent_path(), ec);
        }
        errno = 0;
        std::ofstream out(cand, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            failures += "\n  " + cand.string() + ": " + std::strerror(errno);
            continue;
        }
        out.write(jsonStr.data(), static_cast<std::streamsize>(jsonStr.size()));
        out.close();
        if (out.fail()) {
            failures += "\n  " + cand.string() + ": write error";
            continue;
        }

        m_lastExportPath = fs::weakly_canonical(cand, ec).string();
        if (ec || m_lastExportPath.empty()) m_lastExportPath = cand.string();
        SDL_SetClipboardText(m_lastExportPath.c_str());
        m_exportNotificationText = "Saved JSON (copied): " + m_lastExportPath;
        m_exportNotificationTimer = 12.0f;
        std::cout << "[Export] Saved JSON to " << m_lastExportPath << std::endl;
        return;
    }

    m_exportNotificationText = "Export failed. Tried:" + failures;
    m_exportNotificationTimer = 15.0f;
    std::cerr << "[Export] " << m_exportNotificationText << std::endl;
}

void GuiApp::openLastExportFolder() {
    if (m_lastExportPath.empty()) return;
    std::filesystem::path dir = std::filesystem::path(m_lastExportPath).parent_path();
    std::string url = "file:///" + dir.generic_string();
    SDL_OpenURL(url.c_str());
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
        std::string key = std::to_string(r.deviceIndex) + "|" + r.subcategory + "|" +
                          r.benchmarkName + "|" + std::to_string(r.configIndex);
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

    // 1. Exact subcategory + benchmarkName + configIndex match
    for (const auto& cmp : m_comparisonResults) {
        if (!cmp.isUnsupported && cmp.time_ms > 0.0 &&
            cmp.subcategory == curRes.subcategory &&
            cmp.benchmarkName == curRes.benchmarkName &&
            cmp.configIndex == curRes.configIndex) {
            return &cmp;
        }
    }
    const bool isGenericPrecisionName = (curRes.benchmarkName == "Vector" || curRes.benchmarkName == "Matrix");
    if (!isGenericPrecisionName) {
        // 2. Exact benchmarkName + configIndex match
        for (const auto& cmp : m_comparisonResults) {
            if (!cmp.isUnsupported && cmp.time_ms > 0.0 &&
                cmp.benchmarkName == curRes.benchmarkName &&
                cmp.configIndex == curRes.configIndex) {
                return &cmp;
            }
        }
        // 3. Exact benchmarkName match (ignoring configIndex for legacy v1 files)
        for (const auto& cmp : m_comparisonResults) {
            if (!cmp.isUnsupported && cmp.time_ms > 0.0 &&
                cmp.benchmarkName == curRes.benchmarkName) {
                return &cmp;
            }
        }
    }
    return nullptr;
}

void GuiApp::setInstrumentMode(int mode) {
    m_instrumentMode = std::clamp(mode, 0, 2);
    m_switchToScorecard = false;
    m_switchToRoofline = false;
    m_switchToRtViewport = false;
}

void GuiApp::setPalette(BringUpPalette p) {
    if (m_palette == p && !m_themeDirty) return;
    m_palette = p;
    setActiveBringUpPalette(p);
    m_themeDirty = true;
}

void GuiApp::joinIsaThread() {
    if (m_isaThread.joinable()) {
        m_isaThread.join();
    }
    m_isaLoading.store(false);
}

bool GuiApp::waitForPendingInspections(uint32_t timeoutMs) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (m_isaLoading.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!m_isaLoading.load() && m_isaThread.joinable()) {
        m_isaThread.join();
    }
    return !m_isaLoading.load();
}

const BenchmarkItem* GuiApp::getInspectedItem() const {
    if (m_inspectedCategoryIdx < 0 ||
        m_inspectedCategoryIdx >= static_cast<int>(m_categories.size())) {
        return nullptr;
    }
    const auto& cat = m_categories[static_cast<size_t>(m_inspectedCategoryIdx)];
    if (m_inspectedSubgroupIdx < 0 ||
        m_inspectedSubgroupIdx >= static_cast<int>(cat.subgroups.size())) {
        return nullptr;
    }
    const auto& sg = cat.subgroups[static_cast<size_t>(m_inspectedSubgroupIdx)];
    if (m_inspectedItemIdx < 0 ||
        m_inspectedItemIdx >= static_cast<int>(sg.items.size())) {
        return nullptr;
    }
    return &sg.items[static_cast<size_t>(m_inspectedItemIdx)];
}

void GuiApp::ensureDefaultInspectedBenchmark() {
    if (getInspectedItem() != nullptr) return;

    int firstSupportedC = -1, firstSupportedS = -1, firstSupportedI = -1;
    for (int ci = 0; ci < static_cast<int>(m_categories.size()); ++ci) {
        const auto& cat = m_categories[static_cast<size_t>(ci)];
        for (int si = 0; si < static_cast<int>(cat.subgroups.size()); ++si) {
            const auto& sg = cat.subgroups[static_cast<size_t>(si)];
            for (int ii = 0; ii < static_cast<int>(sg.items.size()); ++ii) {
                const auto& itm = sg.items[static_cast<size_t>(ii)];
                if (itm.isSupported && firstSupportedC < 0) {
                    firstSupportedC = ci;
                    firstSupportedS = si;
                    firstSupportedI = ii;
                }
                if (itm.id == "BF16" && itm.configIndex == 1 && itm.isSupported) {
                    m_inspectedCategoryIdx = ci;
                    m_inspectedSubgroupIdx = si;
                    m_inspectedItemIdx = ii;
                    m_isaExecStageIdx = 0;
                    return;
                }
            }
        }
    }
    if (firstSupportedC >= 0) {
        m_inspectedCategoryIdx = firstSupportedC;
        m_inspectedSubgroupIdx = firstSupportedS;
        m_inspectedItemIdx = firstSupportedI;
        m_isaExecStageIdx = 0;
    } else if (!m_categories.empty() && !m_categories[0].subgroups.empty() &&
               !m_categories[0].subgroups[0].items.empty()) {
        m_inspectedCategoryIdx = 0;
        m_inspectedSubgroupIdx = 0;
        m_inspectedItemIdx = 0;
        m_isaExecStageIdx = 0;
    }
}

bool GuiApp::findBenchmarkItemIndices(const ResultData& res,
                                        int& outCatIdx,
                                        int& outSgIdx,
                                        int& outItemIdx) const {
    for (size_t ci = 0; ci < m_categories.size(); ++ci) {
        const auto& cat = m_categories[ci];
        for (size_t sgi = 0; sgi < cat.subgroups.size(); ++sgi) {
            const auto& sg = cat.subgroups[sgi];
            for (size_t ii = 0; ii < sg.items.size(); ++ii) {
                if (matchesItem(res, sg.items[ii], res.deviceIndex)) {
                    outCatIdx = static_cast<int>(ci);
                    outSgIdx = static_cast<int>(sgi);
                    outItemIdx = static_cast<int>(ii);
                    return true;
                }
            }
        }
    }
    return false;
}

void GuiApp::selectInspectedBenchmark(int catIdx, int sgIdx, int itemIdx, bool switchToF1) {
    if (catIdx < 0 || catIdx >= static_cast<int>(m_categories.size())) return;
    const auto& cat = m_categories[static_cast<size_t>(catIdx)];
    if (sgIdx < 0 || sgIdx >= static_cast<int>(cat.subgroups.size())) return;
    const auto& sg = cat.subgroups[static_cast<size_t>(sgIdx)];
    if (itemIdx < 0 || itemIdx >= static_cast<int>(sg.items.size())) return;

    m_inspectedCategoryIdx = catIdx;
    m_inspectedSubgroupIdx = sgIdx;
    m_inspectedItemIdx = itemIdx;
    m_isaExecStageIdx = 0;
    if (switchToF1) {
        setInstrumentMode(0);
    }
    requestIsaInspection(sg.items[static_cast<size_t>(itemIdx)], false);
}

std::string GuiApp::makeIsaCacheKey(const BenchmarkItem& item) const {
    return item.id + "|" + std::to_string(item.configIndex) + "|" +
           m_selectedBackend + "|" + std::to_string(m_telemetryGpuIndex);
}

void GuiApp::requestIsaInspection(const BenchmarkItem& item, bool forceRefresh) {
    if (m_execState == ExecutionState::Running) {
        return;
    }

    const std::string key = makeIsaCacheKey(item);
    if (!forceRefresh) {
        std::lock_guard<std::mutex> lock(m_isaMutex);
        if (m_isaCache.find(key) != m_isaCache.end()) {
            return;
        }
    }
    if (m_isaLoading.load() && m_isaActiveKey == key && !forceRefresh) {
        return;
    }

    joinIsaThread();
    m_isaActiveKey = key;
    m_isaLoading.store(true);

    IsaDumperOptions opts;
    opts.targets = {item.id};
    if (item.configIndex >= 0) {
        opts.configs = {item.configIndex};
    }
    opts.backends = {m_selectedBackend};
    opts.deviceIndex = m_telemetryGpuIndex;
    opts.fullIsa = true;

    m_isaThread = std::thread([this, key, opts]() mutable {
        IsaInspectionResult res = IsaDumper::inspect(opts);
        if (res.artifacts.empty() && !opts.configs.empty()) {
            opts.configs.clear();
            res = IsaDumper::inspect(opts);
        }
        {
            std::lock_guard<std::mutex> lock(m_isaMutex);
            m_isaCache[key] = std::move(res);
        }
        m_isaLoading.store(false);
    });
}

void GuiApp::stampScopeTrigger(const std::string& label) {
    DeviceTelemetrySnapshot snap{};
    float tSec = 0.0f;
    if (m_telemetryWorker.getSnapshot(m_telemetryGpuIndex, snap) && !snap.timeHistory.empty()) {
        tSec = snap.timeHistory.back();
    }
    if (m_scopeTriggers.size() >= 24) {
        m_scopeTriggers.erase(m_scopeTriggers.begin());
    }
    m_scopeTriggers.push_back({tSec, label});
}

void GuiApp::runSingleBenchmarkItem(const BenchmarkItem& targetItem) {
    if (m_execState == ExecutionState::Running || !targetItem.isSupported) {
        return;
    }

    m_savedSelectionSnapshot.clear();
    for (auto& cat : m_categories) {
        for (auto& sg : cat.subgroups) {
            for (auto& itm : sg.items) {
                m_savedSelectionSnapshot.push_back(itm.selected);
                const bool match = (itm.id == targetItem.id &&
                                    itm.configIndex == targetItem.configIndex &&
                                    itm.name == targetItem.name);
                itm.selected = match;
            }
        }
    }
    m_singleSlotSelectionRestore = true;

    const bool isSys = (targetItem.category == "System");
    bool hasMatchingDev = false;
    for (const auto& d : m_devices) {
        if (d.selected && d.isSystem == isSys) {
            hasMatchingDev = true;
            break;
        }
    }
    if (!hasMatchingDev) {
        for (auto& d : m_devices) {
            if (d.isSystem == isSys && (isSys || d.deviceIndex == m_telemetryGpuIndex)) {
                d.selected = true;
                hasMatchingDev = true;
                break;
            }
        }
        if (!hasMatchingDev) {
            for (auto& d : m_devices) {
                if (d.isSystem == isSys) {
                    d.selected = true;
                    break;
                }
            }
        }
    }

    startBenchmarks();
    if (m_execState != ExecutionState::Running && m_singleSlotSelectionRestore) {
        size_t idx = 0;
        for (auto& cat : m_categories) {
            for (auto& sg : cat.subgroups) {
                for (auto& itm : sg.items) {
                    if (idx < m_savedSelectionSnapshot.size()) {
                        itm.selected = m_savedSelectionSnapshot[idx++];
                    }
                }
            }
        }
        m_savedSelectionSnapshot.clear();
        m_singleSlotSelectionRestore = false;
    }
}

void GuiApp::startSceneVerificationCapture(const std::string& sceneTag) {
    if (m_execState == ExecutionState::Running) {
        return;
    }

    if (m_selectedBackend != "vulkan") {
        setSelectedBackend("vulkan");
    }

    bool hasGpu = false;
    for (const auto& d : m_devices) {
        if (!d.isSystem && d.selected) {
            hasGpu = true;
            break;
        }
    }
    if (!hasGpu) {
        for (auto& d : m_devices) {
            if (!d.isSystem && d.deviceIndex == m_telemetryGpuIndex) {
                d.selected = true;
                hasGpu = true;
                break;
            }
        }
        if (!hasGpu) {
            for (auto& d : m_devices) {
                if (!d.isSystem) {
                    d.selected = true;
                    hasGpu = true;
                    break;
                }
            }
        }
        if (hasGpu) {
            m_lastProbedBackend.clear();
            updateBenchmarkSupport();
        }
    }

    m_savedSelectionSnapshot.clear();
    bool anyPicked = false;
    for (auto& cat : m_categories) {
        for (auto& sg : cat.subgroups) {
            for (auto& itm : sg.items) {
                m_savedSelectionSnapshot.push_back(itm.selected);
                const bool match =
                    (itm.id == "RayScheduling" && itm.isSupported &&
                     (itm.configIndex == 9 || itm.configIndex == 11));
                itm.selected = match;
                if (match) anyPicked = true;
            }
        }
    }
    if (!anyPicked) {
        for (auto& cat : m_categories) {
            for (auto& sg : cat.subgroups) {
                for (auto& itm : sg.items) {
                    if (itm.id == "RayScheduling" && itm.isSupported && itm.configIndex == 0) {
                        itm.selected = true;
                        anyPicked = true;
                        break;
                    }
                }
                if (anyPicked) break;
            }
            if (anyPicked) break;
        }
    }

    if (!anyPicked) {
        size_t idx = 0;
        for (auto& cat : m_categories) {
            for (auto& sg : cat.subgroups) {
                for (auto& itm : sg.items) {
                    if (idx < m_savedSelectionSnapshot.size()) {
                        itm.selected = m_savedSelectionSnapshot[idx++];
                    }
                }
            }
        }
        m_savedSelectionSnapshot.clear();
        m_statusMessage = "RayScheduling requires a Vulkan GPU supporting ray tracing.";
        return;
    }

    m_savedSceneBeforeCapture = m_scene;
    m_restoreSceneAfterRun = true;
    m_scene = (sceneTag == "outdoor") ? "landscape" : sceneTag;
    m_singleSlotSelectionRestore = true;
    m_stayOnRtViewportAfterRun = true;

    startBenchmarks();
    if (m_execState != ExecutionState::Running) {
        size_t idx = 0;
        for (auto& cat : m_categories) {
            for (auto& sg : cat.subgroups) {
                for (auto& itm : sg.items) {
                    if (idx < m_savedSelectionSnapshot.size()) {
                        itm.selected = m_savedSelectionSnapshot[idx++];
                    }
                }
            }
        }
        m_savedSelectionSnapshot.clear();
        m_singleSlotSelectionRestore = false;
        if (m_restoreSceneAfterRun) {
            m_scene = m_savedSceneBeforeCapture;
            m_restoreSceneAfterRun = false;
        }
        m_stayOnRtViewportAfterRun = false;
    }
}

void GuiApp::executeCommand(const std::string& rawCmd) {
    size_t first = rawCmd.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return;
    size_t last = rawCmd.find_last_not_of(" \t\r\n");
    std::string trimmed = rawCmd.substr(first, last - first + 1);
    if (trimmed.empty()) return;

    m_cmdHistory.push_back(trimmed);
    if (m_cmdHistory.size() > 32) m_cmdHistory.erase(m_cmdHistory.begin());

    std::istringstream iss(trimmed);
    std::string verb;
    iss >> verb;
    std::string rest;
    std::getline(iss, rest);
    size_t r0 = rest.find_first_not_of(" \t");
    rest = (r0 == std::string::npos) ? "" : rest.substr(r0);

    auto lowerStr = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return s;
    };
    const std::string lowVerb = lowerStr(verb);
    const std::string lowRest = lowerStr(rest);

    if (lowVerb == "help" || lowVerb == "?") {
        m_cmdStatusText = "CMDS: run [slot|all] | abort | isa <name> | f1 | f2 | f3 | palette <amber|green|paper> | gpu <0|1|all> | backend <vk|rocm|cl> | scene <id> | import <file> | diff <file> | export [file]";
    } else if (lowVerb == "f1") {
        setInstrumentMode(0);
        m_cmdStatusText = "MODE -> F1: ISA & DIE INSPECTOR";
    } else if (lowVerb == "f2" || lowVerb == "charts" || lowVerb == "perf" || lowVerb == "roofline" || lowVerb == "scorecard" || lowVerb == "split") {
        setInstrumentMode(1);
        if (lowVerb == "charts" || lowVerb == "perf" || lowVerb == "roofline") m_f2SubMode = 1;
        else if (lowVerb == "scorecard") m_f2SubMode = 2;
        else if (lowVerb == "split") m_f2SubMode = 0;
        m_cmdStatusText = "MODE -> F2: PERFORMANCE & SCORECARD";
    } else if (lowVerb == "f3" || lowVerb == "rt" || lowVerb == "viewport") {
        setInstrumentMode(2);
        m_cmdStatusText = "MODE -> F3: RT PARITY VIEWPORT";
    } else if (lowVerb == "palette" || lowVerb == "theme") {
        if (lowRest == "green" || lowRest == "p1") {
            setPalette(BringUpPalette::GreenP1);
            m_cmdStatusText = "PALETTE -> GREEN";
        } else if (lowRest == "paper" || lowRest == "light" || lowRest == "datasheet") {
            setPalette(BringUpPalette::PaperDatasheet);
            m_cmdStatusText = "PALETTE -> PAPER";
        } else {
            setPalette(BringUpPalette::AmberP3);
            m_cmdStatusText = "PALETTE -> AMBER";
        }
    } else if (lowVerb == "gpu" || lowVerb == "dev" || lowVerb == "device") {
        if (lowRest == "all" || lowRest == "dual") {
            selectAllGpus();
            m_cmdStatusText = "DUT -> ALL GPUs SELECTED";
        } else if (!lowRest.empty()) {
            try {
                int idx = std::stoi(lowRest);
                setSelectedDevice(idx);
                m_cmdStatusText = "DUT -> GPU " + std::to_string(idx);
            } catch (...) {
                m_cmdStatusText = "ERR: INVALID GPU INDEX";
            }
        }
    } else if (lowVerb == "backend" || lowVerb == "api") {
        std::string b = lowRest;
        if (b == "vk") b = "vulkan";
        else if (b == "cl") b = "opencl";
        else if (b == "hip") b = "rocm";
        if (!b.empty()) {
            setSelectedBackend(b);
            m_cmdStatusText = "BACKEND -> " + m_selectedBackend;
        }
    } else if (lowVerb == "scene") {
        if (!lowRest.empty()) {
            setScene(lowRest);
            m_cmdStatusText = "SCENE -> " + m_scene;
        }
    } else if (lowVerb == "isa" || lowVerb == "inspect") {
        if (rest.empty()) {
            setInstrumentMode(0);
            m_cmdStatusText = "F1: ISA INSPECTOR ACTIVE";
        } else {
            bool found = false;
            for (int ci = 0; ci < static_cast<int>(m_categories.size()) && !found; ++ci) {
                for (int si = 0; si < static_cast<int>(m_categories[static_cast<size_t>(ci)].subgroups.size()) && !found; ++si) {
                    const auto& items = m_categories[static_cast<size_t>(ci)].subgroups[static_cast<size_t>(si)].items;
                    for (int ii = 0; ii < static_cast<int>(items.size()); ++ii) {
                        if (containsCaseInsensitive(items[static_cast<size_t>(ii)].id, rest) ||
                            containsCaseInsensitive(items[static_cast<size_t>(ii)].name, rest)) {
                            selectInspectedBenchmark(ci, si, ii, true);
                            m_cmdStatusText = "ISA INSPECT -> " + items[static_cast<size_t>(ii)].name;
                            found = true;
                            break;
                        }
                    }
                }
            }
            if (!found) {
                setInstrumentMode(0);
                m_cmdStatusText = "NO MATCHING ROM SLOT FOR: " + rest;
            }
        }
    } else if (lowVerb == "abort" || lowVerb == "stop") {
        abortBenchmarks();
        m_cmdStatusText = "ABORT SIGNAL DISPATCHED";
    } else if (lowVerb == "run") {
        if (!rest.empty() && lowRest != "selected" && lowRest != "tests" && lowRest != "armed") {
            selectOnlyBenchmark(rest);
        }
        startBenchmarks();
        m_cmdStatusText = "EXECUTION TRIGGERED";
    } else if (lowVerb == "import" && !rest.empty()) {
        m_cmdStatusText = importResultsFromJson(rest) ? ("IMPORTED: " + rest) : ("IMPORT FAILED: " + rest);
    } else if ((lowVerb == "diff" || lowVerb == "compare") && !rest.empty()) {
        m_cmdStatusText = loadComparisonFromJson(rest) ? ("BASELINE LOADED: " + rest) : ("DIFF LOAD FAILED: " + rest);
    } else if (lowVerb == "export") {
        std::string outPath = rest.empty() ? getDefaultJsonFilename() : rest;
        exportResultsToJson(outPath);
        m_cmdStatusText = m_lastExportPath.empty() ? ("EXPORT FAILED: " + outPath)
                                                   : ("EXPORTED: " + m_lastExportPath);
    } else {
        m_cmdStatusText = "UNKNOWN CMD '" + verb + "' (TYPE 'help')";
    }
}

void GuiApp::updateAndRender() {
    if (m_themeDirty) {
        syncBringUpTheme();
        m_themeDirty = false;
    }

    processIncomingResults();
    updateZoomShortcuts();

    if (m_exportNotificationTimer > 0.0f) {
        m_exportNotificationTimer -= ImGui::GetIO().DeltaTime;
    }

    // Sync swapchain clear color to active theme
    if (m_vulkanContext && m_vulkanContext->isInitialized()) {
        const BringUpColors c = getBringUpColors(m_palette);
        m_vulkanContext->setClearColor(c.bgVoid.x, c.bgVoid.y, c.bgVoid.z, 1.0f);
        m_vulkanContext->setCrtPostFx(false, static_cast<int>(m_palette), 1.0f);
    }

    // Handle pending deferred tab switches
    if (m_switchToRtViewport) {
        setInstrumentMode(2);
    } else if (m_switchToRoofline) {
        setInstrumentMode(1);
        m_f2SubMode = 1;
    } else if (m_switchToScorecard) {
        setInstrumentMode(1);
        m_f2SubMode = 0; // Split view shows both Roofline & Scorecard cleanly
    }

    // Global function-key & prompt shortcuts
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantTextInput) {
        if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) setInstrumentMode(0);
        else if (ImGui::IsKeyPressed(ImGuiKey_F2, false)) setInstrumentMode(1);
        else if (ImGui::IsKeyPressed(ImGuiKey_F3, false)) setInstrumentMode(2);
        else if (ImGui::IsKeyPressed(ImGuiKey_Slash, false) ||
                 (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_K, false))) {
            m_focusCmdInput = true;
        }
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);

    ImGuiWindowFlags rootFlags = ImGuiWindowFlags_NoTitleBar |
                                 ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove |
                                 ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoBringToFrontOnFocus |
                                 ImGuiWindowFlags_NoNavFocus |
                                 ImGuiWindowFlags_NoScrollbar |
                                 ImGuiWindowFlags_NoScrollWithMouse;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(s(8.0f), s(6.0f)));
    ImGui::Begin("GPUBenchRootWindow", nullptr, rootFlags);
    ImGui::PopStyleVar();

    const BringUpColors c = getBringUpColors(m_palette);
    const float availWidth = ImGui::GetContentRegionAvail().x;
    const float availHeight = ImGui::GetContentRegionAvail().y;

    const float headerH = s(68.0f);
    const float scopeH = m_telemetryExpanded ? std::clamp(availHeight * 0.34f, s(220.0f), s(310.0f))
                                             : s(96.0f);
    const float cmdH = s(30.0f);
    const float middleH = std::max(s(220.0f), availHeight - headerH - scopeH - cmdH - s(16.0f));

    // ZONE 1: Top DUT & Instrument Bus Header
    renderTopInstrumentHeader(availWidth);

    ImGui::Spacing();

    // ZONE 2 + ZONE 3: Left ROM-Slot Matrix & Right Main Instrument Stage
    const float minSidebar = std::min(s(320.0f), availWidth * 0.34f);
    const float maxSidebar = std::min(s(440.0f), availWidth * 0.36f);
    const float leftW = std::clamp(availWidth * 0.27f, minSidebar, std::max(minSidebar, maxSidebar));
    const float rightW = std::max(s(320.0f), availWidth - leftW - s(8.0f));

    ImGui::BeginChild("ZoneLeftRomMatrix", ImVec2(leftW, middleH), ImGuiChildFlags_Borders);
    renderBenchmarkSuitePanel();
    ImGui::EndChild();

    ImGui::SameLine(0.0f, s(8.0f));

    ImGui::BeginChild("ZoneRightMainStage", ImVec2(rightW, middleH), ImGuiChildFlags_Borders);
    if (m_instrumentMode == 0) {
        renderIsaInspectorPanel();
    } else if (m_instrumentMode == 1) {
        // F2 Sub-Mode Bar: [SPLIT VIEW | CHARTS | SCORECARD]
        ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bgElevated);
        ImGui::BeginChild("F2SubModeBar", ImVec2(0, s(32.0f)), ImGuiChildFlags_Borders);
        {
            const float bh = s(22.0f);
            if (tacticalButton("##F2Split", "SPLIT VIEW", ImVec2(s(98.0f), bh), m_f2SubMode == 0, c)) m_f2SubMode = 0;
            ImGui::SameLine(0.0f, s(4.0f));
            if (tacticalButton("##F2Roof", "CHARTS", ImVec2(s(86.0f), bh), m_f2SubMode == 1, c)) m_f2SubMode = 1;
            ImGui::SameLine(0.0f, s(4.0f));
            if (tacticalButton("##F2Score", "SCORECARD", ImVec2(s(98.0f), bh), m_f2SubMode == 2, c)) m_f2SubMode = 2;

            ImGui::SameLine(0.0f, s(16.0f));
            if (m_comparisonResults.empty()) {
                ImGui::TextColored(c.textMuted, "SAMPLES: %zu", m_allResults.size());
            } else {
                ImGui::TextColored(c.cyanAccent, "SAMPLES: %zu   COMPARING: %s",
                                   m_allResults.size(), m_comparisonSourceLabel.c_str());
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        const float f2AvailH = std::max(s(140.0f), ImGui::GetContentRegionAvail().y);
        if (m_f2SubMode == 0) {
            const float topH = std::floor(f2AvailH * 0.52f);
            const float botH = std::max(s(100.0f), f2AvailH - topH - s(6.0f));
            ImGui::BeginChild("F2RooflineRegion", ImVec2(0, topH), ImGuiChildFlags_None);
            renderRooflinePanel();
            ImGui::EndChild();
            ImGui::Spacing();
            ImGui::BeginChild("F2ScorecardRegion", ImVec2(0, botH), ImGuiChildFlags_None);
            renderResultsScorecard();
            ImGui::EndChild();
        } else if (m_f2SubMode == 1) {
            renderRooflinePanel();
        } else {
            renderResultsScorecard();
        }
    } else {
        renderRayTracingViewport();
    }
    ImGui::EndChild();

    ImGui::Spacing();

    // ZONE 4A: Persistent 4-Channel Hardware Oscilloscope Strip
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(s(6.0f), s(4.0f)));
    ImGuiWindowFlags scopeFlags = m_telemetryExpanded
        ? ImGuiWindowFlags_None
        : (ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::BeginChild("ZoneBottomScope", ImVec2(availWidth, scopeH), ImGuiChildFlags_Borders, scopeFlags);
    ImGui::PopStyleVar();
    renderLiveTelemetryDock();
    ImGui::EndChild();

    // ZONE 4B: Interactive >_ Hardware Command Prompt Bar
    renderCommandPromptBar(availWidth);

    if (m_showSettingsModal) {
        renderSettingsModal();
    }
    if (m_showJsonSessionModal) {
        renderJsonSessionModal();
    }

    ImGui::End();
    renderZoomToast();
}

void GuiApp::renderTopInstrumentHeader(float width) {
    const BringUpColors c = getBringUpColors(m_palette);
    const bool isRunning = (m_execState == ExecutionState::Running);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bgPanel);
    ImGui::BeginChild("ZoneTopHeader", ImVec2(width, s(68.0f)), ImGuiChildFlags_Borders);
    {
        const float bh = s(24.0f);

        // ---- ROW 1: Brand + DUT Select + Backend + Palette/CRT + Run Control ----
        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "GPUBENCH v" GPUBENCH_VERSION);
        popBoldFont();
        ImGui::SameLine(0.0f, s(6.0f));
        ImGui::TextColored(c.textMuted, "//");
        ImGui::SameLine(0.0f, s(6.0f));

        // DUT GPU selector pills
        if (isRunning) ImGui::BeginDisabled();
        int gpuCount = 0;
        for (const auto& d : m_devices) {
            if (d.isSystem) continue;
            gpuCount++;
            std::string gLbl = "GPU" + std::to_string(d.deviceIndex);
            std::string gId = "##TopGpu" + std::to_string(d.deviceIndex);
            if (tacticalButton(gId.c_str(), gLbl.c_str(), ImVec2(s(46.0f), bh), d.selected, c)) {
                setSelectedDevice(static_cast<int>(d.deviceIndex));
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("%s (%s)", d.name.c_str(), d.architecture.c_str());
            }
            ImGui::SameLine(0.0f, s(3.0f));
        }
        if (gpuCount >= 2) {
            bool allSel = true;
            for (const auto& d : m_devices) {
                if (!d.isSystem && !d.selected) allSel = false;
            }
            if (tacticalButton("##TopGpuAll", "ALL", ImVec2(s(38.0f), bh), allSel, c)) {
                selectAllGpus();
            }
            ImGui::SameLine(0.0f, s(8.0f));
        }

        // Backend pills
        const char* backends[] = {"vulkan", "rocm", "opencl", "auto"};
        const char* bLabels[] = {"VK", "ROCM", "CL", "AUTO"};
        for (int bi = 0; bi < 4; ++bi) {
            bool avail = true;
            const ComputeApiSupportInfo* apiInfo = nullptr;
            for (const auto& info : m_apiSupportList) {
                if (info.name == backends[bi]) {
                    apiInfo = &info;
                    avail = info.isSupported;
                    break;
                }
            }
            std::string bid = "##TopApi" + std::string(bLabels[bi]);
            ImGui::BeginDisabled(!avail);
            if (tacticalButton(bid.c_str(), bLabels[bi], ImVec2(s(46.0f), bh),
                               m_selectedBackend == backends[bi], c)) {
                setSelectedBackend(backends[bi]);
            }
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && apiInfo) {
                ImGui::SetTooltip("%s: %s", apiInfo->label.c_str(), apiInfo->reason.c_str());
            }
            ImGui::SameLine(0.0f, s(3.0f));
        }

        // Compact scene pills
        ImGui::SameLine(0.0f, s(6.0f));
        const char* scIds[] = {"all", "showroom", "indoor", "forest", "landscape"};
        const char* scLbls[] = {"ALL SC", "SHOW", "INDR", "FRST", "OUTD"};
        for (int si = 0; si < 5; ++si) {
            std::string sid = "##TopSc" + std::to_string(si);
            if (tacticalButton(sid.c_str(), scLbls[si], ImVec2(0, bh), m_scene == scIds[si], c)) {
                setScene(scIds[si]);
            }
            ImGui::SameLine(0.0f, s(2.0f));
        }
        if (isRunning) ImGui::EndDisabled();

        // Right cluster on Row 1: Palette + Session + Cfg + Primary RUN/ABORT
        size_t armedCount = 0;
        for (const auto& cat : m_categories) {
            for (const auto& sg : cat.subgroups) {
                for (const auto& itm : sg.items) {
                    if (itm.selected && itm.isSupported) armedCount++;
                }
            }
        }
        char runBtnLbl[48];
        if (isRunning) {
            int pct = (m_totalTasks > 0) ? static_cast<int>((100 * m_completedTasks) / m_totalTasks) : 0;
            std::snprintf(runBtnLbl, sizeof(runBtnLbl), "ABORT %d%%", pct);
        } else {
            std::snprintf(runBtnLbl, sizeof(runBtnLbl), "RUN TESTS (%zu)", armedCount);
        }

        const float palAmberW = s(60.0f);
        const float palGreenW = s(60.0f);
        const float paperW = s(58.0f);
        const float jsonW = s(64.0f);
        const float exportW = s(92.0f);
        const float cfgW = s(44.0f);
        const float execW = s(134.0f);
        const float rightClusterW = palAmberW + palGreenW + paperW + jsonW + exportW + cfgW + execW + s(24.0f);
        const float targetX = ImGui::GetWindowWidth() - rightClusterW - s(10.0f);
        if (targetX > ImGui::GetCursorPosX() + s(8.0f)) {
            ImGui::SetCursorPosX(targetX);
        }

        if (tacticalButton("##PalAmber", "AMBER", ImVec2(palAmberW, bh), m_palette == BringUpPalette::AmberP3, c)) {
            setPalette(BringUpPalette::AmberP3);
        }
        ImGui::SameLine(0.0f, s(3.0f));
        if (tacticalButton("##PalGreen", "GREEN", ImVec2(palGreenW, bh), m_palette == BringUpPalette::GreenP1, c)) {
            setPalette(BringUpPalette::GreenP1);
        }
        ImGui::SameLine(0.0f, s(3.0f));
        if (tacticalButton("##PalPaper", "PAPER", ImVec2(paperW, bh), m_palette == BringUpPalette::PaperDatasheet, c)) {
            setPalette(BringUpPalette::PaperDatasheet);
        }
        ImGui::SameLine(0.0f, s(5.0f));
        if (tacticalButton("##TopJson", "SESSION", ImVec2(jsonW, bh), m_showJsonSessionModal, c)) {
            if (m_jsonPathBuffer[0] == '\0') {
                std::string defName = getDefaultJsonFilename();
                std::filesystem::path docsDir = getDocumentsResultsDir();
                std::string defPath = docsDir.empty() ? defName : (docsDir / defName).string();
                std::snprintf(m_jsonPathBuffer, sizeof(m_jsonPathBuffer), "%s", defPath.c_str());
            }
            m_showJsonSessionModal = true;
        }
        ImGui::SameLine(0.0f, s(3.0f));
        ImGui::BeginDisabled(m_allResults.empty());
        if (tacticalButton("##TopExportJson", "EXPORT JSON", ImVec2(exportW, bh), false, c)) {
            std::string outPath = (m_jsonPathBuffer[0] != '\0')
                ? std::string(m_jsonPathBuffer)
                : getDefaultJsonFilename();
            exportResultsToJson(outPath);
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("%s", m_allResults.empty()
                ? "Run benchmarks or import a session before exporting JSON."
                : "Export all system information and benchmark results to unified v2.0 JSON.");
        }
        ImGui::SameLine(0.0f, s(3.0f));
        if (tacticalButton("##TopCfg", "CFG", ImVec2(cfgW, bh), m_showSettingsModal, c)) {
            m_showSettingsModal = true;
        }
        ImGui::SameLine(0.0f, s(4.0f));
        if (isRunning) {
            if (tacticalButton("##TopAbort", runBtnLbl, ImVec2(execW, bh), true, c, true, false)) {
                abortBenchmarks();
            }
        } else {
            if (tacticalButton("##TopExec", runBtnLbl, ImVec2(execW, bh), false, c, false, true)) {
                startBenchmarks();
            }
        }

        // ---- ROW 2: Hardware Instrument Mode Bus + Status Readout ----
        const float mbH = s(24.0f);
        if (tacticalButton("##ModeF1", "F1: ISA & DIE INSPECTOR", ImVec2(s(200.0f), mbH),
                           m_instrumentMode == 0, c)) {
            setInstrumentMode(0);
        }
        ImGui::SameLine(0.0f, s(4.0f));
        char f2Lbl[64];
        std::snprintf(f2Lbl, sizeof(f2Lbl), "F2: PERFORMANCE & SCORECARD (%zu)", m_allResults.size());
        if (tacticalButton("##ModeF2", f2Lbl, ImVec2(s(246.0f), mbH), m_instrumentMode == 1, c)) {
            setInstrumentMode(1);
        }
        ImGui::SameLine(0.0f, s(4.0f));
        if (tacticalButton("##ModeF3", "F3: RT PARITY VIEWPORT", ImVec2(s(192.0f), mbH),
                           m_instrumentMode == 2, c)) {
            setInstrumentMode(2);
        }

        // Right-aligned live DUT status + contention / notification strip
        ImGui::SameLine(0.0f, s(14.0f));
        std::string statusBanner;
        ImVec4 statusCol = c.textSecondary;
        if (isRunning) {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "RUNNING [%zu/%zu | %.1fs] %s",
                          m_completedTasks, m_totalTasks, m_elapsedSeconds, m_statusMessage.c_str());
            statusBanner = buf;
            statusCol = c.phosphorHi;
        } else if (m_exportNotificationTimer > 0.0f && !m_exportNotificationText.empty()) {
            statusBanner = m_exportNotificationText;
            statusCol = (m_exportNotificationText.rfind("Export failed", 0) == 0) ? c.errColor : c.okColor;
        } else if (!m_statusMessage.empty()) {
            statusBanner = m_statusMessage;
        } else {
            std::string rebarTag;
            for (const auto& d : m_devices) {
                if (!d.isSystem && d.selected && d.deviceIndex == m_telemetryGpuIndex) {
                    for (const auto& prof : m_deviceProfiles) {
                        if (prof.deviceIndex == d.deviceIndex &&
                            (prof.backend == "Vulkan" || prof.rebarSupported || prof.rebarSizeBytes > 0) &&
                            !prof.isApu) {
                            rebarTag = prof.rebarSupported ? " | ReBAR: ON" : " | ReBAR: OFF";
                            break;
                        }
                    }
                    break;
                }
            }
            statusBanner = "DUT READY" + rebarTag + " // " + std::to_string(armedCount) + " TESTS SELECTED";
        }
        const bool showExportActions = !isRunning && !m_lastExportPath.empty();
        const float copyBtnW = s(78.0f);
        const float openBtnW = s(72.0f);
        const float actionReserve = showExportActions ? (copyBtnW + openBtnW + s(12.0f)) : 0.0f;
        const float remW = std::max(s(60.0f), ImGui::GetContentRegionAvail().x - actionReserve);
        ImGui::TextColored(statusCol, "%s", fitTextToPixelWidth(statusBanner, remW).c_str());
        if (showExportActions && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", m_lastExportPath.c_str());
        }
        if (showExportActions) {
            ImGui::SameLine(0.0f, s(6.0f));
            if (tacticalButton("##CopyExportPath", "COPY PATH", ImVec2(copyBtnW, mbH), false, c)) {
                SDL_SetClipboardText(m_lastExportPath.c_str());
                m_exportNotificationText = "Path copied: " + m_lastExportPath;
                m_exportNotificationTimer = 6.0f;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Copy exported JSON path:\n%s", m_lastExportPath.c_str());
            }
            ImGui::SameLine(0.0f, s(4.0f));
            if (tacticalButton("##OpenExportDir", "OPEN DIR", ImVec2(openBtnW, mbH), false, c)) {
                openLastExportFolder();
            }
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void GuiApp::renderCommandPromptBar(float width) {
    const BringUpColors c = getBringUpColors(m_palette);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bgPanel);
    ImGui::BeginChild("ZoneCmdPrompt", ImVec2(width, s(28.0f)), ImGuiChildFlags_Borders);
    {
        pushMonoFont();
        ImGui::TextColored(c.phosphorHi, ">_");
        ImGui::SameLine(0.0f, s(6.0f));

        if (m_focusCmdInput) {
            ImGui::SetKeyboardFocusHere();
            m_focusCmdInput = false;
        }
        const float hintW = s(360.0f);
        const float inputW = std::max(s(220.0f), ImGui::GetContentRegionAvail().x - hintW - s(10.0f));
        ImGui::SetNextItemWidth(inputW);
        if (ImGui::InputTextWithHint("##BringUpCmdPrompt",
                                     "help | run bf16 | isa fp16 | f1 | f2 | f3 | palette paper",
                                     m_cmdBuffer, sizeof(m_cmdBuffer),
                                     ImGuiInputTextFlags_EnterReturnsTrue)) {
            std::string cmd(m_cmdBuffer);
            m_cmdBuffer[0] = '\0';
            executeCommand(cmd);
        }
        ImGui::SameLine(0.0f, s(10.0f));
        const std::string rightInfo = !m_cmdStatusText.empty()
            ? m_cmdStatusText
            : "KEYS: F1/F2/F3 SWITCH   / OR CTRL+K FOCUS PROMPT";
        ImGui::TextColored(c.textMuted, "%s",
                           fitTextToPixelWidth(rightInfo, ImGui::GetContentRegionAvail().x).c_str());
        popMonoFont();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void GuiApp::renderSettingsModal() {
    const BringUpColors c = getBringUpColors(m_palette);
    ImGui::OpenPopup("SIMULATION & RENDER SETTINGS");
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(s(500.0f), 0.0f), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("SIMULATION & RENDER SETTINGS", &m_showSettingsModal, ImGuiWindowFlags_AlwaysAutoResize)) {
        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "RAY TRACING & RASTERIZATION CONFIG");
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
            setScene(scenes[currentSceneIdx]);
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
        ImGui::TextColored(c.textMuted, "When unchecked, F3 RT Viewport still inspects SharedPreviewImages in GPU VRAM.");

        ImGui::Spacing();
        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "INSTRUMENT DISPLAY & DIAGNOSTICS");
        popBoldFont();
        ImGui::Separator();
        ImGui::Spacing();

        float uiScalePct = (m_uiScale / std::max(0.1f, m_baseScale)) * 100.0f;
        if (ImGui::SliderFloat("Interface Zoom (%)", &uiScalePct, 70.0f, 225.0f, "%.0f%%")) {
            setUiScale((uiScalePct / 100.0f) * m_baseScale);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("RESET")) {
            setUiScale(m_baseScale);
        }
        ImGui::Checkbox("Verbose Console Logging", &m_verbose);
        ImGui::Checkbox("Enable Vulkan Validation Layers (--debug)", &m_debug);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (tacticalButton("##CloseCfg", "CLOSE", ImVec2(s(120.0f), s(28.0f)), true, c)) {
            m_showSettingsModal = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void GuiApp::renderJsonSessionModal() {
    const BringUpColors c = getBringUpColors(m_palette);
    ImGui::OpenPopup("UNIFIED JSON SESSION & DIFF");
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(s(560.0f), 0.0f), ImGuiCond_Appearing);

    if (ImGui::BeginPopupModal("UNIFIED JSON SESSION & DIFF", &m_showJsonSessionModal, ImGuiWindowFlags_AlwaysAutoResize)) {
        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "UNIFIED CLI / GUI JSON SCHEMA (v2.0)");
        popBoldFont();
        ImGui::TextWrapped("Load a session exported by gpubench or gpubench-gui as the active telemetry set or baseline comparison.");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::InputText("JSON File Path", m_jsonPathBuffer, sizeof(m_jsonPathBuffer));

        namespace fs = std::filesystem;
        std::error_code ec;
        std::vector<std::pair<std::string, std::string>> quickJsons;
        auto scanJsonDir = [&](const fs::path& dir) {
            if (dir.empty() || !fs::exists(dir, ec)) return;
            for (const auto& entry : fs::directory_iterator(dir, ec)) {
                if (entry.is_regular_file(ec) && entry.path().extension() == ".json") {
                    quickJsons.emplace_back(entry.path().filename().string(), entry.path().string());
                    if (quickJsons.size() >= 8) break;
                }
            }
        };
        scanJsonDir(getDocumentsResultsDir());
        if (quickJsons.size() < 8) scanJsonDir(".");
        if (!quickJsons.empty()) {
            ImGui::TextColored(c.textMuted, "RECENT JSON FILES:");
            for (size_t i = 0; i < quickJsons.size(); ++i) {
                if (i > 0) sameLineIfFits(s(140.0f), s(6.0f));
                if (ImGui::SmallButton(quickJsons[i].first.c_str())) {
                    std::snprintf(m_jsonPathBuffer, sizeof(m_jsonPathBuffer), "%s", quickJsons[i].second.c_str());
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", quickJsons[i].second.c_str());
                }
            }
        }

        ImGui::Spacing();
        float btnW = (ImGui::GetContentRegionAvail().x - s(12.0f)) / 3.0f;

        if (tacticalButton("##JsonImp", "IMPORT SESSION", ImVec2(btnW, s(30.0f)), false, c, false, true)) {
            if (importResultsFromJson(m_jsonPathBuffer)) {
                m_showJsonSessionModal = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::SameLine(0.0f, s(6.0f));
        if (tacticalButton("##JsonDiff", "LOAD BASELINE", ImVec2(btnW, s(30.0f)), false, c)) {
            if (loadComparisonFromJson(m_jsonPathBuffer)) {
                m_showJsonSessionModal = false;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::SameLine(0.0f, s(6.0f));
        bool canExport = !m_allResults.empty();
        ImGui::BeginDisabled(!canExport);
        if (tacticalButton("##JsonExp", "EXPORT ACTIVE", ImVec2(btnW, s(30.0f)), false, c)) {
            exportResultsToJson(m_jsonPathBuffer);
            m_showJsonSessionModal = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndDisabled();

        if (!m_lastExportPath.empty()) {
            ImGui::Spacing();
            if (ImGui::SmallButton("Copy path")) {
                SDL_SetClipboardText(m_lastExportPath.c_str());
                m_exportNotificationText = "Path copied: " + m_lastExportPath;
                m_exportNotificationTimer = 6.0f;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Open folder")) {
                openLastExportFolder();
            }
            ImGui::SameLine();
            ImGui::TextColored(c.textMuted, "Last export: %s",
                               std::filesystem::path(m_lastExportPath).filename().string().c_str());
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", m_lastExportPath.c_str());
        }

        if (!m_comparisonResults.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(c.cyanAccent,
                               "ACTIVE BASELINE: %s (%zu items)",
                               m_comparisonSourceLabel.c_str(), m_comparisonResults.size());
            ImGui::SameLine();
            if (ImGui::SmallButton("CLEAR")) {
                clearComparisonResults();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        if (tacticalButton("##JsonClose", "CLOSE", ImVec2(s(110.0f), s(28.0f)), false, c)) {
            m_showJsonSessionModal = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

} // namespace gpubench::gui
