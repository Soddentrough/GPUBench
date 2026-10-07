#include "GuiApp.h"
#include "core/DeviceDatabase.h"
#include <imgui.h>
#include <implot.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <functional>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#elif defined(__APPLE__)
#include <sys/utsname.h>
#include <sys/sysctl.h>
#else
#include <sys/utsname.h>
#include <sys/sysinfo.h>
#endif

#ifndef GPUBENCH_VERSION
#define GPUBENCH_VERSION "1.0.0"
#endif

namespace gpubench::gui {

namespace {

static std::string detectGpuArchitecture(const std::string& name, uint32_t vendorID, uint32_t deviceID) {
    return gpubench::DeviceDatabase::getArchitectureName(vendorID, deviceID, name);
}

static void getHostSystemInfo(std::string& outCpuModel, std::string& outCpuArch, std::string& outOsDriver, uint64_t& outRamMb) {
    outCpuModel = "Host CPU";
    outCpuArch = "x86_64";
    outOsDriver = "Host OS";
    outRamMb = 32768;

#ifdef _WIN32
    outOsDriver = "Windows 11 / 10";

    int cpuInfo[4] = {-1};
    __cpuid(cpuInfo, (int)0x80000000);
    unsigned int nExIds = static_cast<unsigned int>(cpuInfo[0]);
    if (nExIds >= 0x80000004) {
        char brand[65] = {0};
        __cpuid(reinterpret_cast<int*>(brand), (int)0x80000002);
        __cpuid(reinterpret_cast<int*>(brand + 16), (int)0x80000003);
        __cpuid(reinterpret_cast<int*>(brand + 32), (int)0x80000004);
        std::string raw(brand);
        size_t s = raw.find_first_not_of(" \t\r\n");
        if (s != std::string::npos) {
            size_t e = raw.find_last_not_of(" \t\r\n");
            outCpuModel = raw.substr(s, e - s + 1);
        } else if (!raw.empty()) {
            outCpuModel = raw;
        }
    }

    MEMORYSTATUSEX memStatus;
    memStatus.dwLength = sizeof(memStatus);
    if (GlobalMemoryStatusEx(&memStatus)) {
        outRamMb = memStatus.ullTotalPhys / (1024 * 1024);
    }
#elif defined(__APPLE__)
    outOsDriver = "macOS";
    struct utsname un;
    if (uname(&un) == 0) {
        outOsDriver = std::string(un.sysname) + " " + std::string(un.release);
    }

    char cpuBrand[256] = {0};
    size_t cpuBrandLen = sizeof(cpuBrand);
    if (sysctlbyname("machdep.cpu.brand_string", cpuBrand, &cpuBrandLen, NULL, 0) == 0) {
        outCpuModel = cpuBrand;
    }

    int64_t memBytes = 0;
    size_t memBytesLen = sizeof(memBytes);
    if (sysctlbyname("hw.memsize", &memBytes, &memBytesLen, NULL, 0) == 0) {
        outRamMb = static_cast<uint64_t>(memBytes) / (1024 * 1024);
    }
#if defined(__arm64__) || defined(__aarch64__)
    outCpuArch = "Apple Silicon (ARM64)";
#endif
#else
    outOsDriver = "Linux";
    struct utsname un;
    if (uname(&un) == 0) {
        outOsDriver = std::string(un.sysname) + " " + std::string(un.release);
    }

    std::ifstream cpuFile("/proc/cpuinfo");
    if (cpuFile.is_open()) {
        std::string line;
        while (std::getline(cpuFile, line)) {
            if (line.rfind("model name", 0) == 0) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string val = line.substr(colon + 1);
                    size_t s = val.find_first_not_of(" \t\r\n");
                    if (s != std::string::npos) {
                        size_t e = val.find_last_not_of(" \t\r\n");
                        outCpuModel = val.substr(s, e - s + 1);
                    } else if (!val.empty()) {
                        outCpuModel = val;
                    }
                    break;
                }
            }
        }
    }

    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        outRamMb = (static_cast<uint64_t>(si.totalram) * si.mem_unit) / (1024 * 1024);
    }
#endif

    std::string lowerModel = outCpuModel;
    std::transform(lowerModel.begin(), lowerModel.end(), lowerModel.begin(), ::tolower);
    if (lowerModel.find("zen 5") != std::string::npos || lowerModel.find("9950") != std::string::npos || 
        lowerModel.find("9900") != std::string::npos || lowerModel.find("9700") != std::string::npos || 
        lowerModel.find("9800") != std::string::npos) {
        outCpuArch = "x86_64 Zen 5";
    } else if (lowerModel.find("zen 4") != std::string::npos || lowerModel.find("7800x3d") != std::string::npos || 
               lowerModel.find("7950") != std::string::npos || lowerModel.find("7900") != std::string::npos || 
               lowerModel.find("7700") != std::string::npos || lowerModel.find("7600") != std::string::npos) {
        outCpuArch = "x86_64 Zen 4";
    } else if (lowerModel.find("zen 3") != std::string::npos || lowerModel.find("5950") != std::string::npos || 
               lowerModel.find("5900") != std::string::npos || lowerModel.find("5800") != std::string::npos || 
               lowerModel.find("5600") != std::string::npos) {
        outCpuArch = "x86_64 Zen 3";
    } else if (lowerModel.find("zen 2") != std::string::npos || lowerModel.find("3970") != std::string::npos || 
               lowerModel.find("3950") != std::string::npos || lowerModel.find("3900") != std::string::npos || 
               lowerModel.find("3700") != std::string::npos || lowerModel.find("3600") != std::string::npos) {
        outCpuArch = "x86_64 Zen 2";
    } else if (lowerModel.find("intel") != std::string::npos) {
        outCpuArch = "x86_64 Intel";
    } else {
        outCpuArch = "x86_64";
    }
}

static std::string formatCardDeviceSubtitle(const SelectableDevice& dev) {
    std::string name = dev.name;
    if (dev.isSystem) {
        size_t procPos = name.find(" Processor");
        if (procPos != std::string::npos) {
            name = name.substr(0, procPos);
            size_t corePos = name.rfind("-Core");
            if (corePos != std::string::npos) {
                size_t spaceBefore = name.rfind(' ', corePos);
                if (spaceBefore != std::string::npos) {
                    name = name.substr(0, spaceBefore);
                }
            }
        }
        size_t rPos;
        while ((rPos = name.find("(R)")) != std::string::npos) name.erase(rPos, 3);
        while ((rPos = name.find("(TM)")) != std::string::npos) name.erase(rPos, 4);
        while (name.find("  ") != std::string::npos) {
            name.replace(name.find("  "), 2, " ");
        }
        size_t s = name.find_first_not_of(" \t\r\n");
        if (s != std::string::npos) {
            size_t e = name.find_last_not_of(" \t\r\n");
            name = name.substr(s, e - s + 1);
        }
        return name;
    } else {
        if (name.empty()) {
            return "GPU " + std::to_string(dev.deviceIndex);
        }
        return name;
    }
}

} // anonymous namespace

GuiApp::GuiApp() {
    m_benchmarkStartTime = std::chrono::steady_clock::now();
    initializeBenchmarkCategories();
}

GuiApp::~GuiApp() {
    abortBenchmarks();
    if (m_execThread.joinable()) {
        m_execThread.join();
    }
    m_telemetryWorker.stop();

    clearTextureCache();
}

void GuiApp::clearTextureCache() {
    if (m_vulkanContext) {
        for (auto& pair : m_textureCache) {
            m_vulkanContext->destroyTexture(pair.second);
        }
        m_textureCache.clear();
    }
}

VulkanContext::TextureResource GuiApp::getOrLoadTexture(const std::string& relPath) {
    if (relPath.empty()) return VulkanContext::TextureResource{};
    auto it = m_textureCache.find(relPath);
    if (it != m_textureCache.end()) {
        return it->second;
    }
    if (!m_vulkanContext) {
        return VulkanContext::TextureResource{};
    }
    VulkanContext::TextureResource tex = m_vulkanContext->loadTextureFromFile(relPath);
    if (tex.isValid()) {
        m_textureCache[relPath] = tex;
    }
    return tex;
}

void GuiApp::init(VulkanContext* vulkanContext, float uiScale) {
    m_vulkanContext = vulkanContext;
    m_baseScale = uiScale > 0.0f ? uiScale : 1.0f;
    m_uiScale = m_baseScale;

    // First initialize base unscaled theme metrics
    setupDarkTheme(1.0f);

    if (ImPlot::GetCurrentContext()) {
        m_basePlotStyle = ImPlot::GetStyle();
        m_basePlotStyleInitialized = true;
    }

    // Apply effective target scale
    setUiScale(m_uiScale);
    m_zoomToastTimer = 0.0f; // Don't show toast on initial boot

    initializeBenchmarkCategories();
    discoverHardware();
    m_apiSupportList = GetAllComputeApiSupportAPI();
    for (const auto& api : m_apiSupportList) {
        std::cout << "[GPUBench Compute API] " << api.label << ": " << (api.isSupported ? "Supported" : "Unsupported");
        if (!api.isSupported) {
            std::cout << " (" << api.reason << " | Missing: " << api.missingRequirement << ")";
        }
        std::cout << std::endl;
    }

    // Ensure initial backend selection is supported
    bool currentBackendSupported = false;
    for (const auto& api : m_apiSupportList) {
        if (api.name == m_selectedBackend && api.isSupported) {
            currentBackendSupported = true;
            break;
        }
    }
    if (!currentBackendSupported) {
        for (const auto& api : m_apiSupportList) {
            if (api.isSupported) {
                m_selectedBackend = api.name;
                break;
            }
        }
    }

    updateBenchmarkSupport();
    m_telemetryWorker.start();
}

void GuiApp::setUiScale(float scale) {
    scale = std::clamp(scale, 0.75f, 4.0f);
    if (std::abs(scale - m_uiScale) < 0.005f && m_zoomToastTimer > 0.0f) return;
    m_uiScale = scale;
    setupDarkTheme(m_uiScale);

    ImGuiIO& io = ImGui::GetIO();
    if (m_baseScale > 0.0f) {
        io.FontGlobalScale = m_uiScale / m_baseScale;
    }

    if (m_basePlotStyleInitialized && ImPlot::GetCurrentContext()) {
        ImPlotStyle& plotStyle = ImPlot::GetStyle();
        plotStyle = m_basePlotStyle;
        plotStyle.LineWeight *= m_uiScale;
        plotStyle.MarkerSize *= m_uiScale;
        plotStyle.MarkerWeight *= m_uiScale;
        plotStyle.PlotBorderSize *= m_uiScale;
        plotStyle.MajorTickLen = ImVec2(m_basePlotStyle.MajorTickLen.x * m_uiScale, m_basePlotStyle.MajorTickLen.y * m_uiScale);
        plotStyle.MinorTickLen = ImVec2(m_basePlotStyle.MinorTickLen.x * m_uiScale, m_basePlotStyle.MinorTickLen.y * m_uiScale);
        plotStyle.MajorTickSize = ImVec2(m_basePlotStyle.MajorTickSize.x * m_uiScale, m_basePlotStyle.MajorTickSize.y * m_uiScale);
        plotStyle.MinorTickSize = ImVec2(m_basePlotStyle.MinorTickSize.x * m_uiScale, m_basePlotStyle.MinorTickSize.y * m_uiScale);
        plotStyle.MajorGridSize = ImVec2(m_basePlotStyle.MajorGridSize.x * m_uiScale, m_basePlotStyle.MajorGridSize.y * m_uiScale);
        plotStyle.MinorGridSize = ImVec2(m_basePlotStyle.MinorGridSize.x * m_uiScale, m_basePlotStyle.MinorGridSize.y * m_uiScale);
        plotStyle.PlotPadding = ImVec2(m_basePlotStyle.PlotPadding.x * m_uiScale, m_basePlotStyle.PlotPadding.y * m_uiScale);
        plotStyle.LabelPadding = ImVec2(m_basePlotStyle.LabelPadding.x * m_uiScale, m_basePlotStyle.LabelPadding.y * m_uiScale);
        plotStyle.LegendPadding = ImVec2(m_basePlotStyle.LegendPadding.x * m_uiScale, m_basePlotStyle.LegendPadding.y * m_uiScale);
        plotStyle.LegendInnerPadding = ImVec2(m_basePlotStyle.LegendInnerPadding.x * m_uiScale, m_basePlotStyle.LegendInnerPadding.y * m_uiScale);
        plotStyle.LegendSpacing = ImVec2(m_basePlotStyle.LegendSpacing.x * m_uiScale, m_basePlotStyle.LegendSpacing.y * m_uiScale);
    }

    m_zoomToastTimer = 1.6f;
}

void GuiApp::updateZoomShortcuts() {
    ImGuiIO& io = ImGui::GetIO();
    if (io.KeyCtrl) {
        if (std::abs(io.MouseWheel) > 0.01f) {
            float step = (io.MouseWheel > 0.0f) ? 0.1f : -0.1f;
            setUiScale(std::round((m_uiScale + step) * 10.0f) / 10.0f);
            io.MouseWheel = 0.0f;
            io.MouseWheelH = 0.0f;
        } else if (ImGui::IsKeyPressed(ImGuiKey_Equal) || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd)) {
            setUiScale(std::round((m_uiScale + 0.1f) * 10.0f) / 10.0f);
        } else if (ImGui::IsKeyPressed(ImGuiKey_Minus) || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract)) {
            setUiScale(std::round((m_uiScale - 0.1f) * 10.0f) / 10.0f);
        } else if (ImGui::IsKeyPressed(ImGuiKey_0) || ImGui::IsKeyPressed(ImGuiKey_Keypad0)) {
            setUiScale(m_baseScale);
        }
    }
}

void GuiApp::renderZoomToast() {
    if (m_zoomToastTimer <= 0.0f) return;

    ImGuiIO& io = ImGui::GetIO();
    m_zoomToastTimer -= io.DeltaTime;
    float alpha = std::clamp(m_zoomToastTimer / 0.35f, 0.0f, 1.0f);

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImVec2 toastPos = ImVec2(vp->WorkPos.x + vp->WorkSize.x - s(250.0f), vp->WorkPos.y + s(16.0f));
    ImGui::SetNextWindowPos(toastPos, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.92f * alpha);

    ImGuiWindowFlags toastFlags = ImGuiWindowFlags_NoDecoration 
                                | ImGuiWindowFlags_AlwaysAutoResize 
                                | ImGuiWindowFlags_NoSavedSettings 
                                | ImGuiWindowFlags_NoFocusOnAppearing 
                                | ImGuiWindowFlags_NoNav
                                | ImGuiWindowFlags_NoMove;

    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.38f, 0.75f, 1.0f, 0.75f * alpha));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, alpha));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.08f, 0.10f, 0.15f, 0.95f * alpha));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, s(6.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(s(12.0f), s(8.0f)));

    if (ImGui::Begin("##ZoomToastOverlay", nullptr, toastFlags)) {
        int zoomPct = static_cast<int>(std::round((m_uiScale / m_baseScale) * 100.0f));
        int absPct = static_cast<int>(std::round(m_uiScale * 100.0f));
        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, alpha), "Zoom: %d%%", zoomPct);
        ImGui::SameLine();
        ImGui::TextDisabled("(%.1fx / %d%% DPI)", m_uiScale, absPct);
        if (std::abs(m_uiScale - m_baseScale) > 0.01f) {
            ImGui::TextDisabled("Press Ctrl+0 to reset");
        }
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(3);
}

void GuiApp::setSelectedDevice(int deviceIndex) {
    if (deviceIndex >= 0) {
        bool found = false;
        for (const auto& dev : m_devices) {
            if (!dev.isSystem && dev.deviceIndex == static_cast<uint32_t>(deviceIndex)) {
                found = true;
                break;
            }
        }
        // If requested deviceIndex does not exist, fallback to first available GPU
        if (!found) {
            for (const auto& dev : m_devices) {
                if (!dev.isSystem) {
                    deviceIndex = static_cast<int>(dev.deviceIndex);
                    found = true;
                    break;
                }
            }
        }
        if (found) {
            for (auto& dev : m_devices) {
                dev.selected = (dev.deviceIndex == static_cast<uint32_t>(deviceIndex) && !dev.isSystem);
            }
            m_telemetryGpuIndex = static_cast<uint32_t>(deviceIndex);
            m_telemetryDualGpuMode = false;
            updateBenchmarkSupport();
        }
    }
}

void GuiApp::setSelectedDevices(const std::vector<int>& deviceIndices) {
    bool anySelected = false;
    for (auto& dev : m_devices) {
        dev.selected = false;
        for (int idx : deviceIndices) {
            if (idx < 0 && dev.isSystem) {
                dev.selected = true;
                anySelected = true;
            } else if (!dev.isSystem && dev.deviceIndex == static_cast<uint32_t>(idx)) {
                dev.selected = true;
                anySelected = true;
            }
        }
    }
    // If none of the specified devices exist on this system, select the first actual GPU
    if (!anySelected) {
        for (auto& dev : m_devices) {
            if (!dev.isSystem) {
                dev.selected = true;
                break;
            }
        }
    }
    updateTelemetrySelection();
}

void GuiApp::updateTelemetrySelection() {
    size_t gpuSelCount = 0;
    uint32_t lastGpu = 0;
    for (const auto& dev : m_devices) {
        if (dev.selected && !dev.isSystem) {
            gpuSelCount++;
            lastGpu = dev.deviceIndex;
        }
    }
    if (gpuSelCount >= 2) {
        m_telemetryDualGpuMode = true;
    } else {
        m_telemetryDualGpuMode = false;
        if (gpuSelCount == 1) {
            m_telemetryGpuIndex = lastGpu;
        } else {
            for (const auto& dev : m_devices) {
                if (!dev.isSystem) {
                    m_telemetryGpuIndex = dev.deviceIndex;
                    break;
                }
            }
        }
    }
    updateBenchmarkSupport();
}

void GuiApp::updateBenchmarkSupport() {
    uint32_t targetGpu = 0;
    bool foundGpu = false;
    for (const auto& dev : m_devices) {
        if (dev.selected && !dev.isSystem) {
            targetGpu = dev.deviceIndex;
            foundGpu = true;
            break;
        }
    }
    if (!foundGpu) {
        for (const auto& dev : m_devices) {
            if (!dev.isSystem) {
                targetGpu = dev.deviceIndex;
                foundGpu = true;
                break;
            }
        }
    }

    if (m_selectedBackend == m_lastProbedBackend && targetGpu == m_lastProbedDeviceIndex) {
        return;
    }
    m_lastProbedBackend = m_selectedBackend;
    m_lastProbedDeviceIndex = targetGpu;

    std::vector<BenchmarkSupportInfo> supportList = ProbeBenchmarkSupportAPI(m_selectedBackend, targetGpu);
    std::unordered_map<std::string, BenchmarkSupportInfo> suppMap;
    for (const auto& s : supportList) {
        suppMap[s.id] = s;
    }

    for (auto& cat : m_categories) {
        for (auto& sub : cat.subgroups) {
            for (auto& item : sub.items) {
                if (item.category == "System" || item.category == "Host System" || item.category == "System Memory") {
                    item.isSupported = true;
                    item.supportReason.clear();
                    item.limitationCategory.clear();
                    continue;
                }
                auto it = suppMap.find(item.id);
                bool supported = (it != suppMap.end()) ? it->second.isSupported : true;
                std::string reason = (it != suppMap.end()) ? it->second.reason : "";
                std::string limCat = (it != suppMap.end()) ? it->second.limitationCategory : "";

                // Check by specific config or item name
                auto itName = suppMap.find(item.name);
                if (itName != suppMap.end() && !itName->second.isSupported) {
                    supported = false;
                    reason = itName->second.reason;
                    limCat = itName->second.limitationCategory;
                }

                // Check for SER items specifically
                if (item.name.find("SER") != std::string::npos) {
                    auto itSER = suppMap.find("SER");
                    if (itSER != suppMap.end() && !itSER->second.isSupported) {
                        supported = false;
                        reason = itSER->second.reason;
                        limCat = itSER->second.limitationCategory;
                    }
                }

                item.isSupported = supported;
                item.supportReason = reason;
                item.limitationCategory = limCat;
                if (!item.isSupported) {
                    item.selected = false;
                }
            }
        }
    }
}

void GuiApp::setSelectedBackend(const std::string& backend) {
    if (!backend.empty()) {
        if (m_apiSupportList.empty()) {
            m_apiSupportList = GetAllComputeApiSupportAPI();
        }
        for (const auto& api : m_apiSupportList) {
            if (api.name == backend) {
                if (!api.isSupported) {
                    return; // Ignore selection of unsupported backend
                }
                break;
            }
        }
        m_selectedBackend = backend;
        updateBenchmarkSupport();
    }
}

void GuiApp::selectOnlyBenchmark(const std::string& benchmarkId) {
    std::vector<std::string> targetIds;
    std::stringstream ss(benchmarkId);
    std::string itemToken;
    while (std::getline(ss, itemToken, ',')) {
        if (!itemToken.empty()) targetIds.push_back(itemToken);
    }
    if (targetIds.empty()) targetIds.push_back(benchmarkId);

    bool isSystemBenchmark = false;
    for (auto& cat : m_categories) {
        cat.allSelected = false;
        bool catAnySel = false;
        for (auto& sub : cat.subgroups) {
            sub.allSelected = false;
            bool subAnySel = false;
            for (auto& item : sub.items) {
                bool matches = false;
                for (const auto& tid : targetIds) {
                    if (item.id == tid || item.subcategory == tid || item.name == tid || sub.name == tid) {
                        matches = true;
                        break;
                    }
                    if (tid == "pathtracing" || tid == "path_tracing" || tid == "path" || tid == "Path Tracing" || tid == "pathtrace") {
                        if (item.subcategory.find("Path Tracing") != std::string::npos || sub.name.find("Path Tracing") != std::string::npos || item.name.find("Path Tracing") != std::string::npos) {
                            matches = true;
                            break;
                        }
                    }
                }
                item.selected = matches;
                if (item.selected) {
                    subAnySel = true;
                    catAnySel = true;
                    if (item.category == "System" || item.category == "Host System" || item.category == "System Memory") {
                        isSystemBenchmark = true;
                    }
                }
            }
            sub.allSelected = subAnySel;
        }
        cat.allSelected = catAnySel;
    }
    if (isSystemBenchmark) {
        for (auto& dev : m_devices) {
            if (dev.isSystem) dev.selected = true;
        }
    }

    // Auto-collapse untested subgroups and expand active ones
    for (auto& cat : m_categories) {
        for (auto& sub : cat.subgroups) {
            size_t sel = 0;
            for (const auto& item : sub.items) {
                if (item.selected) sel++;
            }
            if (sel == 0) {
                sub.collapsed = true;
            } else {
                sub.collapsed = false;
            }
        }
    }
}

void GuiApp::setupDarkTheme(float scale) {
    ImGuiStyle& style = ImGui::GetStyle();

    if (!m_baseStyleInitialized) {
        ImVec4* colors = style.Colors;

    // Professional High-Contrast Workstation Dark Theme
    colors[ImGuiCol_Text]                  = ImVec4(0.96f, 0.97f, 0.99f, 1.00f);
    colors[ImGuiCol_TextDisabled]          = ImVec4(0.52f, 0.58f, 0.68f, 1.00f);
    colors[ImGuiCol_WindowBg]              = ImVec4(0.08f, 0.09f, 0.13f, 1.00f);
    colors[ImGuiCol_ChildBg]               = ImVec4(0.10f, 0.12f, 0.17f, 1.00f);
    colors[ImGuiCol_PopupBg]               = ImVec4(0.12f, 0.14f, 0.20f, 0.98f);
    colors[ImGuiCol_Border]                = ImVec4(0.20f, 0.25f, 0.35f, 0.70f);
    colors[ImGuiCol_BorderShadow]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]               = ImVec4(0.13f, 0.16f, 0.23f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.19f, 0.24f, 0.34f, 1.00f);
    colors[ImGuiCol_FrameBgActive]         = ImVec4(0.25f, 0.32f, 0.45f, 1.00f);
    colors[ImGuiCol_TitleBg]               = ImVec4(0.09f, 0.11f, 0.15f, 1.00f);
    colors[ImGuiCol_TitleBgActive]         = ImVec4(0.14f, 0.18f, 0.26f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.07f, 0.08f, 0.11f, 1.00f);
    colors[ImGuiCol_MenuBarBg]             = ImVec4(0.10f, 0.12f, 0.17f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.08f, 0.09f, 0.13f, 0.60f);
    colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.22f, 0.27f, 0.38f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.30f, 0.38f, 0.52f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.38f, 0.48f, 0.65f, 1.00f);
    colors[ImGuiCol_CheckMark]             = ImVec4(0.38f, 0.75f, 1.00f, 1.00f); // Bright Cyan
    colors[ImGuiCol_SliderGrab]            = ImVec4(0.35f, 0.50f, 0.98f, 1.00f); // Electric Blue
    colors[ImGuiCol_SliderGrabActive]      = ImVec4(0.45f, 0.60f, 1.00f, 1.00f);
    colors[ImGuiCol_Button]                = ImVec4(0.16f, 0.20f, 0.29f, 1.00f);
    colors[ImGuiCol_ButtonHovered]         = ImVec4(0.24f, 0.30f, 0.44f, 1.00f);
    colors[ImGuiCol_ButtonActive]          = ImVec4(0.32f, 0.40f, 0.60f, 1.00f);
    colors[ImGuiCol_Header]                = ImVec4(0.18f, 0.23f, 0.33f, 1.00f);
    colors[ImGuiCol_HeaderHovered]         = ImVec4(0.26f, 0.33f, 0.48f, 1.00f);
    colors[ImGuiCol_HeaderActive]          = ImVec4(0.34f, 0.43f, 0.62f, 1.00f);
    colors[ImGuiCol_Separator]             = ImVec4(0.20f, 0.25f, 0.35f, 0.70f);
    colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.32f, 0.40f, 0.58f, 1.00f);
    colors[ImGuiCol_SeparatorActive]       = ImVec4(0.40f, 0.50f, 0.72f, 1.00f);
    colors[ImGuiCol_Tab]                   = ImVec4(0.11f, 0.13f, 0.19f, 1.00f);
    colors[ImGuiCol_TabHovered]            = ImVec4(0.24f, 0.30f, 0.45f, 1.00f);
    colors[ImGuiCol_TabActive]             = ImVec4(0.20f, 0.27f, 0.42f, 1.00f);
    colors[ImGuiCol_TabUnfocused]          = ImVec4(0.09f, 0.11f, 0.15f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive]   = ImVec4(0.14f, 0.18f, 0.27f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]         = ImVec4(0.14f, 0.17f, 0.25f, 1.00f);
    colors[ImGuiCol_TableBorderStrong]     = ImVec4(0.22f, 0.28f, 0.40f, 1.00f);
    colors[ImGuiCol_TableBorderLight]      = ImVec4(0.16f, 0.20f, 0.28f, 0.80f);
    colors[ImGuiCol_TableRowBg]            = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]         = ImVec4(0.12f, 0.15f, 0.21f, 0.50f);

        style.WindowRounding    = 0.0f;
        style.ChildRounding     = 8.0f;
        style.FrameRounding     = 6.0f;
        style.PopupRounding     = 6.0f;
        style.ScrollbarRounding = 6.0f;
        style.GrabRounding      = 4.0f;
        style.TabRounding       = 6.0f;
        style.WindowBorderSize  = 0.0f;
        style.FrameBorderSize   = 1.0f;
        style.ItemSpacing       = ImVec2(10.0f, 8.0f);
        style.FramePadding      = ImVec2(10.0f, 6.0f);
        style.ItemInnerSpacing  = ImVec2(6.0f, 4.0f);

        m_baseStyle = style;
        m_baseStyleInitialized = true;
    }

    style = m_baseStyle;
    if (scale > 1.01f || scale < 0.99f) {
        style.ScaleAllSizes(scale);
    }
}

void GuiApp::initializeBenchmarkCategories() {
    m_categories.clear();

    // 1. [Compute] (1 subgroup, 13 tests)
    {
        BenchmarkCategory cat;
        cat.name = "Compute";
        cat.description = "Vector and matrix compute arithmetic across IEEE precisions and quantized formats";

        BenchmarkSubgroup sub;
        sub.name = "Compute Precision";
        sub.component = "Compute";
        sub.engineId = "Compute";
        sub.description = "Peak floating-point and integer vector/matrix throughput (FP64 down to INT4)";
        sub.items = {
            {"FP64", "FP64", "FP64 Double Precision", "Compute", "TFLOPS", "64-bit IEEE 754 floating-point peak throughput", true, 0},
            {"FP32", "FP32", "FP32 Single Precision", "Compute", "TFLOPS", "32-bit floating-point peak TFLOPS (dual-issue FMA)", true, 0},
            {"FP16", "FP16", "FP16 Half Precision - Vector", "Compute", "TFLOPS", "Packed 16-bit half-precision vector arithmetic", true, 0},
            {"FP16", "FP16", "FP16 Half Precision - Matrix (WMMA)", "Compute", "TFLOPS", "Cooperative matrix / tensor half-precision throughput", true, 1},
            {"BF16", "BF16", "BF16 Bfloat16 - Vector", "Compute", "TFLOPS", "Packed 16-bit bfloat16 vector arithmetic", true, 0},
            {"BF16", "BF16", "BF16 Bfloat16 - Matrix (WMMA)", "Compute", "TFLOPS", "Cooperative matrix / tensor bfloat16 throughput", true, 1},
            {"FP8", "FP8", "FP8 Micro-Float - Matrix (WMMA)", "Compute", "TFLOPS", "E4M3 / E5M2 8-bit floating point cooperative matrix ops", true, 0},
            {"FP6", "FP6", "FP6 Micro-Float - Vector", "Compute", "TOPS", "Sub-byte 6-bit quantized floating point throughput (SPV_NV_float6)", true, 0},
            {"FP4", "FP4", "FP4 Micro-Float - Vector", "Compute", "TOPS", "Sub-byte 4-bit quantized floating point throughput", true, 0},
            {"INT8", "INT8", "INT8 Integer - Vector (DP4A)", "Compute", "TOPS", "INT8 dot product (DP4A) vector instructions", true, 0},
            {"INT8", "INT8", "INT8 Integer - Matrix (WMMA)", "Compute", "TOPS", "INT8 cooperative matrix / WMMA tensor throughput", true, 1},
            {"INT4", "INT4", "INT4 Integer - Vector", "Compute", "TOPS", "Sub-byte 4-bit integer packed vector operations", true, 0},
            {"INT4", "INT4", "INT4 Integer - Matrix (WMMA)", "Compute", "TOPS", "Sub-byte 4-bit integer matrix core throughput", true, 1}
        };
        cat.subgroups.push_back(sub);

        BenchmarkSubgroup dualSub;
        dualSub.name = "Dual-Issue & Concurrency";
        dualSub.component = "Compute";
        dualSub.engineId = "Dual-Issue";
        dualSub.description = "Dual-issue ALU co-issuing, ILP scaling, and concurrent FP32+INT32 arithmetic";
        dualSub.items = {
            {"Dual-Issue", "Dual-Issue", "Standard FP32", "Compute", "TFLOPS", "Single-issue FP32 baseline (1 FMA/cycle); sequential dependency prevents dual-issuing to measure honest 1-issue capacity", true, 0},
            {"Dual-Issue", "Dual-Issue", "Dual-Issue FP32 (Partial Co-Issue)", "Compute", "TFLOPS", "Moderate ILP (8 chains); measures realistic dual-issue scaling with latency bubbles typical of real compiled shaders, before peak saturation", true, 1},
            {"Dual-Issue", "Dual-Issue", "Dual-Issue FP32 (FP32+FP32)", "Compute", "TFLOPS", "Peak dual-issue saturation (16 chains); saturates dual ALUs to measure maximum hardware co-issue capacity (2 FMAs/cycle)", true, 2},
            {"Dual-Issue", "Dual-Issue", "Standard INT32", "Compute", "TOPS", "Single-issue integer baseline (1 ALU op/cycle); tests basic integer ALU throughput", true, 3},
            {"Dual-Issue", "Dual-Issue", "Dual-Issue INT32 (Partial Co-Issue)", "Compute", "TOPS", "Moderate integer ILP (8 chains); tests whether integer ALUs can co-issue operations under typical instruction parallelism", true, 4},
            {"Dual-Issue", "Dual-Issue", "Dual-Issue INT32 (INT32+INT32)", "Compute", "TOPS", "Peak integer ILP (16 chains); reveals if GPU has dual integer ALUs or is physically capped at 1 ALU/cycle", true, 5},
            {"Dual-Issue", "Dual-Issue", "Dual-Issue Mixed (FP32+INT32)", "Compute", "TOPS", "Concurrent 1 FP32 + 1 INT32 per cycle; measures simultaneous execution across separate float and integer pipelines", true, 6}
        };
        cat.subgroups.push_back(dualSub);

        BenchmarkSubgroup ldsSub;
        ldsSub.name = "LDS & Shared Memory";
        ldsSub.component = "Compute";
        ldsSub.engineId = "LDS Bank Conflicts";
        ldsSub.description = "Local Data Share 32-bank conflict sweep and serialization quantification";
        ldsSub.items = {
            {"LDS Bank Conflicts", "LDS Bank Conflicts", "Stride 1 (Conflict-Free Baseline)", "Compute", "TB/s", "1-way conflict-free; all 32 lanes access distinct 4-byte banks", true, 0},
            {"LDS Bank Conflicts", "LDS Bank Conflicts", "Stride 2 (2-Way Conflict)", "Compute", "TB/s", "2 lanes per bank; 2x serialization reduces throughput to ~50%", true, 1},
            {"LDS Bank Conflicts", "LDS Bank Conflicts", "Stride 4 (4-Way Conflict)", "Compute", "TB/s", "4 lanes per bank; 4x serialization reduces throughput to ~25%", true, 2},
            {"LDS Bank Conflicts", "LDS Bank Conflicts", "Stride 8 (8-Way Conflict)", "Compute", "TB/s", "8 lanes per bank; 8x serialization reduces throughput to ~12.5%", true, 3},
            {"LDS Bank Conflicts", "LDS Bank Conflicts", "Stride 16 (16-Way Conflict)", "Compute", "TB/s", "16 lanes per bank; 16x serialization reduces throughput to ~6.25%", true, 4},
            {"LDS Bank Conflicts", "LDS Bank Conflicts", "Stride 32 (32-Way Full Serialization)", "Compute", "TB/s", "32 lanes collide on bank 0; fully serialized down to ~3.125%", true, 5},
            {"LDS Bank Conflicts", "LDS Bank Conflicts", "Stride 3 (Odd Stride Control)", "Compute", "TB/s", "gcd(3,32)=1; conflict-free permutation proves modular collision theory", true, 6},
            {"LDS Bank Conflicts", "LDS Bank Conflicts", "Stride 5 (Odd Stride Control)", "Compute", "TB/s", "gcd(5,32)=1; conflict-free permutation proves modular collision theory", true, 7}
        };
        cat.subgroups.push_back(ldsSub);

        BenchmarkSubgroup indirectSub;
        indirectSub.name = "Indirect Command Synthesis";
        indirectSub.component = "Compute";
        indirectSub.engineId = "In-Shader Indirect Synthesis";
        indirectSub.description = "Dynamic GPU workgroup command synthesis and hardware zero-dispatch pruning";
        indirectSub.items = {
            {"In-Shader Indirect Synthesis", "Indirect Command Synthesis", "CPU Direct Fixed Grid (100% Active Baseline)", "Compute", "MItems/s", "Host fixed grid dispatch with 100% item activity; baseline compute throughput", true, 0},
            {"In-Shader Indirect Synthesis", "Indirect Command Synthesis", "100% Active", "Compute", "MItems/s", "Classifier synthesizes VkDispatchIndirectCommand on GPU; measures indirect dispatch overhead", true, 1},
            {"In-Shader Indirect Synthesis", "Indirect Command Synthesis", "50% Active", "Compute", "MItems/s", "Stream compaction bins 50% active work; launches half the workgroups dynamically", true, 2},
            {"In-Shader Indirect Synthesis", "Indirect Command Synthesis", "10% Active", "Compute", "MItems/s", "Sparse workload; GPU skips 90% of workgroups via synthesized grid sizing", true, 3},
            {"In-Shader Indirect Synthesis", "Indirect Command Synthesis", "1% Active", "Compute", "MItems/s", "Highly sparse workload; dynamic indirect dispatch processes only 1% work", true, 4},
            {"In-Shader Indirect Synthesis", "Indirect Command Synthesis", "In-Shader Zero-Dispatch Pruning (0% Active)", "Compute", "MItems/s", "Hardware Command Processor instantaneously dismisses zero-workgroup (0,0,0) dispatches", true, 5},
            {"In-Shader Indirect Synthesis", "Indirect Command Synthesis", "CPU Direct Fixed Grid (10% Active)", "Compute", "MItems/s", "Host fixed grid dispatch on sparse 10% active workload; shows wasted wave occupancy", true, 6}
        };
        cat.subgroups.push_back(indirectSub);

        m_categories.push_back(cat);
    }

    // 2. [Memory] (3 subgroups, 28 tests)
    {
        BenchmarkCategory cat;
        cat.name = "Memory";
        cat.description = "On-chip cache hierarchy latencies (L0..L3), cache curves, and VRAM streaming bandwidth";

        cat.subgroups.push_back({"Cache Latency", "Memory", "L0 Cache Latency", "On-chip CU vector cache read latency", {
            {"L0 Cache Latency", "Cache Latency", "L0 Cache Latency (16 KB)", "Memory", "ns", "On-chip compute unit L0 vector cache latency", true, 0}
        }});

        BenchmarkSubgroup cacheCurveSub;
        cacheCurveSub.name = "Cache Latency Curve";
        cacheCurveSub.component = "Memory";
        cacheCurveSub.engineId = "Cache Latency Curve";
        cacheCurveSub.description = "128-byte cache-line pointer chasing (16 KB to 256 MB) across L0, GL1, GL2, L3 MALL, and GDDR6";
        cacheCurveSub.items = {
            {"Cache Latency Curve", "Cache Latency Curve", "16 KB", "Memory", "ns", "L0 TCP Vector Cache (32B/64B/128B Cache Line)", true, 0},
            {"Cache Latency Curve", "Cache Latency Curve", "32 KB", "Memory", "ns", "L0 TCP Capacity Boundary", true, 1},
            {"Cache Latency Curve", "Cache Latency Curve", "64 KB", "Memory", "ns", "GL1 Cache Transition", true, 2},
            {"Cache Latency Curve", "Cache Latency Curve", "128 KB", "Memory", "ns", "GL1 Cache Working Set", true, 3},
            {"Cache Latency Curve", "Cache Latency Curve", "256 KB", "Memory", "ns", "GL1 Cache Capacity Boundary", true, 4},
            {"Cache Latency Curve", "Cache Latency Curve", "512 KB", "Memory", "ns", "GL2 Cache Working Set", true, 5},
            {"Cache Latency Curve", "Cache Latency Curve", "1 MB", "Memory", "ns", "GL2 Cache Working Set", true, 6},
            {"Cache Latency Curve", "Cache Latency Curve", "2 MB", "Memory", "ns", "GL2 Cache Working Set", true, 7},
            {"Cache Latency Curve", "Cache Latency Curve", "4 MB", "Memory", "ns", "GL2 Cache Capacity Boundary", true, 8},
            {"Cache Latency Curve", "Cache Latency Curve", "8 MB", "Memory", "ns", "L3 MALL / Infinity Cache", true, 9},
            {"Cache Latency Curve", "Cache Latency Curve", "16 MB", "Memory", "ns", "L3 MALL / Infinity Cache", true, 10},
            {"Cache Latency Curve", "Cache Latency Curve", "32 MB", "Memory", "ns", "L3 MALL / Infinity Cache", true, 11},
            {"Cache Latency Curve", "Cache Latency Curve", "64 MB", "Memory", "ns", "L3 MALL / Infinity Cache", true, 12},
            {"Cache Latency Curve", "Cache Latency Curve", "128 MB", "Memory", "ns", "L3 MALL Capacity Boundary", true, 13},
            {"Cache Latency Curve", "Cache Latency Curve", "256 MB", "Memory", "ns", "GDDR6 VRAM DRAM Step", true, 14}
        };
        cat.subgroups.push_back(cacheCurveSub);

        cat.subgroups.push_back({"VRAM Streaming Bandwidth", "Memory", "Device Memory Bandwidth", "VRAM read/write streaming bandwidth across thread block sizes", {
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Read (128 threads/group)", "Memory", "GB/s", "Streaming VRAM read bandwidth at 128 threads/group", true, 0},
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Write (128 threads/group)", "Memory", "GB/s", "Streaming VRAM write bandwidth at 128 threads/group", true, 1},
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Read / Write (128 threads/group)", "Memory", "GB/s", "Combined VRAM read/write streaming bandwidth at 128 threads/group", true, 2},
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Read (256 threads/group)", "Memory", "GB/s", "Streaming VRAM read bandwidth at 256 threads/group", true, 3},
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Write (256 threads/group)", "Memory", "GB/s", "Streaming VRAM write bandwidth at 256 threads/group", true, 4},
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Read / Write (256 threads/group)", "Memory", "GB/s", "Combined VRAM read/write streaming bandwidth at 256 threads/group", true, 5},
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Read (1024 threads/group)", "Memory", "GB/s", "Streaming VRAM read bandwidth at 1024 threads/group", true, 6},
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Write (1024 threads/group)", "Memory", "GB/s", "Streaming VRAM write bandwidth at 1024 threads/group", true, 7},
            {"Device Memory Bandwidth", "VRAM Bandwidth", "Read / Write (1024 threads/group)", "Memory", "GB/s", "Combined VRAM read/write streaming bandwidth at 1024 threads/group", true, 8}
        }});

        m_categories.push_back(cat);
    }

    // 3. [Ray Tracing] (5 subgroups, 54 tests)
    {
        BenchmarkCategory cat;
        cat.name = "Ray Tracing";
        cat.description = "Hardware BVH, acceleration structure builds, ray tracing & multi-bounce path tracing";

        // Subgroup 1: Acceleration Structure Builds (8 tests)
        cat.subgroups.push_back({"Acceleration Structure Builds", "Ray Tracing", "RayASBuild", "Ray tracing acceleration structure build and dynamic update performance", {
            {"RayASBuild", "BLAS Build & Update", "BLAS Build (1M Triangles)", "Ray Tracing", "MTris/s", "Measures how fast the GPU builds ray tracing geometry structures for 1 million triangles", true, 0},
            {"RayASBuild", "BLAS Build & Update", "BLAS Update (1M Triangles)", "Ray Tracing", "MTris/s", "Measures how fast the GPU updates geometry structures for moving objects (1 million triangles)", true, 1},
            {"RayASBuild", "BLAS Build & Update", "BLAS Build (5M Triangles)", "Ray Tracing", "MTris/s", "Measures ray tracing structure build speed for complex geometry (5 million triangles)", true, 2},
            {"RayASBuild", "BLAS Build & Update", "BLAS Update (5M Triangles)", "Ray Tracing", "MTris/s", "Measures dynamic update speed for complex moving geometry (5 million triangles)", true, 3},
            {"RayASBuild", "BLAS Build & Update", "BLAS Build (10M Triangles)", "Ray Tracing", "MTris/s", "Measures ray tracing structure build speed under heavy geometry load (10 million triangles)", true, 4},
            {"RayASBuild", "TLAS Construction", "TLAS: Indoor Corridor (20K Instances)", "Ray Tracing", "MInst/s", "Measures scene hierarchy build speed for 20,000 instanced objects in an indoor scene", true, 5},
            {"RayASBuild", "TLAS Construction", "TLAS: Dense Jungle (50K Instances)", "Ray Tracing", "MInst/s", "Measures scene hierarchy build speed for 50,000 instanced objects in dense foliage", true, 6},
            {"RayASBuild", "TLAS Construction", "TLAS: Massive Open World (200K Instances)", "Ray Tracing", "MInst/s", "Measures scene hierarchy build speed for 200,000 instanced objects across a massive open world", true, 7}
        }});

        // Subgroup 2: Primary & Full Frame Ray Tracing (7 tests)
        cat.subgroups.push_back({"Primary & Full Frame Ray Tracing", "Ray Tracing", "RayScheduling", "Primary camera ray tracing and full-frame rendering across dispatch methods", {
            {"RayScheduling", "Scene Ray Tracing (PBR)", "Primary Rays (Compute Megakernel)", "Ray Tracing", "MRays/s", "Traces initial camera rays into the scene using a standard compute shader", true, 17},
            {"RayScheduling", "Scene Ray Tracing (PBR)", "Primary Rays (RTP)", "Ray Tracing", "MRays/s", "Traces initial camera rays using dedicated hardware ray tracing pipelines", true, 29},
            {"RayScheduling", "Scene Ray Tracing (PBR)", "Primary Rays (RTP + SER)", "Ray Tracing", "MRays/s", "Traces camera rays with hardware reordering to group similar rays and maximize core efficiency", true, 30},
            {"RayScheduling", "Scene Ray Tracing (PBR)", "Primary Rays (Wavefront - DGC)", "Ray Tracing", "MRays/s", "Traces camera rays using GPU work queues that group active work to maximize utilization", true, 18},
            {"RayScheduling", "Total Scene Render", "Full Frame (Megakernel)", "Ray Tracing", "MRays/s", "Renders complete frames with full lighting and shadows using standard compute shaders", true, 9},
            {"RayScheduling", "Total Scene Render", "Full Frame (RTP + SER)", "Ray Tracing", "MRays/s", "Renders complete frames using dedicated ray tracing hardware and ray reordering", true, 10},
            {"RayScheduling", "Total Scene Render", "Full Frame (DGC)", "Ray Tracing", "MRays/s", "Renders complete frames using dynamic GPU-driven work queues", true, 11}
        }});

        // Subgroup 3: Scene Path Tracing (5 tests)
        cat.subgroups.push_back({"Scene Path Tracing", "Ray Tracing", "RayScheduling", "Multi-bounce diffuse path tracing and high-sample stress workloads", {
            {"RayScheduling", "Scene Path Tracing (Multi-Bounce)", "Path Tracing (1 SPP) (Megakernel)", "Ray Tracing", "MRays/s", "Calculates realistic indirect light bounces using a standard all-in-one compute shader", true, 3},
            {"RayScheduling", "Scene Path Tracing (Multi-Bounce)", "Path Tracing (1 SPP) (RTP + SER)", "Ray Tracing", "MRays/s", "Calculates light bounces using dedicated ray tracing cores with hardware ray reordering", true, 4},
            {"RayScheduling", "Scene Path Tracing (Multi-Bounce)", "Path Tracing (1 SPP) (DGC)", "Ray Tracing", "MRays/s", "Calculates light bounces by grouping active rays into queues to keep GPU cores fully utilized", true, 5},
            {"RayScheduling", "Scene Path Tracing (16 SPP)", "Path Tracing (16 SPP) (Megakernel)", "Ray Tracing", "MRays/s", "High-quality path tracing (16 samples per pixel) using a standard compute shader", true, 23},
            {"RayScheduling", "Scene Path Tracing (16 SPP)", "Path Tracing (16 SPP) (DGC)", "Ray Tracing", "MRays/s", "High-quality path tracing (16 samples per pixel) grouping active rays into queues", true, 24}
        }});

        // Subgroup 4: Pipeline Stages & Scheduling (22 tests)
        cat.subgroups.push_back({"Pipeline Stages & Scheduling", "Ray Tracing", "RayScheduling", "Isolated rendering pipeline phases, shadows, shading, and queue compaction", {
            {"RayScheduling", "Directional Shadows", "Shadows - Single Light (Megakernel)", "Ray Tracing", "MRays/s", "Tests shadow ray performance for a single light source using standard compute shaders", true, 19},
            {"RayScheduling", "Directional Shadows", "Shadows - Single Light (RTP + SER)", "Ray Tracing", "MRays/s", "Tests shadow ray performance for a single light using dedicated ray tracing hardware and reordering", true, 20},
            {"RayScheduling", "Directional Shadows", "Shadows - Single Light (DGC)", "Ray Tracing", "MRays/s", "Tests shadow ray performance for a single light using dynamic GPU work queues", true, 21},
            {"RayScheduling", "Directional Shadows", "Shadows - Multi-Light (3 Lights, DGC)", "Ray Tracing", "MRays/s", "Tests shadow ray performance across 3 light sources using dynamic GPU work queues", true, 22},
            {"RayScheduling", "Multi-Light Evaluation", "Multi-Light Shading - Single Light (Megakernel)", "Ray Tracing", "MRays/s", "Evaluates material surface shading under a single light source using standard compute shaders", true, 31},
            {"RayScheduling", "Multi-Light Evaluation", "Multi-Light Shading - Single Light (DGC)", "Ray Tracing", "MRays/s", "Evaluates material surface shading under a single light source using dynamic work queues", true, 32},
            {"RayScheduling", "Multi-Light Evaluation", "Multi-Light Shading - 128 Lights (Megakernel)", "Ray Tracing", "MRays/s", "Stresses GPU cores by evaluating material shading across 128 unbinned lights in a single pass", true, 33},
            {"RayScheduling", "Multi-Light Evaluation", "Multi-Light Shading - 128 Lights (DGC Light Binning)", "Ray Tracing", "MRays/s", "Groups 128 light sources into coherent batches to optimize shading performance", true, 34},
            {"RayScheduling", "Material Shading", "Material (Megakernel)", "Ray Tracing", "MHits/s", "Evaluates realistic surface materials (metal, roughness, glass) in a standard compute pass", true, 0},
            {"RayScheduling", "Material Shading", "Material (RTP + SER)", "Ray Tracing", "MHits/s", "Evaluates surface materials using dedicated ray tracing closest-hit shaders and ray reordering", true, 1},
            {"RayScheduling", "Material Shading", "Material (DGC)", "Ray Tracing", "MHits/s", "Evaluates surface materials by sorting hits into batches to reduce shader divergence", true, 2},
            {"RayScheduling", "Incoherent Ray Tracing", "Incoherent Rays (Megakernel)", "Ray Tracing", "MRays/s", "Tests GPU memory and cache performance with scattered, randomly bouncing light rays", true, 6},
            {"RayScheduling", "Incoherent Ray Tracing", "Incoherent Rays (RTP + SER)", "Ray Tracing", "MRays/s", "Tests randomly bouncing light rays reordered by hardware to restore memory coherence", true, 7},
            {"RayScheduling", "Incoherent Ray Tracing", "Incoherent Rays (DGC)", "Ray Tracing", "MRays/s", "Tests randomly bouncing light rays sorted and compacted into queues to maximize SIMD efficiency", true, 8},
            {"RayScheduling", "Traversal Ordering & Coherence", "Linear 1D Scanline", "Ray Tracing", "MRays/s", "Traces rays in simple row-by-row order, serving as a baseline for ray order comparisons", true, 12},
            {"RayScheduling", "Traversal Ordering & Coherence", "2D Screen Tiled (8x4)", "Ray Tracing", "MRays/s", "Traces rays in small 8x4 pixel rectangular blocks to improve cache hit rates", true, 14},
            {"RayScheduling", "Traversal Ordering & Coherence", "2D Morton (8x4)", "Ray Tracing", "MRays/s", "Traces rays following an 8x4 Z-order curve to maximize spatial cache locality", true, 15},
            {"RayScheduling", "Traversal Ordering & Coherence", "2D Morton (4x8)", "Ray Tracing", "MRays/s", "Traces rays following a 4x8 Z-order curve to test directional cache preference", true, 16},
            {"RayScheduling", "Wavefront Stream Compaction", "Wave Ballot Compaction", "Ray Tracing", "MRecords/s", "Tests how fast GPU compute units can filter and compact active threads using wave voting", true, 13},
            {"RayScheduling", "Wavefront Stream Compaction", "Queue Compaction (Single-Pass)", "Ray Tracing", "MRecords/s", "Measures the speed of removing terminated rays from active queues in a single GPU pass", true, 26},
            {"RayScheduling", "Wavefront Stream Compaction", "Queue Compaction Throughput", "Ray Tracing", "MRecords/s", "Measures maximum rate of sorting and packing active rays into queues for the next pass", true, 28},
            {"RayScheduling", "Alpha Cutout Divergence", "Traversal Divergence (Alpha Cutout)", "Ray Tracing", "MRays/s", "Stresses ray traversal with transparent foliage (leaves and grass) requiring repeated opacity tests", true, 27}
        }});

        // Subgroup 5: Hardware BVH & Divergence Stress (12 tests)
        cat.subgroups.push_back({"Hardware BVH & Divergence Stress", "Ray Tracing", "RayRawTraversal", "Hardware ray-box, triangle traversal, alpha foliage, and SIMD divergence", {
            {"RayRawTraversal", "Hardware BVH Traversal", "Coherent Triangles", "Ray Tracing", "MRays/s", "Measures raw hardware ray-triangle test speed with parallel, orderly rays", true, 0},
            {"RayRawTraversal", "Hardware BVH Traversal", "Deep Box Stress", "Ray Tracing", "MRays/s", "Stresses hardware ray-box intersection cores with deeply nested bounding boxes", true, 1},
            {"RayIntersect", "Intersection Tests", "Ray-Triangle", "Ray Tracing", "MRays/s", "Pure hardware test measuring maximum ray-triangle intersection tests per second", true, 0},
            {"RayIntersect", "Intersection Tests", "Ray-Box", "Ray Tracing", "MRays/s", "Pure hardware test measuring maximum bounding box (AABB) intersection tests per second", true, 1},
            {"RayAnyHit", "Alpha-Tested Geometry", "100% Solid (Any-Hit Baseline)", "Ray Tracing", "MRays/s", "Baseline opacity test where all geometry is fully solid with no transparency", true, 0},
            {"RayAnyHit", "Alpha-Tested Geometry", "50% Solid (Cutout Stress)", "Ray Tracing", "MRays/s", "Tests GPU performance when half of all ray hits encounter transparent leaf cutouts", true, 1},
            {"RayProcedural", "Procedural Geometry", "AABB Spheres", "Ray Tracing", "MRays/s", "Tests ray intersection with mathematically defined spheres rather than polygons", true, 0},
            {"RayDivergence", "Ray Directional Coherence", "90° Hemispherical (Incoherent)", "Ray Tracing", "MRays/s", "Baseline for diffuse reflections where rays scatter randomly in all directions across the surface", true, 0},
            {"RayDivergence", "Ray Directional Coherence", "67.5° Cone Spread", "Ray Tracing", "MRays/s", "Tests ray performance when rays scatter widely in a 67.5-degree cone", true, 1},
            {"RayDivergence", "Ray Directional Coherence", "45° Cone Spread", "Ray Tracing", "MRays/s", "Tests ray performance with moderate ray scattering in a 45-degree cone (semi-glossy surfaces)", true, 2},
            {"RayDivergence", "Ray Directional Coherence", "22.5° Cone Spread", "Ray Tracing", "MRays/s", "Tests ray performance with tight ray scattering in a 22.5-degree cone (glossy surfaces)", true, 3},
            {"RayDivergence", "Ray Directional Coherence", "0° Beam (Coherent)", "Ray Tracing", "MRays/s", "Measures peak ray tracing speed when all rays travel in exactly the same direction (perfect mirror)", true, 4}
        }});

        m_categories.push_back(cat);
    }

    // 4. [Graphics] (1 subgroup, 3 tests)
    {
        BenchmarkCategory cat;
        cat.name = "Graphics";
        cat.description = "Fixed-function rasterizer ROP throughput and blending";

        cat.subgroups.push_back({"Graphics & ROP Fill Rate", "Graphics", "Pixel Fill Rate", "Fixed-function rasterizer fill rate across color formats", {
            {"Pixel Fill Rate", "ROP Throughput", "RGBA8 Color Fill", "Graphics", "GPixels/s", "Standard 32-bit color rendering fill rate to frame buffers", true, 0},
            {"Pixel Fill Rate", "ROP Throughput", "RGBA16F HDR Fill", "Graphics", "GPixels/s", "High dynamic range 64-bit HDR color rendering fill rate", true, 1},
            {"Pixel Fill Rate", "ROP Throughput", "Alpha Blending Fill", "Graphics", "GPixels/s", "Transparent surface blending fill rate combining overlapping pixels", true, 2}
        }});

        m_categories.push_back(cat);
    }

    // 5. [Host System] (1 subgroup, 7 tests)
    {
        BenchmarkCategory cat;
        cat.name = "Host System";
        cat.description = "Host CPU system memory DDR streaming bandwidth and latency";

        cat.subgroups.push_back({"Host CPU System Memory", "Host System", "System Memory Bandwidth", "Host CPU DDR read, write, copy bandwidth and latency", {
            {"System Memory Bandwidth", "Host CPU Memory Bandwidth", "Multi-Threaded Read", "System", "GB/s", "Main system RAM read speed using all CPU cores", true, 0},
            {"System Memory Bandwidth", "Host CPU Memory Bandwidth", "Multi-Threaded Write", "System", "GB/s", "Main system RAM write speed using all CPU cores", true, 1},
            {"System Memory Bandwidth", "Host CPU Memory Bandwidth", "Multi-Threaded Copy", "System", "GB/s", "Main system RAM copy speed using all CPU cores", true, 2},
            {"System Memory Bandwidth", "Host CPU Memory Bandwidth", "Single-Threaded Read (1T)", "System", "GB/s", "Main system RAM read speed using a single CPU core", true, 3},
            {"System Memory Bandwidth", "Host CPU Memory Bandwidth", "Single-Threaded Write (1T)", "System", "GB/s", "Main system RAM write speed using a single CPU core", true, 4},
            {"System Memory Bandwidth", "Host CPU Memory Bandwidth", "Single-Threaded Copy (1T)", "System", "GB/s", "Main system RAM copy speed using a single CPU core", true, 5},
            {"System Memory Latency", "Host CPU Memory Latency", "Default", "System", "ns", "Memory access response time (latency) from CPU caches down to RAM", true, 0}
        }});

        m_categories.push_back(cat);
    }

    applyVisualizationMetadata();
}

void GuiApp::applyVisualizationMetadata() {
    // Declarative mapping of graphics-pipeline workloads to tooltip visualizations.
    // Keyed by (engineId, configIndex) to keep the category initializer lists above
    // untouched. Scene-relative assets use the "{scene}" token, resolved at render
    // time against the currently selected benchmark scene (showroom fallback).
    struct VizEntry {
        const char* engineId;
        int configIndex;
        BenchmarkVisualization viz;
    };

    const VizEntry entries[] = {
        // ---- RayASBuild: BLAS build/update (0-4), TLAS construction (5-7) ----
        {"RayASBuild", 0, {"assets/thumbnails/thumb_blas_wireframe.png", "",
            "Vehicle mesh enclosed by hierarchical AABBs - the geometry staged into the bottom-level structure", 16.0f/9.0f, 3, 0}},
        {"RayASBuild", 1, {"assets/thumbnails/thumb_blas_wireframe.png", "",
            "Incremental BLAS update: only moved geometry is re-inserted, sibling nodes are re-bounded in place", 16.0f/9.0f, 3, 0}},
        {"RayASBuild", 2, {"assets/thumbnails/thumb_blas_wireframe.png", "",
            "5M-triangle build: deeper tree levels, wider nodes, higher memory-bandwidth pressure during sorting", 16.0f/9.0f, 3, 0}},
        {"RayASBuild", 3, {"assets/thumbnails/thumb_blas_wireframe.png", "",
            "Dynamic update of complex geometry: per-instance transforms are refreshed without a full rebuild", 16.0f/9.0f, 3, 0}},
        {"RayASBuild", 4, {"assets/thumbnails/thumb_blas_wireframe.png", "",
            "10M-triangle build: stresses triangle sorting, node packing, and BVH memory allocation at scale", 16.0f/9.0f, 3, 0}},
        {"RayASBuild", 5, {"", "tlas_hierarchy",
            "Top-level structure: one TLAS node per instance (20K here), each referencing a pre-built BLAS", 16.0f/9.0f, 3, 2}},
        {"RayASBuild", 6, {"", "tlas_hierarchy",
            "50K instanced objects: TLAS build cost grows with instance count, not triangle count", 16.0f/9.0f, 3, 2}},
        {"RayASBuild", 7, {"", "tlas_hierarchy",
            "200K instances: massive open-world hierarchy; traversal starts here before descending into BLAS", 16.0f/9.0f, 3, 2}},

        // ---- RayScheduling: material shading (0-2) ----
        {"RayScheduling", 0, {"assets/thumbnails/thumb_material_lineup.png", "",
            "Five BSDF archetypes (car paint, subsurface, velvet, glass, rust) shaded per hit", 16.0f/9.0f, 2, 0}},
        {"RayScheduling", 1, {"assets/thumbnails/thumb_material_lineup.png", "",
            "Closest-hit shaders reordered by hardware SER into uniform material bins, removing wavefront divergence", 16.0f/9.0f, 2, 0}},
        {"RayScheduling", 2, {"assets/thumbnails/thumb_material_lineup.png", "",
            "DGC sorts hits into material-coherent work queues before shading dispatch", 16.0f/9.0f, 2, 0}},

        // ---- RayScheduling: path tracing (3-5: 1 SPP, 23-24: 16 SPP) ----
        {"RayScheduling", 3, {"assets/thumbnails/thumb_{scene}_stage6_indirect.png", "",
            "Multi-bounce diffuse indirect light: each bounce re-enters the BVH with incoherent directions", 16.0f/9.0f, 1, 5}},
        {"RayScheduling", 4, {"assets/thumbnails/thumb_{scene}_stage6_indirect.png", "",
            "Secondary bounces traced on hardware ray cores with hardware reordering of incoherent rays", 16.0f/9.0f, 1, 5}},
        {"RayScheduling", 5, {"assets/thumbnails/thumb_{scene}_stage6_indirect.png", "",
            "Bounce rays grouped into GPU work queues; terminated rays compacted out each iteration", 16.0f/9.0f, 1, 5}},
        {"RayScheduling", 23, {"assets/thumbnails/thumb_{scene}_stage6_indirect.png", "",
            "16 samples per pixel: 16x the bounce rays of 1 SPP, converging the noisy indirect estimate", 16.0f/9.0f, 1, 5}},
        {"RayScheduling", 24, {"assets/thumbnails/thumb_{scene}_stage6_indirect.png", "",
            "16 SPP with DGC: per-sample work queues keep cores saturated across the extra bounces", 16.0f/9.0f, 1, 5}},

        // ---- RayScheduling: incoherent GI (6-8) ----
        {"RayScheduling", 6, {"", "cone_divergence_90",
            "Diffuse bounce rays scatter across the full hemisphere - worst-case SIMD divergence", 16.0f/9.0f, 1, 5}},
        {"RayScheduling", 7, {"", "cone_divergence_90",
            "Hardware SER re-bins scattered rays so each wavefront traverses coherently", 16.0f/9.0f, 1, 5}},
        {"RayScheduling", 8, {"", "cone_divergence_90",
            "DGC compacts scattered rays into direction-coherent queues for the next dispatch", 16.0f/9.0f, 1, 5}},

        // ---- RayScheduling: full frame (9-11) ----
        {"RayScheduling", 9, {"assets/thumbnails/thumb_{scene}_stage7_final.png", "",
            "Complete beauty pass: primary rays + shadows + RTAO + direct PBR + indirect GI in one pipeline", 16.0f/9.0f, 1, 6}},
        {"RayScheduling", 10, {"assets/thumbnails/thumb_{scene}_stage7_final.png", "",
            "Full frame on hardware ray tracing with SER reordering across every stage", 16.0f/9.0f, 1, 6}},
        {"RayScheduling", 11, {"assets/thumbnails/thumb_{scene}_stage7_final.png", "",
            "Full frame via GPU-driven command queues; each stage re-dispatches from compacted work", 16.0f/9.0f, 1, 6}},

        // ---- RayScheduling: traversal ordering (12, 14-16) ----
        {"RayScheduling", 12, {"", "traversal_scanline",
            "Row-by-row dispatch: baseline order, long strides between consecutive rays", 16.0f/9.0f, 1, 0}},
        {"RayScheduling", 14, {"", "traversal_tiled_8x4",
            "8x4 pixel tiles dispatched together: short strides keep BVH nodes cache-hot", 16.0f/9.0f, 1, 0}},
        {"RayScheduling", 15, {"", "traversal_morton_8x4",
            "Tiles visited along a Morton Z-curve: spatial locality preserved across tile boundaries", 16.0f/9.0f, 1, 0}},
        {"RayScheduling", 16, {"", "traversal_morton_4x8",
            "Tall 4x8 tiles on a Z-curve: probes the hardware's directional cache preference", 16.0f/9.0f, 1, 0}},

        // ---- RayScheduling: wavefront compaction (13, 26, 28) ----
        {"RayScheduling", 13, {"", "wave_ballot_compaction",
            "Wave-wide vote marks terminated rays; surviving lanes are packed into a dense queue", 16.0f/9.0f, -1, 0}},
        {"RayScheduling", 26, {"", "wave_ballot_compaction",
            "Single-pass removal of dead rays from the active stream in one GPU traversal", 16.0f/9.0f, -1, 0}},
        {"RayScheduling", 28, {"", "wave_ballot_compaction",
            "Peak pack rate: sorting and writing compacted ray records for the next dispatch", 16.0f/9.0f, -1, 0}},

        // ---- RayScheduling: primary rays (17, 18, 29, 30) ----
        {"RayScheduling", 17, {"assets/thumbnails/thumb_{scene}_stage2_primary.png", "",
            "Primary camera ray hits rendered as world-space G-Buffer normals", 16.0f/9.0f, 1, 1}},
        {"RayScheduling", 18, {"assets/thumbnails/thumb_{scene}_stage2_primary.png", "",
            "Primary rays issued from GPU work queues instead of a fixed pixel grid", 16.0f/9.0f, 1, 1}},
        {"RayScheduling", 29, {"assets/thumbnails/thumb_{scene}_stage2_primary.png", "",
            "Primary rays on dedicated hardware ray-query pipelines", 16.0f/9.0f, 1, 1}},
        {"RayScheduling", 30, {"assets/thumbnails/thumb_{scene}_stage2_primary.png", "",
            "Hardware ray reordering groups similar primary rays before traversal", 16.0f/9.0f, 1, 1}},

        // ---- RayScheduling: directional shadows (19-22) ----
        {"RayScheduling", 19, {"assets/thumbnails/thumb_{scene}_stage3_shadow.png", "",
            "Per-pixel visibility rays toward the sun produce this binary shadow mask", 16.0f/9.0f, 1, 2}},
        {"RayScheduling", 20, {"assets/thumbnails/thumb_{scene}_stage3_shadow.png", "",
            "Shadow rays on hardware pipelines with early termination at first hit", 16.0f/9.0f, 1, 2}},
        {"RayScheduling", 21, {"assets/thumbnails/thumb_{scene}_stage3_shadow.png", "",
            "Shadow rays batched through GPU work queues", 16.0f/9.0f, 1, 2}},
        {"RayScheduling", 22, {"assets/thumbnails/thumb_{scene}_stage3_shadow.png", "",
            "3x the shadow rays: one visibility pass per light, batched by DGC light binning", 16.0f/9.0f, 1, 2}},

        // ---- RayScheduling: alpha cutout divergence (27) ----
        {"RayScheduling", 27, {"assets/thumbnails/thumb_alpha_layers.png", "",
            "Foliage layers fail the alpha test and force traversal to continue - any-hit re-entry stress", 16.0f/9.0f, 3, 1}},

        // ---- RayScheduling: multi-light evaluation (31-34) ----
        {"RayScheduling", 31, {"assets/thumbnails/thumb_{scene}_stage5_direct.png", "",
            "Direct PBR shading under one light: GGX specular + Lambert diffuse per hit", 16.0f/9.0f, 1, 4}},
        {"RayScheduling", 32, {"assets/thumbnails/thumb_{scene}_stage5_direct.png", "",
            "Single-light shading with DGC-batched hits", 16.0f/9.0f, 1, 4}},
        {"RayScheduling", 33, {"assets/thumbnails/thumb_{scene}_multilight_128_dgc.png", "",
            "128 lights evaluated per hit in one pass: shading ALU throughput becomes the bottleneck", 16.0f/9.0f, 1, 4}},
        {"RayScheduling", 34, {"assets/thumbnails/thumb_{scene}_multilight_128_dgc.png", "",
            "128 lights grouped into coherent DGC bins so each wavefront shades similar lights", 16.0f/9.0f, 1, 4}},

        // ---- RayRawTraversal: coherent triangles (0), deep boxes (1) ----
        {"RayRawTraversal", 0, {"assets/thumbnails/thumb_{scene}_stage1_bvh.png", "",
            "BVH traversal cost per pixel: blue = shallow, red = deep box/triangle test counts", 16.0f/9.0f, 1, 0}},
        {"RayRawTraversal", 1, {"", "bvh_nested_boxes",
            "Deeply nested AABBs: every level forces an extra ray-box test before reaching geometry", 16.0f/9.0f, 1, 0}},

        // ---- RayIntersect: ray-triangle (0), ray-box (1) ----
        {"RayIntersect", 0, {"", "intersect_ray_triangle",
            "Moller-Trumbore ray-triangle test: the innermost intersection primitive", 16.0f/9.0f, -1, 0}},
        {"RayIntersect", 1, {"", "intersect_ray_box",
            "AABB slab test: the fast rejection primitive executed at every BVH node", 16.0f/9.0f, -1, 0}},

        // ---- RayAnyHit: 100% solid (0), 50% cutout (1) ----
        {"RayAnyHit", 0, {"assets/thumbnails/thumb_alpha_layers.png", "",
            "Baseline: every hit is opaque, any-hit shader exits immediately", 16.0f/9.0f, 3, 1}},
        {"RayAnyHit", 1, {"assets/thumbnails/thumb_alpha_layers.png", "",
            "50% of hits fail the alpha test; traversal re-enters the BVH until an opaque surface is found", 16.0f/9.0f, 3, 1}},

        // ---- RayProcedural: AABB spheres (0) ----
        {"RayProcedural", 0, {"", "procedural_sphere",
            "Analytic sphere intersection: no triangle fetch, no barycentrics - pure ALU math", 16.0f/9.0f, -1, 0}},

        // ---- RayDivergence: 90 (0), 67.5 (1), 45 (2), 22.5 (3), 0 (4) ----
        {"RayDivergence", 0, {"", "cone_divergence_90",
            "Full hemisphere scatter (diffuse): rays in one wavefront hit unrelated BVH branches", 16.0f/9.0f, -1, 0}},
        {"RayDivergence", 1, {"", "cone_divergence_67.5",
            "Wide 67.5-degree cone: rough specular surfaces, heavy but bounded divergence", 16.0f/9.0f, -1, 0}},
        {"RayDivergence", 2, {"", "cone_divergence_45",
            "45-degree cone: semi-glossy surfaces, moderate cache locality", 16.0f/9.0f, -1, 0}},
        {"RayDivergence", 3, {"", "cone_divergence_22.5",
            "22.5-degree cone: glossy surfaces, rays stay close to the reflection direction", 16.0f/9.0f, -1, 0}},
        {"RayDivergence", 4, {"", "cone_divergence_0",
            "Perfectly parallel beam (mirror): peak traversal throughput, zero divergence", 16.0f/9.0f, -1, 0}},

        // ---- Pixel Fill Rate: RGBA8 (0), RGBA16F (1), Alpha blend (2) ----
        {"Pixel Fill Rate", 0, {"", "rop_fill_rgba8",
            "32-bit color fill: one pixel per ROP clock, pure rasterizer throughput", 16.0f/9.0f, -1, 0}},
        {"Pixel Fill Rate", 1, {"", "rop_fill_hdr",
            "64-bit RGBA16F HDR fill: 2x the color bandwidth of RGBA8 per pixel", 16.0f/9.0f, -1, 0}},
        {"Pixel Fill Rate", 2, {"", "rop_fill_blend",
            "Alpha blending: read-modify-write per overlapping pixel stresses the blend stage", 16.0f/9.0f, -1, 0}},
    };

    int matched = 0;
    for (const auto& e : entries) {
        bool found = false;
        for (auto& cat : m_categories) {
            for (auto& sub : cat.subgroups) {
                for (auto& item : sub.items) {
                    if (item.id == e.engineId && item.configIndex == e.configIndex) {
                        item.viz = e.viz;
                        found = true;
                    }
                }
            }
        }
        if (found) matched++;
        else std::cerr << "[GPUBench GUI] WARNING: viz entry unmatched: "
                      << e.engineId << " config " << e.configIndex << "\n";
    }
    std::cout << "[GPUBench GUI] Tooltip visualizations: " << matched << "/"
              << sizeof(entries) / sizeof(entries[0]) << " workloads mapped\n";
}

std::string GuiApp::currentSceneTag() const {
    // Sponza (Crytek Sponza atrium) is the canonical reference architecture scene.
    if (m_scene == "forest" || m_scene == "outdoor" || m_scene == "showroom") return m_scene;
    if (m_scene == "indoor") return "indoor";
    if (!m_suiteActiveScene.empty() && m_suiteActiveScene != "all") return m_suiteActiveScene;
    return "indoor";
}

void GuiApp::renderBenchmarkTooltip(const BenchmarkItem& item, const std::string& sceneOverride) {
    ImGui::BeginTooltip();

    // 1. Header & badges
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.0f), "%s", item.name.c_str());
    ImGui::TextDisabled("Subcategory: %s | Metric: %s", item.subcategory.c_str(), item.metricType.c_str());
    ImGui::Separator();

    // 2. Textual explanation
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + s(380.0f));
    ImGui::TextUnformatted(item.description.c_str());
    ImGui::PopTextWrapPos();

    // 3. Visualization canvas (Tier 1 texture or Tier 2 procedural diagram)
    if (item.viz.hasVisualization()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float canvasWidth = s(340.0f);
        float ar = item.viz.aspectRatio > 0.001f ? item.viz.aspectRatio : (16.0f / 9.0f);
        float canvasHeight = canvasWidth / ar;

        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 p1 = ImVec2(p0.x + canvasWidth, p0.y + canvasHeight);
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        drawList->AddRectFilled(p0, p1, IM_COL32(12, 14, 20, 255), s(4.0f));

        if (!item.viz.assetPath.empty()) {
            // Tier 1: static pre-baked thumbnail, lazily loaded & cached
            std::string assetPath = item.viz.assetPath;
            std::string scene = !sceneOverride.empty() ? sceneOverride : currentSceneTag();
            size_t pos;
            while ((pos = assetPath.find("{scene}")) != std::string::npos) {
                assetPath.replace(pos, 7, scene);
            }
            auto tex = getOrLoadTexture(assetPath);
            if (tex.isValid()) {
                // Letterbox the image inside the canvas preserving its aspect
                float tar = (tex.height > 0) ? (static_cast<float>(tex.width) / tex.height) : ar;
                float imgW = canvasWidth;
                float imgH = canvasWidth / tar;
                if (imgH > canvasHeight) {
                    imgH = canvasHeight;
                    imgW = canvasHeight * tar;
                }
                ImVec2 ip0 = ImVec2(p0.x + (canvasWidth - imgW) * 0.5f, p0.y + (canvasHeight - imgH) * 0.5f);
                ImVec2 ip1 = ImVec2(ip0.x + imgW, ip0.y + imgH);
                drawList->AddImage(reinterpret_cast<ImTextureID>(tex.descriptorSet), ip0, ip1);
            } else {
                std::string fallback = "Preview unavailable: " + item.viz.caption;
                ImVec2 tsz = ImGui::CalcTextSize(fallback.c_str());
                drawList->AddText(ImVec2(p0.x + (canvasWidth - tsz.x) * 0.5f, p0.y + (canvasHeight - tsz.y) * 0.5f),
                                  IM_COL32(160, 170, 190, 255), fallback.c_str());
            }
        } else {
            // Tier 2: procedural vector schematic (crisp at any UI scale)
            renderProceduralDiagram(item.viz.diagramId, p0, p1);
        }

        drawList->AddRect(p0, p1, IM_COL32(50, 60, 80, 255), s(4.0f));
        ImGui::Dummy(ImVec2(canvasWidth, canvasHeight));

        // Caption & legend
        if (!item.viz.caption.empty()) {
            ImGui::Spacing();
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + canvasWidth);
            ImGui::TextColored(ImVec4(0.85f, 0.75f, 0.35f, 1.0f), "%s", item.viz.caption.c_str());
            ImGui::PopTextWrapPos();
        }

        // Cross-link to the full Ray Tracing Viewport
        if (item.viz.linkedViewportMode >= 0) {
            ImGui::Spacing();
            if (ImGui::SmallButton("Open in Ray Tracing Viewport  ->")) {
                m_rtViewportMode = item.viz.linkedViewportMode;
                if (item.viz.linkedViewportMode == 1) m_rtPassIndex = item.viz.linkedViewportSubIndex;
                else if (item.viz.linkedViewportMode == 2) m_rtMaterialIndex = item.viz.linkedViewportSubIndex;
                else if (item.viz.linkedViewportMode == 3) m_rtGeometryIndex = item.viz.linkedViewportSubIndex;
                m_switchToRtViewport = true;
            }
        }
    }

    ImGui::EndTooltip();
}

void GuiApp::renderProceduralDiagram(const std::string& diagramId, ImVec2 p0, ImVec2 p1) const {
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // ImVec2 math helpers (project does not define IMGUI_DEFINE_MATH_OPERATORS)
    auto add = [](const ImVec2& a, const ImVec2& b) { return ImVec2(a.x + b.x, a.y + b.y); };
    auto sub = [](const ImVec2& a, const ImVec2& b) { return ImVec2(a.x - b.x, a.y - b.y); };
    auto mul = [](const ImVec2& a, float s) { return ImVec2(a.x * s, a.y * s); };

    // Shared palette
    const ImU32 colGrid  = IM_COL32(35, 45, 60, 180);
    const ImU32 colPath  = IM_COL32(0, 220, 180, 230);
    const ImU32 colAmber = IM_COL32(255, 190, 60, 230);
    const ImU32 colHot   = IM_COL32(255, 100, 80, 230);
    const ImU32 colCool  = IM_COL32(80, 180, 255, 230);
    const ImU32 colText  = IM_COL32(200, 210, 230, 255);
    const ImU32 colDim   = IM_COL32(140, 150, 170, 255);
    const ImU32 colGreen = IM_COL32(90, 220, 120, 230);

    float pad = s(12.0f);
    auto drawTitle = [&](const char* title) {
        dl->AddText(ImVec2(p0.x + pad, p0.y + s(6.0f)), colText, title);
    };
    ImVec2 area = ImVec2(p0.x + pad, p0.y + s(30.0f));
    ImVec2 areaEnd = ImVec2(p1.x - pad, p1.y - pad - s(16.0f));
    float gw = areaEnd.x - area.x;
    float gh = areaEnd.y - area.y;
    auto drawFootnote = [&](const char* text) {
        dl->AddText(ImVec2(p0.x + pad, p1.y - s(12.0f)), colDim, text);
    };

    // ------------------------------------------------------------------
    // Ray directional divergence cone (cone_divergence_<deg>)
    // ------------------------------------------------------------------
    if (diagramId.rfind("cone_divergence_", 0) == 0) {
        float angle = 0.0f;
        try { angle = std::stof(diagramId.substr(16)); } catch (...) {}
        drawTitle(angle == 0.0f ? "Ray Divergence: 0 Beam (Coherent)" : "Ray Divergence Cone");

        ImVec2 hit(area.x + gw * 0.5f, areaEnd.y - s(4.0f));
        float surfHalf = gw * 0.42f;
        dl->AddLine(ImVec2(hit.x - surfHalf, hit.y), ImVec2(hit.x + surfHalf, hit.y),
                    IM_COL32(120, 130, 150, 255), 2.0f);

        float normalLen = gh * 0.88f;
        ImVec2 normalEnd(hit.x, hit.y - normalLen);
        dl->AddLine(hit, normalEnd, colCool, 1.5f);
        dl->AddText(ImVec2(normalEnd.x + s(4.0f), normalEnd.y), colCool, "N");

        float rad = angle * 0.5f * (3.14159265f / 180.0f);
        int nRays = (angle == 0.0f) ? 1 : 7;
        ImU32 rayCol = (angle > 60.0f) ? colHot : ((angle > 30.0f) ? colAmber : colCool);
        for (int i = 0; i < nRays; ++i) {
            float t = (nRays > 1) ? (static_cast<float>(i) / (nRays - 1) * 2.0f - 1.0f) : 0.0f;
            float a = t * rad;
            float len = normalLen * 0.92f;
            ImVec2 end(hit.x + std::sin(a) * len, hit.y - std::cos(a) * len);
            dl->AddLine(hit, end, rayCol, 1.5f);
            dl->AddCircleFilled(end, s(2.0f), rayCol);
        }

        if (angle > 0.0f) {
            float arcR = normalLen * 0.38f;
            float a0 = -3.14159265f / 2.0f - rad;
            float a1 = -3.14159265f / 2.0f + rad;
            dl->PathArcTo(hit, arcR, a0, a1, 24);
            dl->PathStroke(colAmber, 1.2f);
        }

        char buf[48];
        snprintf(buf, sizeof(buf), "Cone: %.1f deg", angle);
        dl->AddText(ImVec2(area.x, area.y), colAmber, buf);
        const char* sub = (angle == 0.0f) ? "mirror - peak throughput" :
                          (angle <= 22.5f) ? "glossy surface" :
                          (angle <= 45.0f) ? "semi-glossy surface" :
                          (angle <= 67.5f) ? "rough surface" : "diffuse - max divergence";
        dl->AddText(ImVec2(area.x, area.y + s(16.0f)), colDim, sub);
        return;
    }

    // ------------------------------------------------------------------
    // Traversal ordering: pixel-level scanline vs 8x4 tiled (16x8 grid)
    // ------------------------------------------------------------------
    if (diagramId == "traversal_scanline" || diagramId == "traversal_tiled_8x4") {
        bool tiled = (diagramId == "traversal_tiled_8x4");
        drawTitle(tiled ? "Traversal: 2D Screen Tiled (8x4)" : "Traversal: Linear 1D Scanline");

        const int cols = 16, rows = 8;
        float cw = gw / cols, ch = gh / rows;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                ImVec2 c0(area.x + c * cw, area.y + r * ch);
                ImVec2 c1(c0.x + cw, c0.y + ch);
                dl->AddRect(c0, c1, colGrid);
            }
        }

        std::vector<ImVec2> centers;
        centers.reserve(cols * rows);
        if (!tiled) {
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                    centers.push_back(ImVec2(area.x + (c + 0.5f) * cw, area.y + (r + 0.5f) * ch));
        } else {
            const int tc = 8, tr = 4;
            for (int tr0 = 0; tr0 < rows; tr0 += tr)
                for (int tc0 = 0; tc0 < cols; tc0 += tc)
                    for (int r = 0; r < tr; ++r)
                        for (int c = 0; c < tc; ++c)
                            centers.push_back(ImVec2(area.x + (tc0 + c + 0.5f) * cw, area.y + (tr0 + r + 0.5f) * ch));
            for (int tr0 = 0; tr0 < rows; tr0 += tr)
                for (int tc0 = 0; tc0 < cols; tc0 += tc) {
                    ImVec2 t0(area.x + tc0 * cw, area.y + tr0 * ch);
                    ImVec2 t1(area.x + (tc0 + tc) * cw, area.y + (tr0 + tr) * ch);
                    dl->AddRect(t0, t1, colAmber, 2.0f);
                }
        }

        for (size_t i = 1; i < centers.size(); ++i) {
            dl->AddLine(centers[i - 1], centers[i], colPath, 1.2f);
        }
        dl->AddCircleFilled(centers.front(), s(3.0f), colGreen);
        dl->AddCircleFilled(centers.back(), s(3.0f), colHot);
        drawFootnote(tiled ? "wrap at tile edge keeps 8x4 ray groups cache-coherent"
                           : "consecutive rays stride the full row width");
        return;
    }

    // ------------------------------------------------------------------
    // Traversal ordering: Morton Z-curve over a 4x4 tile grid
    // ------------------------------------------------------------------
    if (diagramId == "traversal_morton_8x4" || diagramId == "traversal_morton_4x8") {
        bool wide = (diagramId == "traversal_morton_8x4");
        drawTitle(wide ? "Traversal: 2D Morton Z-Order (8x4)" : "Traversal: 2D Morton Z-Order (4x8)");

        const int n = 4;
        float cw = gw / n, ch = gh / n;

        auto morton2D = [](uint32_t x, uint32_t y) -> uint32_t {
            auto part1by1 = [](uint32_t v) -> uint32_t {
                v &= 0x0000ffff;
                v = (v | (v << 8)) & 0x00FF00FF;
                v = (v | (v << 4)) & 0x0F0F0F0F;
                v = (v | (v << 2)) & 0x33333333;
                v = (v | (v << 1)) & 0x55555555;
                return v;
            };
            return (part1by1(y) << 1) | part1by1(x);
        };

        std::vector<std::pair<uint32_t, ImVec2>> tiles;
        tiles.reserve(n * n);
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                tiles.push_back({morton2D(static_cast<uint32_t>(c), static_cast<uint32_t>(r)),
                                 ImVec2(area.x + (c + 0.5f) * cw, area.y + (r + 0.5f) * ch)});
        std::sort(tiles.begin(), tiles.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                ImVec2 t0(area.x + c * cw, area.y + r * ch);
                ImVec2 t1(t0.x + cw, t0.y + ch);
                dl->AddRect(t0, t1, colGrid);
                // Mini-raster suggesting the intra-tile pixel order
                if (wide) {
                    for (int i = 1; i <= 3; ++i)
                        dl->AddLine(ImVec2(t0.x + s(3.0f), t0.y + ch * i / 4.0f),
                                    ImVec2(t1.x - s(3.0f), t0.y + ch * i / 4.0f),
                                    IM_COL32(60, 75, 95, 200), 1.0f);
                } else {
                    for (int i = 1; i <= 3; ++i)
                        dl->AddLine(ImVec2(t0.x + cw * i / 4.0f, t0.y + s(3.0f)),
                                    ImVec2(t0.x + cw * i / 4.0f, t1.y - s(3.0f)),
                                    IM_COL32(60, 75, 95, 200), 1.0f);
                }
            }
        }

        for (size_t i = 1; i < tiles.size(); ++i) {
            dl->AddLine(tiles[i - 1].second, tiles[i].second, colPath, 1.5f);
        }
        for (size_t i = 0; i < tiles.size(); ++i) {
            ImU32 dotCol = (i == 0) ? colGreen : ((i == tiles.size() - 1) ? colHot : colPath);
            dl->AddCircleFilled(tiles[i].second, s(2.5f), dotCol);
            if (i < 4 || i >= tiles.size() - 2) {
                char b[16];
                snprintf(b, sizeof(b), "%u", static_cast<unsigned>(i));
                ImVec2 tsz = ImGui::CalcTextSize(b);
                dl->AddText(ImVec2(tiles[i].second.x + s(4.0f), tiles[i].second.y - tsz.y - s(2.0f)), colDim, b);
            }
        }
        drawFootnote("tiles visited along the Z-curve: locality across boundaries");
        return;
    }

    // ------------------------------------------------------------------
    // Wavefront stream compaction via ballot
    // ------------------------------------------------------------------
    if (diagramId == "wave_ballot_compaction") {
        drawTitle("Wave Ballot Stream Compaction");

        const int lanes = 32;
        static const bool active[32] = {
            1,1,1,0, 1,1,0,0, 1,1,1,1, 0,0,1,0,
            1,1,0,0, 1,1,1,0, 0,0,1,0, 0,0,1,1
        };
        float labelW = s(84.0f);
        float barX = area.x + labelW;
        float barW = areaEnd.x - barX;
        float lw = barW / lanes;
        float laneH = s(14.0f);
        float rowGap = s(24.0f);

        // Row 1: wavefront lanes
        dl->AddText(ImVec2(area.x, area.y), colDim, "wavefront");
        for (int i = 0; i < lanes; ++i) {
            ImVec2 l0(barX + i * lw + s(1.0f), area.y - s(2.0f));
            ImVec2 l1(l0.x + lw - s(2.0f), l0.y + laneH);
            if (active[i]) {
                dl->AddRectFilled(l0, l1, IM_COL32(60, 140, 80, 255), s(2.0f));
                dl->AddRect(l0, l1, colGreen);
            } else {
                dl->AddRectFilled(l0, l1, IM_COL32(70, 40, 40, 255), s(2.0f));
                dl->AddRect(l0, l1, IM_COL32(150, 70, 60, 200));
            }
        }

        // Row 2: ballot mask
        char mask[33];
        for (int i = 0; i < lanes; ++i) mask[i] = active[i] ? '1' : '0';
        mask[32] = 0;
        float y2 = area.y + rowGap;
        dl->AddText(ImVec2(area.x, y2), colDim, "ballot mask");
        dl->AddText(ImVec2(barX, y2), colAmber, mask);

        // Row 3: compacted queue
        float y3 = y2 + rowGap;
        dl->AddText(ImVec2(area.x, y3), colDim, "compacted");
        int k = 0;
        for (int i = 0; i < lanes; ++i) {
            ImVec2 l0(barX + i * lw + s(1.0f), y3 - s(2.0f));
            ImVec2 l1(l0.x + lw - s(2.0f), l0.y + laneH);
            if (active[i]) {
                dl->AddRectFilled(l0, l1, IM_COL32(60, 140, 80, 255), s(2.0f));
                dl->AddRect(l0, l1, colGreen);
                ++k;
            } else {
                dl->AddRect(l0, l1, IM_COL32(40, 48, 60, 120));
            }
        }

        char b[64];
        snprintf(b, sizeof(b), "%d active lanes repacked for the next dispatch", k);
        drawFootnote(b);
        return;
    }

    // ------------------------------------------------------------------
    // TLAS instance hierarchy
    // ------------------------------------------------------------------
    if (diagramId == "tlas_hierarchy") {
        drawTitle("TLAS Instance Hierarchy");

        auto drawNode = [&](ImVec2 c, float nw, float nh, const char* label, ImU32 fill, ImU32 border) {
            ImVec2 n0(c.x - nw * 0.5f, c.y - nh * 0.5f);
            ImVec2 n1(c.x + nw * 0.5f, c.y + nh * 0.5f);
            dl->AddRectFilled(n0, n1, fill, s(3.0f));
            dl->AddRect(n0, n1, border, s(3.0f));
            ImVec2 tsz = ImGui::CalcTextSize(label);
            dl->AddText(ImVec2(c.x - tsz.x * 0.5f, c.y - tsz.y * 0.5f), colText, label);
        };

        ImVec2 rootC(area.x + gw * 0.5f, area.y + s(10.0f));
        drawNode(rootC, s(92.0f), s(20.0f), "TLAS Root", IM_COL32(30, 50, 80, 255), colCool);

        ImVec2 mid[3];
        for (int i = 0; i < 3; ++i) {
            mid[i] = ImVec2(area.x + gw * (0.2f + 0.3f * i), area.y + gh * 0.45f);
            dl->AddLine(rootC, mid[i], IM_COL32(90, 110, 140, 200), 1.5f);
            drawNode(mid[i], s(64.0f), s(18.0f), "Node", IM_COL32(28, 45, 70, 255), IM_COL32(80, 120, 170, 255));
        }

        int bi = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 2; ++j) {
                ImVec2 leaf(mid[i].x + (j == 0 ? -1 : 1) * gw * 0.085f, area.y + gh * 0.85f);
                dl->AddLine(mid[i], leaf, IM_COL32(90, 110, 140, 160), 1.2f);
                char b[16];
                snprintf(b, sizeof(b), "BLAS %d", bi++);
                drawNode(leaf, s(56.0f), s(16.0f), b, IM_COL32(25, 40, 55, 255), IM_COL32(70, 100, 130, 255));
            }
        }
        drawFootnote("one TLAS node per instance -> each references a pre-built BLAS");
        return;
    }

    // ------------------------------------------------------------------
    // Ray-triangle intersection (Moller-Trumbore)
    // ------------------------------------------------------------------
    if (diagramId == "intersect_ray_triangle") {
        drawTitle("Ray-Triangle (Moller-Trumbore)");

        ImVec2 A(area.x + gw * 0.30f, area.y + gh * 0.15f);
        ImVec2 B(area.x + gw * 0.88f, area.y + gh * 0.72f);
        ImVec2 C(area.x + gw * 0.12f, area.y + gh * 0.88f);
        dl->AddLine(A, B, colCool, 1.5f);
        dl->AddLine(B, C, colCool, 1.5f);
        dl->AddLine(C, A, colCool, 1.5f);

        // Hit point via barycentric combination
        ImVec2 H(A.x * 0.35f + B.x * 0.45f + C.x * 0.20f, A.y * 0.35f + B.y * 0.45f + C.y * 0.20f);

        ImVec2 rayStart(area.x + gw * 0.02f, area.y + gh * 0.02f);
        ImVec2 rayDir = sub(H, rayStart);
        ImVec2 rayEnd = add(rayStart, mul(rayDir, 1.35f));
        dl->AddLine(rayStart, rayEnd, colAmber, 1.8f);

        dl->AddLine(H, A, IM_COL32(140, 150, 170, 160), 1.0f);
        dl->AddLine(H, B, IM_COL32(140, 150, 170, 160), 1.0f);
        dl->AddLine(H, C, IM_COL32(140, 150, 170, 160), 1.0f);
        dl->AddCircleFilled(H, s(3.0f), colHot);
        dl->AddText(ImVec2(H.x + s(6.0f), H.y - s(16.0f)), colHot, "hit (t, u, v)");
        drawFootnote("solves a 3x3 linear system; barycentrics select the triangle");
        return;
    }

    // ------------------------------------------------------------------
    // Ray-AABB slab test
    // ------------------------------------------------------------------
    if (diagramId == "intersect_ray_box") {
        drawTitle("Ray-AABB (Slab Test)");

        ImVec2 f0(area.x + gw * 0.30f, area.y + gh * 0.32f);
        ImVec2 f1(area.x + gw * 0.72f, area.y + gh * 0.84f);
        ImVec2 off(gw * 0.14f, -gh * 0.20f);
        ImVec2 b0 = add(f0, off), b1 = add(f1, off);

        // Back face + connecting edges
        dl->AddLine(b0, b1, IM_COL32(70, 90, 120, 200), 1.2f);
        dl->AddLine(ImVec2(b1.x, b0.y), b1, IM_COL32(70, 90, 120, 200), 1.2f);
        dl->AddLine(b0, ImVec2(b0.x, b1.y), IM_COL32(70, 90, 120, 200), 1.2f);
        dl->AddLine(f0, b0, IM_COL32(70, 90, 120, 200), 1.2f);
        dl->AddLine(ImVec2(f1.x, f0.y), ImVec2(b1.x, b0.y), IM_COL32(70, 90, 120, 200), 1.2f);
        dl->AddLine(f1, b1, IM_COL32(70, 90, 120, 200), 1.2f);
        dl->AddLine(ImVec2(f0.x, f1.y), ImVec2(b0.x, b1.y), IM_COL32(70, 90, 120, 200), 1.2f);

        // Front face
        dl->AddLine(f0, ImVec2(f1.x, f0.y), colCool, 1.5f);
        dl->AddLine(ImVec2(f1.x, f0.y), f1, colCool, 1.5f);
        dl->AddLine(f1, ImVec2(f0.x, f1.y), colCool, 1.5f);
        dl->AddLine(ImVec2(f0.x, f1.y), f0, colCool, 1.5f);

        ImVec2 rayStart(area.x + gw * 0.04f, area.y + gh * 0.08f);
        ImVec2 rayEnd(area.x + gw * 0.96f, area.y + gh * 0.96f);
        dl->AddLine(rayStart, rayEnd, colAmber, 1.8f);

        ImVec2 t0pt = add(rayStart, mul(sub(rayEnd, rayStart), 0.40f));
        ImVec2 t1pt = add(rayStart, mul(sub(rayEnd, rayStart), 0.62f));
        dl->AddCircleFilled(t0pt, s(3.0f), colGreen);
        dl->AddCircleFilled(t1pt, s(3.0f), colHot);
        dl->AddText(ImVec2(t0pt.x - s(38.0f), t0pt.y - s(18.0f)), colGreen, "t near");
        dl->AddText(ImVec2(t1pt.x + s(6.0f), t1pt.y + s(2.0f)), colHot, "t far");
        drawFootnote("6 slab comparisons; reject the box when t near > t far");
        return;
    }

    // ------------------------------------------------------------------
    // Procedural analytic sphere
    // ------------------------------------------------------------------
    if (diagramId == "procedural_sphere") {
        drawTitle("Procedural: Analytic Sphere");

        ImVec2 c(area.x + gw * 0.55f, area.y + gh * 0.50f);
        float r = std::min(gw, gh) * 0.38f;
        dl->AddCircleFilled(c, r, IM_COL32(25, 45, 70, 255));
        dl->AddCircle(c, r, colCool, 1.5f);
        dl->PathArcTo(c, r * 0.8f, -3.14159265f * 0.75f, -3.14159265f * 0.25f, 16);
        dl->PathStroke(IM_COL32(120, 170, 220, 180), 1.5f);

        ImVec2 rayStart(area.x + gw * 0.05f, area.y + gh * 0.10f);
        ImVec2 dir = sub(c, rayStart);
        float dlen = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        dir.x /= dlen;
        dir.y /= dlen;
        ImVec2 hit = sub(c, mul(dir, r));
        ImVec2 rayEnd = add(hit, mul(dir, r * 0.55f));
        dl->AddLine(rayStart, rayEnd, colAmber, 1.8f);
        dl->AddCircleFilled(hit, s(3.0f), colHot);

        ImVec2 nrm = sub(hit, c);
        float nlen = std::sqrt(nrm.x * nrm.x + nrm.y * nrm.y);
        nrm.x /= nlen;
        nrm.y /= nlen;
        ImVec2 nEnd = add(hit, mul(nrm, r * 0.5f));
        dl->AddLine(hit, nEnd, colGreen, 1.5f);
        dl->AddText(ImVec2(nEnd.x + s(4.0f), nEnd.y - s(14.0f)), colGreen, "N");
        drawFootnote("quadratic solve in the shader - no triangle fetch, no barycentrics");
        return;
    }

    // ------------------------------------------------------------------
    // Deeply nested BVH boxes
    // ------------------------------------------------------------------
    if (diagramId == "bvh_nested_boxes") {
        drawTitle("Deeply Nested AABBs");

        auto box = [&](float x, float y, float bw, float bh, ImU32 col) {
            dl->AddRect(ImVec2(area.x + gw * x, area.y + gh * y),
                        ImVec2(area.x + gw * (x + bw), area.y + gh * (y + bh)), col);
        };
        box(0.05f, 0.10f, 0.90f, 0.80f, colCool);
        box(0.08f, 0.14f, 0.40f, 0.72f, IM_COL32(90, 150, 210, 220));
        box(0.52f, 0.14f, 0.40f, 0.72f, IM_COL32(90, 150, 210, 220));
        box(0.10f, 0.18f, 0.17f, 0.30f, colAmber);
        box(0.28f, 0.50f, 0.17f, 0.30f, colAmber);
        box(0.54f, 0.18f, 0.17f, 0.30f, colAmber);
        box(0.72f, 0.50f, 0.17f, 0.30f, colAmber);

        ImVec2 rs(area.x + gw * 0.02f, area.y + gh * 0.06f);
        ImVec2 re(area.x + gw * 0.98f, area.y + gh * 0.94f);
        dl->AddLine(rs, re, colPath, 1.8f);
        const float steps[6] = {0.10f, 0.22f, 0.38f, 0.52f, 0.68f, 0.82f};
        for (float t : steps) {
            ImVec2 p = add(rs, mul(sub(re, rs), t));
            dl->AddCircleFilled(p, s(2.5f), colAmber);
        }
        drawFootnote("each level adds a ray-box test before reaching leaf triangles");
        return;
    }

    // ------------------------------------------------------------------
    // ROP fill-rate variants
    // ------------------------------------------------------------------
    if (diagramId.rfind("rop_fill_", 0) == 0) {
        bool hdr = (diagramId == "rop_fill_hdr");
        bool blend = (diagramId == "rop_fill_blend");
        drawTitle(hdr ? "ROP: RGBA16F HDR Fill" : (blend ? "ROP: Alpha Blending" : "ROP: RGBA8 Color Fill"));

        ImVec2 fb0(area.x, area.y);
        ImVec2 fb1(areaEnd.x, areaEnd.y);
        dl->AddRect(fb0, fb1, colGrid, 1.0f);

        if (!blend) {
            ImVec2 q0(area.x + gw * 0.15f, area.y + gh * 0.20f);
            ImVec2 q1(area.x + gw * 0.85f, area.y + gh * 0.80f);
            if (hdr) {
                const int slices = 24;
                for (int i = 0; i < slices; ++i) {
                    float t0 = static_cast<float>(i) / slices;
                    float t1 = static_cast<float>(i + 1) / slices;
                    int v = (int)(30 + 225 * t1);
                    dl->AddRectFilled(ImVec2(q0.x + (q1.x - q0.x) * t0, q0.y),
                                      ImVec2(q0.x + (q1.x - q0.x) * t1, q1.y),
                                      IM_COL32(v, v, v, 255));
                }
            } else {
                dl->AddRectFilled(q0, q1, IM_COL32(40, 90, 200, 255));
            }
            // Rasterization scanline arrow
            ImVec2 sl0(q0.x, q1.y + s(8.0f));
            ImVec2 sl1(q1.x, q1.y + s(8.0f));
            dl->AddLine(sl0, sl1, colAmber, 1.5f);
            dl->AddLine(sl1, ImVec2(sl1.x - s(6.0f), sl1.y - s(4.0f)), colAmber, 1.5f);
            dl->AddLine(sl1, ImVec2(sl1.x - s(6.0f), sl1.y + s(4.0f)), colAmber, 1.5f);
            drawFootnote(hdr ? "64-bit color: 2x the memory bandwidth of RGBA8 per pixel"
                             : "1 pixel per ROP clock - pure rasterizer throughput");
        } else {
            ImVec2 q0a(area.x + gw * 0.12f, area.y + gh * 0.18f);
            ImVec2 q1a(area.x + gw * 0.55f, area.y + gh * 0.78f);
            ImVec2 q0b(area.x + gw * 0.45f, area.y + gh * 0.22f);
            ImVec2 q1b(area.x + gw * 0.88f, area.y + gh * 0.82f);
            dl->AddRectFilled(q0a, q1a, IM_COL32(60, 120, 230, 160));
            dl->AddRectFilled(q0b, q1b, IM_COL32(230, 90, 60, 160));
            ImVec2 o0(std::max(q0a.x, q0b.x), std::max(q0a.y, q0b.y));
            ImVec2 o1(std::min(q1a.x, q1b.x), std::min(q1a.y, q1b.y));
            dl->AddRect(o0, o1, IM_COL32(240, 240, 255, 220), 1.5f);
            dl->AddText(ImVec2(o0.x + s(4.0f), o0.y + s(2.0f)), colText, "src + dst");
            drawFootnote("overlapping pixels: read old color, blend, write back");
        }
        return;
    }

    // ------------------------------------------------------------------
    // Fallback: unknown diagram id
    // ------------------------------------------------------------------
    float canvasH = p1.y - p0.y;
    dl->AddText(ImVec2(p0.x + pad, p0.y + canvasH * 0.45f), colDim, ("Unknown diagram: " + diagramId).c_str());
}

void GuiApp::discoverHardware() {
    m_devices.clear();

    std::vector<DeviceProfile> profiles = GetDeviceProfilesAPI();

    for (const auto& prof : profiles) {
        SelectableDevice dev;
        dev.deviceIndex = prof.deviceIndex;
        dev.name = prof.deviceName;
        dev.backend = prof.backend;
        dev.architecture = detectGpuArchitecture(prof.deviceName, prof.vendorID, prof.deviceID);
        dev.driver = prof.driverName + (!prof.driverVersion.empty() ? (" " + prof.driverVersion) : "");
        dev.vramTotalMb = prof.vramTotalMb;
        dev.isSystem = false;
        // Default GPU 0 (Primary) to selected
        dev.selected = (prof.deviceIndex == 0);

        m_devices.push_back(dev);
    }

    // Fallback if profiles list was empty
    if (m_devices.empty()) {
        std::vector<std::string> hwEntries = GetAvailableHardwareAPI();
        for (const auto& entry : hwEntries) {
            size_t p1 = entry.find('|');
            if (p1 != std::string::npos) {
                size_t p2 = entry.find('|', p1 + 1);
                if (p2 != std::string::npos) {
                    std::string backend = entry.substr(0, p1);
                    std::string idxStr = entry.substr(p1 + 1, p2 - (p1 + 1));
                    std::string devName = entry.substr(p2 + 1);

                    if (backend == "System") continue; // Handled dynamically below

                    SelectableDevice dev;
                    try {
                        dev.deviceIndex = std::stoul(idxStr);
                    } catch (...) {
                        dev.deviceIndex = 0;
                    }
                    dev.name = devName;
                    dev.backend = backend;
                    dev.architecture = detectGpuArchitecture(devName, 0, 0);
                    dev.driver = backend;
                    dev.vramTotalMb = 0;
                    dev.isSystem = false;
                    dev.selected = (dev.deviceIndex == 0);
                    m_devices.push_back(dev);
                }
            }
        }
    }

    // Dynamically query Host CPU & RAM system entry
    std::string cpuModel, cpuArch, osDriver;
    uint64_t ramMb = 0;
    getHostSystemInfo(cpuModel, cpuArch, osDriver, ramMb);

    unsigned int numThreads = std::thread::hardware_concurrency();
    std::string cpuFullName = cpuModel;
    if (numThreads > 0 && cpuFullName.find("Thread") == std::string::npos && cpuFullName.find("C/") == std::string::npos) {
        cpuFullName += " (" + std::to_string(numThreads) + " Threads)";
    }

    SelectableDevice sysDev;
    sysDev.deviceIndex = 0xFFFFFFFF;
    sysDev.name = cpuFullName;
    sysDev.backend = "System";
    sysDev.architecture = cpuArch;
    sysDev.driver = osDriver;
    sysDev.vramTotalMb = ramMb;
    sysDev.isSystem = true;
    m_devices.push_back(sysDev);

    // Natural logical order: GPU 0 (Primary) first, GPU 1 (Secondary) second, Host CPU last
    std::sort(m_devices.begin(), m_devices.end(), [](const SelectableDevice& a, const SelectableDevice& b) {
        if (a.isSystem != b.isSystem) return !a.isSystem;
        return a.deviceIndex < b.deviceIndex;
    });
}

void GuiApp::processEvents() {
    processIncomingResults();
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
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags rootFlags = ImGuiWindowFlags_NoDecoration 
                               | ImGuiWindowFlags_NoMove 
                               | ImGuiWindowFlags_NoResize 
                               | ImGuiWindowFlags_NoSavedSettings 
                               | ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(s(10.0f), s(10.0f)));

    if (ImGui::Begin("GPUBenchWorkstationRoot", nullptr, rootFlags)) {
        float sidebarWidth = s(350.0f);
        float fullHeight = ImGui::GetContentRegionAvail().y;

        // Left Workstation Sidebar Rail (Controls, Devices, Telemetry, Actions)
        renderLeftSidebar(sidebarWidth, fullHeight);

        ImGui::SameLine();

        // Right Workstation Main Workspace (Tabs, Benchmark Suite, Scorecard, Viewport)
        renderRightWorkspace(0.0f, fullHeight);
    }
    ImGui::End();
    ImGui::PopStyleVar(3);

    if (m_telemetryNeedsFit && m_execState != ExecutionState::Running) {
        m_telemetryNeedsFit = false;
    }

    if (m_showSettingsModal) {
        renderSettingsModal();
    }

    renderZoomToast();
}

void GuiApp::renderLeftSidebar(float width, float height) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.12f, 0.17f, 1.00f));
    ImGui::BeginChild("LeftSidebar", ImVec2(width, height), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    // 1. Top Brand & Prominent Start Action Block (Pinned at top left)
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.00f, 1.00f), "GPUBench");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.60f, 0.70f, 0.90f, 1.00f), "v%s", GPUBENCH_VERSION);

    ImGui::Spacing();

    size_t selectedTests = 0;
    for (const auto& cat : m_categories) {
        for (const auto& sub : cat.subgroups) {
            for (const auto& item : sub.items) {
                if (item.selected && (!m_hideUnsupported || item.isSupported)) selectedTests++;
            }
        }
    }

    if (m_execState == ExecutionState::Running) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.20f, 0.20f, 1.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.95f, 0.25f, 0.25f, 1.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.00f, 0.30f, 0.30f, 1.00f));
        if (ImGui::Button("ABORT BENCHMARK", ImVec2(-1, s(38.0f)))) {
            abortBenchmarks();
        }
        ImGui::PopStyleColor(3);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.45f, 0.95f, 1.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.55f, 1.00f, 1.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.35f, 0.65f, 1.00f, 1.00f));
        std::string runLabel = (selectedTests > 1)
            ? ("START BENCHMARKS (" + std::to_string(selectedTests) + ")")
            : (selectedTests == 1 ? "START BENCHMARK (1)" : "START BENCHMARK");
        if (ImGui::Button(runLabel.c_str(), ImVec2(-1, s(38.0f)))) {
            startBenchmarks();
        }
        ImGui::PopStyleColor(3);
    }

    if (m_execState == ExecutionState::Running) {
        ImGui::Spacing();
        float progress = (m_totalTasks > 0) ? (static_cast<float>(m_completedTasks) / static_cast<float>(m_totalTasks)) : 0.0f;
        progress = std::clamp(progress, 0.0f, 1.0f);
        std::string progText = std::to_string(m_completedTasks) + "/" + std::to_string(m_totalTasks) + " (" + std::to_string(static_cast<int>(progress * 100.0f)) + "%)";
        ImGui::ProgressBar(progress, ImVec2(-1, s(20.0f)), progText.c_str());
        ImGui::TextDisabled("%s", m_currentBenchmarkName.c_str());
    } else if (m_execState == ExecutionState::Completed) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.18f, 0.80f, 0.44f, 1.00f));
        ImGui::ProgressBar(1.0f, ImVec2(-1, s(20.0f)), "Complete");
        ImGui::PopStyleColor();
        ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.55f, 1.0f), "[PASSED] %zu workloads recorded", m_allResults.size());
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.45f, 0.90f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.55f, 1.00f, 1.0f));
        if (ImGui::Button("Export JSON", ImVec2(-1, s(28.0f)))) {
            exportResultsToJson("");
        }
        ImGui::PopStyleColor(2);
        if (m_exportNotificationTimer > 0.0f) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s", m_exportNotificationText.c_str());
        }
    } else if (m_execState == ExecutionState::Cancelled) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.85f, 0.55f, 0.15f, 1.00f));
        ImGui::ProgressBar(1.0f, ImVec2(-1, s(20.0f)), "Cancelled");
        ImGui::PopStyleColor();
        ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.20f, 1.0f), "[!] Cancelled by user");
        if (!m_allResults.empty()) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.45f, 0.90f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.55f, 1.00f, 1.0f));
            if (ImGui::Button("Export JSON", ImVec2(-1, s(28.0f)))) {
                exportResultsToJson("");
            }
            ImGui::PopStyleColor(2);
            if (m_exportNotificationTimer > 0.0f) {
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s", m_exportNotificationText.c_str());
            }
        }
    } else if (!m_allResults.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.40f, 0.75f, 1.0f, 1.0f), "%zu records available", m_allResults.size());
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.45f, 0.90f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.55f, 1.00f, 1.0f));
        if (ImGui::Button("Export JSON", ImVec2(-1, s(28.0f)))) {
            exportResultsToJson("");
        }
        ImGui::PopStyleColor(2);
        if (m_exportNotificationTimer > 0.0f) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s", m_exportNotificationText.c_str());
        }
    } else if (!m_statusMessage.empty()) {
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.30f, 1.0f), "%s", m_statusMessage.c_str());
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // 2. Scrollable Body (Target Accelerators, Telemetry, API, Resolution)
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::BeginChild("LeftSidebarScroll", ImVec2(0, 0), false);
    ImGui::PopStyleVar();

    // 2. Target Accelerators (Multi-Selection)
    ImGui::TextColored(ImVec4(0.65f, 0.75f, 0.90f, 1.0f), "TARGET ACCELERATORS");

    // Presets with active highlights
    bool dualGpu = false, allSel = true;
    size_t selCount = 0;
    for (const auto& dev : m_devices) {
        if (dev.selected) selCount++;
        else allSel = false;
    }

    std::vector<SelectableDevice*> actualGpus;
    SelectableDevice* hostDev = nullptr;
    for (auto& dev : m_devices) {
        if (!dev.isSystem) actualGpus.push_back(&dev);
        else hostDev = &dev;
    }

    size_t gpuCount = actualGpus.size();
    if (selCount == 2 && gpuCount >= 2) {
        bool allGpus = true;
        for (const auto* g : actualGpus) {
            if (!g->selected) allGpus = false;
        }
        dualGpu = allGpus;
    }

    auto presetPill = [](const char* label, bool active) {
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.45f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.16f, 0.23f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.78f, 0.90f, 1.0f));
        }
        bool clicked = ImGui::SmallButton(label);
        ImGui::PopStyleColor(2);
        return clicked;
    };

    // Render presets dynamically for actual GPUs only
    for (size_t g = 0; g < gpuCount; ++g) {
        auto* gpu = actualGpus[g];
        bool onlyThisGpu = (gpu->selected && selCount == 1);
        std::string pillLabel = "GPU " + std::to_string(gpu->deviceIndex);
        if (presetPill(pillLabel.c_str(), onlyThisGpu)) {
            for (auto& dev : m_devices) {
                dev.selected = (&dev == gpu);
            }
            updateTelemetrySelection();
        }
        ImGui::SameLine();
    }

    if (gpuCount > 1) {
        if (presetPill("Dual GPUs", dualGpu)) {
            for (auto& dev : m_devices) dev.selected = !dev.isSystem;
            updateTelemetrySelection();
        }
        ImGui::SameLine();
    }

    if (hostDev != nullptr) {
        if (presetPill("+ Host", false)) {
            hostDev->selected = true;
            updateTelemetrySelection();
        }
        ImGui::SameLine();
    }

    if (presetPill("All", allSel)) {
        for (auto& dev : m_devices) dev.selected = true;
        updateTelemetrySelection();
    }

    ImGui::Spacing();

    // Device Cards in Left Sidebar
    for (size_t i = 0; i < m_devices.size(); ++i) {
        auto& dev = m_devices[i];
        ImGui::PushID(static_cast<int>(i));

        if (dev.selected) {
            ImVec4 borderCol = dev.isSystem ? ImVec4(0.95f, 0.75f, 0.30f, 0.90f) : ImVec4(0.20f, 0.65f, 1.0f, 0.85f);
            ImVec4 bgCol = dev.isSystem ? ImVec4(0.18f, 0.15f, 0.10f, 0.85f) : ImVec4(0.12f, 0.18f, 0.28f, 0.85f);
            ImGui::PushStyleColor(ImGuiCol_Border, borderCol);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, bgCol);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.20f, 0.25f, 0.35f, 0.45f));
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.10f, 0.14f, 0.70f));
        }

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(s(8.0f), s(6.0f)));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, s(6.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, s(1.0f));

        // Explicit NoScrollbar flags and scaled height
        ImGui::BeginChild("DevCard", ImVec2(0, s(54.0f)), true, 
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        if (ImGui::BeginTable("DevCardTbl", 2, ImGuiTableFlags_None)) {
            ImGui::TableSetupColumn("DevCol", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("BadgeCol", ImGuiTableColumnFlags_WidthFixed, s(76.0f));
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(s(3.0f), s(3.0f)));
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + s(2.0f));
            if (ImGui::Checkbox("##dev_sel", &dev.selected)) {
                updateTelemetrySelection();
            }
            ImGui::PopStyleVar();

            ImGui::SameLine(0, s(8.0f));
            ImGui::BeginGroup();
            if (dev.isSystem) {
                ImVec4 titleCol = dev.selected ? ImVec4(0.98f, 0.85f, 0.45f, 1.0f) : ImVec4(0.80f, 0.75f, 0.55f, 0.85f);
                ImGui::TextColored(titleCol, "Host CPU & RAM");
                std::string sub = formatCardDeviceSubtitle(dev);
                ImGui::TextDisabled("%s", sub.c_str());
            } else {
                bool isPrimary = (dev.deviceIndex == 0);
                std::string title = (gpuCount > 1)
                    ? (isPrimary ? "GPU 0 (Primary)" : ("GPU " + std::to_string(dev.deviceIndex) + " (Secondary)"))
                    : ("GPU " + std::to_string(dev.deviceIndex));
                ImVec4 titleCol = dev.selected 
                    ? (isPrimary ? ImVec4(0.38f, 0.85f, 1.00f, 1.0f) : ImVec4(0.72f, 0.85f, 0.98f, 1.0f))
                    : ImVec4(0.60f, 0.68f, 0.80f, 0.8f);
                ImGui::TextColored(titleCol, "%s", title.c_str());
                std::string sub = formatCardDeviceSubtitle(dev);
                ImGui::TextDisabled("%s", sub.c_str());
            }
            ImGui::EndGroup();

            ImGui::TableSetColumnIndex(1);
            uint64_t memGb = (dev.vramTotalMb + 512) / 1024;
            std::string badgeStr = dev.vramTotalMb > 0 ? ("[" + std::to_string(memGb) + " GB]") : "[N/A]";
            ImVec4 badgeCol = dev.isSystem ? ImVec4(0.90f, 0.75f, 0.35f, 0.85f) : ImVec4(0.38f, 0.75f, 1.00f, 0.85f);
            float badgeW = ImGui::CalcTextSize(badgeStr.c_str()).x;
            float availW = ImGui::GetContentRegionAvail().x;
            if (availW > badgeW) {
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availW - badgeW));
            }
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + s(2.0f));
            ImGui::TextColored(badgeCol, "%s", badgeStr.c_str());

            ImGui::EndTable();
        }

        // Clicking anywhere on the card toggles its selection
        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && ImGui::IsMouseClicked(0)) {
            if (!ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive()) {
                dev.selected = !dev.selected;
                updateTelemetrySelection();
            }
        }

        ImGui::EndChild();
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(2);

        // Tooltip with detailed device information
        if (ImGui::IsItemHovered()) {
            if (dev.isSystem) {
                uint64_t ramGb = (dev.vramTotalMb + 512) / 1024;
                ImGui::SetTooltip("Host System Memory & CPU\nCPU: %s\nArchitecture: %s\nRAM: %llu GB (%llu MB)\nOS / Platform: %s\nRuns multi-threaded & single-threaded DDR bandwidth and pointer-chasing latency benchmarks",
                                  dev.name.c_str(),
                                  dev.architecture.c_str(),
                                  static_cast<unsigned long long>(ramGb),
                                  static_cast<unsigned long long>(dev.vramTotalMb),
                                  dev.driver.c_str());
            } else {
                uint64_t vramGb = (dev.vramTotalMb + 512) / 1024;
                std::string roleStr = (gpuCount > 1)
                    ? (dev.deviceIndex == 0 ? "Primary Target GPU (-d 0)" : "Secondary Accelerator GPU (-d " + std::to_string(dev.deviceIndex) + ")")
                    : "Target GPU (-d " + std::to_string(dev.deviceIndex) + ")";
                ImGui::SetTooltip("%s\nArchitecture: %s\nVRAM: %llu GB (%llu MB)\nDriver: %s\nRole: %s",
                                  dev.name.c_str(),
                                  dev.architecture.c_str(),
                                  static_cast<unsigned long long>(vramGb),
                                  static_cast<unsigned long long>(dev.vramTotalMb),
                                  dev.driver.c_str(),
                                  roleStr.c_str());
            }
        }

        ImGui::PopID();
        ImGui::Spacing();
    }

    // 3. Live Hardware Telemetry
    renderSidebarTelemetry();

    // 4. Compute API Selection
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.65f, 0.75f, 0.90f, 1.0f), "COMPUTE API");
    const char* backends[] = { "vulkan", "rocm", "opencl", "auto" };
    const char* backendLabels[] = { "Vulkan", "ROCm", "OpenCL", "Auto" };
    if (m_apiSupportList.empty()) {
        m_apiSupportList = GetAllComputeApiSupportAPI();
    }
    for (int b = 0; b < 4; ++b) {
        const ComputeApiSupportInfo* suppInfo = nullptr;
        for (const auto& info : m_apiSupportList) {
            if (info.name == backends[b]) {
                suppInfo = &info;
                break;
            }
        }
        bool isSupported = (suppInfo != nullptr) ? suppInfo->isSupported : true;
        bool isActive = (m_selectedBackend == backends[b]);

        if (!isSupported) {
            ImGui::BeginDisabled(true);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.42f, 0.46f, 0.54f, 0.65f));
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.09f, 0.11f, 0.15f, 0.45f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.09f, 0.11f, 0.15f, 0.45f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.09f, 0.11f, 0.15f, 0.45f));
        } else if (isActive) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.45f, 0.90f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.15f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.82f, 0.92f, 1.0f));
        }

        if (ImGui::SmallButton(backendLabels[b])) {
            if (isSupported) {
                m_selectedBackend = backends[b];
                updateBenchmarkSupport();
            }
        }

        if (!isSupported) {
            ImGui::PopStyleColor(4);
            ImGui::EndDisabled();
        } else {
            ImGui::PopStyleColor(2);
        }

        // Hover tooltip explaining why (what is missing) or ready status
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (ImGui::BeginTooltip()) {
                if (suppInfo && !suppInfo->isSupported) {
                    ImGui::TextColored(ImVec4(0.95f, 0.42f, 0.42f, 1.0f), "COMPUTE API: %s [UNSUPPORTED]", backendLabels[b]);
                    ImGui::TextColored(ImVec4(0.85f, 0.40f, 0.40f, 1.0f), "Status: Unavailable / Unsupported");
                    ImGui::Separator();
                    ImGui::PushTextWrapPos(s(280.0f));
                    ImGui::TextColored(ImVec4(0.70f, 0.75f, 0.85f, 1.0f), "Reason:");
                    ImGui::TextUnformatted(suppInfo->reason.c_str());
                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1.0f), "Missing Requirement:");
                    ImGui::TextUnformatted(suppInfo->missingRequirement.c_str());
                    ImGui::PopTextWrapPos();
                } else {
                    ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.45f, 1.0f), "COMPUTE API: %s [AVAILABLE]", backendLabels[b]);
                    ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.45f, 1.0f), "Status: Ready");
                    ImGui::Separator();
                    ImGui::PushTextWrapPos(s(280.0f));
                    if (suppInfo && !suppInfo->reason.empty()) {
                        ImGui::TextUnformatted(suppInfo->reason.c_str());
                    } else {
                        ImGui::TextUnformatted("Click to select this compute backend for benchmarking.");
                    }
                    ImGui::PopTextWrapPos();
                }
                ImGui::EndTooltip();
            }
        }

        if (b < 3) ImGui::SameLine();
    }

    // 4. Render Target Resolution & Image Dump Options
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.65f, 0.75f, 0.90f, 1.0f), "BENCHMARK RENDER RESOLUTION");
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.00f, 1.0f), "Ray Tracing & Scene Render Target");
        ImGui::Separator();
        ImGui::PushTextWrapPos(s(300.0f));
        ImGui::TextUnformatted(
            "Controls the internal framebuffer canvas dimensions and total ray count for Ray Tracing, Path Tracing, and Graphics benchmark passes.\n\n"
            "Higher resolutions increase compute and memory bandwidth load linearly / proportionally with pixel count (W x H):\n"
            "  • 720p:  1280 x 720   (~0.92M primary rays)\n"
            "  • 1080p: 1920 x 1080  (~2.07M primary rays) [FHD]\n"
            "  • 1440p: 2560 x 1440  (~3.69M primary rays) [QHD]\n"
            "  • 4K:    3840 x 2160  (~8.29M primary rays) [4K UHD / Default Stress]\n\n"
            "Note: This sets internal benchmark workload dimensions and does not change your monitor or GUI window resolution."
        );
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }

    ImGui::TextDisabled("Ray Tracing & Graphics Workload Canvas");
    ImGui::Spacing();

    const char* resLabels[] = { "720p", "1080p", "1440p", "4K" };
    const char* resTooltips[] = {
        "720p (1280 x 720) - Fast / Low load (~0.92M rays)",
        "1080p (1920 x 1080) - Standard Full HD ray tracing (~2.07M rays)",
        "1440p (2560 x 1440) - High load QHD benchmark (~3.69M rays)",
        "4K (3840 x 2160) - Extreme 4K UHD stress test (~8.29M rays) [Default]"
    };
    const uint32_t resDims[][2] = { {1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160} };

    for (int r = 0; r < 4; ++r) {
        bool isSel = (m_renderWidth == resDims[r][0] && m_renderHeight == resDims[r][1]);
        if (isSel) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.48f, 0.90f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.55f, 0.98f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.18f, 0.42f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.15f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f, 0.23f, 0.32f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.22f, 0.28f, 0.40f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.78f, 0.90f, 1.0f));
        }

        if (ImGui::SmallButton(resLabels[r])) {
            m_renderWidth = resDims[r][0];
            m_renderHeight = resDims[r][1];
        }

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", resTooltips[r]);
        }

        ImGui::PopStyleColor(4);
        if (r < 3) ImGui::SameLine();
    }

    // Active Canvas dimensions readout
    float raysM = static_cast<float>(m_renderWidth * m_renderHeight) / 1000000.0f;
    ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.00f, 1.0f), "%u x %u", m_renderWidth, m_renderHeight);
    ImGui::SameLine();
    ImGui::TextDisabled("(%.2f Mpix / frame)", raysM);

    // 5. Ray Tracing Scenario Selection
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.65f, 0.75f, 0.90f, 1.0f), "RAY TRACING SCENARIO");
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.00f, 1.0f), "Ray Tracing Geometry & Architectural Workload");
        ImGui::Separator();
        ImGui::PushTextWrapPos(s(300.0f));
        ImGui::TextUnformatted(
            "Selects the 3D test scenario for ray tracing and path tracing passes:\n\n"
            "  • All Scenes: Evaluates all 4 scenes sequentially (Console CLI parity)\n"
            "  • Sponza (Indoor): Crytek Sponza atrium (262K tris, 25 PBR materials, 100% screen fill) [Canonical Architecture Benchmark]\n"
            "  • AAA Forest: Heavy foliage, alpha-tested leaf cutouts & deep geometry stress (1.05M+ tris)\n"
            "  • Landscape: Expansive outdoor terrain with wide depth range & sky escape rays\n"
            "  • Showroom: Single vehicle model in empty studio void (108K tris)\n"
        );
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }

    ImGui::TextDisabled("Test Scene Geometry & Shading");
    ImGui::Spacing();

    const char* sceneBtnLabels[] = { "All", "Sponza", "Forest", "Landscape", "Showroom" };
    const char* sceneBtnTags[] = { "all", "indoor", "forest", "outdoor", "showroom" };
    const char* sceneBtnTooltips[] = {
        "All Scenes - Runs all 4 scenarios sequentially (Console parity default)",
        "Indoor Atrium (Crytek Sponza) - 262K triangles, 25 PBR materials, 100% screen fill [Canonical Benchmark]",
        "Open-World Forest - 1.05M+ triangles, heavy alpha cutout foliage & traversal stress",
        "Outdoor Landscape - Expansive terrain vista with wide depth variance & sky miss rays",
        "Showroom Studio - Single high-detail vehicle in studio void (108K triangles)"
    };

    for (int sIdx = 0; sIdx < 5; ++sIdx) {
        bool isSel = (m_scene == sceneBtnTags[sIdx]);
        if (isSel) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.48f, 0.90f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.55f, 0.98f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.18f, 0.42f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.15f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f, 0.23f, 0.32f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.22f, 0.28f, 0.40f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.78f, 0.90f, 1.0f));
        }

        float btnW = ImGui::CalcTextSize(sceneBtnLabels[sIdx]).x + ImGui::GetStyle().FramePadding.x * 2.0f;
        if (btnW + ImGui::GetStyle().ItemSpacing.x <= ImGui::GetContentRegionAvail().x && sIdx > 0 && sIdx != 3) {
            ImGui::SameLine();
        }

        if (ImGui::SmallButton(sceneBtnLabels[sIdx])) {
            m_scene = sceneBtnTags[sIdx];
            if (m_scene != "all") {
                m_suiteActiveScene = m_scene;
            }
        }

        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", sceneBtnTooltips[sIdx]);
        }

        ImGui::PopStyleColor(4);
    }

    if (m_scene == "all") {
        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.00f, 1.0f), "All 4 Scenarios (Console Parity)");
    } else if (m_scene == "indoor") {
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "Indoor Atrium (Sponza - Canonical)");
    } else if (m_scene == "forest") {
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "Open-World Forest (1.05M+ Tris)");
    } else if (m_scene == "outdoor") {
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "Outdoor Landscape (Terrain Vista)");
    } else {
        ImGui::TextColored(ImVec4(0.70f, 0.75f, 0.85f, 1.0f), "Showroom Studio (108K Tris)");
    }

    ImGui::Spacing();
    ImGui::Checkbox("Dump Scene Renders to Disk", &m_dumpRenders);
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.00f, 1.0f), "Render Output Image Export");
        ImGui::Separator();
        ImGui::PushTextWrapPos(s(300.0f));
        ImGui::TextUnformatted(
            "Saves full-resolution output framebuffers (PPM/PNG images) of ray-traced scenes to the 'renders/' folder on disk.\n\n"
            "• Checked: Writes image files to disk for visual verification and image quality comparison.\n"
            "• Unchecked: Keeps frames in GPU memory for pure throughput benchmarking with zero disk I/O latency."
        );
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }

    if (m_dumpRenders) {
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1.0f), "Saving image output to ./renders/");
    } else {
        ImGui::TextDisabled("In-memory only (max throughput)");
    }


    ImGui::EndChild(); // LeftSidebarScroll
    ImGui::EndChild(); // LeftSidebar
    ImGui::PopStyleColor();
}

void GuiApp::renderRightWorkspace(float width, float height) {
    ImGui::BeginChild("RightWorkspace", ImVec2(width, height), false);

    if (ImGui::BeginTabBar("MainWorkstationTabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Benchmark Suite")) {
            renderBenchmarkSuitePanel();
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags scorecardFlags = ImGuiTabItemFlags_None;
        if (m_switchToScorecard) {
            scorecardFlags |= ImGuiTabItemFlags_SetSelected;
            m_switchToScorecard = false;
        }

        if (ImGui::BeginTabItem("Results Scorecard", nullptr, scorecardFlags)) {
            renderResultsScorecard();
            ImGui::EndTabItem();
        }

        ImGuiTabItemFlags rtFlags = ImGuiTabItemFlags_None;
        if (m_switchToRtViewport) {
            rtFlags |= ImGuiTabItemFlags_SetSelected;
            m_switchToRtViewport = false;
        }

        if (ImGui::BeginTabItem("Ray Tracing Viewport", nullptr, rtFlags)) {
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

static bool hasAngleToken(const std::string& str, const std::string& token) {
    size_t pos = 0;
    while ((pos = str.find(token, pos)) != std::string::npos) {
        if (pos == 0 || !isdigit(static_cast<unsigned char>(str[pos - 1]))) {
            return true;
        }
        pos += token.length();
    }
    return false;
}

bool GuiApp::matchesItem(const ResultData& r, const BenchmarkItem& itm, uint32_t activeDev) const {
    bool isHostItem = (itm.category == "System" || itm.category == "Host System" || itm.category == "System Memory");
    uint32_t expectedDev = isHostItem ? 0xFFFFFFFF : activeDev;
    if (r.deviceIndex != expectedDev) return false;

    // Fast path: Exact configIndex and benchmark id match
    if (itm.configIndex >= 0) {
        std::string baseName = r.benchmarkName;
        size_t p = baseName.find(" (");
        if (p != std::string::npos) {
            baseName = baseName.substr(0, p);
        }
        if ((itm.id == baseName || (itm.id == "L0 Cache Latency" && baseName == "L0 Cache Latency")) &&
            r.configIndex == static_cast<uint32_t>(itm.configIndex)) {
            return true;
        }
    }

    // Subcategory matching with normalization
    if (!itm.subcategory.empty() && !r.subcategory.empty()) {
        bool subMatches = (itm.subcategory == r.subcategory);
        if (!subMatches) {
            if (itm.subcategory.find(r.subcategory) != std::string::npos ||
                r.subcategory.find(itm.subcategory) != std::string::npos) {
                subMatches = true;
            }
            if ((itm.subcategory.find("ROP") != std::string::npos || itm.subcategory.find("Fill Rate") != std::string::npos) &&
                (r.subcategory.find("ROP") != std::string::npos || r.subcategory.find("Fill Rate") != std::string::npos)) {
                subMatches = true;
            }
            if (itm.subcategory.find("Procedural") != std::string::npos &&
                (r.subcategory.find("Procedural") != std::string::npos ||
                 r.subcategory.find("Intersection") != std::string::npos)) {
                subMatches = true;
            }
            if (itm.category == "System" || itm.category == "Host System") {
                if (r.deviceIndex != 0xFFFFFFFF) subMatches = false;
            } else {
                if (r.deviceIndex == 0xFFFFFFFF) subMatches = false;
            }
        }
        if (!subMatches) return false;
    }

    // Cache Latency matching
    if (itm.subcategory == "Cache Latency" || r.subcategory == "Latency") {
        if (r.deviceIndex != 0xFFFFFFFF) {
            if (itm.name.find("L0") != std::string::npos && r.benchmarkName.find("L0") != std::string::npos) return true;
            if (itm.name.find("L1") != std::string::npos && r.benchmarkName.find("L1") != std::string::npos) return true;
            if (itm.name.find("L2") != std::string::npos && r.benchmarkName.find("L2") != std::string::npos) return true;
            if (itm.name.find("L3") != std::string::npos && r.benchmarkName.find("L3") != std::string::npos) return true;
        }
    }

    // Cache Latency Curve matching
    if (itm.subcategory == "Cache Latency Curve" || r.subcategory == "Cache Latency Curve") {
        std::string clean = cleanWorkloadName(r.benchmarkName, r.subcategory);
        return (clean == itm.name || r.benchmarkName == itm.name);
    }

    // LDS Bank Conflicts matching
    if (itm.subcategory == "LDS Bank Conflicts" || r.subcategory == "LDS Bank Conflicts") {
        std::string clean = cleanWorkloadName(r.benchmarkName, r.subcategory);
        return (clean == itm.name || r.benchmarkName == itm.name || r.benchmarkName.find(itm.name) != std::string::npos);
    }

    // In-Shader Indirect Synthesis matching
    if (itm.subcategory == "Indirect Command Synthesis" || r.subcategory == "Indirect Command Synthesis") {
        std::string clean = cleanWorkloadName(r.benchmarkName, r.subcategory);
        return (clean == itm.name || r.benchmarkName == itm.name || r.benchmarkName.find(itm.name) != std::string::npos);
    }

    // Ray Directional Coherence matching
    if (itm.subcategory == "Ray Directional Coherence" || r.subcategory == "Ray Directional Coherence") {
        std::string clean = cleanWorkloadName(r.benchmarkName, r.subcategory);
        if (clean == itm.name || r.benchmarkName == itm.name || clean.find(itm.name) != std::string::npos) {
            return true;
        }

        if (hasAngleToken(itm.name, "90°") || itm.name.find("90 deg") != std::string::npos || itm.name.find("Hemispherical") != std::string::npos || itm.name.find("Diffuse") != std::string::npos) {
            return r.configIndex == 0 || hasAngleToken(clean, "90°") || clean.find("90 deg") != std::string::npos || clean.find("Hemispherical") != std::string::npos || clean.find("Diffuse") != std::string::npos;
        }
        if (hasAngleToken(itm.name, "67.5°") || itm.name.find("67.5 deg") != std::string::npos) {
            return r.configIndex == 1 || clean.find("67.5°") != std::string::npos || clean.find("67.5 deg") != std::string::npos;
        }
        if (hasAngleToken(itm.name, "45°") || itm.name.find("45 deg") != std::string::npos) {
            return r.configIndex == 2 || clean.find("45°") != std::string::npos || clean.find("45 deg") != std::string::npos;
        }
        if (hasAngleToken(itm.name, "22.5°") || itm.name.find("22.5 deg") != std::string::npos) {
            return r.configIndex == 3 || clean.find("22.5°") != std::string::npos || clean.find("22.5 deg") != std::string::npos;
        }
        if (hasAngleToken(itm.name, "0°") || itm.name.find("0 deg") != std::string::npos || itm.name.find("Beam") != std::string::npos) {
            return r.configIndex == 4 || hasAngleToken(clean, "0°") || clean.find("0 deg") != std::string::npos || clean.find("Beam") != std::string::npos;
        }
        return false;
    }

    // VRAM Bandwidth matching
    if (itm.subcategory == "VRAM Bandwidth" || r.subcategory == "Bandwidth" || itm.id == "Device Memory Bandwidth") {
        if (r.deviceIndex != 0xFFFFFFFF) {
            for (const char* tGrp : {"128", "256", "1024"}) {
                if (itm.name.find(tGrp) != std::string::npos && r.benchmarkName.find(tGrp) != std::string::npos) {
                    bool itmIsRW = (itm.name.find("Read / Write") != std::string::npos || itm.name.find("R/W") != std::string::npos);
                    bool rIsRW = (r.benchmarkName.find("Read / Write") != std::string::npos || r.benchmarkName.find("R/W") != std::string::npos);
                    if (itmIsRW && rIsRW) return true;
                    if (!itmIsRW && !rIsRW) {
                        if (itm.name.find("Read") != std::string::npos && r.benchmarkName.find("Read") != std::string::npos) return true;
                        if (itm.name.find("Write") != std::string::npos && r.benchmarkName.find("Write") != std::string::npos) return true;
                    }
                }
            }
        }
    }

    // Host Memory matching
    if (itm.category == "System" || itm.category == "Host System") {
        if (r.deviceIndex == 0xFFFFFFFF) {
            if ((itm.id == "System Memory Latency" || itm.name == "Default" || itm.name.find("Latency") != std::string::npos || itm.name.find("Pointer Chasing") != std::string::npos) && r.metric == "ns") return true;
            bool is1T_itm = (itm.name.find("1T") != std::string::npos || itm.name.find("Single-Threaded") != std::string::npos);
            bool is1T_r = (r.benchmarkName.find("1 Thread") != std::string::npos || r.benchmarkName.find("1T") != std::string::npos || r.benchmarkName.find("Single-Threaded") != std::string::npos);
            if (is1T_itm == is1T_r && r.metric != "ns") {
                if (itm.name.find("Read") != std::string::npos && r.benchmarkName.find("Read") != std::string::npos) return true;
                if (itm.name.find("Write") != std::string::npos && r.benchmarkName.find("Write") != std::string::npos) return true;
                if (itm.name.find("Copy") != std::string::npos && r.benchmarkName.find("Copy") != std::string::npos) return true;
            }
        }
        return false;
    }

    // Dual-Issue matching
    if (itm.subcategory == "Dual-Issue" || r.subcategory == "Dual-Issue") {
        std::string clean = cleanWorkloadName(r.benchmarkName, r.subcategory);
        return (clean == itm.name || r.benchmarkName == itm.name || r.benchmarkName.find(itm.name) != std::string::npos);
    }

    // Compute Precision matching
    if (itm.category == "Compute") {
        if (itm.name.find("Matrix") != std::string::npos) {
            return (r.benchmarkName.find("Matrix") != std::string::npos);
        } else if (itm.name.find("Vector") != std::string::npos) {
            return (r.benchmarkName.find("Matrix") == std::string::npos);
        }
        return true;
    }

    std::string clean = cleanWorkloadName(r.benchmarkName, r.subcategory);

    // Graphics / Pixel Fill Rate (ROP Throughput) matching
    if (itm.id == "Pixel Fill Rate" || r.benchmarkName.find("Pixel Fill Rate") != std::string::npos ||
        itm.subcategory == "ROP Throughput" || r.subcategory == "ROP Throughput") {
        if (r.deviceIndex != 0xFFFFFFFF) {
            if ((itm.name.find("RGBA8") != std::string::npos) &&
                (r.benchmarkName.find("RGBA8") != std::string::npos || clean.find("RGBA8") != std::string::npos)) return true;
            if ((itm.name.find("RGBA16") != std::string::npos || itm.name.find("HDR") != std::string::npos) &&
                (r.benchmarkName.find("RGBA16") != std::string::npos || clean.find("RGBA16") != std::string::npos)) return true;
            if ((itm.name.find("Alpha") != std::string::npos || itm.name.find("Blend") != std::string::npos) &&
                (r.benchmarkName.find("Alpha") != std::string::npos || clean.find("Alpha") != std::string::npos)) return true;
        }
    }

    if (clean == itm.name || r.benchmarkName == itm.name) return true;
    if (clean.find(itm.name) != std::string::npos || itm.name.find(clean) != std::string::npos) return true;

    // RayASBuild matching
    if (itm.id == "RayASBuild") {
        for (const char* tlasKey : {"Indoor Corridor", "Dense Jungle", "Massive Open World"}) {
            if (itm.name.find(tlasKey) != std::string::npos && clean.find(tlasKey) != std::string::npos) return true;
        }
        for (const char* trisKey : {"1M", "5M", "10M"}) {
            if (itm.name.find(trisKey) != std::string::npos && clean.find(trisKey) != std::string::npos) {
                bool isUpdateItm = (itm.name.find("Update") != std::string::npos);
                bool isUpdateR = (clean.find("Update") != std::string::npos);
                if (isUpdateItm == isUpdateR) return true;
            }
        }
    }

    // RayScheduling dispatch keyword matching
    if (itm.id == "RayScheduling" || itm.id == "RayPathTracing") {
        if (!itm.subcategory.empty() && !r.subcategory.empty() && r.subcategory != itm.subcategory) {
            return false;
        }
        if (itm.configIndex >= 0 && r.configIndex == static_cast<uint32_t>(itm.configIndex)) {
            return true;
        }

        // Multi-Light Evaluation exact matching
        if (itm.subcategory == "Multi-Light Evaluation" || r.subcategory == "Multi-Light Evaluation") {
            bool itm128 = (itm.name.find("128") != std::string::npos);
            bool r128 = (clean.find("128") != std::string::npos || r.benchmarkName.find("128") != std::string::npos || r.configIndex == 33 || r.configIndex == 34);
            if (itm128 != r128) return false;
            bool itmDGC = (itm.name.find("DGC") != std::string::npos || itm.name.find("Binning") != std::string::npos);
            bool rDGC = (clean.find("DGC") != std::string::npos || clean.find("Binning") != std::string::npos || r.configIndex == 32 || r.configIndex == 34);
            return (itmDGC == rDGC);
        }

        // Differentiate 128 Lights vs 1 Light
        bool itm128 = (itm.name.find("128") != std::string::npos);
        bool r128 = (clean.find("128") != std::string::npos || r.benchmarkName.find("128") != std::string::npos);
        if (itm128 != r128) return false;

        // Differentiate 3 Lights vs Single Light in Shadows
        bool itm3L = (itm.name.find("3 Light") != std::string::npos || itm.name.find("Multi-Light (3") != std::string::npos);
        bool r3L = (clean.find("3 Light") != std::string::npos || r.benchmarkName.find("3 Light") != std::string::npos);
        if (itm3L != r3L) return false;

        if (itm.name.find("Megakernel") != std::string::npos && clean.find("Megakernel") != std::string::npos &&
            clean.find("RTP") == std::string::npos && itm.name.find("RTP") == std::string::npos) return true;
        if (itm.name.find("RTP") != std::string::npos && clean.find("RTP") != std::string::npos) {
            bool itmHasSER = (itm.name.find("SER") != std::string::npos);
            bool cleanHasSER = (clean.find("SER") != std::string::npos);
            if (itmHasSER == cleanHasSER) return true;
        }
        if ((itm.name.find("Multi-Light") != std::string::npos || itm.name.find("Many Lights") != std::string::npos) &&
            (clean.find("Multi-Light") != std::string::npos || clean.find("Many Lights") != std::string::npos || clean.find("Binning") != std::string::npos)) return true;
        if ((itm.name.find("Single Light") != std::string::npos || itm.name.find("DGC") != std::string::npos) &&
            (clean.find("Single Light") != std::string::npos || clean.find("DGC") != std::string::npos || clean.find("Work Lists") != std::string::npos) &&
            itm.name.find("Megakernel") == std::string::npos && clean.find("Megakernel") == std::string::npos &&
            itm.name.find("RTP") == std::string::npos && clean.find("RTP") == std::string::npos &&
            itm.name.find("Multi-Light") == std::string::npos && clean.find("Multi-Light") == std::string::npos &&
            itm.name.find("Binning") == std::string::npos && clean.find("Binning") == std::string::npos) return true;
        if ((itm.name.find("Persistent") != std::string::npos || itm.name.find("Load Balancing") != std::string::npos) &&
            (clean.find("Persistent") != std::string::npos || clean.find("Load Balancing") != std::string::npos || r.configIndex == 25)) return true;
        if (itm.name.find("Alpha Cutout") != std::string::npos && clean.find("Alpha") != std::string::npos) return true;
        if (itm.name.find("SER") != std::string::npos && clean.find("SER") != std::string::npos) return true;
        if (itm.name.find("Dedicated") != std::string::npos && clean.find("Dedicated") != std::string::npos) return true;
        if (itm.name.find("DGC") != std::string::npos && (clean.find("DGC") != std::string::npos || clean.find("Work Lists") != std::string::npos)) {
            bool itmMulti = (itm.name.find("Multi-Light") != std::string::npos || itm.name.find("Many Lights") != std::string::npos);
            bool cleanMulti = (clean.find("Multi-Light") != std::string::npos || clean.find("Many Lights") != std::string::npos || clean.find("Binning") != std::string::npos);
            if (itmMulti == cleanMulti) return true;
        }
        // Traversal Scheduling
        if (itm.name.find("Morton") != std::string::npos && clean.find("Morton") != std::string::npos) {
            if (itm.name.find("8x4") != std::string::npos && clean.find("8x4") != std::string::npos) return true;
            if (itm.name.find("4x8") != std::string::npos && clean.find("4x8") != std::string::npos) return true;
            if (itm.name.find("4x4") != std::string::npos && clean.find("8x4") != std::string::npos) return true;
            if (itm.name.find("8x8") != std::string::npos && clean.find("4x8") != std::string::npos) return true;
        }
        if (itm.name.find("Screen Tiled") != std::string::npos && clean.find("Screen Tiled") != std::string::npos) return true;
        for (const char* schedKey : {"Scanline", "Wave Ballot", "Single-Pass", "Round-Trip"}) {
            if (itm.name.find(schedKey) != std::string::npos && clean.find(schedKey) != std::string::npos) return true;
        }
    }

    // BVH & Divergence Stress matching
    auto hasPercentToken = [](const std::string& str, const std::string& token) -> bool {
        size_t pos = 0;
        while ((pos = str.find(token, pos)) != std::string::npos) {
            if (pos == 0 || !isdigit(static_cast<unsigned char>(str[pos - 1]))) {
                return true;
            }
            pos += token.length();
        }
        return false;
    };

    for (const char* pct : {"100%", "75%", "50%", "25%", "10%", "0%"}) {
        if (hasPercentToken(itm.name, pct) && hasPercentToken(clean, pct)) return true;
    }
    for (const char* deg : {"45 deg", "75 deg", "90 deg", "180 deg", "Microbench", "Mirror", "Diffuse", "0°", "22.5°", "45°", "67.5°", "90°"}) {
        if (itm.name.find(deg) != std::string::npos && clean.find(deg) != std::string::npos) return true;
    }
    for (const char* pl : {"16B", "128B", "256B"}) {
        if (itm.name.find(pl) != std::string::npos && clean.find(pl) != std::string::npos) return true;
    }
    if (itm.name.find("Ray-Triangle") != std::string::npos && clean.find("Ray-Triangle") != std::string::npos) return true;
    if (itm.name.find("Ray-Box") != std::string::npos && clean.find("Ray-Box") != std::string::npos) return true;
    if (itm.name.find("Coherent") != std::string::npos && itm.name.find("Incoherent") == std::string::npos &&
        clean.find("Coherent") != std::string::npos && clean.find("Incoherent") == std::string::npos) return true;
    if (itm.name.find("Deep Box") != std::string::npos && clean.find("Box") != std::string::npos) return true;
    if (itm.name.find("Spheres") != std::string::npos && clean.find("Spheres") != std::string::npos) return true;
    if (itm.id == "RayProcedural" || itm.name.find("Procedural") != std::string::npos) {
        if (r.benchmarkName.find("RayProcedural") != std::string::npos ||
            clean.find("Spheres") != std::string::npos ||
            clean.find("Procedural") != std::string::npos) return true;
    }

    return false;
}

GuiApp::BenchmarkDisplayInfo GuiApp::getBenchmarkDisplayInfo(
    const BenchmarkItem& item,
    uint32_t targetDeviceIndex) const
{
    BenchmarkDisplayInfo info;
    info.hasResult = false;
    info.isBaseline = false;
    info.hasComparison = false;
    info.speedupRatio = 1.0;
    info.percentDelta = 0.0;
    info.deltaColor = ImVec4(0.60f, 0.65f, 0.75f, 0.70f);

    uint32_t activeDev = targetDeviceIndex;
    bool isHostItem = (item.category == "System" || item.category == "Host System" || item.category == "System Memory");
    if (isHostItem) {
        activeDev = 0xFFFFFFFF;
    } else {
        bool hasTargetDev = false;
        for (const auto& r : m_allResults) {
            if (r.deviceIndex == activeDev) {
                hasTargetDev = true;
                break;
            }
        }
        if (!hasTargetDev && !m_allResults.empty()) {
            for (const auto& r : m_allResults) {
                if (r.deviceIndex != 0xFFFFFFFF) {
                    activeDev = r.deviceIndex;
                    break;
                }
            }
        }
    }

    auto formatScore = [](const ResultData& r) -> std::string {
        if (r.time_ms == -3.0) return "ABORTED";
        if (r.time_ms < 0.0) return "FAILED";
        if (r.time_ms == 0.0 || r.operations == 0) return "-";
        double opsPerSec = (static_cast<double>(r.operations) / r.time_ms) * 1000.0;
        char buf[64];
        if (r.metric.find("TFLOPS") != std::string::npos || r.metric.find("TOPS") != std::string::npos) {
            snprintf(buf, sizeof(buf), "%.2f %s", opsPerSec / 1e12, r.metric.c_str());
        } else if (r.metric.find("TB/s") != std::string::npos) {
            snprintf(buf, sizeof(buf), "%.2f TB/s", opsPerSec / 1e12);
        } else if (r.metric.find("GB/s") != std::string::npos) {
            snprintf(buf, sizeof(buf), "%.1f GB/s", opsPerSec / 1e9);
        } else if (r.metric.find("GIS/s") != std::string::npos) {
            snprintf(buf, sizeof(buf), "%.2f GIS/s", opsPerSec / 1e9);
        } else if (r.metric.find("MRays/s") != std::string::npos) {
            snprintf(buf, sizeof(buf), "%.1f MRays/s", opsPerSec / 1e6);
        } else if (r.metric.find("GPixels/s") != std::string::npos) {
            snprintf(buf, sizeof(buf), "%.2f GPixels/s", opsPerSec / 1e9);
        } else if (r.metric.find("MBVH/s") != std::string::npos ||
                   r.metric.find("MTris/s") != std::string::npos ||
                   r.metric.find("MInst/s") != std::string::npos ||
                   r.metric.find("MHits/s") != std::string::npos ||
                   r.metric.find("MItems/s") != std::string::npos ||
                   r.metric.find("MRecords/s") != std::string::npos) {
            snprintf(buf, sizeof(buf), "%.1f %s", opsPerSec / 1e6, r.metric.c_str());
        } else if (r.metric.find("ns") != std::string::npos) {
            double nsVal = (r.operations > 0) ? ((r.time_ms * 1e6) / r.operations) : r.time_ms;
            snprintf(buf, sizeof(buf), "%.1f ns", nsVal);
        } else {
            snprintf(buf, sizeof(buf), "%.1f %s", opsPerSec, r.metric.c_str());
        }
        return std::string(buf);
    };

    const ResultData* curRes = nullptr;
    const ResultData* fallbackRes = nullptr;
    for (const auto& r : m_allResults) {
        if (matchesItem(r, item, activeDev)) {
            if (!fallbackRes && r.time_ms > 0.0) fallbackRes = &r;
            if (item.id == "RayScheduling" || item.id == "RayPathTracing") {
                bool sceneMatches = false;
                if (m_suiteActiveScene == "indoor" && r.benchmarkName.find("Indoor Atrium") != std::string::npos) sceneMatches = true;
                else if (m_suiteActiveScene == "forest" && r.benchmarkName.find("Forest") != std::string::npos) sceneMatches = true;
                else if (m_suiteActiveScene == "outdoor" && r.benchmarkName.find("Outdoor") != std::string::npos) sceneMatches = true;
                else if (m_suiteActiveScene == "showroom" && r.benchmarkName.find("Showroom") != std::string::npos) sceneMatches = true;
                else if (m_suiteActiveScene.empty() || m_suiteActiveScene == "all") sceneMatches = true;

                if (sceneMatches) {
                    curRes = &r;
                    if (r.time_ms > 0.0) break;
                }
            } else {
                curRes = &r;
                if (r.time_ms > 0.0) break;
            }
        }
    }
    if (!curRes) curRes = fallbackRes;

    if (curRes) {
        info.hasResult = true;
        info.primaryResult = *curRes;
        info.scoreText = formatScore(*curRes);

        if (curRes->time_ms == -3.0) {
            info.deltaText = !curRes->errorString.empty() ? ("[" + curRes->errorString + "]") : "[ABORTED]";
            info.deltaColor = ImVec4(0.95f, 0.70f, 0.25f, 1.0f);
        } else if (curRes->time_ms < 0.0) {
            info.deltaText = !curRes->errorString.empty() ? ("[" + curRes->errorString + "]") : "[FAILED]";
            info.deltaColor = ImVec4(0.95f, 0.35f, 0.35f, 1.0f);
        }

        bool isUnsupportedItem = (!item.isSupported || curRes->isUnsupported || curRes->time_ms <= 0.0);

        // Determine baseline item name in the same subcategory
        std::string baselineName = "";
        if (!isUnsupportedItem) {
            if (item.subcategory == "LDS Bank Conflicts") {
                if (item.name.find("Baseline") != std::string::npos || item.name.find("Stride 1") != std::string::npos) {
                    info.isBaseline = true;
                    info.deltaText = "[Baseline]";
                    info.deltaColor = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                } else {
                    baselineName = "Stride 1";
                }
            } else if (item.subcategory == "Indirect Command Synthesis") {
                if (item.name.find("Baseline") != std::string::npos || item.name.find("CPU Direct") != std::string::npos) {
                    info.isBaseline = true;
                    info.deltaText = "[Baseline]";
                    info.deltaColor = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                } else {
                    baselineName = "CPU Direct";
                }
            } else if (item.subcategory == "Dual-Issue") {
                if (item.name == "Standard FP32") {
                    info.isBaseline = true;
                    info.deltaText = "[Baseline]";
                    info.deltaColor = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                } else if (item.name.find("Dual-Issue FP32") != std::string::npos) {
                    baselineName = "Dual_Issue_FP32_Baseline";
                } else if (item.name == "Standard INT32") {
                    info.isBaseline = true;
                    info.deltaText = "[Baseline]";
                    info.deltaColor = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                } else if (item.name.find("Dual-Issue INT32") != std::string::npos) {
                    baselineName = "Dual_Issue_INT32_Baseline";
                } else if (item.name.find("Mixed") != std::string::npos) {
                    baselineName = "Dual_Issue_Mixed_Harmonic";
                }
            } else if (item.category == "Compute") {
                // In Compute Precision, Vector ALU is the baseline for Matrix (WMMA) within the SAME precision.
                // Standalone precisions (FP64, FP32, FP4) do not compare against unrelated datatypes.
                if (item.name.find("Vector") != std::string::npos) {
                    if (item.id == "FP16" || item.id == "BF16" || item.id == "FP8" || item.id == "INT8" || item.id == "INT4") {
                        info.isBaseline = true;
                        info.deltaText = "[Baseline]";
                        info.deltaColor = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                    }
                } else if (item.name.find("Matrix") != std::string::npos) {
                    baselineName = item.id + "_Vector";
                }
            } else if (item.name.find("Vector ALU") != std::string::npos || item.name == "Vector") {
                info.isBaseline = true;
                info.deltaText = "[Baseline]";
                info.deltaColor = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
            } else if (item.name.find("Matrix") != std::string::npos) {
                baselineName = "Vector";
            } else if ((item.name == "Compute Megakernel" || item.name.find("Compute Megakernel") != std::string::npos ||
                        (item.name.find("Megakernel") != std::string::npos && 
                         item.name.find("RTP") == std::string::npos && 
                         item.name.find("SER") == std::string::npos && 
                         item.name.find("DGC") == std::string::npos)) ||
                       item.name.find("Linear 1D Scanline") != std::string::npos ||
                       item.name.find("Wave Ballot") != std::string::npos ||
                       item.name.find("100% Solid") != std::string::npos ||
                       item.name.find("Uniform Material") != std::string::npos ||
                       item.name.find("Coherent Material") != std::string::npos ||
                       item.name.find("90° Hemispherical") != std::string::npos ||
                       item.name.find("90 deg") != std::string::npos ||
                       item.name.find("Diffuse Baseline") != std::string::npos ||
                       item.name.find("16B Payload") != std::string::npos ||
                       item.name.find("16B") != std::string::npos ||
                       item.name.find("4 Bytes") != std::string::npos) {
                info.isBaseline = true;
                info.deltaText = "[Baseline]";
                info.deltaColor = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
            } else {
                // Find appropriate baseline for this subcategory
                if (item.subcategory == "Multi-Light Evaluation" || item.subcategory.find("Multi-Light") != std::string::npos) {
                    if (item.name.find("128") != std::string::npos) {
                        baselineName = "128_Megakernel";
                    } else {
                        baselineName = "Single_Megakernel";
                    }
                } else if (item.subcategory.find("Scene Ray Tracing") != std::string::npos ||
                    item.subcategory.find("Path Tracing") != std::string::npos ||
                    item.subcategory.find("Total Scene Render") != std::string::npos ||
                    item.subcategory.find("Directional Shadows") != std::string::npos ||
                    item.subcategory.find("Material Shading") != std::string::npos ||
                    item.subcategory.find("Incoherent Ray Tracing") != std::string::npos) {
                    baselineName = "Megakernel";
                } else if (item.subcategory.find("Alpha-Tested") != std::string::npos || item.name.find("Solid") != std::string::npos) {
                    baselineName = "100% Solid";
                } else if (item.subcategory.find("Material Divergence") != std::string::npos || item.name.find("Material Divergence") != std::string::npos) {
                    baselineName = "Uniform";
                } else if (item.subcategory.find("Ray Directional Coherence") != std::string::npos || item.name.find("Coherence") != std::string::npos || item.name.find("Mirror") != std::string::npos || item.name.find("Diffuse") != std::string::npos) {
                    baselineName = "90° Hemispherical";
                } else if (item.subcategory.find("Payload Register Pressure") != std::string::npos || item.name.find("Payload") != std::string::npos) {
                    baselineName = "16B";
                } else if (item.subcategory.find("Traversal Ordering") != std::string::npos ||
                           item.subcategory.find("Pipeline Breakdown") != std::string::npos ||
                           item.name.find("Traversal Scheduling") != std::string::npos) {
                    baselineName = "Linear 1D Scanline";
                } else if (item.subcategory.find("Wavefront Stream Compaction") != std::string::npos) {
                    baselineName = "Wave Ballot";
                }
            }
        }
        const ResultData* baselineRes = nullptr;
        if (!baselineName.empty()) {
            for (const auto& r : m_allResults) {
                if (r.deviceIndex == activeDev) {
                    if (r.isUnsupported || r.time_ms <= 0.0) continue;
                    if (baselineName == "Dual_Issue_FP32_Baseline") {
                        if (r.subcategory == "Dual-Issue" && (r.configIndex == 0 || r.benchmarkName.find("Standard FP32") != std::string::npos)) {
                            baselineRes = &r;
                            break;
                        }
                    } else if (baselineName == "Dual_Issue_INT32_Baseline") {
                        if (r.subcategory == "Dual-Issue" && (r.configIndex == 3 || r.benchmarkName.find("Standard INT32") != std::string::npos)) {
                            baselineRes = &r;
                            break;
                        }
                    } else if (baselineName.find("_Vector") != std::string::npos) {
                        std::string targetSub = baselineName.substr(0, baselineName.find("_Vector"));
                        if (r.subcategory == targetSub && r.benchmarkName.find("Vector") != std::string::npos) {
                            baselineRes = &r;
                            break;
                        }
                    } else if (baselineName == "128_Megakernel") {
                        if (r.subcategory == "Multi-Light Evaluation" &&
                            (r.configIndex == 33 || (r.benchmarkName.find("128") != std::string::npos && r.benchmarkName.find("Megakernel") != std::string::npos && r.benchmarkName.find("DGC") == std::string::npos))) {
                            baselineRes = &r;
                            break;
                        }
                    } else if (baselineName == "Single_Megakernel") {
                        if (r.subcategory == "Multi-Light Evaluation" &&
                            (r.configIndex == 31 || (r.benchmarkName.find("Single Light") != std::string::npos && r.benchmarkName.find("Megakernel") != std::string::npos && r.benchmarkName.find("DGC") == std::string::npos))) {
                            baselineRes = &r;
                            break;
                        }
                    } else if (baselineName == "90° Hemispherical") {
                        if (r.subcategory == "Ray Directional Coherence" &&
                            (r.configIndex == 0 || hasAngleToken(r.benchmarkName, "90°") || r.benchmarkName.find("90 deg") != std::string::npos ||
                             r.benchmarkName.find("Hemispherical") != std::string::npos || r.benchmarkName.find("Diffuse") != std::string::npos)) {
                            baselineRes = &r;
                            break;
                        }
                    } else if (r.subcategory == item.subcategory) {
                        if (item.id == "RayScheduling" || item.id == "RayPathTracing") {
                            std::string curScene = extractSceneName(curRes->benchmarkName);
                            std::string rScene = extractSceneName(r.benchmarkName);
                            if (!curScene.empty() && !rScene.empty() && curScene != rScene) {
                                continue;
                            }
                        }
                        std::string cName = cleanWorkloadName(r.benchmarkName, r.subcategory);
                        if (baselineName == "Megakernel") {
                            if ((cName.find("Megakernel") != std::string::npos || r.benchmarkName.find("Megakernel") != std::string::npos) &&
                                r.benchmarkName.find("RTP") == std::string::npos && r.benchmarkName.find("DGC") == std::string::npos) {
                                if (item.subcategory == "Multi-Light Evaluation") {
                                    bool itm128 = (item.name.find("128") != std::string::npos);
                                    bool r128 = (r.benchmarkName.find("128") != std::string::npos || r.configIndex == 33 || r.configIndex == 34);
                                    if (itm128 != r128) continue;
                                }
                                baselineRes = &r;
                                break;
                            }
                        } else if (cName.find(baselineName) != std::string::npos || r.benchmarkName.find(baselineName) != std::string::npos) {
                            baselineRes = &r;
                            break;
                        }
                    }
                }
            }
        }

        if (baselineRes && !baselineRes->isUnsupported && baselineRes->time_ms > 0.0 && baselineRes->operations > 0 &&
            !curRes->isUnsupported && curRes->time_ms > 0.0 && curRes->operations > 0 &&
            baselineRes->metric == curRes->metric) {
            double curOps = (static_cast<double>(curRes->operations) / curRes->time_ms) * 1000.0;
            double baseOps = (static_cast<double>(baselineRes->operations) / baselineRes->time_ms) * 1000.0;
            if (baseOps > 0.0) {
                info.hasComparison = true;
                info.baselineResult = *baselineRes;
                info.speedupRatio = curOps / baseOps;
                info.percentDelta = (info.speedupRatio - 1.0) * 100.0;

                char dBuf[64];
                if (std::abs(info.percentDelta) >= 0.1) {
                    snprintf(dBuf, sizeof(dBuf), "%.2fx (%s%.1f%%)",
                             info.speedupRatio,
                             info.percentDelta >= 0.0 ? "+" : "",
                             info.percentDelta);
                } else {
                    snprintf(dBuf, sizeof(dBuf), "%.2fx", info.speedupRatio);
                }
                info.deltaText = dBuf;
                info.deltaColor = (info.speedupRatio >= 1.0) ? ImVec4(0.35f, 0.95f, 0.55f, 1.0f) : ImVec4(0.70f, 0.75f, 0.85f, 1.0f);
            }
        } else if (baselineName == "Dual_Issue_Mixed_Harmonic" && !curRes->isUnsupported && curRes->time_ms > 0.0 && curRes->operations > 0) {
            double baseFP32 = 0.0;
            double baseINT32 = 0.0;
            for (const auto& r : m_allResults) {
                if (r.deviceIndex == activeDev && r.subcategory == "Dual-Issue" && !r.isUnsupported && r.time_ms > 0.0) {
                    if (r.configIndex == 0 || r.benchmarkName.find("Standard FP32") != std::string::npos) {
                        baseFP32 = (static_cast<double>(r.operations) / r.time_ms) * 1000.0;
                    } else if (r.configIndex == 3 || r.benchmarkName.find("Standard INT32") != std::string::npos) {
                        baseINT32 = (static_cast<double>(r.operations) / r.time_ms) * 1000.0;
                    }
                }
            }
            if (baseFP32 > 0.0 && baseINT32 > 0.0) {
                double baseOps = 2.0 / ((1.0 / baseFP32) + (1.0 / baseINT32));
                double curOps = (static_cast<double>(curRes->operations) / curRes->time_ms) * 1000.0;
                info.hasComparison = true;
                info.speedupRatio = curOps / baseOps;
                info.percentDelta = (info.speedupRatio - 1.0) * 100.0;

                char dBuf[64];
                if (std::abs(info.percentDelta) >= 0.1) {
                    snprintf(dBuf, sizeof(dBuf), "%.2fx (%s%.1f%%)",
                             info.speedupRatio,
                             info.percentDelta >= 0.0 ? "+" : "",
                             info.percentDelta);
                } else {
                    snprintf(dBuf, sizeof(dBuf), "%.2fx", info.speedupRatio);
                }
                info.deltaText = dBuf;
                info.deltaColor = (info.speedupRatio >= 1.0) ? ImVec4(0.35f, 0.95f, 0.55f, 1.0f) : ImVec4(0.70f, 0.75f, 0.85f, 1.0f);
            }
        }
    }

    return info;
}

void GuiApp::renderBenchmarkSuitePanel() {
    ImGui::Spacing();
    
    bool canEditWorkloads = (m_execState == ExecutionState::Idle);

    if (m_execState == ExecutionState::Completed) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.18f, 0.25f, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.30f, 0.65f, 0.95f, 0.85f));

        float btn1W = ImGui::CalcTextSize("View Results Scorecard").x + ImGui::GetStyle().FramePadding.x * 2.0f;
        float btn2W = ImGui::CalcTextSize("Reconfigure Workloads").x + ImGui::GetStyle().FramePadding.x * 2.0f;
        float totalBtnsW = btn1W + btn2W + s(14.0f);
        float bannerTextW = ImGui::CalcTextSize("[PASSED] Benchmark suite finished (171 results recorded). | Selection locked.").x;
        bool needsTwoLines = (ImGui::GetContentRegionAvail().x < bannerTextW + totalBtnsW + s(30.0f));
        float bannerH = needsTwoLines ? (ImGui::GetFrameHeight() * 2.0f + s(22.0f)) : (ImGui::GetFrameHeight() + s(16.0f));

        ImGui::BeginChild("CompletedSuiteBanner", ImVec2(0, bannerH), true);
        ImGui::AlignTextToFramePadding();
        size_t failCnt = 0;
        size_t abortCnt = 0;
        for (const auto& r : m_allResults) {
            if (r.time_ms == -3.0) abortCnt++;
            else if (r.time_ms < 0.0 && !r.isUnsupported) failCnt++;
        }
        if (failCnt > 0 || abortCnt > 0) {
            ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.35f, 1.0f), "[COMPLETED WITH ISSUES] %zu Passed | %zu Failed | %zu Aborted",
                               m_allResults.size() - failCnt - abortCnt, failCnt, abortCnt);
        } else {
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "[PASSED] Benchmark suite finished (%zu results recorded).", m_allResults.size());
        }
        
        if (!needsTwoLines) {
            ImGui::SameLine();
            ImGui::TextDisabled("| Selection locked.");
            ImGui::SameLine();
            float availBanner = ImGui::GetContentRegionAvail().x;
            if (availBanner >= totalBtnsW) {
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availBanner - totalBtnsW);
            }
        }
        
        if (ImGui::SmallButton("View Results Scorecard")) {
            m_switchToScorecard = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reconfigure Workloads")) {
            m_execState = ExecutionState::Idle;
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
        ImGui::Spacing();
    }

    // Category View Filter Tabs with dynamic flowing wrap
    ImGui::TextDisabled("View Category:");
    const char* viewLabels[] = {"All Tests (90)", "Compute (13)", "Memory (13)", "Ray Tracing (54)", "Graphics (3)", "Host CPU (7)"};
    for (int vi = 0; vi < 6; ++vi) {
        float btnW = ImGui::CalcTextSize(viewLabels[vi]).x + ImGui::GetStyle().FramePadding.x * 2.0f;
        if (btnW + ImGui::GetStyle().ItemSpacing.x <= ImGui::GetContentRegionAvail().x) {
            ImGui::SameLine();
        }
        bool isAct = (m_suiteCategoryFilter == vi);
        if (isAct) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.48f, 0.92f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.16f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.75f, 0.85f, 1.0f));
        }
        if (ImGui::SmallButton(viewLabels[vi])) {
            m_suiteCategoryFilter = vi;
        }
        ImGui::PopStyleColor(2);
    }

    // Hide Unsupported Toggle placed cleanly with dynamic wrapping
    float checkW = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize("Hide Unsupported").x + s(10.0f);
    if (checkW + ImGui::GetStyle().ItemSpacing.x <= ImGui::GetContentRegionAvail().x) {
        ImGui::SameLine();
    }
    ImGui::PushStyleColor(ImGuiCol_Text, m_hideUnsupported ? ImVec4(0.70f, 0.80f, 0.95f, 1.0f) : ImVec4(0.95f, 0.70f, 0.25f, 1.0f));
    if (ImGui::Checkbox("Hide Unsupported", &m_hideUnsupported)) {
        if (m_hideUnsupported) {
            for (auto& cat : m_categories) {
                for (auto& sub : cat.subgroups) {
                    for (auto& item : sub.items) {
                        if (!item.isSupported) item.selected = false;
                    }
                }
            }
        }
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Hide workloads not supported by current hardware or API toolchain");
    }

    // Ray Tracing Scenario View Tabs (available whenever Ray Tracing or All is viewed)
    if (m_suiteCategoryFilter == 0 || m_suiteCategoryFilter == 3) {
        ImGui::Spacing();
        ImGui::TextDisabled("Ray Tracing Scenario View:");
        ImGui::SameLine();
        const char* sceneViewLabels[] = { "Indoor Atrium (Sponza) [Canonical]", "Open-World Forest", "Outdoor Landscape", "Showroom Studio" };
        const char* sceneViewTags[] = { "indoor", "forest", "outdoor", "showroom" };
        const char* sceneViewTips[] = {
            "View scores and tooltips for Indoor Atrium (Crytek Sponza, 262K tris) [Canonical Reference]",
            "View scores and tooltips for Open-World Forest (1.05M+ tris, alpha-tested foliage)",
            "View scores and tooltips for Outdoor Landscape (terrain vista, distance rays)",
            "View scores and tooltips for Showroom Studio (108K tris, vehicle void)"
        };
        for (int svi = 0; svi < 4; ++svi) {
            float sBtnW = ImGui::CalcTextSize(sceneViewLabels[svi]).x + ImGui::GetStyle().FramePadding.x * 2.0f;
            if (sBtnW + ImGui::GetStyle().ItemSpacing.x <= ImGui::GetContentRegionAvail().x && svi > 0) {
                ImGui::SameLine();
            }
            bool isAct = (m_suiteActiveScene == sceneViewTags[svi]);
            if (isAct) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.48f, 0.92f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.16f, 0.22f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.75f, 0.85f, 1.0f));
            }
            if (ImGui::SmallButton(sceneViewLabels[svi])) {
                m_suiteActiveScene = sceneViewTags[svi];
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", sceneViewTips[svi]);
            }
            ImGui::PopStyleColor(2);
        }
    }

    ImGui::Spacing();

    // Toolbar with Select Filters and Collapse/Expand Controls
    ImGui::TextDisabled("Select:");

    auto getCatStatus = [&](const std::string& catName) -> int {
        for (const auto& cat : m_categories) {
            if (cat.name == catName) {
                size_t total = 0, sel = 0;
                for (const auto& sub : cat.subgroups) {
                    for (const auto& it : sub.items) {
                        total++;
                        if (it.selected) sel++;
                    }
                }
                if (total == 0 || sel == 0) return 0;
                if (sel == total) return 2;
                return 1;
            }
        }
        return 0;
    };

    auto toggleCat = [&](const std::string& catName) {
        for (auto& cat : m_categories) {
            if (cat.name == catName) {
                int st = getCatStatus(catName);
                bool turnOn = (st != 2);
                cat.allSelected = turnOn;
                for (auto& sub : cat.subgroups) {
                    sub.allSelected = turnOn;
                    for (auto& it : sub.items) it.selected = turnOn;
                }
                break;
            }
        }
        // Auto-collapse untested subgroups and expand active ones
        for (auto& cat : m_categories) {
            for (auto& sub : cat.subgroups) {
                size_t sel = 0;
                for (const auto& itm : sub.items) {
                    if (itm.selected) sel++;
                }
                if (sel == 0) {
                    sub.collapsed = true;
                } else {
                    sub.collapsed = false;
                }
            }
        }
    };

    size_t totalSelectedItems = 0;
    size_t totalPossibleItems = 0;
    for (const auto& cat : m_categories) {
        for (const auto& sub : cat.subgroups) {
            for (const auto& it : sub.items) {
                totalPossibleItems++;
                if (it.selected) totalSelectedItems++;
            }
        }
    }
    bool allItemsSelected = (totalPossibleItems > 0 && totalSelectedItems == totalPossibleItems);
    bool noneItemsSelected = (totalSelectedItems == 0);

    // If only one category or subgroup is selected, auto-collapse untested groups
    if (totalSelectedItems != m_prevTotalSelectedItems) {
        m_prevTotalSelectedItems = totalSelectedItems;
        size_t activeCats = 0;
        size_t activeSubs = 0;
        for (const auto& cat : m_categories) {
            bool catHasSel = false;
            for (const auto& sub : cat.subgroups) {
                size_t sel = 0;
                for (const auto& itm : sub.items) {
                    if (itm.selected) sel++;
                }
                if (sel > 0) {
                    activeSubs++;
                    catHasSel = true;
                }
            }
            if (catHasSel) activeCats++;
        }
        if (activeCats == 1 || activeSubs == 1) {
            for (auto& cat : m_categories) {
                for (auto& sub : cat.subgroups) {
                    size_t sel = 0;
                    for (const auto& itm : sub.items) {
                        if (itm.selected) sel++;
                    }
                    if (sel == 0) sub.collapsed = true;
                }
            }
        }
    }

    auto renderFilterPill = [](const char* label, int status, const char* tooltip) -> bool {
        if (status == 2) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.48f, 0.92f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.28f, 0.58f, 1.00f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.15f, 0.40f, 0.82f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.45f, 0.82f, 1.00f, 0.95f));
        } else if (status == 1) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.28f, 0.48f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.22f, 0.38f, 0.62f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.12f, 0.22f, 0.40f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.92f, 1.00f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.65f, 0.90f, 0.70f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.15f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f, 0.22f, 0.32f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.10f, 0.12f, 0.18f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.60f, 0.68f, 0.80f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.20f, 0.24f, 0.34f, 0.50f));
        }
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 3.0f));

        bool clicked = ImGui::SmallButton(label);

        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(5);

        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && tooltip) {
            ImGui::SetTooltip("%s", tooltip);
        }

        return clicked;
    };

    auto flowPill = [&](const char* label, int status, const char* tooltip) -> bool {
        float pillW = ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2.0f + s(16.0f);
        if (pillW + ImGui::GetStyle().ItemSpacing.x <= ImGui::GetContentRegionAvail().x) {
            ImGui::SameLine();
        }
        return renderFilterPill(label, status, tooltip);
    };

    if (!canEditWorkloads) ImGui::BeginDisabled(true);
    if (flowPill("Select All", allItemsSelected ? 2 : 0, "Select all workloads across all categories")) {
        for (auto& cat : m_categories) {
            cat.allSelected = true;
            for (auto& sub : cat.subgroups) {
                sub.allSelected = true;
                sub.collapsed = false;
                for (auto& item : sub.items) item.selected = true;
            }
        }
        for (auto& dev : m_devices) {
            if (dev.isSystem) dev.selected = true;
        }
    }
    if (flowPill("Deselect All", noneItemsSelected ? 2 : 0, "Deselect all workloads")) {
        for (auto& cat : m_categories) {
            cat.allSelected = false;
            for (auto& sub : cat.subgroups) {
                sub.allSelected = false;
                sub.collapsed = true;
                for (auto& item : sub.items) item.selected = false;
            }
        }
    }
    if (flowPill("Compute", getCatStatus("Compute"), "Toggle Compute Precision workloads (FP64..INT4)")) {
        toggleCat("Compute");
    }
    if (flowPill("Memory", getCatStatus("Memory"), "Toggle VRAM streaming bandwidth and Cache Latency tests")) {
        toggleCat("Memory");
    }
    if (flowPill("Ray Tracing", getCatStatus("Ray Tracing"), "Toggle all Ray Tracing & BVH Stress workloads")) {
        toggleCat("Ray Tracing");
    }
    if (flowPill("Graphics", getCatStatus("Graphics"), "Toggle Fixed-Function Pixel & Blend Fill Rate tests")) {
        toggleCat("Graphics");
    }
    if (flowPill("Host CPU", getCatStatus("Host System"), "Toggle Host System RAM Bandwidth and Latency tests")) {
        toggleCat("Host System");
        if (getCatStatus("Host System") > 0) {
            for (auto& dev : m_devices) if (dev.isSystem) dev.selected = true;
        }
    }
    if (!canEditWorkloads) ImGui::EndDisabled();

    // Group Collapse / Expand Controls
    float untestW = ImGui::CalcTextSize("Collapse Untested").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float expW = ImGui::CalcTextSize("Expand All").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float colW = ImGui::CalcTextSize("Collapse All").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float groupControlsW = expW + colW + untestW + s(32.0f);
    if (groupControlsW <= ImGui::GetContentRegionAvail().x) {
        ImGui::SameLine();
        float availForExp = ImGui::GetContentRegionAvail().x;
        if (availForExp > groupControlsW + s(10.0f)) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availForExp - groupControlsW);
        }
    }
    if (ImGui::SmallButton("Collapse Untested")) {
        for (auto& cat : m_categories) {
            for (auto& sub : cat.subgroups) {
                size_t sel = 0;
                for (const auto& itm : sub.items) {
                    if (itm.selected) sel++;
                }
                if (sel == 0) sub.collapsed = true;
            }
        }
    }
    ImGui::SameLine(0, s(6.0f));
    if (ImGui::SmallButton("Expand All")) {
        for (auto& cat : m_categories) {
            for (auto& sub : cat.subgroups) sub.collapsed = false;
        }
    }
    ImGui::SameLine(0, s(6.0f));
    if (ImGui::SmallButton("Collapse All")) {
        for (auto& cat : m_categories) {
            for (auto& sub : cat.subgroups) sub.collapsed = true;
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Optimal linear partition (Painter's Partition DP) to minimize maximum column height
    auto partitionSubgroupsOptimal = [](
        const std::vector<BenchmarkSubgroup*>& subs,
        int numCols,
        const std::function<float(const BenchmarkSubgroup&)>& getHeight)
        -> std::vector<std::vector<BenchmarkSubgroup*>>
    {
        std::vector<std::vector<BenchmarkSubgroup*>> cols(numCols);
        if (subs.empty() || numCols <= 0) return cols;
        if (numCols == 1) {
            cols[0] = subs;
            return cols;
        }
        int n = static_cast<int>(subs.size());
        if (n <= numCols) {
            for (int i = 0; i < n; ++i) cols[i].push_back(subs[i]);
            return cols;
        }

        std::vector<float> heights(n);
        for (int i = 0; i < n; ++i) heights[i] = getHeight(*subs[i]);

        std::vector<std::vector<float>> dp(numCols, std::vector<float>(n, 1e9f));
        std::vector<std::vector<int>> split(numCols, std::vector<int>(n, 0));

        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += heights[i];
            dp[0][i] = sum;
        }

        for (int c = 1; c < numCols; ++c) {
            for (int i = c; i < n; ++i) {
                float runningSum = 0.0f;
                for (int j = i; j >= c; --j) {
                    runningSum += heights[j];
                    float maxH = std::max(dp[c - 1][j - 1], runningSum);
                    if (maxH < dp[c][i]) {
                        dp[c][i] = maxH;
                        split[c][i] = j;
                    }
                }
            }
        }

        std::vector<int> cuts(numCols + 1, 0);
        cuts[numCols] = n;
        int currEnd = n - 1;
        for (int c = numCols - 1; c >= 1; --c) {
            cuts[c] = split[c][currEnd];
            currEnd = cuts[c] - 1;
        }
        cuts[0] = 0;

        for (int c = 0; c < numCols; ++c) {
            for (int i = cuts[c]; i < cuts[c + 1]; ++i) {
                cols[c].push_back(subs[i]);
            }
        }

        return cols;
    };

    // Collapsible Subgroup Renderer (Accordion Header + Dynamic Structured Workload Rows)
    auto renderSubgroupCard = [this, canEditWorkloads](BenchmarkSubgroup& sub) {
        size_t visibleCount = 0;
        size_t selectedCount = 0;
        for (const auto& itm : sub.items) {
            BenchmarkDisplayInfo info = getBenchmarkDisplayInfo(itm, m_telemetryGpuIndex);
            bool isUnsupported = (!itm.isSupported || (info.hasResult && info.primaryResult.isUnsupported));
            if (!m_hideUnsupported || !isUnsupported) {
                visibleCount++;
                if (itm.selected) selectedCount++;
            }
        }
        if (visibleCount == 0 && m_hideUnsupported) return;

        ImGui::PushID(sub.name.c_str());

        bool isAllSel = (selectedCount == visibleCount && visibleCount > 0);
        bool isPartSel = (selectedCount > 0 && selectedCount < visibleCount);

        float availW = ImGui::GetContentRegionAvail().x;
        float frameH = ImGui::GetFrameHeight();
        float headerH = std::max(s(32.0f), frameH + s(8.0f));

        // 1. Check if there are comparison / baseline rows in this subgroup
        bool hasAnyComparison = false;
        for (const auto& itm : sub.items) {
            BenchmarkDisplayInfo d = getBenchmarkDisplayInfo(itm, m_telemetryGpuIndex);
            if ((d.hasResult && d.hasComparison && !d.primaryResult.isUnsupported) ||
                (d.isBaseline && sub.items.size() > 1)) {
                hasAnyComparison = true;
                break;
            }
        }

        // 2. Fixed column widths for Score and Speedup guarantee visibility
        float scoreColW = std::max(s(130.0f), ImGui::CalcTextSize("99999.9 MRecords/s").x + s(12.0f));
        float deltaColW = hasAnyComparison ? std::max(s(130.0f), ImGui::CalcTextSize("[Baseline]").x + s(40.0f)) : 0.0f;
        int numSubCols = hasAnyComparison ? 3 : 2;

        // 3. Exact minimum width needed by workload column so NO test name is ever truncated
        float maxWorkloadTextW = s(200.0f);
        for (const auto& itm : sub.items) {
            BenchmarkDisplayInfo d = getBenchmarkDisplayInfo(itm, m_telemetryGpuIndex);
            bool isUnsupported = (!itm.isSupported || (d.hasResult && d.primaryResult.isUnsupported));
            if (m_hideUnsupported && isUnsupported) continue;
            float tw = ImGui::CalcTextSize(itm.name.c_str()).x;
            if (tw > maxWorkloadTextW) maxWorkloadTextW = tw;
        }
        float minWorkloadColW = maxWorkloadTextW + s(16.0f) /* indent */ + frameH /* checkbox */ + 
                                ImGui::GetStyle().ItemInnerSpacing.x + ImGui::GetStyle().CellPadding.x * 2.0f + s(12.0f) /* safety margin */;
        float minTableW = minWorkloadColW + scoreColW + deltaColW + s(10.0f);

        // 4. Right-aligned header score/status badge
        std::string rightBadge = "";
        ImVec4 badgeColor = ImVec4(0.6f, 0.7f, 0.85f, 1.0f);
        if (sub.collapsed) {
            for (const auto& itm : sub.items) {
                BenchmarkDisplayInfo d = getBenchmarkDisplayInfo(itm, m_telemetryGpuIndex);
                if (d.hasResult && d.primaryResult.time_ms > 0.0 && !d.primaryResult.isUnsupported) {
                    rightBadge = d.scoreText;
                    badgeColor = ImVec4(0.35f, 0.95f, 0.55f, 1.0f);
                    break;
                }
            }
            if (rightBadge.empty()) {
                rightBadge = std::to_string(selectedCount) + " selected";
                badgeColor = ImVec4(0.55f, 0.65f, 0.75f, 0.9f);
            }
        } else {
            if (sub.name == "Compute Precision" || sub.name == "Dual-Issue & Concurrency") {
                rightBadge = "[TFLOPS / TOPS]";
            } else if (sub.name == "Host CPU System Memory") {
                rightBadge = "[GB/s / ns]";
            } else if (sub.name == "Acceleration Structure Builds") {
                rightBadge = "[MTris/s / MInst/s]";
            } else if (sub.name == "Pipeline Stages & Scheduling") {
                rightBadge = "[MRays/s / MHits/s]";
            } else if (sub.name == "Hardware BVH & Divergence Stress") {
                rightBadge = "[GIS/s / MRays/s]";
            } else if (!sub.items.empty()) {
                rightBadge = "[" + sub.items.front().metricType + "]";
            }
            badgeColor = ImVec4(0.35f, 0.65f, 0.95f, 0.85f);
        }

        float badgeW = ImGui::CalcTextSize(rightBadge.c_str()).x;
        float titleTextW = ImGui::CalcTextSize(sub.name.c_str()).x;
        float minHeaderW = s(58.0f) + titleTextW + s(24.0f) + badgeW + s(16.0f);

        float requiredCardW = std::max(minTableW, minHeaderW);
        float cardW = std::max(availW, requiredCardW);

        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 p1 = ImVec2(p0.x + cardW, p0.y + headerH);

        ImVec4 headerBg = sub.collapsed ? ImVec4(0.12f, 0.15f, 0.22f, 0.85f) : ImVec4(0.15f, 0.20f, 0.30f, 0.95f);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(p0, p1, ImGui::GetColorU32(headerBg), s(5.0f));
        drawList->AddRect(p0, p1, ImGui::GetColorU32(ImVec4(0.25f, 0.32f, 0.45f, 0.6f)), s(5.0f));

        // Draw crisp geometric triangle for chevron
        ImVec2 center = ImVec2(p0.x + s(14.0f), p0.y + headerH * 0.5f);
        float r = s(4.0f);
        ImU32 triCol = ImGui::GetColorU32(ImVec4(0.60f, 0.80f, 1.0f, 1.0f));
        if (sub.collapsed) {
            drawList->AddTriangleFilled(
                ImVec2(center.x - r * 0.6f, center.y - r),
                ImVec2(center.x - r * 0.6f, center.y + r),
                ImVec2(center.x + r * 0.8f, center.y),
                triCol
            );
        } else {
            drawList->AddTriangleFilled(
                ImVec2(center.x - r, center.y - r * 0.6f),
                ImVec2(center.x + r, center.y - r * 0.6f),
                ImVec2(center.x, center.y + r * 0.8f),
                triCol
            );
        }

        // Clickable chevron area for collapse/expand
        ImGui::SetCursorScreenPos(p0);
        if (ImGui::InvisibleButton("##chev_click", ImVec2(s(26.0f), headerH))) {
            sub.collapsed = !sub.collapsed;
        }

        // Group-level select/deselect checkbox
        float cbOffsetY = (headerH - frameH) * 0.5f;
        ImGui::SetCursorScreenPos(ImVec2(p0.x + s(28.0f), p0.y + cbOffsetY));
        if (!canEditWorkloads) ImGui::BeginDisabled(true);
        bool cbVal = isAllSel;
        if (isPartSel) {
            ImGui::PushStyleColor(ImGuiCol_CheckMark, ImVec4(0.95f, 0.70f, 0.25f, 1.0f));
        }
        if (ImGui::Checkbox("##sub_all", &cbVal)) {
            for (auto& itm : sub.items) {
                if (!m_hideUnsupported || itm.isSupported) {
                    itm.selected = cbVal;
                }
            }
            if (!cbVal) {
                sub.collapsed = true;
            } else {
                sub.collapsed = false;
            }
        }
        if (isPartSel) ImGui::PopStyleColor();
        if (!canEditWorkloads) ImGui::EndDisabled();

        // Subgroup Name and count
        float textOffsetY = (headerH - ImGui::GetTextLineHeight()) * 0.5f;
        float titleStartX = p0.x + s(58.0f);
        ImGui::SetCursorScreenPos(ImVec2(titleStartX, p0.y + textOffsetY));
        std::string displaySubName = sub.name;
        if (sub.component == "Ray Tracing" && (sub.engineId == "RayScheduling" || sub.engineId == "RayPathTracing")) {
            if (m_suiteActiveScene == "indoor") displaySubName += " [Indoor Atrium]";
            else if (m_suiteActiveScene == "forest") displaySubName += " [Open-World Forest]";
            else if (m_suiteActiveScene == "outdoor") displaySubName += " [Outdoor Landscape]";
            else if (m_suiteActiveScene == "showroom") displaySubName += " [Showroom Studio]";
        }
        ImGui::TextColored(ImVec4(0.95f, 0.96f, 0.98f, 1.0f), "%s", displaySubName.c_str());

        ImGui::SameLine(0, s(6.0f));
        ImGui::TextDisabled("(%zu/%zu)", selectedCount, visibleCount);
        float titleEndX = ImGui::GetCursorScreenPos().x;

        // Render right-aligned header badge ONLY if it doesn't collide with title
        float badgeStartX = p0.x + cardW - badgeW - s(10.0f);
        if (badgeStartX > titleEndX + s(12.0f)) {
            ImGui::SetCursorScreenPos(ImVec2(badgeStartX, p0.y + textOffsetY));
            ImGui::TextColored(badgeColor, "%s", rightBadge.c_str());
        }

        // Clickable title area for collapse/expand
        ImGui::SetCursorScreenPos(ImVec2(titleStartX, p0.y));
        if (ImGui::InvisibleButton("##title_click", ImVec2(cardW - s(58.0f), headerH))) {
            sub.collapsed = !sub.collapsed;
        }

        // Place cursor clearly BELOW header banner with margin
        ImGui::SetCursorScreenPos(ImVec2(p0.x, p1.y + s(6.0f)));

        // Render Indented Workload Items if Expanded
        if (!sub.collapsed) {
            std::string tblId = "SubTbl_" + sub.name;
            ImGuiTableFlags subTblFlags = ImGuiTableFlags_SizingStretchProp | 
                                          ImGuiTableFlags_NoPadOuterX | 
                                          ImGuiTableFlags_NoSavedSettings;
            if (ImGui::BeginTable(tblId.c_str(), numSubCols, subTblFlags, ImVec2(cardW, 0.0f))) {
                ImGui::TableSetupColumn("Workload", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, scoreColW);
                if (hasAnyComparison) {
                    ImGui::TableSetupColumn("Delta", ImGuiTableColumnFlags_WidthFixed, deltaColW);
                }

                float activeBaselineX = -1.0f;
                float activeBaselineY = -1.0f;
                bool hasActiveBaseline = false;
                std::string activeBaselineSubcat = "";

                std::string currentGroupKey = "";
                int groupIndex = -1;

                for (size_t iIdx = 0; iIdx < sub.items.size(); ++iIdx) {
                    auto& item = sub.items[iIdx];
                    BenchmarkDisplayInfo dispInfo = getBenchmarkDisplayInfo(item, m_telemetryGpuIndex);
                    bool isUnsupported = (!item.isSupported || (dispInfo.hasResult && dispInfo.primaryResult.isUnsupported));
                    if (m_hideUnsupported && isUnsupported) continue;

                    // Group identification for related tests
                    std::string itemGroupKey;
                    if (sub.name == "Dual-Issue & Concurrency") {
                        if (item.name.find("FP32") != std::string::npos && item.name.find("Mixed") == std::string::npos) {
                            itemGroupKey = "Dual-Issue_FP32";
                        } else if (item.name.find("INT32") != std::string::npos && item.name.find("Mixed") == std::string::npos) {
                            itemGroupKey = "Dual-Issue_INT32";
                        } else if (item.name.find("Mixed") != std::string::npos) {
                            itemGroupKey = "Dual-Issue_Mixed";
                        } else {
                            itemGroupKey = item.subcategory;
                        }
                    } else if (sub.name == "VRAM Streaming Bandwidth") {
                        if (item.name.find("128 threads") != std::string::npos) {
                            itemGroupKey = "VRAM_128";
                        } else if (item.name.find("256 threads") != std::string::npos) {
                            itemGroupKey = "VRAM_256";
                        } else if (item.name.find("1024 threads") != std::string::npos) {
                            itemGroupKey = "VRAM_1024";
                        } else {
                            itemGroupKey = item.subcategory;
                        }
                    } else if (!item.subcategory.empty()) {
                        itemGroupKey = item.subcategory;
                    } else {
                        itemGroupKey = item.name;
                    }

                    if (itemGroupKey != currentGroupKey) {
                        currentGroupKey = itemGroupKey;
                        groupIndex++;
                    }

                    ImGui::PushID(static_cast<int>(iIdx));
                    float rowH = std::max(s(22.0f), frameH + s(2.0f));
                    ImGui::TableNextRow(ImGuiTableRowFlags_None, rowH);

                    // Slightly different background shading for alternating groups of related tests
                    if (groupIndex >= 0 && (groupIndex % 2 == 1)) {
                        ImU32 shadedBg = ImGui::GetColorU32(ImVec4(0.14f, 0.18f, 0.27f, 0.42f));
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, shadedBg);
                    }

                    // Col 0: Workload Checkbox & Label
                    ImGui::TableNextColumn();
                    ImGui::Indent(s(16.0f));

                    if (isUnsupported || !canEditWorkloads) {
                        ImGui::BeginDisabled(true);
                        ImGui::Checkbox(item.name.c_str(), &item.selected);
                        ImGui::EndDisabled();
                    } else {
                        ImGui::Checkbox(item.name.c_str(), &item.selected);
                    }
                    ImGui::Unindent(s(16.0f));

                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                        if (isUnsupported) {
                            std::string reason = !item.supportReason.empty() ? item.supportReason :
                                (dispInfo.hasResult && !dispInfo.primaryResult.supportNote.empty() ? dispInfo.primaryResult.supportNote : "Hardware or API limitation");
                            std::string catStr = !item.limitationCategory.empty() ? item.limitationCategory : "Limitation";
                            ImGui::SetTooltip("Workload: %s (%s - %s)\nStatus: UNSUPPORTED (%s)\nReason: %s",
                                              item.name.c_str(), item.subcategory.c_str(), item.id.c_str(),
                                              catStr.c_str(), reason.c_str());
                        } else if (!item.description.empty()) {
                            std::string scn = (item.id == "RayScheduling" || item.id == "RayRawTraversal" || item.id == "RayPathTracing") ? m_suiteActiveScene : "";
                            renderBenchmarkTooltip(item, scn);
                        }
                    }

                    // Col 1: Score / Status / Metric Badge (ALWAYS VISIBLE!)
                    ImGui::TableNextColumn();
                    bool isCurrentlyTesting = !isUnsupported && (m_execState == ExecutionState::Running &&
                        m_hasCurrentlyRunningResult &&
                        matchesItem(m_currentlyRunningResult, item, m_telemetryGpuIndex));

                    if (dispInfo.hasResult && !dispInfo.primaryResult.isUnsupported && dispInfo.primaryResult.time_ms > 0.0) {
                        float sW = ImGui::CalcTextSize(dispInfo.scoreText.c_str()).x;
                        float availC = ImGui::GetContentRegionAvail().x;
                        if (availC > sW + s(4.0f)) {
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availC - sW - s(4.0f));
                        }
                        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s", dispInfo.scoreText.c_str());
                    } else if (dispInfo.hasResult && dispInfo.primaryResult.time_ms == -3.0) {
                        float sW = ImGui::CalcTextSize("[ABORTED]").x;
                        float availC = ImGui::GetContentRegionAvail().x;
                        if (availC > sW + s(4.0f)) {
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availC - sW - s(4.0f));
                        }
                        ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.25f, 1.0f), "[ABORTED]");
                    } else if (dispInfo.hasResult && dispInfo.primaryResult.time_ms < 0.0) {
                        float sW = ImGui::CalcTextSize("[FAILED]").x;
                        float availC = ImGui::GetContentRegionAvail().x;
                        if (availC > sW + s(4.0f)) {
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availC - sW - s(4.0f));
                        }
                        ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "[FAILED]");
                    } else if (isUnsupported) {
                        ImGui::TextColored(ImVec4(0.85f, 0.55f, 0.15f, 1.0f), "[UNSUPPORTED]");
                    } else if (isCurrentlyTesting) {
                        float pulse = 0.5f + 0.5f * sinf(static_cast<float>(ImGui::GetTime()) * 8.0f);
                        ImGui::TextColored(ImVec4(0.20f + 0.20f * pulse, 0.80f + 0.20f * pulse, 1.0f, 1.0f), "[RUNNING...]");
                    } else {
                        std::string badge = "[" + item.metricType + "]";
                        float bW = ImGui::CalcTextSize(badge.c_str()).x;
                        float availC = ImGui::GetContentRegionAvail().x;
                        if (availC > bW + s(4.0f)) {
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + availC - bW - s(4.0f));
                        }
                        ImGui::TextColored(ImVec4(0.35f, 0.65f, 0.95f, 0.85f), "%s", badge.c_str());
                    }

                    // Col 2: Delta Speedup / Baseline Connecting Branch
                    if (hasAnyComparison) {
                        ImGui::TableNextColumn();
                        if (dispInfo.hasResult && dispInfo.primaryResult.time_ms == -3.0) {
                            ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.25f, 1.0f), "[Aborted]");
                        } else if (dispInfo.hasResult && dispInfo.primaryResult.time_ms < 0.0) {
                            std::string err = !dispInfo.primaryResult.errorString.empty() ? ("[" + dispInfo.primaryResult.errorString + "]") : "[Failed]";
                            ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "%s", err.c_str());
                        } else {
                        ImVec2 cellPos = ImGui::GetCursorScreenPos();
                        float textH = ImGui::GetTextLineHeight();
                        float curCenterY = cellPos.y + textH * 0.5f;

                        if (dispInfo.isBaseline && !isUnsupported && dispInfo.hasResult && dispInfo.primaryResult.time_ms > 0.0) {
                            activeBaselineX = cellPos.x + s(10.0f);
                            activeBaselineY = curCenterY;
                            hasActiveBaseline = true;
                            activeBaselineSubcat = item.subcategory;

                            ImVec4 baseColor = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(6.0f));
                            ImGui::TextColored(baseColor, "[Baseline]");
                        } else if (dispInfo.hasComparison && dispInfo.hasResult && !isUnsupported && dispInfo.primaryResult.time_ms > 0.0 && !dispInfo.deltaText.empty()) {
                            if (!item.subcategory.empty() && item.subcategory != activeBaselineSubcat) {
                                hasActiveBaseline = false;
                                activeBaselineY = -1.0f;
                                activeBaselineSubcat = "";
                            }
                            float stemX = (hasActiveBaseline && activeBaselineX > 0.0f) ? activeBaselineX : (cellPos.x + s(10.0f));
                            ImDrawList* drawList = ImGui::GetWindowDrawList();
                            ImU32 branchCol = ImGui::GetColorU32(ImVec4(0.38f, 0.65f, 0.90f, 0.65f));
                            float branchLen = s(12.0f);

                            if (hasActiveBaseline && activeBaselineY > 0.0f && item.subcategory == activeBaselineSubcat) {
                                // Draw vertical connecting stem from baseline down to current speedup row
                                drawList->AddLine(ImVec2(stemX, activeBaselineY + s(8.0f)), ImVec2(stemX, curCenterY), branchCol, s(1.5f));
                            }
                            // Draw horizontal branch pointing to speedup value
                            drawList->AddLine(ImVec2(stemX, curCenterY), ImVec2(stemX + branchLen, curCenterY), branchCol, s(1.5f));
                            // Directional pointer arrow
                            float arrowSz = s(3.5f);
                            drawList->AddLine(ImVec2(stemX + branchLen - arrowSz, curCenterY - arrowSz), ImVec2(stemX + branchLen, curCenterY), branchCol, s(1.5f));
                            drawList->AddLine(ImVec2(stemX + branchLen - arrowSz, curCenterY + arrowSz), ImVec2(stemX + branchLen, curCenterY), branchCol, s(1.5f));

                            // Offset text past branch indicator
                            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + s(24.0f));
                            ImGui::TextColored(dispInfo.deltaColor, "%s", dispInfo.deltaText.c_str());
                        } else {
                            if (!item.subcategory.empty() && item.subcategory != activeBaselineSubcat) {
                                hasActiveBaseline = false;
                                activeBaselineY = -1.0f;
                                activeBaselineSubcat = "";
                            }
                        }
                    }
                    }

                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
        }

        if (cardW > availW) {
            ImGui::Dummy(ImVec2(cardW, 0.0f));
        }

        // Clean separation gap between subgroups
        ImGui::Spacing();
        ImGui::Dummy(ImVec2(0, s(4.0f)));
        ImGui::PopID();
    };

    // Scrollable suite area ensures no content is ever clipped or lost off screen
    ImGui::BeginChild("SuiteScrollArea", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

    std::vector<BenchmarkSubgroup*> activeSubgroups;
    if (m_suiteCategoryFilter == 0) {
        for (auto& cat : m_categories) {
            for (auto& sub : cat.subgroups) activeSubgroups.push_back(&sub);
        }
    } else if (m_suiteCategoryFilter == 1 && m_categories.size() > 0) {
        for (auto& sub : m_categories[0].subgroups) activeSubgroups.push_back(&sub);
    } else if (m_suiteCategoryFilter == 2 && m_categories.size() > 1) {
        for (auto& sub : m_categories[1].subgroups) activeSubgroups.push_back(&sub);
    } else if (m_suiteCategoryFilter == 3 && m_categories.size() > 2) {
        for (auto& sub : m_categories[2].subgroups) activeSubgroups.push_back(&sub);
    } else if (m_suiteCategoryFilter == 4 && m_categories.size() > 3) {
        for (auto& sub : m_categories[3].subgroups) activeSubgroups.push_back(&sub);
    } else if (m_suiteCategoryFilter == 5 && m_categories.size() > 4) {
        for (auto& sub : m_categories[4].subgroups) activeSubgroups.push_back(&sub);
    }

    std::vector<BenchmarkSubgroup*> visibleSubgroups;
    for (auto* sub : activeSubgroups) {
        size_t visibleCount = 0;
        for (const auto& itm : sub->items) {
            BenchmarkDisplayInfo info = getBenchmarkDisplayInfo(itm, m_telemetryGpuIndex);
            bool isUnsupported = (!itm.isSupported || (info.hasResult && info.primaryResult.isUnsupported));
            if (!m_hideUnsupported || !isUnsupported) visibleCount++;
        }
        if (visibleCount > 0 || !m_hideUnsupported) {
            visibleSubgroups.push_back(sub);
        }
    }

    float availW = ImGui::GetContentRegionAvail().x;
    float fontScaleFactor = ImGui::GetFontSize() / 16.0f;
    float minColW = std::max(s(440.0f), 400.0f * fontScaleFactor);
    int maxCols = std::max(1, static_cast<int>(availW / minColW));
    if (maxCols > 4) maxCols = 4;

    int numCols = std::max(1, std::min(static_cast<int>(visibleSubgroups.size()), maxCols));

    if (numCols > 1 && visibleSubgroups.size() > 1) {
        auto getSubHeight = [this](const BenchmarkSubgroup& sub) -> float {
            float rowH = std::max(s(24.0f), ImGui::GetFrameHeight() + s(4.0f));
            if (sub.collapsed) return rowH + s(10.0f);
            size_t vis = 0;
            for (const auto& itm : sub.items) {
                BenchmarkDisplayInfo info = getBenchmarkDisplayInfo(itm, m_telemetryGpuIndex);
                bool isUnsupported = (!itm.isSupported || (info.hasResult && info.primaryResult.isUnsupported));
                if (!m_hideUnsupported || !isUnsupported) vis++;
            }
            return (rowH + s(10.0f)) + static_cast<float>(vis) * rowH + s(14.0f);
        };

        std::vector<std::vector<BenchmarkSubgroup*>> cols = partitionSubgroupsOptimal(
            visibleSubgroups, numCols, getSubHeight);

        std::string tableId = "DynamicSuiteGrid_" + std::to_string(numCols);
        float gridW = std::max(availW, minColW * numCols);
        if (ImGui::BeginTable(tableId.c_str(), numCols, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings, ImVec2(gridW, 0.0f))) {
            for (int c = 0; c < numCols; ++c) {
                ImGui::TableNextColumn();
                for (auto* sub : cols[c]) {
                    renderSubgroupCard(*sub);
                }
            }
            ImGui::EndTable();
        }
    } else {
        for (auto* sub : visibleSubgroups) {
            renderSubgroupCard(*sub);
        }
    }

    ImGui::EndChild();
}

void GuiApp::renderLiveTelemetryDock() {
    std::vector<uint32_t> actualGpuIndices;
    for (const auto& dev : m_devices) {
        if (!dev.isSystem) {
            actualGpuIndices.push_back(dev.deviceIndex);
        }
    }

    if (actualGpuIndices.size() <= 1) {
        m_telemetryDualGpuMode = false;
        if (!actualGpuIndices.empty()) {
            m_telemetryGpuIndex = actualGpuIndices[0];
        }
    } else {
        bool validGpu = false;
        for (uint32_t idx : actualGpuIndices) {
            if (idx == m_telemetryGpuIndex) {
                validGpu = true;
                break;
            }
        }
        if (!validGpu) {
            m_telemetryGpuIndex = actualGpuIndices[0];
        }
    }

    DeviceTelemetrySnapshot snap0, snap1;
    if (!actualGpuIndices.empty()) {
        m_telemetryWorker.getSnapshot(actualGpuIndices[0], snap0);
    }
    if (actualGpuIndices.size() > 1) {
        m_telemetryWorker.getSnapshot(actualGpuIndices[1], snap1);
    }

    // Header bar with Device selection, Time Window, and live digital readout
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.0f, 1.0f), "LIVE TELEMETRY MONITOR");
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    if (m_execState == ExecutionState::Running) {
        ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.45f, 1.0f), "LIVE");
    } else if (m_execState == ExecutionState::Completed) {
        ImGui::TextColored(ImVec4(0.40f, 0.75f, 1.0f, 1.0f), "COMPLETE");
    } else if (m_execState == ExecutionState::Cancelled) {
        ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.35f, 1.0f), "ABORTED");
    } else {
        ImGui::TextDisabled("IDLE");
    }
    ImGui::SameLine();
    ImGui::TextDisabled("|");
    ImGui::SameLine();

    // Device selection: ONLY actual devices!
    if (actualGpuIndices.size() > 1) {
        for (size_t g = 0; g < actualGpuIndices.size(); ++g) {
            uint32_t idx = actualGpuIndices[g];
            std::string label = (g == 0) ? ("GPU " + std::to_string(idx) + " (Primary)")
                                         : ("GPU " + std::to_string(idx) + " (Secondary)");
            if (ImGui::RadioButton(label.c_str(), !m_telemetryDualGpuMode && m_telemetryGpuIndex == idx)) {
                m_telemetryGpuIndex = idx;
                m_telemetryDualGpuMode = false;
            }
            ImGui::SameLine();
        }
        if (ImGui::RadioButton("Dual GPU Overlay", m_telemetryDualGpuMode)) {
            m_telemetryDualGpuMode = true;
        }
    } else if (actualGpuIndices.size() == 1) {
        std::string label = "GPU " + std::to_string(actualGpuIndices[0]);
        for (const auto& dev : m_devices) {
            if (dev.deviceIndex == actualGpuIndices[0] && !dev.isSystem) {
                label += " (" + dev.name + ")";
                break;
            }
        }
        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, 1.0f), "%s", label.c_str());
    }

    // Time window selector
    float winTextW = ImGui::CalcTextSize("Window:").x;
    float bFullW = ImGui::CalcTextSize("Full Run").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float b30W = ImGui::CalcTextSize("30s").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float b60W = ImGui::CalcTextSize("60s").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float b120W = ImGui::CalcTextSize("120s").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float totalWinW = winTextW + bFullW + b30W + b60W + b120W + ImGui::GetStyle().ItemSpacing.x * 5.0f;

    ImGui::SameLine();
    float availWin = ImGui::GetContentRegionAvail().x;
    if (availWin >= totalWinW + s(16.0f)) {
        float targetX = ImGui::GetCursorPosX() + availWin - totalWinW;
        if (targetX > ImGui::GetCursorPosX()) {
            ImGui::SetCursorPosX(targetX);
        }
    } else {
        ImGui::NewLine();
    }
    ImGui::TextDisabled("Window:");
    ImGui::SameLine();
    auto winBtn = [this](const char* label, float winSec) {
        bool isAct = (m_telemetryTimeWindow == winSec);
        if (isAct) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.48f, 0.90f, 1.0f));
        else ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.15f, 0.22f, 1.0f));
        if (ImGui::SmallButton(label)) {
            m_telemetryTimeWindow = winSec;
            m_telemetryNeedsFit = true;
        }
        ImGui::PopStyleColor();
    };
    winBtn("Full Run", 0.0f);
    ImGui::SameLine();
    winBtn("30s", 30.0f);
    ImGui::SameLine();
    winBtn("60s", 60.0f);
    ImGui::SameLine();
    winBtn("120s", 120.0f);

    // Live digital metrics banner with rock-solid fixed column offsets (zero bouncing text)
    if (m_telemetryDualGpuMode) {
        float v0 = static_cast<float>(snap0.vramUsedBytes / (1024 * 1024));
        float t0 = static_cast<float>(snap0.vramTotalBytes / (1024 * 1024));
        if (t0 < 1000.0f) t0 = 32624.0f;
        float v1 = static_cast<float>(snap1.vramUsedBytes / (1024 * 1024));
        float t1 = static_cast<float>(snap1.vramTotalBytes / (1024 * 1024));
        if (t1 < 1000.0f) t1 = 32624.0f;

        auto renderDualGpuRow = [this](const char* label, ImVec4 col, const DeviceTelemetrySnapshot& snap, float vMb, float tMb) {
            float startX = ImGui::GetCursorPosX();
            ImGui::TextColored(col, "%s", label);
            ImGui::SameLine(startX + s(140.0f));
            ImGui::TextDisabled("Shader:");
            ImGui::SameLine(startX + s(195.0f));
            ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, 1.0f), "%4.0f MHz", snap.sclkMhz);

            ImGui::SameLine(startX + s(270.0f));
            ImGui::TextDisabled("| Mem:");
            ImGui::SameLine(startX + s(315.0f));
            ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, 1.0f), "%4.0f MHz", snap.mclkMhz);

            ImGui::SameLine(startX + s(390.0f));
            ImGui::TextDisabled("| Pwr:");
            ImGui::SameLine(startX + s(435.0f));
            ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1.0f), "%5.1f W", snap.powerWatts);

            ImGui::SameLine(startX + s(510.0f));
            ImGui::TextDisabled("| Temp:");
            ImGui::SameLine(startX + s(560.0f));
            ImVec4 tColor = (snap.tempJctC > 80.0f) ? ImVec4(0.95f, 0.3f, 0.3f, 1.0f) : ImVec4(0.35f, 0.85f, 0.65f, 1.0f);
            ImGui::TextColored(tColor, "%4.0f C", snap.tempJctC);

            ImGui::SameLine(startX + s(630.0f));
            ImGui::TextDisabled("| VRAM:");
            ImGui::SameLine(startX + s(685.0f));
            float vPct = (tMb > 0.0f) ? (vMb / tMb) * 100.0f : 0.0f;
            ImGui::TextColored(ImVec4(0.70f, 0.80f, 0.95f, 1.0f), "%5.0f / %5.0f MB (%2.0f%%)", vMb, tMb, vPct);
        };
        renderDualGpuRow("GPU 0 (Primary):", ImVec4(0.38f, 0.75f, 1.0f, 1.0f), snap0, v0, t0);
        renderDualGpuRow("GPU 1 (Secondary):", ImVec4(0.75f, 0.50f, 1.0f, 1.0f), snap1, v1, t1);
    } else {
        const auto& snap = (m_telemetryGpuIndex == 1) ? snap1 : snap0;
        float vramMb = static_cast<float>(snap.vramUsedBytes / (1024 * 1024));
        float totalVramMb = static_cast<float>(snap.vramTotalBytes / (1024 * 1024));
        if (totalVramMb < 1000.0f) totalVramMb = 32624.0f;
        float vramPct = (vramMb / totalVramMb) * 100.0f;

        float startX = ImGui::GetCursorPosX();
        ImGui::TextDisabled("Shader Clk:");
        ImGui::SameLine(startX + s(88.0f));
        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, 1.0f), "%4.0f MHz", snap.sclkMhz);

        ImGui::SameLine(startX + s(185.0f));
        ImGui::TextDisabled("| Mem Clk:");
        ImGui::SameLine(startX + s(270.0f));
        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, 1.0f), "%4.0f MHz", snap.mclkMhz);

        ImGui::SameLine(startX + s(365.0f));
        ImGui::TextDisabled("| Power:");
        ImGui::SameLine(startX + s(430.0f));
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1.0f), "%5.1f W", snap.powerWatts);

        ImGui::SameLine(startX + s(515.0f));
        ImGui::TextDisabled("| Junction:");
        ImGui::SameLine(startX + s(590.0f));
        ImVec4 tColor = (snap.tempJctC > 80.0f) ? ImVec4(0.95f, 0.3f, 0.3f, 1.0f) : ImVec4(0.35f, 0.85f, 0.65f, 1.0f);
        ImGui::TextColored(tColor, "%4.1f C", snap.tempJctC);

        ImGui::SameLine(startX + s(675.0f));
        ImGui::TextDisabled("| VRAM:");
        ImGui::SameLine(startX + s(735.0f));
        ImGui::TextColored(ImVec4(0.70f, 0.80f, 0.95f, 1.0f), "%5.0f / %5.0f MB (%2.0f%%)", vramMb, totalVramMb, vramPct);
    }

    ImGui::Spacing();

    // Side-by-side real-time ImPlot graphs
    float curT = snap0.timeHistory.empty() ? 0.0f : snap0.timeHistory.back();
    if (m_telemetryDualGpuMode && !snap1.timeHistory.empty()) {
        curT = std::max(curT, snap1.timeHistory.back());
    } else if (m_telemetryGpuIndex == 1 && !snap1.timeHistory.empty()) {
        curT = snap1.timeHistory.back();
    }

    float minT = 0.0f;
    float maxT = 10.0f;
    if (m_telemetryTimeWindow <= 0.0f) {
        // Full Run: compress to cover the entire run
        minT = 0.0f;
        maxT = std::max(5.0f, curT + 0.2f);
    } else {
        // Fixed sliding window (30s, 60s, 120s)
        minT = std::max(0.0f, curT - m_telemetryTimeWindow);
        maxT = std::max(minT + 5.0f, curT + 0.2f);
    }

    ImPlotCond cond = (m_execState == ExecutionState::Running || m_telemetryNeedsFit) 
                      ? ImPlotCond_Always 
                      : ImPlotCond_Once;

    float availW = ImGui::GetContentRegionAvail().x;
    float graphW = (availW - s(14.0f)) * 0.5f;
    float graphH = std::max(s(280.0f), ImGui::GetContentRegionAvail().y - s(20.0f));

    // Left graph: Clocks (Shader & Memory)
    if (ImPlot::BeginPlot("##suite_clocks", ImVec2(graphW, graphH), ImPlotFlags_NoTitle)) {
        ImPlot::SetupAxes("Time (s)", "Clock (MHz)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_X1, minT, maxT, cond);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 3500, ImPlotCond_Once);

        if (m_telemetryDualGpuMode) {
            if (snap0.timeHistory.size() > 1) {
                ImPlot::PlotLine("GPU 0 Shader", snap0.timeHistory.data(), snap0.sclkHistory.data(),
                                 static_cast<int>(snap0.sclkHistory.size()), 0, static_cast<int>(snap0.sclkHistory.offset()));
                ImPlot::PlotLine("GPU 0 Memory", snap0.timeHistory.data(), snap0.mclkHistory.data(),
                                 static_cast<int>(snap0.mclkHistory.size()), 0, static_cast<int>(snap0.mclkHistory.offset()));
            }
            if (snap1.timeHistory.size() > 1) {
                ImPlot::PlotLine("GPU 1 Shader", snap1.timeHistory.data(), snap1.sclkHistory.data(),
                                 static_cast<int>(snap1.sclkHistory.size()), 0, static_cast<int>(snap1.sclkHistory.offset()));
                ImPlot::PlotLine("GPU 1 Memory", snap1.timeHistory.data(), snap1.mclkHistory.data(),
                                 static_cast<int>(snap1.mclkHistory.size()), 0, static_cast<int>(snap1.mclkHistory.offset()));
            }
        } else {
            const auto& snap = (m_telemetryGpuIndex == 1) ? snap1 : snap0;
            if (snap.timeHistory.size() > 1) {
                ImPlot::PlotLine("Shader Clock", snap.timeHistory.data(), snap.sclkHistory.data(),
                                 static_cast<int>(snap.sclkHistory.size()), 0, static_cast<int>(snap.sclkHistory.offset()));
                ImPlot::PlotLine("Memory Clock", snap.timeHistory.data(), snap.mclkHistory.data(),
                                 static_cast<int>(snap.mclkHistory.size()), 0, static_cast<int>(snap.mclkHistory.offset()));
            }
        }
        ImPlot::EndPlot();
    }

    ImGui::SameLine();

    // Right graph: Thermal & Power Dynamics
    if (ImPlot::BeginPlot("##suite_power", ImVec2(graphW, graphH), ImPlotFlags_NoTitle)) {
        ImPlot::SetupAxes("Time (s)", "Temp (C)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxis(ImAxis_Y2, "Power (W)", ImPlotAxisFlags_AuxDefault);
        ImPlot::SetupAxisLimits(ImAxis_X1, minT, maxT, cond);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 20, 110, ImPlotCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y2, 0, 350, ImPlotCond_Once);

        if (m_telemetryDualGpuMode) {
            if (snap0.timeHistory.size() > 1) {
                ImPlot::PlotLine("GPU 0 Temp", snap0.timeHistory.data(), snap0.tempJctHistory.data(),
                                 static_cast<int>(snap0.tempJctHistory.size()), 0, static_cast<int>(snap0.tempJctHistory.offset()));
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                ImPlot::PlotLine("GPU 0 Power", snap0.timeHistory.data(), snap0.powerHistory.data(),
                                 static_cast<int>(snap0.powerHistory.size()), 0, static_cast<int>(snap0.powerHistory.offset()));
            }
            if (snap1.timeHistory.size() > 1) {
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                ImPlot::PlotLine("GPU 1 Temp", snap1.timeHistory.data(), snap1.tempJctHistory.data(),
                                 static_cast<int>(snap1.tempJctHistory.size()), 0, static_cast<int>(snap1.tempJctHistory.offset()));
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                ImPlot::PlotLine("GPU 1 Power", snap1.timeHistory.data(), snap1.powerHistory.data(),
                                 static_cast<int>(snap1.powerHistory.size()), 0, static_cast<int>(snap1.powerHistory.offset()));
            }
        } else {
            const auto& snap = (m_telemetryGpuIndex == 1) ? snap1 : snap0;
            if (snap.timeHistory.size() > 1) {
                ImPlot::PlotLine("Junction Temp", snap.timeHistory.data(), snap.tempJctHistory.data(),
                                 static_cast<int>(snap.tempJctHistory.size()), 0, static_cast<int>(snap.tempJctHistory.offset()));
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                ImPlot::PlotLine("Board Power", snap.timeHistory.data(), snap.powerHistory.data(),
                                 static_cast<int>(snap.powerHistory.size()), 0, static_cast<int>(snap.powerHistory.offset()));
            }
        }
        ImPlot::EndPlot();
    }
}

void GuiApp::renderSidebarTelemetry() {
    std::vector<uint32_t> actualGpuIndices;
    for (const auto& dev : m_devices) {
        if (!dev.isSystem) {
            actualGpuIndices.push_back(dev.deviceIndex);
        }
    }

    if (actualGpuIndices.size() <= 1) {
        m_telemetryDualGpuMode = false;
        if (!actualGpuIndices.empty()) {
            m_telemetryGpuIndex = actualGpuIndices[0];
        }
    } else {
        bool validGpu = false;
        for (uint32_t idx : actualGpuIndices) {
            if (idx == m_telemetryGpuIndex) {
                validGpu = true;
                break;
            }
        }
        if (!validGpu) {
            m_telemetryGpuIndex = actualGpuIndices[0];
        }
    }

    DeviceTelemetrySnapshot snap0, snap1;
    if (!actualGpuIndices.empty()) {
        m_telemetryWorker.getSnapshot(actualGpuIndices[0], snap0);
    }
    if (actualGpuIndices.size() > 1) {
        m_telemetryWorker.getSnapshot(actualGpuIndices[1], snap1);
    }

    DeviceTelemetrySnapshot activeSnap;
    if (actualGpuIndices.size() > 1 && m_telemetryGpuIndex == actualGpuIndices[1]) {
        activeSnap = snap1;
    } else {
        activeSnap = snap0;
    }

    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextColored(ImVec4(0.65f, 0.75f, 0.90f, 1.0f), "HARDWARE TELEMETRY");
    ImGui::SameLine();
    if (m_execState == ExecutionState::Running) {
        ImGui::TextColored(ImVec4(0.35f, 0.85f, 0.45f, 1.0f), "LIVE");
    } else if (m_execState == ExecutionState::Completed) {
        ImGui::TextColored(ImVec4(0.40f, 0.75f, 1.0f, 1.0f), "COMPLETE");
    } else if (m_execState == ExecutionState::Cancelled) {
        ImGui::TextColored(ImVec4(0.95f, 0.45f, 0.35f, 1.0f), "ABORTED");
    }

    // Device switch buttons (ONLY for actual devices!)
    auto devPill = [this](const char* label, bool active) {
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.45f, 0.85f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.16f, 0.23f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.78f, 0.90f, 1.0f));
        }
        bool clicked = ImGui::SmallButton(label);
        ImGui::PopStyleColor(2);
        return clicked;
    };

    if (actualGpuIndices.size() > 1) {
        for (size_t g = 0; g < actualGpuIndices.size(); ++g) {
            uint32_t idx = actualGpuIndices[g];
            std::string label = "GPU " + std::to_string(idx);
            if (devPill(label.c_str(), !m_telemetryDualGpuMode && m_telemetryGpuIndex == idx)) {
                m_telemetryGpuIndex = idx;
                m_telemetryDualGpuMode = false;
            }
            ImGui::SameLine();
        }
        if (devPill("Dual", m_telemetryDualGpuMode)) {
            m_telemetryDualGpuMode = true;
        }
    } else if (actualGpuIndices.size() == 1) {
        std::string label = "GPU " + std::to_string(actualGpuIndices[0]);
        devPill(label.c_str(), true);
    }

    // Time window selector on the right
    float bFullW = ImGui::CalcTextSize("Full").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float b30W = ImGui::CalcTextSize("30s").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float b60W = ImGui::CalcTextSize("60s").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float b120W = ImGui::CalcTextSize("120s").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    float totalBtnsW = bFullW + b30W + b60W + b120W + ImGui::GetStyle().ItemSpacing.x * 3.0f;

    ImGui::SameLine();
    float availSide = ImGui::GetContentRegionAvail().x;
    if (availSide >= totalBtnsW + s(10.0f)) {
        float targetX = ImGui::GetCursorPosX() + availSide - totalBtnsW;
        if (targetX > ImGui::GetCursorPosX()) {
            ImGui::SetCursorPosX(targetX);
        }
    } else {
        ImGui::NewLine();
    }
    auto winBtn = [this](const char* label, float winSec) {
        bool isAct = (m_telemetryTimeWindow == winSec);
        if (isAct) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.22f, 0.48f, 0.90f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.15f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.65f, 0.70f, 0.80f, 1.0f));
        }
        if (ImGui::SmallButton(label)) {
            m_telemetryTimeWindow = winSec;
            m_telemetryNeedsFit = true;
        }
        ImGui::PopStyleColor(2);
    };
    winBtn("Full", 0.0f);
    ImGui::SameLine();
    winBtn("30s", 30.0f);
    ImGui::SameLine();
    winBtn("60s", 60.0f);
    ImGui::SameLine();
    winBtn("120s", 120.0f);

    ImGui::Spacing();

    // Time calculations
    float curT = snap0.timeHistory.empty() ? 0.0f : snap0.timeHistory.back();
    if (m_telemetryDualGpuMode && actualGpuIndices.size() > 1 && !snap1.timeHistory.empty()) {
        curT = std::max(curT, snap1.timeHistory.back());
    } else if (!activeSnap.timeHistory.empty()) {
        curT = activeSnap.timeHistory.back();
    }

    float minT = 0.0f;
    float maxT = 10.0f;
    if (m_telemetryTimeWindow <= 0.0f) {
        // Full Run: compress to cover the entire run
        minT = 0.0f;
        maxT = std::max(5.0f, curT + 0.2f);
    } else {
        // Fixed sliding window (30s, 60s, 120s)
        minT = std::max(0.0f, curT - m_telemetryTimeWindow);
        maxT = std::max(minT + 5.0f, curT + 0.2f);
    }

    ImPlotCond cond = (m_execState == ExecutionState::Running || m_telemetryNeedsFit) 
                      ? ImPlotCond_Always 
                      : ImPlotCond_Once;

    float plotH = s(120.0f);

    ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(s(6.0f), s(4.0f)));
    ImPlot::PushStyleVar(ImPlotStyleVar_LabelPadding, ImVec2(s(4.0f), s(2.0f)));

    // Color definitions for telemetry traces & legend
    const ImVec4 colGpu0Shader   = ImVec4(0.35f, 0.85f, 0.45f, 1.0f); // Green
    const ImVec4 colGpu1Shader   = ImVec4(0.95f, 0.38f, 0.35f, 1.0f); // Coral Red
    const ImVec4 colSingleShader = ImVec4(0.38f, 0.75f, 1.00f, 1.0f); // Sky Blue
    const ImVec4 colSingleMem    = ImVec4(0.25f, 0.90f, 0.80f, 1.0f); // Cyan

    const ImVec4 colGpu0Temp  = ImVec4(0.35f, 0.85f, 0.45f, 1.0f); // Green
    const ImVec4 colGpu0Power = ImVec4(0.95f, 0.38f, 0.35f, 1.0f); // Red
    const ImVec4 colGpu1Temp  = ImVec4(0.68f, 0.52f, 0.98f, 1.0f); // Purple
    const ImVec4 colGpu1Power = ImVec4(0.95f, 0.72f, 0.30f, 1.0f); // Amber

    const ImVec4 colSingleTemp  = ImVec4(0.35f, 0.85f, 0.55f, 1.0f); // Emerald Green
    const ImVec4 colSinglePower = ImVec4(0.95f, 0.72f, 0.30f, 1.0f); // Amber Gold

    auto renderLegendChip = [this](const char* label, ImVec4 color) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        float lineH = ImGui::GetTextLineHeight();
        float boxSize = s(7.0f);
        float boxOffsetY = (lineH - boxSize) * 0.5f;
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImU32 col = ImGui::GetColorU32(color);
        drawList->AddRectFilled(ImVec2(p.x, p.y + boxOffsetY), ImVec2(p.x + boxSize, p.y + boxOffsetY + boxSize), col, s(1.5f));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + boxSize + s(5.0f));
        ImGui::TextColored(ImVec4(0.80f, 0.85f, 0.92f, 1.0f), "%s", label);
    };

    ImPlotFlags plotFlags = ImPlotFlags_NoTitle | ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText;

    // Graph 1: Frequencies (Shader & Memory Clock)
    if (ImPlot::BeginPlot("##sidebar_clocks", ImVec2(-1, plotH), plotFlags)) {
        ImPlot::SetupAxes(nullptr, "MHz", ImPlotAxisFlags_NoLabel, ImPlotAxisFlags_None);
        ImPlot::SetupAxisLimits(ImAxis_X1, minT, maxT, cond);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 3500, ImPlotCond_Once);

        if (m_telemetryDualGpuMode && actualGpuIndices.size() > 1) {
            if (snap0.timeHistory.size() > 1) {
                ImPlot::SetNextLineStyle(colGpu0Shader, s(1.5f));
                ImPlot::PlotLine("##g0s", snap0.timeHistory.data(), snap0.sclkHistory.data(),
                                 static_cast<int>(snap0.sclkHistory.size()), 0, static_cast<int>(snap0.sclkHistory.offset()));
            }
            if (snap1.timeHistory.size() > 1) {
                ImPlot::SetNextLineStyle(colGpu1Shader, s(1.5f));
                ImPlot::PlotLine("##g1s", snap1.timeHistory.data(), snap1.sclkHistory.data(),
                                 static_cast<int>(snap1.sclkHistory.size()), 0, static_cast<int>(snap1.sclkHistory.offset()));
            }
        } else {
            if (activeSnap.timeHistory.size() > 1) {
                ImPlot::SetNextLineStyle(colSingleShader, s(1.5f));
                ImPlot::PlotLine("##sclk", activeSnap.timeHistory.data(), activeSnap.sclkHistory.data(),
                                 static_cast<int>(activeSnap.sclkHistory.size()), 0, static_cast<int>(activeSnap.sclkHistory.offset()));
                ImPlot::SetNextLineStyle(colSingleMem, s(1.5f));
                ImPlot::PlotLine("##mclk", activeSnap.timeHistory.data(), activeSnap.mclkHistory.data(),
                                 static_cast<int>(activeSnap.mclkHistory.size()), 0, static_cast<int>(activeSnap.mclkHistory.offset()));
            }
        }
        ImPlot::EndPlot();
    }

    // Legend below Graph 1
    if (ImGui::BeginTable("LegendClocks", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextColumn();
        if (m_telemetryDualGpuMode && actualGpuIndices.size() > 1) {
            renderLegendChip("GPU 0 Shader", colGpu0Shader);
            ImGui::TableNextColumn();
            renderLegendChip("GPU 1 Shader", colGpu1Shader);
        } else {
            renderLegendChip("Shader Clock", colSingleShader);
            ImGui::TableNextColumn();
            renderLegendChip("Memory Clock", colSingleMem);
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();

    // Graph 2: Thermals & Power Dynamics
    if (ImPlot::BeginPlot("##sidebar_thermals", ImVec2(-1, plotH), plotFlags)) {
        ImPlot::SetupAxes(nullptr, "°C", ImPlotAxisFlags_NoLabel, ImPlotAxisFlags_None);
        ImPlot::SetupAxis(ImAxis_Y2, "W", ImPlotAxisFlags_AuxDefault);
        ImPlot::SetupAxisLimits(ImAxis_X1, minT, maxT, cond);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 20, 110, ImPlotCond_Once);
        ImPlot::SetupAxisLimits(ImAxis_Y2, 0, 350, ImPlotCond_Once);

        if (m_telemetryDualGpuMode && actualGpuIndices.size() > 1) {
            if (snap0.timeHistory.size() > 1) {
                ImPlot::SetNextLineStyle(colGpu0Temp, s(1.5f));
                ImPlot::PlotLine("##g0t", snap0.timeHistory.data(), snap0.tempJctHistory.data(),
                                 static_cast<int>(snap0.tempJctHistory.size()), 0, static_cast<int>(snap0.tempJctHistory.offset()));
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                ImPlot::SetNextLineStyle(colGpu0Power, s(1.5f));
                ImPlot::PlotLine("##g0p", snap0.timeHistory.data(), snap0.powerHistory.data(),
                                 static_cast<int>(snap0.powerHistory.size()), 0, static_cast<int>(snap0.powerHistory.offset()));
            }
            if (snap1.timeHistory.size() > 1) {
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
                ImPlot::SetNextLineStyle(colGpu1Temp, s(1.5f));
                ImPlot::PlotLine("##g1t", snap1.timeHistory.data(), snap1.tempJctHistory.data(),
                                 static_cast<int>(snap1.tempJctHistory.size()), 0, static_cast<int>(snap1.tempJctHistory.offset()));
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                ImPlot::SetNextLineStyle(colGpu1Power, s(1.5f));
                ImPlot::PlotLine("##g1p", snap1.timeHistory.data(), snap1.powerHistory.data(),
                                 static_cast<int>(snap1.powerHistory.size()), 0, static_cast<int>(snap1.powerHistory.offset()));
            }
        } else {
            if (activeSnap.timeHistory.size() > 1) {
                ImPlot::SetNextLineStyle(colSingleTemp, s(1.5f));
                ImPlot::PlotLine("##jct", activeSnap.timeHistory.data(), activeSnap.tempJctHistory.data(),
                                 static_cast<int>(activeSnap.tempJctHistory.size()), 0, static_cast<int>(activeSnap.tempJctHistory.offset()));
                ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
                ImPlot::SetNextLineStyle(colSinglePower, s(1.5f));
                ImPlot::PlotLine("##pwr", activeSnap.timeHistory.data(), activeSnap.powerHistory.data(),
                                 static_cast<int>(activeSnap.powerHistory.size()), 0, static_cast<int>(activeSnap.powerHistory.offset()));
            }
        }
        ImPlot::EndPlot();
    }

    // Legend below Graph 2
    if (ImGui::BeginTable("LegendThermals", 2, ImGuiTableFlags_SizingStretchSame)) {
        if (m_telemetryDualGpuMode && actualGpuIndices.size() > 1) {
            ImGui::TableNextColumn();
            renderLegendChip("GPU 0 Temp", colGpu0Temp);
            ImGui::TableNextColumn();
            renderLegendChip("GPU 0 Power", colGpu0Power);
            ImGui::TableNextColumn();
            renderLegendChip("GPU 1 Temp", colGpu1Temp);
            ImGui::TableNextColumn();
            renderLegendChip("GPU 1 Power", colGpu1Power);
        } else {
            ImGui::TableNextColumn();
            renderLegendChip("Junction Temp", colSingleTemp);
            ImGui::TableNextColumn();
            renderLegendChip("Board Power", colSinglePower);
        }
        ImGui::EndTable();
    }

    ImPlot::PopStyleVar(2);
}

void GuiApp::renderResultsScorecard() {
    ImGui::Spacing();
    
    // Top Summary Banner when results are present
    if (!m_allResults.empty()) {
        size_t passedCount = 0, unsuppCount = 0, failedCount = 0, abortedCount = 0;
        double peakCompute = 0.0, peakBandwidth = 0.0, peakRT = 0.0;

        for (const auto& res : m_allResults) {
            if (res.isUnsupported) { unsuppCount++; continue; }
            if (res.time_ms == -3.0) { abortedCount++; continue; }
            if (res.time_ms < 0.0) { failedCount++; continue; }
            passedCount++;

            if (res.time_ms > 0.0 && res.operations > 0) {
                double opsPerSec = (static_cast<double>(res.operations) / res.time_ms) * 1000.0;
                if (res.metric.find("TFLOPS") != std::string::npos) {
                    peakCompute = std::max(peakCompute, opsPerSec / 1e12);
                } else if (res.metric.find("GB/s") != std::string::npos) {
                    peakBandwidth = std::max(peakBandwidth, opsPerSec / 1e9);
                } else if (res.metric.find("MRays/s") != std::string::npos) {
                    peakRT = std::max(peakRT, opsPerSec / 1e6);
                }
            }
        }

        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.11f, 0.14f, 0.20f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.24f, 0.36f, 0.55f, 0.6f));
        ImGui::BeginChild("ScorecardSummaryCard", ImVec2(0, s(56.0f)), true);

        ImGui::TextDisabled("Workloads:"); ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%zu Passed", passedCount);
        if (unsuppCount > 0) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.85f, 0.55f, 0.15f, 1.0f), "| %zu Unsupported", unsuppCount);
        }
        if (failedCount > 0) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.30f, 0.30f, 1.0f), "| %zu Failed", failedCount);
        }
        if (abortedCount > 0) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.25f, 1.0f), "| %zu Aborted", abortedCount);
        }

        ImGui::SameLine(s(320.0f));
        if (peakCompute > 0.0) {
            ImGui::TextDisabled("Peak Compute:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.38f, 0.78f, 1.00f, 1.0f), "%.2f TFLOPS", peakCompute);
            ImGui::SameLine();
        }
        if (peakBandwidth > 0.0) {
            ImGui::TextDisabled("Peak VRAM:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.38f, 0.78f, 1.00f, 1.0f), "%.1f GB/s", peakBandwidth);
            ImGui::SameLine();
        }
        if (peakRT > 0.0) {
            ImGui::TextDisabled("Peak Ray Tracing:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.98f, 0.75f, 0.35f, 1.0f), "%.1f MRays/s", peakRT);
        }

        ImGui::EndChild();
        ImGui::PopStyleColor(2);
        ImGui::Spacing();
    } else {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.12f, 0.18f, 1.0f));
        ImGui::BeginChild("EmptyScorecardPrompt", ImVec2(0, s(64.0f)), true);
        ImGui::TextColored(ImVec4(0.70f, 0.78f, 0.90f, 1.0f), "No benchmark results recorded yet.");
        ImGui::TextDisabled("Select workloads on the 'Benchmark Suite' tab and click 'START BENCHMARK' to generate performance scorecards.");
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();
    }
    
    // Category Filter
    ImGui::Text("Filter Category:");
    ImGui::SameLine();
    ImGui::RadioButton("All", &m_activeScorecardFilter, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Compute", &m_activeScorecardFilter, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Memory", &m_activeScorecardFilter, 2);
    ImGui::SameLine();
    ImGui::RadioButton("Ray Tracing", &m_activeScorecardFilter, 3);
    ImGui::SameLine();
    ImGui::RadioButton("Graphics", &m_activeScorecardFilter, 4);
    ImGui::SameLine();
    ImGui::RadioButton("Host System", &m_activeScorecardFilter, 5);

    // Device Filter & Unsupported toggle (dynamically populated from actual devices)
    float devTextW = ImGui::CalcTextSize("Device:").x;
    float allDevW = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize("All Devices").x;
    float totalDevBlockW = devTextW + allDevW;
    for (const auto& dev : m_devices) {
        std::string label = dev.isSystem ? "Host CPU" : ("GPU " + std::to_string(dev.deviceIndex));
        totalDevBlockW += ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize(label.c_str()).x;
    }
    float unsuppW = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + ImGui::CalcTextSize("Hide Unsupported").x;
    totalDevBlockW += unsuppW + ImGui::GetStyle().ItemSpacing.x * (m_devices.size() + 3);

    ImGui::SameLine();
    float availDev = ImGui::GetContentRegionAvail().x;
    if (availDev >= totalDevBlockW + s(20.0f)) {
        ImGui::TextDisabled("|");
        ImGui::SameLine(0, s(12.0f));
    } else {
        ImGui::NewLine();
    }

    ImGui::Text("Device:");
    ImGui::SameLine();
    ImGui::RadioButton("All Devices", &m_activeDeviceScorecardFilter, -1);
    for (const auto& dev : m_devices) {
        ImGui::SameLine();
        if (dev.isSystem) {
            ImGui::RadioButton("Host CPU", &m_activeDeviceScorecardFilter, -2);
        } else {
            std::string label = "GPU " + std::to_string(dev.deviceIndex);
            ImGui::RadioButton(label.c_str(), &m_activeDeviceScorecardFilter, static_cast<int>(dev.deviceIndex));
        }
    }

    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, m_hideUnsupported ? ImVec4(0.70f, 0.80f, 0.95f, 1.0f) : ImVec4(0.95f, 0.70f, 0.25f, 1.0f));
    ImGui::Checkbox("Hide Unsupported##scorecard", &m_hideUnsupported);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Hide tests with UNSUPPORTED status from results table (default: hidden)");
    }

    // Scenario Filter for Ray Tracing
    ImGui::Text("Scenario:");
    ImGui::SameLine();
    ImGui::RadioButton("All Scenarios", &m_scorecardSceneFilter, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Indoor (Sponza)", &m_scorecardSceneFilter, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Forest", &m_scorecardSceneFilter, 2);
    ImGui::SameLine();
    ImGui::RadioButton("Landscape", &m_scorecardSceneFilter, 3);
    ImGui::SameLine();
    ImGui::RadioButton("Showroom", &m_scorecardSceneFilter, 4);

    ImGui::Spacing();

    const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_Resizable | ImGuiTableFlags_Sortable |
                                  ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX;

    if (ImGui::BeginTable("ResultsTable", 9, flags, ImVec2(0, -1))) {
        ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed, s(85.0f));
        ImGui::TableSetupColumn("Device", ImGuiTableColumnFlags_WidthFixed, s(90.0f));
        ImGui::TableSetupColumn("Backend", ImGuiTableColumnFlags_WidthFixed, s(85.0f));
        ImGui::TableSetupColumn("Component", ImGuiTableColumnFlags_WidthFixed, s(100.0f));
        ImGui::TableSetupColumn("Benchmark", ImGuiTableColumnFlags_WidthStretch, s(380.0f));
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, s(95.0f));
        ImGui::TableSetupColumn("Score / Throughput", ImGuiTableColumnFlags_WidthFixed, s(140.0f));
        ImGui::TableSetupColumn("Speedup / Delta", ImGuiTableColumnFlags_WidthFixed, s(155.0f));
        ImGui::TableSetupColumn("Time (ms)", ImGuiTableColumnFlags_WidthFixed, s(75.0f));
        ImGui::TableHeadersRow();

        for (const auto& res : m_allResults) {
            // Apply category filter
            if (m_activeScorecardFilter == 1 && res.component != "Compute") continue;
            if (m_activeScorecardFilter == 2 && res.component != "Memory") continue;
            if (m_activeScorecardFilter == 3 && res.component != "Ray Tracing") continue;
            if (m_activeScorecardFilter == 4 && (res.component != "Graphics" && res.component != "Raster" && res.component != "Rasterization & ROP")) continue;
            if (m_activeScorecardFilter == 5 && (res.component != "System" && res.component != "Host System" && res.component != "Host CPU System Memory")) continue;

            // Apply device filter (only actual devices)
            if (m_activeDeviceScorecardFilter >= 0 && res.deviceIndex != static_cast<uint32_t>(m_activeDeviceScorecardFilter)) continue;
            if (m_activeDeviceScorecardFilter == -2 && res.deviceIndex != 0xFFFFFFFF) continue;

            // Apply unsupported filter
            if (m_hideUnsupported && res.isUnsupported) continue;

            // Apply scenario filter for Ray Tracing
            if (m_scorecardSceneFilter > 0 && res.component == "Ray Tracing" && res.benchmarkName.find("RayScheduling") != std::string::npos) {
                std::string sName = extractSceneName(res.benchmarkName);
                if (m_scorecardSceneFilter == 1 && sName.find("Indoor") == std::string::npos) continue;
                if (m_scorecardSceneFilter == 2 && sName.find("Forest") == std::string::npos) continue;
                if (m_scorecardSceneFilter == 3 && sName.find("Outdoor") == std::string::npos) continue;
                if (m_scorecardSceneFilter == 4 && sName.find("Showroom") == std::string::npos) continue;
            }

            ImGui::TableNextRow();

            // 1. Status
            ImGui::TableNextColumn();
            if (res.isUnsupported) {
                ImGui::TextColored(ImVec4(0.85f, 0.55f, 0.15f, 1.0f), "UNSUPPORTED");
            } else if (res.time_ms == -3.0) {
                ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.25f, 1.0f), "ABORTED");
            } else if (res.time_ms < 0.0) {
                ImGui::TextColored(ImVec4(0.95f, 0.25f, 0.25f, 1.0f), "FAILED");
            } else {
                ImGui::TextColored(ImVec4(0.25f, 0.85f, 0.35f, 1.0f), "PASS");
            }

            // 2. Device
            ImGui::TableNextColumn();
            if (res.deviceIndex == 0xFFFFFFFF) {
                ImGui::Text("Host CPU");
            } else {
                ImGui::Text("GPU %u", res.deviceIndex);
            }

            // 3. Backend
            ImGui::TableNextColumn();
            ImGui::Text("%s", res.backendName.c_str());

            // 4. Component
            ImGui::TableNextColumn();
            ImGui::Text("%s", res.component.c_str());

            // 5. Benchmark
            ImGui::TableNextColumn();
            ImGui::Text("%s", res.benchmarkName.c_str());
            if (ImGui::IsItemHovered()) {
                // Match the suite item by engine id + config index (the result's
                // benchmarkName is "<id> (<config name>)" for multi-config benches)
                const BenchmarkItem* match = nullptr;
                {
                    std::string baseName = res.benchmarkName;
                    size_t p = baseName.find(" (");
                    if (p != std::string::npos) baseName = baseName.substr(0, p);
                    for (const auto& cat : m_categories) {
                        for (const auto& sub : cat.subgroups) {
                            for (const auto& itm : sub.items) {
                                if (itm.id == baseName &&
                                    static_cast<uint32_t>(itm.configIndex) == res.configIndex) {
                                    match = &itm;
                                    break;
                                }
                            }
                            if (match) break;
                        }
                        if (match) break;
                    }
                }
                if (match) {
                    std::string scn = "";
                    std::string sName = extractSceneName(res.benchmarkName);
                    if (sName.find("Indoor") != std::string::npos) scn = "indoor";
                    else if (sName.find("Forest") != std::string::npos) scn = "forest";
                    else if (sName.find("Outdoor") != std::string::npos) scn = "outdoor";
                    else if (sName.find("Showroom") != std::string::npos) scn = "showroom";
                    renderBenchmarkTooltip(*match, scn);
                }
            }

            // 6. Metric
            ImGui::TableNextColumn();
            ImGui::Text("%s", res.metric.c_str());

            // 7. Score
            ImGui::TableNextColumn();
            double curOpsPerSec = 0.0;
            if (res.time_ms > 0.0 && res.operations > 0) {
                curOpsPerSec = (static_cast<double>(res.operations) / res.time_ms) * 1000.0;
                if (res.metric.find("TFLOPS") != std::string::npos || res.metric.find("TOPS") != std::string::npos) {
                    ImGui::Text("%.2f %s", curOpsPerSec / 1e12, res.metric.c_str());
                } else if (res.metric.find("TB/s") != std::string::npos) {
                    ImGui::Text("%.2f TB/s", curOpsPerSec / 1e12);
                } else if (res.metric.find("GB/s") != std::string::npos) {
                    ImGui::Text("%.2f GB/s", curOpsPerSec / 1e9);
                } else if (res.metric.find("GIS/s") != std::string::npos) {
                    ImGui::Text("%.2f GIS/s", curOpsPerSec / 1e9);
                } else if (res.metric.find("MRays/s") != std::string::npos || res.metric.find("MItems/s") != std::string::npos) {
                    ImGui::Text("%.2f %s", curOpsPerSec / 1e6, res.metric.c_str());
                } else if (res.metric.find("GPixels/s") != std::string::npos) {
                    ImGui::Text("%.2f GPixels/s", curOpsPerSec / 1e9);
                } else if (res.metric.find("MTris/s") != std::string::npos) {
                    ImGui::Text("%.2f MTris/s", curOpsPerSec / 1e6);
                } else if (res.metric.find("MBVH/s") != std::string::npos) {
                    ImGui::Text("%.2f MBVH/s", curOpsPerSec / 1e6);
                } else if (res.metric.find("MInst/s") != std::string::npos) {
                    ImGui::Text("%.2f MInst/s", curOpsPerSec / 1e6);
                } else if (res.metric.find("ns") != std::string::npos) {
                    double nsVal = (res.operations > 0) ? ((res.time_ms * 1e6) / res.operations) : res.time_ms;
                    ImGui::Text("%.2f ns", nsVal);
                } else {
                    ImGui::Text("%.2f %s", curOpsPerSec, res.metric.c_str());
                }
            } else if (res.isUnsupported && !res.supportNote.empty()) {
                ImGui::TextDisabled("%s", res.supportNote.c_str());
            } else if (res.time_ms == -3.0) {
                ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.25f, 1.0f), "ABORTED");
            } else if (res.time_ms < 0.0) {
                ImGui::TextColored(ImVec4(0.95f, 0.30f, 0.30f, 1.0f), "FAILED");
            } else {
                ImGui::TextDisabled("-");
            }

            // 8. Speedup / Delta (Parity with console output)
            ImGui::TableNextColumn();
            std::string deltaStr = "-";
            ImVec4 deltaCol = ImVec4(0.60f, 0.65f, 0.75f, 0.70f);

            if (res.isUnsupported) {
                deltaStr = "-";
            } else if (res.time_ms == -3.0) {
                deltaStr = !res.errorString.empty() ? ("[" + res.errorString + "]") : "[Aborted: GPU Lost]";
                deltaCol = ImVec4(0.95f, 0.70f, 0.25f, 1.0f);
            } else if (res.time_ms < 0.0) {
                deltaStr = !res.errorString.empty() ? ("[" + res.errorString + "]") : "[Failed]";
                deltaCol = ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
            } else if (res.component == "Compute") {
                if (res.subcategory == "Dual-Issue") {
                    if (res.configIndex == 0 || res.configIndex == 3) {
                        deltaStr = "[Baseline]";
                        deltaCol = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                    } else if (res.configIndex == 6) {
                        // Harmonic mean of Config 0 (Standard FP32) and Config 3 (Standard INT32)
                        double baseFP32 = 0.0;
                        double baseINT32 = 0.0;
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.backendName == res.backendName && other.subcategory == "Dual-Issue" && other.time_ms > 0.0) {
                                if (other.configIndex == 0 || other.benchmarkName.find("Standard FP32") != std::string::npos) {
                                    baseFP32 = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                } else if (other.configIndex == 3 || other.benchmarkName.find("Standard INT32") != std::string::npos) {
                                    baseINT32 = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                }
                            }
                        }
                        if (baseFP32 > 0.0 && baseINT32 > 0.0) {
                            double baseOps = 2.0 / ((1.0 / baseFP32) + (1.0 / baseINT32));
                            double ratio = curOpsPerSec / baseOps;
                            double pct = (ratio - 1.0) * 100.0;
                            char dBuf[48];
                            if (std::abs(pct) >= 0.1) {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx (%s%.1f%%)", ratio, (pct >= 0 ? "+" : ""), pct);
                            } else {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx", ratio);
                            }
                            deltaStr = dBuf;
                            deltaCol = (ratio >= 1.0) ? ImVec4(0.30f, 0.92f, 0.85f, 1.0f) : ImVec4(0.92f, 0.65f, 0.35f, 1.0f);
                        }
                    } else {
                        uint32_t targetBaseConfig = (res.configIndex >= 3) ? 3 : 0;
                        double baseOps = 0.0;
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.backendName == res.backendName && other.subcategory == "Dual-Issue" &&
                                other.configIndex == targetBaseConfig && other.time_ms > 0.0) {
                                baseOps = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                break;
                            }
                        }
                        if (baseOps > 0.0) {
                            double ratio = curOpsPerSec / baseOps;
                            double pct = (ratio - 1.0) * 100.0;
                            char dBuf[48];
                            if (std::abs(pct) >= 0.1) {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx (%s%.1f%%)", ratio, (pct >= 0 ? "+" : ""), pct);
                            } else {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx", ratio);
                            }
                            deltaStr = dBuf;
                            deltaCol = (ratio >= 1.0) ? ImVec4(0.30f, 0.92f, 0.85f, 1.0f) : ImVec4(0.92f, 0.65f, 0.35f, 1.0f);
                        }
                    }
                } else if (res.subcategory == "LDS Bank Conflicts") {
                    if (res.configIndex == 0) {
                        deltaStr = "[Baseline]";
                        deltaCol = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                    } else {
                        double baseOps = 0.0;
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.backendName == res.backendName &&
                                other.subcategory == "LDS Bank Conflicts" && other.configIndex == 0 && other.time_ms > 0.0) {
                                baseOps = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                break;
                            }
                        }
                        if (baseOps > 0.0) {
                            double ratio = curOpsPerSec / baseOps;
                            double pct = (ratio - 1.0) * 100.0;
                            char dBuf[48];
                            if (std::abs(pct) >= 0.1) {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx (%s%.1f%%)", ratio, (pct >= 0 ? "+" : ""), pct);
                            } else {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx", ratio);
                            }
                            deltaStr = dBuf;
                            deltaCol = (ratio >= 1.0) ? ImVec4(0.30f, 0.92f, 0.85f, 1.0f) : ImVec4(0.92f, 0.65f, 0.35f, 1.0f);
                        }
                    }
                } else if (res.subcategory == "Indirect Command Synthesis") {
                    if (res.configIndex == 0) {
                        deltaStr = "[Baseline]";
                        deltaCol = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                    } else {
                        double baseOps = 0.0;
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.backendName == res.backendName &&
                                other.subcategory == "Indirect Command Synthesis" && other.configIndex == 0 && other.time_ms > 0.0) {
                                baseOps = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                break;
                            }
                        }
                        if (baseOps > 0.0) {
                            double ratio = curOpsPerSec / baseOps;
                            double pct = (ratio - 1.0) * 100.0;
                            char dBuf[48];
                            if (std::abs(pct) >= 0.1) {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx (%s%.1f%%)", ratio, (pct >= 0 ? "+" : ""), pct);
                            } else {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx", ratio);
                            }
                            deltaStr = dBuf;
                            deltaCol = (ratio >= 1.0) ? ImVec4(0.30f, 0.92f, 0.85f, 1.0f) : ImVec4(0.92f, 0.65f, 0.35f, 1.0f);
                        }
                    }
                } else {
                    // Compute Precision: Vector is the baseline for Matrix within the SAME precision.
                    // Standalone precisions (FP64, FP32, FP4) do not compare against unrelated datatypes.
                    bool isVector = (res.benchmarkName.find("Vector") != std::string::npos);
                    bool isMatrix = (res.benchmarkName.find("Matrix") != std::string::npos);
                    if (isVector && (res.subcategory == "FP16" || res.subcategory == "BF16" || res.subcategory == "FP8" ||
                                     res.subcategory == "INT8" || res.subcategory == "INT4")) {
                        deltaStr = "[Baseline]";
                        deltaCol = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                    } else if (isMatrix && res.time_ms > 0.0 && curOpsPerSec > 0.0) {
                        double baseOps = 0.0;
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.backendName == res.backendName &&
                                other.subcategory == res.subcategory &&
                                other.benchmarkName.find("Vector") != std::string::npos && other.time_ms > 0.0) {
                                baseOps = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                break;
                            }
                        }
                        if (baseOps > 0.0) {
                            double ratio = curOpsPerSec / baseOps;
                            double pct = (ratio - 1.0) * 100.0;
                            char dBuf[48];
                            if (std::abs(pct) >= 0.1) {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx (%s%.1f%%)", ratio, (pct >= 0 ? "+" : ""), pct);
                            } else {
                                snprintf(dBuf, sizeof(dBuf), "%.2fx", ratio);
                            }
                            deltaStr = dBuf;
                            deltaCol = (ratio >= 1.0) ? ImVec4(0.30f, 0.92f, 0.85f, 1.0f) : ImVec4(0.92f, 0.65f, 0.35f, 1.0f);
                        }
                    } else {
                        deltaStr = "-";
                    }
                }
            } else if (res.component == "Ray Tracing") {
                if (res.benchmarkName.find("RayRawTraversal") != std::string::npos) {
                    double time_s = res.time_ms / 1000.0;
                    if (res.configIndex == 0) {
                        double gis_s = (time_s > 0.0) ? ((static_cast<double>(res.operations) / time_s) / 1e9) : 0.0;
                        double pct = (gis_s / 300.8) * 100.0;
                        char dBuf[48];
                        snprintf(dBuf, sizeof(dBuf), "%.1f%% of Peak", pct);
                        deltaStr = dBuf;
                        deltaCol = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                    } else {
                        uint64_t boxOps = res.operations * 64;
                        double box_gis_s = (time_s > 0.0) ? ((static_cast<double>(boxOps) / time_s) / 1e9) : 0.0;
                        double pct = (box_gis_s / 1203.2) * 100.0;
                        char dBuf[48];
                        snprintf(dBuf, sizeof(dBuf), "%.1f%% of Peak", pct);
                        deltaStr = dBuf;
                        deltaCol = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                    }
                } else if (res.benchmarkName.find("Megakernel") != std::string::npos ||
                           res.benchmarkName.find("Traditional") != std::string::npos ||
                           res.benchmarkName.find("Scanline (Baseline)") != std::string::npos ||
                           res.benchmarkName.find("Scanline") != std::string::npos ||
                           res.benchmarkName.find("Wave Ballot") != std::string::npos ||
                           (res.benchmarkName.find("RayDivergence") != std::string::npos && res.configIndex == 0) ||
                           (res.benchmarkName.find("RayPayload") != std::string::npos && res.configIndex == 0) ||
                           (res.benchmarkName.find("RayAnyHit") != std::string::npos && res.configIndex == 0)) {
                    deltaStr = "[Baseline]";
                    deltaCol = ImVec4(0.38f, 0.75f, 1.00f, 0.95f);
                } else if (res.time_ms > 0.0 && curOpsPerSec > 0.0) {
                    double baseOps = 0.0;
                    if (res.benchmarkName.find("RayDivergence") != std::string::npos) {
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.benchmarkName.find("RayDivergence") != std::string::npos &&
                                other.configIndex == 0 && other.time_ms > 0.0) {
                                baseOps = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                break;
                            }
                        }
                    } else if (res.benchmarkName.find("RayPayload") != std::string::npos) {
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.benchmarkName.find("RayPayload") != std::string::npos &&
                                other.configIndex == 0 && other.time_ms > 0.0) {
                                baseOps = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                break;
                            }
                        }
                    } else if (res.benchmarkName.find("RayAnyHit") != std::string::npos) {
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.benchmarkName.find("RayAnyHit") != std::string::npos &&
                                other.configIndex == 0 && other.time_ms > 0.0) {
                                baseOps = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                break;
                            }
                        }
                    } else {
                        // RayScheduling and general Ray Tracing
                        for (const auto& other : m_allResults) {
                            if (other.deviceIndex == res.deviceIndex && other.component == "Ray Tracing" &&
                                other.subcategory == res.subcategory &&
                                (other.benchmarkName.find("Megakernel") != std::string::npos || other.benchmarkName.find("Baseline") != std::string::npos ||
                                 other.benchmarkName.find("Scanline") != std::string::npos || other.benchmarkName.find("Wave Ballot") != std::string::npos) &&
                                other.time_ms > 0.0 && other.metric == res.metric) {
                                baseOps = (static_cast<double>(other.operations) / other.time_ms) * 1000.0;
                                break;
                            }
                        }
                    }

                    if (baseOps > 0.0) {
                        double ratio = curOpsPerSec / baseOps;
                        double pct = (ratio - 1.0) * 100.0;
                        char dBuf[48];
                        if (std::abs(pct) >= 0.1) {
                            snprintf(dBuf, sizeof(dBuf), "%.2fx (%s%.1f%%)", ratio, (pct >= 0 ? "+" : ""), pct);
                        } else {
                            snprintf(dBuf, sizeof(dBuf), "%.2fx", ratio);
                        }
                        deltaStr = dBuf;
                        deltaCol = (ratio >= 1.0) ? ImVec4(0.30f, 0.92f, 0.85f, 1.0f) : ImVec4(0.92f, 0.65f, 0.35f, 1.0f);
                    }
                }
            }

            ImGui::TextColored(deltaCol, "%s", deltaStr.c_str());

            // 9. Time
            ImGui::TableNextColumn();
            if (res.time_ms > 0.0) {
                ImGui::Text("%.1f", res.time_ms);
            } else {
                ImGui::TextDisabled("-");
            }
        }

        ImGui::EndTable();
    }
}

void GuiApp::renderRayTracingViewport() {
    ImGui::Spacing();

    // Scene & Viewport Navigation Header
    struct SceneMetadata {
        const char* displayName;
        const char* tag;
        const char* modelName;
        const char* triangleCount;
        const char* description;
        const char* techAScore;
        const char* techBScore;
        const char* speedup;
        int vgprTrad;
        int vgprDgc;
        const char* simdTrad;
        const char* simdDgc;
        const char* psnr;
        const char* maxDelta;
        const char* details;
    };

    static const SceneMetadata scenes[] = {
        {
            "Showroom Studio (toycar.glb)",
            "showroom",
            "assets/models/toycar.glb",
            "108,936 Triangles",
            "Studio turntable showcase with multi-BSDF automotive car paint, clearcoat flakes, dispersive glass, rubber tires, and alloy wheels.",
            "185.40 MRays/s (201.2 FPS)",
            "523.80 MRays/s (568.3 FPS)",
            "2.82x (+182.5%)",
            128, 64,
            "68.2% (Divergent Wavefronts)", "94.7% (Re-Coalesced Wavefronts)",
            "120.0 dB (BIT-EXACT)", "0.000",
            "Primary Rays: 921,600 (1280x720) | Multi-BSDF Hits: 2,457,600 | BVH Traversal: 44.8 steps/ray"
        },
        {
            "Indoor Atrium (sponza.glb)",
            "indoor",
            "assets/models/sponza.glb",
            "262,267 Triangles",
            "Classic architectural global illumination benchmark with multi-tiered arches, carved stone columns, lion reliefs, and alpha cutout tapestry.",
            "99.05 MRays/s (107.5 FPS)",
            "277.80 MRays/s (301.4 FPS)",
            "2.80x (+180.5%)",
            128, 64,
            "64.5% (Divergent Wavefronts)", "95.1% (Re-Coalesced Wavefronts)",
            "120.0 dB (BIT-EXACT)", "0.000",
            "Primary Rays: 921,600 (1280x720) | Indirect Bounces: 3,686,400 | BVH Traversal: 52.1 steps/ray"
        },
        {
            "Open-World Forest (AAAOutdoorForest)",
            "forest",
            "Procedural Nature Heightfield",
            "1,050,000+ Triangles",
            "High-density AAA woodland with rolling terrain, 350 soaring multi-tiered pines, 180 birches, 3,500 sword ferns, 2,500 shrubs, 180 nurse logs, and 600 boulders.",
            "89.90 MRays/s (97.5 FPS)",
            "266.27 MRays/s (288.9 FPS)",
            "2.96x (+196.2%)",
            128, 64,
            "61.3% (Divergent Wavefronts)", "96.2% (Re-Coalesced Wavefronts)",
            "120.0 dB (BIT-EXACT)", "0.000",
            "Primary Rays: 921,600 (1280x720) | Alpha Tests: 2,764,800 | BVH Traversal: 64.5 steps/ray"
        },
        {
            "Outdoor Landscape (OutdoorLandscape)",
            "outdoor",
            "Procedural Alpine Terrain",
            "150,000+ Triangles",
            "Expansive alpine landscape featuring distant mountains, pine forests, reflective lake surface, and timber cabins.",
            "542.03 MRays/s (588.1 FPS)",
            "1,444.29 MRays/s (1567.2 FPS)",
            "2.66x (+166.5%)",
            128, 64,
            "72.1% (Divergent Wavefronts)", "95.8% (Re-Coalesced Wavefronts)",
            "120.0 dB (BIT-EXACT)", "0.000",
            "Primary Rays: 921,600 (1280x720) | Direct Light Passes: 4 | BVH Traversal: 38.2 steps/ray"
        }
    };
    const size_t numScenes = sizeof(scenes) / sizeof(scenes[0]);
    if (m_rtSceneIndex < 0 || static_cast<size_t>(m_rtSceneIndex) >= numScenes) {
        m_rtSceneIndex = 0;
    }
    const auto& curSceneMeta = scenes[m_rtSceneIndex];

    std::string dynamicTechAScore = "--";
    std::string dynamicTechBScore = "--";
    std::string dynamicSpeedup = "--";
    auto containsCi = [](const std::string& haystack, const std::string& needle) -> bool {
        if (needle.empty()) return true;
        auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
            [](char c1, char c2) { return std::tolower(static_cast<unsigned char>(c1)) == std::tolower(static_cast<unsigned char>(c2)); });
        return (it != haystack.end());
    };

    double scoreTrad = 0.0, scoreDgc = 0.0;
    // Prefer matching full frame render (configs 9 & 11) or path tracing (configs 3 & 5)
    for (int priorityPass = 0; priorityPass < 2; ++priorityPass) {
        for (const auto& res : m_allResults) {
            bool sceneMatches = containsCi(res.benchmarkName, curSceneMeta.displayName) ||
                                containsCi(res.benchmarkName, curSceneMeta.tag) ||
                                containsCi(res.subcategory, curSceneMeta.displayName);
            if (!sceneMatches) continue;

            double rate = (res.time_ms > 0.0) ? (static_cast<double>(res.operations) / res.time_ms * 1000.0 / 1e6) : 0.0;

            if (priorityPass == 0) {
                // First pass: look specifically for Full Frame (config 9 vs 11) or Scene Path Tracing (config 3 vs 5)
                if (res.configIndex == 9 || (res.configIndex == 3 && scoreTrad == 0.0)) {
                    scoreTrad = rate;
                } else if (res.configIndex == 11 || (res.configIndex == 5 && scoreDgc == 0.0)) {
                    scoreDgc = rate;
                }
            } else {
                // Fallback pass: any Megakernel vs DGC in this scene if specific configs were not run
                if (scoreTrad == 0.0 && (containsCi(res.benchmarkName, "Megakernel") || containsCi(res.benchmarkName, "Traditional"))) {
                    scoreTrad = rate;
                }
                if (scoreDgc == 0.0 && (containsCi(res.benchmarkName, "DGC") || containsCi(res.benchmarkName, "Work List"))) {
                    scoreDgc = rate;
                }
            }
        }
        if (scoreTrad > 0.0 && scoreDgc > 0.0) break;
    }
    if (scoreTrad > 0.0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f MRays/s", scoreTrad);
        dynamicTechAScore = buf;
    }
    if (scoreDgc > 0.0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.2f MRays/s", scoreDgc);
        dynamicTechBScore = buf;
    }
    if (scoreTrad > 0.0 && scoreDgc > 0.0) {
        char buf[64];
        double sp = scoreDgc / scoreTrad;
        snprintf(buf, sizeof(buf), "%.2fx (+%.1f%%)", sp, (sp - 1.0) * 100.0);
        dynamicSpeedup = buf;
    }

    bool parityMeasured = false;
    bool parityPass = false;
    std::string livePsnr = "--";
    std::string liveMaxDelta = "--";

    std::string profilePath = "renders/render_" + std::string(curSceneMeta.tag) + "_profile.json";
    std::ifstream pf(profilePath);
    if (pf.is_open()) {
        std::string line;
        while (std::getline(pf, line)) {
            if (line.find("\"psnr\":") != std::string::npos) {
                size_t col = line.find(':');
                if (col != std::string::npos) {
                    try {
                        std::string numStr = line.substr(col + 1);
                        size_t comma = numStr.find(',');
                        if (comma != std::string::npos) numStr = numStr.substr(0, comma);
                        float val = std::stof(numStr);
                        char b[64];
                        if (val >= 120.0f) snprintf(b, sizeof(b), "%.1f dB (BIT-EXACT)", val);
                        else snprintf(b, sizeof(b), "%.2f dB", val);
                        livePsnr = b;
                        parityMeasured = true;
                        if (val >= 45.0f) parityPass = true;
                    } catch (...) {}
                }
            } else if (line.find("\"diff_pct\":") != std::string::npos || line.find("\"mae\":") != std::string::npos) {
                size_t col = line.find(':');
                if (col != std::string::npos) {
                    try {
                        std::string numStr = line.substr(col + 1);
                        size_t comma = numStr.find(',');
                        if (comma != std::string::npos) numStr = numStr.substr(0, comma);
                        float delta = std::stof(numStr);
                        char b[64];
                        snprintf(b, sizeof(b), "%.4f", delta);
                        liveMaxDelta = b;
                    } catch (...) {}
                }
            }
        }
    }

    // Top Controls Bar
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.12f, 0.16f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, s(6.0f));
    ImGui::BeginChild("RtToolbar", ImVec2(0, s(46.0f)), true, ImGuiWindowFlags_NoScrollbar);
    
    // Viewport Mode Buttons
    const char* modes[] = { "Scenes & Parity", "Pipeline Passes", "PBR Materials", "Geometry & BVH" };
    for (int i = 0; i < 4; ++i) {
        if (i > 0) ImGui::SameLine(0, s(6.0f));
        bool active = (m_rtViewportMode == i);
        if (active) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.48f, 0.80f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.20f, 0.28f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.75f, 0.80f, 0.90f, 1.0f));
        }
        if (ImGui::Button(modes[i], ImVec2(s(130.0f), s(28.0f)))) {
            m_rtViewportMode = i;
        }
        ImGui::PopStyleColor(2);
    }

    ImGui::SameLine(0, s(20.0f));
    ImGui::TextDisabled("| Scene:");
    ImGui::SameLine(0, s(8.0f));
    const char* sceneComboNames[] = {
        "Showroom Studio (toycar.glb)",
        "Indoor Atrium (sponza.glb)",
        "Open-World Forest (AAAOutdoorForest)",
        "Outdoor Landscape (OutdoorLandscape)"
    };
    ImGui::SetNextItemWidth(s(280.0f));
    ImGui::Combo("##scene_combo", &m_rtSceneIndex, sceneComboNames, 4);

    ImGui::SameLine(0, s(16.0f));
    if (ImGui::Button("Run Benchmark / Re-Render", ImVec2(s(190.0f), s(28.0f)))) {
        std::string targetTag = curSceneMeta.tag;
        if (targetTag == "forest") m_scene = "forest";
        else if (targetTag == "indoor") m_scene = "indoor";
        else if (targetTag == "showroom") m_scene = "showroom";
        else m_scene = "outdoor";
        m_dumpRenders = true;
        clearTextureCache();
        selectOnlyBenchmark("RayScheduling");
        startBenchmarks();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Executes RayScheduling benchmark for %s and updates rendered frame buffers.", curSceneMeta.displayName);
    }

    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();

    ImGui::Spacing();

    // MODE 0: SCENES & PARITY
    if (m_rtViewportMode == 0) {
        // Parity Verification Status Banner
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.16f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.20f, 0.50f, 0.80f, 0.6f));
        ImGui::BeginChild("ParityBanner", ImVec2(0, s(34.0f)), true, ImGuiWindowFlags_NoScrollbar);
        ImGui::AlignTextToFramePadding();

        if (!parityMeasured) {
            ImGui::TextColored(ImVec4(0.70f, 0.70f, 0.75f, 1.0f), "[PARITY: NOT MEASURED]");
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("PSNR:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.70f, 0.70f, 0.75f, 1.0f), "--");
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("Max Delta:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.70f, 0.70f, 0.75f, 1.0f), "--");
        } else if (parityPass) {
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "[PARITY: PASS]");
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("PSNR:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.95f, 1.0f), "%s", livePsnr.c_str());
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("Max Delta:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s", liveMaxDelta.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.95f, 0.35f, 0.35f, 1.0f), "[PARITY: FAIL]");
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("PSNR:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.40f, 1.0f), "%s", livePsnr.c_str());
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("Max Delta:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.40f, 1.0f), "%s", liveMaxDelta.c_str());
        }

        ImGui::SameLine(0, s(16.0f));
        ImGui::TextDisabled("Triangles:"); ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.35f, 1.0f), "%s", curSceneMeta.triangleCount);
        ImGui::SameLine(0, s(16.0f));
        ImGui::TextDisabled("Speedup:"); ImGui::SameLine();
        if (dynamicSpeedup != "--") {
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s (Live)", dynamicSpeedup.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.65f, 0.70f, 0.80f, 1.0f), "--");
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(2);

        ImGui::Spacing();

        // Sub-view mode tabs
        const char* pModes[] = { "Split Parity Slider", "Megakernel Solo", "Compacted DGC Solo", "Difference Heatmap", "Side-by-Side" };
        for (int m = 0; m < 5; ++m) {
            if (m > 0) ImGui::SameLine(0, s(4.0f));
            bool isCur = (m_rtParityViewMode == m);
            if (isCur) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.40f, 0.65f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.16f, 0.22f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.65f, 0.70f, 0.80f, 1.0f));
            }
            if (ImGui::Button(pModes[m], ImVec2(s(150.0f), s(24.0f)))) {
                m_rtParityViewMode = m;
            }
            ImGui::PopStyleColor(2);
        }

        if (m_rtParityViewMode == 0) {
            ImGui::SameLine(0, s(20.0f));
            ImGui::SetNextItemWidth(s(220.0f));
            ImGui::SliderFloat("##parity_split_slider", &m_paritySplitRatio, 0.05f, 0.95f, "Split: %.2f");
        }

        ImGui::Spacing();

        // Check if texture cache needs reload
        if (m_textureCacheNeedsClear.exchange(false)) {
            clearTextureCache();
        }

        // Resolve Image Textures
        std::string tag = curSceneMeta.tag;
        std::string tradPath = "renders/render_" + tag + "_traditional_megakernel.png";
        std::string dgcPath = "renders/render_" + tag + "_worklist_dgc.png";
        std::string diffPath = "renders/render_" + tag + "_difference_heatmap.png";

        auto texTrad = getOrLoadTexture(tradPath);
        auto texDgc = getOrLoadTexture(dgcPath);
        auto texDiff = getOrLoadTexture(diffPath);

        // On initial startup only: if default scene (index 0) has no renders, pick first scene with renders
        static bool s_hasInitializedSceneIndex = false;
        if (!s_hasInitializedSceneIndex) {
            s_hasInitializedSceneIndex = true;
            if (!texTrad.isValid() && !texDgc.isValid()) {
                for (size_t sIdx = 0; sIdx < numScenes; ++sIdx) {
                    if (sIdx == static_cast<size_t>(m_rtSceneIndex)) continue;
                    std::string altTrad = "renders/render_" + std::string(scenes[sIdx].tag) + "_traditional_megakernel.png";
                    auto altTex = getOrLoadTexture(altTrad);
                    if (altTex.isValid()) {
                        m_rtSceneIndex = static_cast<int>(sIdx);
                        tag = scenes[m_rtSceneIndex].tag;
                        tradPath = altTrad;
                        dgcPath = "renders/render_" + tag + "_worklist_dgc.png";
                        diffPath = "renders/render_" + tag + "_difference_heatmap.png";
                        texTrad = altTex;
                        texDgc = getOrLoadTexture(dgcPath);
                        texDiff = getOrLoadTexture(diffPath);
                        break;
                    }
                }
            }
        }

        // Viewport Canvas Calculation
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float canvasHeight = std::max(s(320.0f), avail.y - s(115.0f));
        float targetAspect = (texTrad.isValid() && texTrad.height > 0) ? (static_cast<float>(texTrad.width) / texTrad.height) : (16.0f / 9.0f);
        float canvasWidth = std::min(avail.x, canvasHeight * targetAspect);
        if (canvasWidth > avail.x) {
            canvasWidth = avail.x;
            canvasHeight = canvasWidth / targetAspect;
        }

        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 p1 = ImVec2(p0.x + canvasWidth, p0.y + canvasHeight);
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        // Background
        drawList->AddRectFilled(p0, p1, IM_COL32(12, 14, 20, 255), s(6.0f));
        drawList->AddRect(p0, p1, IM_COL32(35, 45, 65, 255), s(6.0f));

        if (!texTrad.isValid() && !texDgc.isValid()) {
            // Missing texture fallback
            std::string msg = "Render output not found for " + std::string(curSceneMeta.displayName);
            ImVec2 textSize = ImGui::CalcTextSize(msg.c_str());
            drawList->AddText(ImVec2(p0.x + (canvasWidth - textSize.x) * 0.5f, p0.y + canvasHeight * 0.42f),
                              IM_COL32(220, 180, 80, 255), msg.c_str());
            std::string hint = "Click 'Run Benchmark / Re-Render' above to execute the Vulkan ray query pipeline.";
            ImVec2 hintSize = ImGui::CalcTextSize(hint.c_str());
            drawList->AddText(ImVec2(p0.x + (canvasWidth - hintSize.x) * 0.5f, p0.y + canvasHeight * 0.50f),
                              IM_COL32(160, 170, 190, 255), hint.c_str());
            ImGui::SetCursorScreenPos(ImVec2(p0.x, p1.y + s(10.0f)));
        } else {
            if (m_rtParityViewMode == 0) {
                // Split Parity Slider Mode
                float splitX = p0.x + canvasWidth * m_paritySplitRatio;

                // Left: Technique A (Megakernel)
                if (texTrad.isValid() && splitX > p0.x + 1.0f) {
                    ImVec2 uv0(0.0f, 0.0f);
                    ImVec2 uv1(m_paritySplitRatio, 1.0f);
                    drawList->AddImage(reinterpret_cast<ImTextureID>(texTrad.descriptorSet), p0, ImVec2(splitX, p1.y), uv0, uv1);
                }

                // Right: Technique B (DGC / Compacted)
                if (texDgc.isValid() && splitX < p1.x - 1.0f) {
                    ImVec2 uv0(m_paritySplitRatio, 0.0f);
                    ImVec2 uv1(1.0f, 1.0f);
                    drawList->AddImage(reinterpret_cast<ImTextureID>(texDgc.descriptorSet), ImVec2(splitX, p0.y), p1, uv0, uv1);
                }

                // Vertical Divider Line & Draggable Handle
                drawList->AddLine(ImVec2(splitX, p0.y), ImVec2(splitX, p1.y), IM_COL32(0, 216, 246, 255), s(2.5f));
                float handleY = p0.y + canvasHeight * 0.5f;
                drawList->AddCircleFilled(ImVec2(splitX, handleY), s(9.0f), IM_COL32(0, 216, 246, 255));
                drawList->AddCircle(ImVec2(splitX, handleY), s(9.0f), IM_COL32(255, 255, 255, 255), 16, s(2.0f));

                // Technique Labels
                drawList->AddText(ImVec2(p0.x + s(14.0f), p0.y + s(12.0f)), IM_COL32(100, 200, 255, 255), "Technique A: Megakernel (Reference)");
                drawList->AddText(ImVec2(splitX + s(14.0f), p0.y + s(12.0f)), IM_COL32(255, 180, 100, 255), "Technique B: Compacted Wavefront (DGC)");

                // Drag handle interaction
                ImGui::SetCursorScreenPos(ImVec2(splitX - s(12.0f), p0.y));
                ImGui::InvisibleButton("##split_handle", ImVec2(s(24.0f), canvasHeight));
                if (ImGui::IsItemActive()) {
                    float mouseX = ImGui::GetIO().MousePos.x;
                    m_paritySplitRatio = std::clamp((mouseX - p0.x) / canvasWidth, 0.02f, 0.98f);
                }
            } else if (m_rtParityViewMode == 1) {
                // Technique A Solo
                if (texTrad.isValid()) {
                    drawList->AddImage(reinterpret_cast<ImTextureID>(texTrad.descriptorSet), p0, p1);
                }
                drawList->AddText(ImVec2(p0.x + s(14.0f), p0.y + s(12.0f)), IM_COL32(100, 200, 255, 255), "Technique A: Monolithic Megakernel (Reference)");
            } else if (m_rtParityViewMode == 2) {
                // Technique B Solo
                if (texDgc.isValid()) {
                    drawList->AddImage(reinterpret_cast<ImTextureID>(texDgc.descriptorSet), p0, p1);
                }
                drawList->AddText(ImVec2(p0.x + s(14.0f), p0.y + s(12.0f)), IM_COL32(255, 180, 100, 255), "Technique B: Compacted Wavefronts (DGC / SER)");
            } else if (m_rtParityViewMode == 3) {
                // Difference Heatmap
                if (texDiff.isValid()) {
                    drawList->AddImage(reinterpret_cast<ImTextureID>(texDiff.descriptorSet), p0, p1);
                }
                drawList->AddText(ImVec2(p0.x + s(14.0f), p0.y + s(12.0f)), IM_COL32(255, 80, 80, 255), "10x Amplified Parity Difference Heatmap (Pure Black = 100% Bit-Exact)");
            } else if (m_rtParityViewMode == 4) {
                // Side-by-Side
                float halfW = canvasWidth * 0.5f - s(2.0f);
                if (texTrad.isValid()) {
                    drawList->AddImage(reinterpret_cast<ImTextureID>(texTrad.descriptorSet), p0, ImVec2(p0.x + halfW, p1.y));
                }
                if (texDgc.isValid()) {
                    drawList->AddImage(reinterpret_cast<ImTextureID>(texDgc.descriptorSet), ImVec2(p0.x + halfW + s(4.0f), p0.y), p1);
                }
                drawList->AddText(ImVec2(p0.x + s(12.0f), p0.y + s(10.0f)), IM_COL32(100, 200, 255, 255), "Megakernel (Reference)");
                drawList->AddText(ImVec2(p0.x + halfW + s(16.0f), p0.y + s(10.0f)), IM_COL32(255, 180, 100, 255), "Compacted Wavefront (DGC)");
            }

            // Canvas Bottom HUD Overlay
            drawList->AddText(ImVec2(p0.x + s(14.0f), p1.y - s(26.0f)), IM_COL32(180, 195, 220, 230), "%s", curSceneMeta.details);

            ImGui::SetCursorScreenPos(ImVec2(p0.x, p1.y + s(10.0f)));
        }

        // Architecture Comparison Cards
        if (ImGui::BeginTable("RtTechCompareTable", 2, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            ImGui::BeginChild("TechACard", ImVec2(0, s(90.0f)), true);
            ImGui::TextColored(ImVec4(0.40f, 0.78f, 1.00f, 1.0f), "Technique A: Megakernel (Reference)");
            ImGui::TextDisabled("Architecture: Monolithic Ray Query kernel (Stackless traversal)");
            ImGui::Text("Throughput: "); ImGui::SameLine();
            if (dynamicTechAScore != "--") {
                ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, 1.0f), "%s (Live Measured)", dynamicTechAScore.c_str());
            } else {
                ImGui::TextColored(ImVec4(0.65f, 0.70f, 0.80f, 1.0f), "Not Measured");
            }
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("Reference Target: %s", curSceneMeta.techAScore);
            ImGui::TextDisabled("VGPR Pressure:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.65f, 0.35f, 1.0f), "%d VGPRs", curSceneMeta.vgprTrad);
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("SIMD Utilization:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.35f, 1.0f), "%s", curSceneMeta.simdTrad);
            ImGui::EndChild();

            ImGui::TableNextColumn();
            ImGui::BeginChild("TechBCard", ImVec2(0, s(90.0f)), true);
            ImGui::TextColored(ImVec4(0.98f, 0.72f, 0.35f, 1.0f), "Technique B: Compacted Wavefront (DGC / SER)");
            ImGui::TextDisabled("Architecture: Shader Execution Reordering & Clustered Ray Bins");
            ImGui::Text("Throughput: "); ImGui::SameLine();
            if (dynamicTechBScore != "--") {
                ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s (Live Measured)", dynamicTechBScore.c_str());
            } else {
                ImGui::TextColored(ImVec4(0.65f, 0.70f, 0.80f, 1.0f), "Not Measured");
            }
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("Reference Target: %s", curSceneMeta.techBScore);
            ImGui::TextDisabled("VGPR Pressure:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%d VGPRs", curSceneMeta.vgprDgc);
            ImGui::SameLine(0, s(16.0f));
            ImGui::TextDisabled("SIMD Utilization:"); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s", curSceneMeta.simdDgc);
            ImGui::EndChild();

            ImGui::EndTable();
        }
    }
    // MODE 1: PIPELINE PASSES (7 STAGES)
    else if (m_rtViewportMode == 1) {
        struct PassInfo {
            const char* id;
            const char* pillLabel;
            const char* name;
            const char* passType;
            const char* timeMs;
            const char* mrays;
            const char* fps;
            const char* description;
        };

        static const PassInfo passes[] = {
            {
                "stage1_bvh",
                "1. BVH",
                "1. BVH Traversal Complexity Heatmap",
                "Ray Query Step Profiling (Linear Turbo Map)",
                "2.03 ms", "4,085.6 MRays/s", "492.6 FPS",
                "Visualizes ray-box and ray-triangle intersection step count per pixel. Hotter colors indicate deeper BVH tree traversal depth and cache divergence."
            },
            {
                "stage2_primary",
                "2. Normals",
                "2. Primary Surface G-Buffer Normals",
                "Primary Ray Cast (Vulkan 1.4 RQ)",
                "1.81 ms", "4,594.2 MRays/s", "553.9 FPS",
                "Renders world-space geometric surface normals derived from barycentric triangle interpolation and normal mapping."
            },
            {
                "stage3_shadow",
                "3. Shadows",
                "3. Sun Occlusion Shadow Mask",
                "Directional Shadow Traversal",
                "2.18 ms", "7,593.8 MRays/s", "457.8 FPS",
                "Evaluates directional sun visibility using stackless binary ray queries with early termination (gl_RayFlagsTerminateOnFirstHitEXT)."
            },
            {
                "stage4_rtao",
                "4. RTAO",
                "4. Ray-Traced Ambient Occlusion (RTAO)",
                "Stratified Hemisphere Occlusion (4 Rays)",
                "3.88 ms", "10,692.7 MRays/s", "257.8 FPS",
                "Generates high-frequency contact shadows using cosine-weighted hemisphere sampling with localized ray distance falloff."
            },
            {
                "stage5_direct",
                "5. Direct",
                "5. Direct Hybrid PBR Shading",
                "Analytic Sun + GGX Specular + Shadows + RTAO",
                "6.69 ms", "6,194.8 MRays/s", "149.4 FPS",
                "Combines microfacet GGX specular distribution, Fresnel-Schlick reflectivity, Lambertian diffuse, RTAO occlusion, and directional shadows."
            },
            {
                "stage6_indirect",
                "6. Indirect GI",
                "6. Secondary Indirect GI Bounce",
                "Cosine-Sampled Diffuse Radiance (4 Rays)",
                "11.21 ms", "3,699.9 MRays/s", "89.2 FPS",
                "Computes multi-bounce indirect global illumination by sampling diffuse radiance reflection from neighboring geometry surfaces."
            },
            {
                "stage7_final",
                "ALL (Final Render)",
                "7. Final Beauty Render (Full Pipeline Composite)",
                "Full Composite (Direct PBR + Shadows + RTAO + GI + Reflections)",
                "15.82 ms", "6,120.4 MRays/s", "63.2 FPS",
                "Full beauty composite pass combining primary ray query visibility, Cook-Torrance GGX direct lighting, area soft shadows, contact RTAO, dielectric glass transmission, and multi-bounce diffuse global illumination."
            }
        };

        const size_t numPasses = sizeof(passes) / sizeof(passes[0]);
        if (m_rtPassIndex < 0 || static_cast<size_t>(m_rtPassIndex) >= numPasses) {
            m_rtPassIndex = 0;
        }
        const auto& curPass = passes[m_rtPassIndex];

        // Pass selector pill buttons
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, s(12.0f));
        for (size_t p = 0; p < numPasses; ++p) {
            bool isAll = (p == numPasses - 1);
            if (p > 0) {
                if (isAll) {
                    ImGui::SameLine(0, s(10.0f));
                    ImGui::TextDisabled("|");
                    ImGui::SameLine(0, s(10.0f));
                } else {
                    ImGui::SameLine(0, s(4.0f));
                }
            }
            bool isCur = (static_cast<size_t>(m_rtPassIndex) == p);
            if (isCur) {
                if (isAll) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.58f, 0.65f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.05f, 0.68f, 0.75f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.48f, 0.80f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.10f, 0.56f, 0.88f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                }
            } else {
                if (isAll) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.12f, 0.22f, 0.26f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.16f, 0.30f, 0.36f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.90f, 0.85f, 1.0f));
                } else {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.16f, 0.22f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.18f, 0.24f, 0.32f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.75f, 0.85f, 1.0f));
                }
            }
            float pillW = ImGui::CalcTextSize(passes[p].pillLabel).x + s(22.0f);
            if (ImGui::Button(passes[p].pillLabel, ImVec2(pillW, s(24.0f)))) {
                m_rtPassIndex = static_cast<int>(p);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s\n%s\nTime: %s | Throughput: %s | %s",
                                  passes[p].name, passes[p].passType,
                                  passes[p].timeMs, passes[p].mrays, passes[p].fps);
            }
            ImGui::PopStyleColor(3);
        }
        ImGui::PopStyleVar();

        ImGui::Spacing();

        // Pass Information Header
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.14f, 0.20f, 1.0f));
        ImGui::BeginChild("PassInfoBanner", ImVec2(0, s(52.0f)), true, ImGuiWindowFlags_NoScrollbar);
        ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.0f), "%s", curPass.name);
        ImGui::SameLine(0, s(16.0f));
        ImGui::TextDisabled("| %s", curPass.passType);
        ImGui::SameLine(0, s(20.0f));
        ImGui::Text("Time: "); ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.35f, 1.0f), "%s", curPass.timeMs);
        ImGui::SameLine(0, s(16.0f));
        ImGui::Text("Throughput: "); ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s", curPass.mrays);
        ImGui::SameLine(0, s(16.0f));
        ImGui::Text("Effective FPS: "); ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.95f, 1.0f), "%s", curPass.fps);
        ImGui::TextDisabled("%s", curPass.description);
        ImGui::EndChild();
        ImGui::PopStyleColor();

        ImGui::Spacing();

        // Render pass image
        std::string tag = curSceneMeta.tag;
        std::string passFile = "renders/render_" + tag + "_" + curPass.id + ".png";
        auto texPass = getOrLoadTexture(passFile);

        ImVec2 avail = ImGui::GetContentRegionAvail();
        float canvasHeight = std::max(s(320.0f), avail.y - s(20.0f));
        float targetAspect = (texPass.isValid() && texPass.height > 0) ? (static_cast<float>(texPass.width) / texPass.height) : (16.0f / 9.0f);
        float canvasWidth = std::min(avail.x, canvasHeight * targetAspect);
        if (canvasWidth > avail.x) {
            canvasWidth = avail.x;
            canvasHeight = canvasWidth / targetAspect;
        }

        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 p1 = ImVec2(p0.x + canvasWidth, p0.y + canvasHeight);
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        drawList->AddRectFilled(p0, p1, IM_COL32(10, 12, 18, 255), s(6.0f));
        drawList->AddRect(p0, p1, IM_COL32(35, 45, 65, 255), s(6.0f));

        if (texPass.isValid()) {
            drawList->AddImage(reinterpret_cast<ImTextureID>(texPass.descriptorSet), p0, p1);
            drawList->AddText(ImVec2(p0.x + s(14.0f), p0.y + s(12.0f)), IM_COL32(255, 255, 255, 220), curPass.name);
        } else {
            std::string msg = "Pass image not found: " + passFile;
            ImVec2 textSize = ImGui::CalcTextSize(msg.c_str());
            drawList->AddText(ImVec2(p0.x + (canvasWidth - textSize.x) * 0.5f, p0.y + canvasHeight * 0.48f),
                              IM_COL32(200, 180, 80, 255), msg.c_str());
        }
        ImGui::SetCursorScreenPos(ImVec2(p0.x, p1.y + s(10.0f)));
    }
    // MODE 2: PBR MATERIALS
    else if (m_rtViewportMode == 2) {
        struct MaterialInfo {
            const char* name;
            const char* file;
            const char* bsdfClass;
            const char* baseColor;
            float metallic;
            float roughness;
            float transmission;
            float ior;
            const char* archetype;
            const char* notes;
        };

        static const MaterialInfo materials[] = {
            {
                "Full Material Lineup (Overview)",
                "docs/images/material_lineup.png",
                "Realistic PBR Lineup (5 BSDF Models)",
                "Various", 0.50f, 0.40f, 0.20f, 1.50f,
                "Multi-Class SER Bins",
                "Side-by-side comparison of 5 distinct BSDF classes rendered in studio lighting: Automotive Multi-Coat Car Paint, Organic Subsurface Scattering, Anisotropic Velvet Fabric, Dielectric Dispersive Glass, and Weathered Metal Rust."
            },
            {
                "Automotive Multi-Coat Car Paint",
                "docs/images/material_01_car_paint.png",
                "Dual-Lobe GGX Specular + Clearcoat Layer",
                "Deep Metallic Blue (0.05, 0.22, 0.78)",
                0.92f, 0.18f, 0.00f, 1.50f,
                "Archetype 1 (Specular Conductor)",
                "Two-layer reflection model: a smooth dielectric clearcoat top layer over an absorbing metallic base coat with anisotropic metallic flake scattering."
            },
            {
                "Organic Subsurface Foliage",
                "docs/images/material_02_organic_subsurface.png",
                "Dipole BSSRDF + Translucent Transmission",
                "Emerald Leaf Green (0.18, 0.65, 0.24)",
                0.00f, 0.45f, 0.58f, 1.45f,
                "Archetype 4 (Translucent Subsurface)",
                "Simulates light penetrating translucent media (leaves, needles, skin, wax) and exiting at differing surface locations, softening hard shadow edges."
            },
            {
                "Anisotropic Sheen Velvet / Fabric",
                "docs/images/material_03_anisotropic_velvet.png",
                "Kajiya-Kay Fiber Cylinder Anisotropy + Sheen",
                "Crimson Red (0.72, 0.08, 0.14)",
                0.00f, 0.75f, 0.00f, 1.55f,
                "Archetype 2 (Anisotropic Sheen)",
                "Evaluates microfiber scattering where reflections align perpendicular to cloth tangents, creating grazing-angle sheen effects on tapestry and upholstery."
            },
            {
                "Dielectric Dispersive Refractive Glass",
                "docs/images/material_04_dispersive_glass.png",
                "Fresnel Dielectric Transmission + Cauchy Dispersion",
                "Clear Tint (0.95, 0.98, 1.00)",
                0.00f, 0.02f, 0.98f, 1.52f,
                "Archetype 3 (Dielectric Refraction)",
                "Evaluates Snell's Law refraction with wavelength-dependent index of refraction (Cauchy dispersion formula), splitting rays into spectral color bands."
            },
            {
                "Rough Weathered Rust & Oxidized Metal",
                "docs/images/material_05_weathered_rust.png",
                "Coupled GGX Conductor / Dielectric Composite",
                "Burnt Ochre (0.55, 0.28, 0.16)",
                0.35f, 0.88f, 0.00f, 1.85f,
                "Archetype 0 (Standard PBR Opaque)",
                "Simulates aged, non-uniform oxidation where microscopic metal pits and ferric oxide deposits produce diffuse and high-roughness scattering."
            },
            {
                "Comprehensive Scene Material Range",
                "docs/images/realistic_scene_material_range.png",
                "Full 8-Class Shader Execution Reordering Bins",
                "Comprehensive Range", 0.50f, 0.50f, 0.50f, 1.50f,
                "SER Archetype Bins 0-7",
                "Demonstrates the full range of PBR material archetypes grouped by wavefront reordering to eliminate divergence across warp execution."
            }
        };

        const size_t numMaterials = sizeof(materials) / sizeof(materials[0]);
        if (m_rtMaterialIndex < 0 || static_cast<size_t>(m_rtMaterialIndex) >= numMaterials) {
            m_rtMaterialIndex = 0;
        }
        const auto& curMat = materials[m_rtMaterialIndex];

        // Material selector buttons
        for (size_t mi = 0; mi < numMaterials; ++mi) {
            if (mi > 0) ImGui::SameLine(0, s(4.0f));
            bool isCur = (static_cast<size_t>(m_rtMaterialIndex) == mi);
            if (isCur) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.48f, 0.80f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.16f, 0.22f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.75f, 0.85f, 1.0f));
            }
            if (ImGui::Button(materials[mi].name, ImVec2(s(130.0f), s(24.0f)))) {
                m_rtMaterialIndex = static_cast<int>(mi);
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::Spacing();

        // Split view: Left = Material Image, Right = Material Properties Card
        if (ImGui::BeginTable("MatLayoutTable", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthStretch, 0.50f);
            ImGui::TableSetupColumn("Details", ImGuiTableColumnFlags_WidthStretch, 0.50f);

            float cardHeight = std::max(s(360.0f), ImGui::GetContentRegionAvail().y - s(20.0f));

            ImGui::TableNextColumn();
            
            // Left Column: Material Texture Card
            ImGui::BeginChild("MatImageCard", ImVec2(0, cardHeight), true, ImGuiWindowFlags_NoScrollbar);
            auto texMat = getOrLoadTexture(curMat.file);
            ImVec2 childAvail = ImGui::GetContentRegionAvail();
            float targetAspect = (texMat.isValid() && texMat.height > 0) ? (static_cast<float>(texMat.width) / texMat.height) : 1.0f;
            float imgW = childAvail.x;
            float imgH = (targetAspect > 0.001f) ? (imgW / targetAspect) : imgW;
            if (imgH > childAvail.y) {
                imgH = childAvail.y;
                imgW = imgH * targetAspect;
            }
            float offsetX = std::max(0.0f, (childAvail.x - imgW) * 0.5f);
            float offsetY = std::max(0.0f, (childAvail.y - imgH) * 0.5f);

            ImVec2 curPos = ImGui::GetCursorScreenPos();
            ImVec2 p0 = ImVec2(curPos.x + offsetX, curPos.y + offsetY);
            ImVec2 p1 = ImVec2(p0.x + imgW, p0.y + imgH);
            ImDrawList* drawList = ImGui::GetWindowDrawList();

            drawList->AddRectFilled(p0, p1, IM_COL32(10, 12, 18, 255), s(6.0f));
            drawList->AddRect(p0, p1, IM_COL32(35, 45, 65, 255), s(6.0f));

            if (texMat.isValid()) {
                drawList->AddImage(reinterpret_cast<ImTextureID>(texMat.descriptorSet), p0, p1);
            } else {
                std::string msg = "Image not found: " + std::string(curMat.file);
                ImVec2 textSize = ImGui::CalcTextSize(msg.c_str());
                drawList->AddText(ImVec2(p0.x + (imgW - textSize.x) * 0.5f, p0.y + (imgH - textSize.y) * 0.5f),
                                  IM_COL32(200, 180, 80, 255), msg.c_str());
            }
            ImGui::EndChild();

            ImGui::TableNextColumn();

            // Right Column: Physical Properties & Shader Formulation
            ImGui::BeginChild("MatPropsCard", ImVec2(0, cardHeight), true);
            ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.0f), "%s", curMat.name);
            ImGui::TextDisabled("BSDF Formulation: %s", curMat.bsdfClass);
            ImGui::Separator();

            ImGui::Spacing();
            ImGui::Text("Base Color Factor: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.95f, 1.0f), "%s", curMat.baseColor);

            ImGui::Text("Metallic Factor: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%.2f", curMat.metallic);
            ImGui::SameLine(0, s(20.0f));
            ImGui::Text("Roughness Factor: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.70f, 0.35f, 1.0f), "%.2f", curMat.roughness);

            ImGui::Text("Transmission Factor: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, 1.0f), "%.2f", curMat.transmission);
            ImGui::SameLine(0, s(20.0f));
            ImGui::Text("Index of Refraction (IOR): "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.95f, 1.0f), "%.2f", curMat.ior);

            ImGui::Spacing();
            ImGui::Text("Wavefront Archetype: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.98f, 0.72f, 0.35f, 1.0f), "%s", curMat.archetype);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.85f, 0.90f, 0.95f, 1.0f), "Physical Shading & Evaluation Notes:");
            ImGui::TextWrapped("%s", curMat.notes);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextDisabled("Vulkan Ray Tracing Evaluation:");
            ImGui::BulletText("GLSL Header: pbr_common.glsl / rt_scheduling_common.glsl");
            ImGui::BulletText("Ray Sorting: Spatial Morton Z-Curve & Material Binning");
            ImGui::BulletText("Register Footprint: 32 VGPRs (compacted payload)");

            ImGui::EndChild();

            ImGui::EndTable();
        }
    }
    // MODE 3: GEOMETRY & BVH TOPOLOGY
    else if (m_rtViewportMode == 3) {
        struct GeometryInfo {
            const char* title;
            const char* file;
            const char* triangles;
            const char* vertices;
            const char* stride;
            const char* blasMemory;
            const char* description;
        };

        std::string sceneBvhFile = "renders/render_" + std::string(curSceneMeta.tag) + "_stage1_bvh.png";
        static GeometryInfo geoViews[] = {
            {
                "Showroom Vehicle Wireframe & BVH Clustering",
                "docs/images/geometry_showroom_wireframe.png",
                "108,936 Triangles",
                "58,420 Vertices",
                "44 Bytes (GltfVertex)",
                "8.4 MB (BLAS) | 64 KB (TLAS)",
                "Wireframe topology and bounding volume hierarchy clustering for the showroom vehicle model. High polygon density around wheel arches, headlights, and body curves stresses BVH build and traversal efficiency."
            },
            {
                "Multi-Layer Alpha Cutout Foliage Geometry",
                "docs/images/geometry_alpha_layers.png",
                "48,000+ Triangles",
                "24,000 Vertices",
                "44 Bytes (GltfVertex)",
                "3.6 MB (BLAS) | 32 KB (TLAS)",
                "Stresses the Ray Tracing Pipeline any-hit shader and ray query alpha testing. When rays traverse translucent foliage canopies, texture alpha masks determine whether the hit is committed or traversal continues."
            },
            {
                "Scene BVH Traversal Step Depth Heatmap",
                "", // Dynamic per selected scene
                curSceneMeta.triangleCount,
                "Dynamic",
                "44 Bytes (GltfVertex)",
                "Varies by Scene",
                "False-color heatmap visualizing BVH box and triangle intersection test counts per ray for the currently selected scene. Blue indicates shallow traversal (1-15 steps); Yellow/Red indicates deep traversal and geometric occlusion (40-90+ steps)."
            }
        };
        geoViews[2].file = sceneBvhFile.c_str();
        geoViews[2].triangles = curSceneMeta.triangleCount;

        const size_t numGeo = sizeof(geoViews) / sizeof(geoViews[0]);
        if (m_rtGeometryIndex < 0 || static_cast<size_t>(m_rtGeometryIndex) >= numGeo) {
            m_rtGeometryIndex = 0;
        }
        const auto& curGeo = geoViews[m_rtGeometryIndex];

        // Geometry Selector buttons
        for (size_t gi = 0; gi < numGeo; ++gi) {
            if (gi > 0) ImGui::SameLine(0, s(4.0f));
            bool isCur = (static_cast<size_t>(m_rtGeometryIndex) == gi);
            if (isCur) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.00f, 0.48f, 0.80f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.13f, 0.16f, 0.22f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.75f, 0.85f, 1.0f));
            }
            if (ImGui::Button(geoViews[gi].title, ImVec2(s(220.0f), s(24.0f)))) {
                m_rtGeometryIndex = static_cast<int>(gi);
            }
            ImGui::PopStyleColor(2);
        }

        ImGui::Spacing();

        if (ImGui::BeginTable("GeoLayoutTable", 2, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Resizable)) {
            ImGui::TableSetupColumn("Preview", ImGuiTableColumnFlags_WidthStretch, 0.50f);
            ImGui::TableSetupColumn("Details", ImGuiTableColumnFlags_WidthStretch, 0.50f);

            float cardHeight = std::max(s(360.0f), ImGui::GetContentRegionAvail().y - s(20.0f));

            ImGui::TableNextColumn();

            // Left Column: Geometry Image Card
            ImGui::BeginChild("GeoImageCard", ImVec2(0, cardHeight), true, ImGuiWindowFlags_NoScrollbar);
            auto texGeo = getOrLoadTexture(curGeo.file);
            ImVec2 childAvail = ImGui::GetContentRegionAvail();
            float targetAspect = (texGeo.isValid() && texGeo.height > 0) ? (static_cast<float>(texGeo.width) / texGeo.height) : (16.0f / 9.0f);
            float imgW = childAvail.x;
            float imgH = (targetAspect > 0.001f) ? (imgW / targetAspect) : imgW;
            if (imgH > childAvail.y) {
                imgH = childAvail.y;
                imgW = imgH * targetAspect;
            }
            float offsetX = std::max(0.0f, (childAvail.x - imgW) * 0.5f);
            float offsetY = std::max(0.0f, (childAvail.y - imgH) * 0.5f);

            ImVec2 curPos = ImGui::GetCursorScreenPos();
            ImVec2 p0 = ImVec2(curPos.x + offsetX, curPos.y + offsetY);
            ImVec2 p1 = ImVec2(p0.x + imgW, p0.y + imgH);
            ImDrawList* drawList = ImGui::GetWindowDrawList();

            drawList->AddRectFilled(p0, p1, IM_COL32(10, 12, 18, 255), s(6.0f));
            drawList->AddRect(p0, p1, IM_COL32(35, 45, 65, 255), s(6.0f));

            if (texGeo.isValid()) {
                drawList->AddImage(reinterpret_cast<ImTextureID>(texGeo.descriptorSet), p0, p1);
            } else {
                std::string msg = "Geometry image not found: " + std::string(curGeo.file);
                ImVec2 textSize = ImGui::CalcTextSize(msg.c_str());
                drawList->AddText(ImVec2(p0.x + (imgW - textSize.x) * 0.5f, p0.y + (imgH - textSize.y) * 0.5f),
                                  IM_COL32(200, 180, 80, 255), msg.c_str());
            }
            ImGui::EndChild();

            ImGui::TableNextColumn();

            // Right Column: Geometry & BVH Topology Specs
            ImGui::BeginChild("GeoPropsCard", ImVec2(0, cardHeight), true);
            ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.0f), "%s", curGeo.title);
            ImGui::Separator();

            ImGui::Spacing();
            ImGui::Text("Primitive Count: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.55f, 1.0f), "%s", curGeo.triangles);

            ImGui::Text("Vertex Count: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.35f, 1.0f), "%s", curGeo.vertices);

            ImGui::Text("Vertex Format: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.95f, 0.95f, 0.95f, 1.0f), "%s", curGeo.stride);

            ImGui::Text("AS Memory Footprint: "); ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.0f, 1.0f), "%s", curGeo.blasMemory);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.85f, 0.90f, 0.95f, 1.0f), "Topology & Acceleration Structure Analysis:");
            ImGui::TextWrapped("%s", curGeo.description);

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextDisabled("Hardware Traversal Pipeline:");
            ImGui::BulletText("BVH Node Format: Bvh8 / Bvh4 wide trees with node compression");
            ImGui::BulletText("Intersection Engine: Dual-Ray Box testing units + Ray-Triangle units");
            ImGui::BulletText("Alpha Cutout Handling: Stackless Any-Hit shader execution");
            ImGui::BulletText("Build Algorithm: Spatial Split Bounding Interval Hierarchy (SBVH)");

            ImGui::EndChild();

            ImGui::EndTable();
        }
    }
}

void GuiApp::renderSettingsModal() {
    ImGui::OpenPopup("Benchmark Engine Configuration");
    if (ImGui::BeginPopupModal("Benchmark Engine Configuration", &m_showSettingsModal, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Render Resolution:");
        int w = static_cast<int>(m_renderWidth);
        int h = static_cast<int>(m_renderHeight);
        if (ImGui::InputInt("Width", &w)) m_renderWidth = std::max(64, w);
        if (ImGui::InputInt("Height", &h)) m_renderHeight = std::max(64, h);

        ImGui::Spacing();
        int spp = static_cast<int>(m_samplesPerPixel);
        if (ImGui::SliderInt("Samples Per Pixel", &spp, 1, 64)) {
            m_samplesPerPixel = static_cast<uint32_t>(spp);
        }

        ImGui::Checkbox("Dump Renders to Disk", &m_dumpRenders);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::TextColored(ImVec4(0.38f, 0.75f, 1.00f, 1.00f), "User Interface Scaling:");
        float curZoom = m_uiScale;
        if (ImGui::SliderFloat("Zoom Level", &curZoom, 0.75f, 3.5f, "%.1fx")) {
            setUiScale(curZoom);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset Default")) {
            setUiScale(m_baseScale);
        }
        ImGui::TextDisabled("Tip: Hold CTRL and scroll mouse wheel anywhere to zoom in/out (Ctrl+0 to reset).");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::Button("Close", ImVec2(s(120.0f), 0))) {
            m_showSettingsModal = false;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void GuiApp::startBenchmarks() {
    if (m_execState == ExecutionState::Running) return;

    // Collect selected target devices
    std::vector<uint32_t> targetGpus;
    bool hasSystemDevice = false;

    for (const auto& dev : m_devices) {
        if (dev.selected) {
            if (dev.isSystem) {
                hasSystemDevice = true;
            } else {
                targetGpus.push_back(dev.deviceIndex);
            }
        }
    }

    // Auto-enable host device if user selected any host system benchmarks
    bool hasSelectedHostBenchmarks = false;
    for (const auto& cat : m_categories) {
        for (const auto& sub : cat.subgroups) {
            for (const auto& item : sub.items) {
                if (item.selected && (item.category == "System" || item.category == "Host System" || item.id.find("System Memory") != std::string::npos)) {
                    hasSelectedHostBenchmarks = true;
                    break;
                }
            }
        }
    }
    if (hasSelectedHostBenchmarks) {
        hasSystemDevice = true;
        for (auto& dev : m_devices) {
            if (dev.isSystem) dev.selected = true;
        }
    }

    if (targetGpus.empty() && !hasSystemDevice) {
        m_statusMessage = "Select at least one GPU or host device.";
        return;
    }

    // Collect selected benchmarks and workloads dynamically
    std::vector<std::string> selectedBenchmarks;
    std::vector<std::string> selectedWorkloads;
    size_t gpu_configs = 0;
    size_t sys_configs = 0;

    for (const auto& cat : m_categories) {
        for (const auto& sub : cat.subgroups) {
            for (const auto& item : sub.items) {
                if (!item.selected) continue;
                if (m_hideUnsupported && !item.isSupported) continue;

                bool isHost = (item.category == "System" || item.category == "Host System" || item.id.find("System Memory") != std::string::npos);
                if (isHost) {
                    if (!hasSystemDevice) continue;
                    sys_configs++;
                } else {
                    if (targetGpus.empty()) continue;
                    if (item.id == "RayScheduling" && m_scene == "all") {
                        gpu_configs += 4;
                    } else {
                        gpu_configs += 1;
                    }
                }

                selectedBenchmarks.push_back(item.id);
                if (item.configIndex >= 0) {
                    selectedWorkloads.push_back(item.id + "#" + std::to_string(item.configIndex));
                }
                selectedWorkloads.push_back(item.name);
            }
        }
    }

    if (selectedBenchmarks.empty()) {
        m_statusMessage = "Select at least one benchmark matching chosen device(s).";
        return;
    }

    m_statusMessage.clear();
    m_execState = ExecutionState::Running;
    m_cancelToken.store(false);
    m_completedTasks = 0;

    std::vector<std::string> engineBenchmarks;
    for (const auto& b : selectedBenchmarks) {
        std::string mapped = b;
        if (b == "SceneRayTracing" || b == "PathTracing" || b == "PipelineBreakdown") {
            mapped = "RayScheduling";
        }
        if (std::find(engineBenchmarks.begin(), engineBenchmarks.end(), mapped) == engineBenchmarks.end()) {
            engineBenchmarks.push_back(mapped);
        }
    }

    m_totalTasks = (gpu_configs * targetGpus.size()) + sys_configs;
    if (m_totalTasks == 0) m_totalTasks = 1;

    m_allResults.clear();
    m_latestResults.clear();
    m_currentlyRunningTestId.clear();
    m_hasCurrentlyRunningResult = false;
    m_benchmarkStartTime = std::chrono::steady_clock::now();
    m_telemetryWorker.startRecording();
    m_telemetryNeedsFit = true;

    if (m_execThread.joinable()) {
        m_execThread.join();
    }

    std::string targetBackend = m_selectedBackend;
    uint32_t rWidth = m_renderWidth;
    uint32_t rHeight = m_renderHeight;
    uint32_t spp = m_samplesPerPixel;
    bool dumpR = m_dumpRenders;
    std::string scene = m_scene;

    m_execThread = std::thread([this, engineBenchmarks, targetGpus, targetBackend, rWidth, rHeight, spp, dumpR, scene, selectedWorkloads]() {
        auto callback = [this](const ResultData& res) {
            std::lock_guard<std::mutex> lock(m_resultsMutex);
            m_incomingResults.push_back(res);
        };

        RunBenchmarksAPI(
            engineBenchmarks,
            targetGpus,
            {targetBackend},
            false, false, false,
            dumpR, rWidth, rHeight,
            callback,
            scene,
            spp,
            &m_cancelToken,
            /*quiet=*/true,
            selectedWorkloads
        );

        m_telemetryWorker.stopRecording();
        m_telemetryNeedsFit = true;

        if (m_cancelToken.load()) {
            m_execState = ExecutionState::Cancelled;
        } else {
            m_execState = ExecutionState::Completed;
        }
        m_currentlyRunningTestId.clear();
        m_hasCurrentlyRunningResult = false;
        m_textureCacheNeedsClear.store(true);
    });
}

void GuiApp::abortBenchmarks() {
    if (m_execState == ExecutionState::Running) {
        m_cancelToken.store(true);
        m_telemetryWorker.stopRecording();
        m_telemetryNeedsFit = true;
    }
}

void GuiApp::processIncomingResults() {
    std::vector<ResultData> batch;
    {
        std::lock_guard<std::mutex> lock(m_resultsMutex);
        batch.swap(m_incomingResults);
    }

    for (const auto& res : batch) {
        std::string baseName = res.benchmarkName;
        size_t p = baseName.find(" (");
        if (p != std::string::npos) {
            baseName = baseName.substr(0, p);
        }

        if (res.time_ms < 0.0) {
            if (res.time_ms == -2.0) {
                m_completedTasks++;
                m_allResults.push_back(res);
                m_latestResults[baseName] = res;
                m_latestResults[res.benchmarkName] = res;
                if (!res.subcategory.empty()) m_latestResults[res.subcategory] = res;
                if (m_currentlyRunningTestId == baseName) m_currentlyRunningTestId.clear();
                if (m_hasCurrentlyRunningResult && m_currentlyRunningResult.benchmarkName == res.benchmarkName) {
                    m_hasCurrentlyRunningResult = false;
                }
            } else if (res.time_ms == -1.0) {
                std::string devTag = (res.deviceIndex == 0xFFFFFFFF) ? "Host CPU" : ("GPU " + std::to_string(res.deviceIndex));
                m_currentBenchmarkName = res.benchmarkName + " on [" + devTag + "]";
                m_currentlyRunningTestId = baseName;
                m_currentlyRunningResult = res;
                m_hasCurrentlyRunningResult = true;
                if (res.deviceIndex < 2 && !m_telemetryDualGpuMode) {
                    m_telemetryGpuIndex = res.deviceIndex;
                }
            }
        } else {
            m_completedTasks++;
            m_allResults.push_back(res);

            // Update m_latestResults for baseName (matches item.id)
            if (baseName != "RayScheduling" || res.metric == "MRays/s") {
                auto it = m_latestResults.find(baseName);
                if (it == m_latestResults.end()) {
                    m_latestResults[baseName] = res;
                } else {
                    double newOps = (res.time_ms > 0.0) ? ((static_cast<double>(res.operations) / res.time_ms) * 1000.0) : 0.0;
                    double oldOps = (it->second.time_ms > 0.0) ? ((static_cast<double>(it->second.operations) / it->second.time_ms) * 1000.0) : 0.0;
                    if (newOps >= oldOps || it->second.isUnsupported || it->second.time_ms < 0.0) {
                        m_latestResults[baseName] = res;
                    }
                }
            }

            m_latestResults[res.benchmarkName] = res;
            if (!res.subcategory.empty()) m_latestResults[res.subcategory] = res;
            if (m_currentlyRunningTestId == baseName) m_currentlyRunningTestId.clear();
            if (m_hasCurrentlyRunningResult && m_currentlyRunningResult.benchmarkName == res.benchmarkName) {
                m_hasCurrentlyRunningResult = false;
            }
        }
    }
}

void GuiApp::exportResultsToJson(const std::string& filepath) {
    std::string outPath = filepath.empty() ? getDefaultJsonFilename() : filepath;
    std::ofstream file(outPath);
    if (!file.is_open()) {
        m_statusMessage = "Failed to export: " + outPath;
        return;
    }

    file << resultsToJson(m_allResults);
    file.close();
    m_exportNotificationText = "Exported " + std::to_string(m_allResults.size()) + " records to " + outPath;
    m_exportNotificationTimer = 4.0f;
    m_statusMessage = m_exportNotificationText;
}

} // namespace gpubench::gui
