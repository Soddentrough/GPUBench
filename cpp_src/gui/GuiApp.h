#pragma once

#include "GuiBringUpTheme.h"
#include "TelemetryWorker.h"
#include "VulkanContext.h"
#include "core/RunnerAPI.h"
#include "core/ResultFormatter.h"
#include "utils/IsaDumper.h"
#include <imgui.h>
#include <implot.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace gpubench::gui {

enum class ExecutionState {
    Idle,
    Running,
    Completed,
    Cancelled,
    Error
};

struct SelectableDevice {
    uint32_t deviceIndex; // 0, 1, or 0xFFFFFFFF for System
    std::string name;
    std::string backend;
    std::string architecture;
    std::string driver;
    uint64_t vramTotalMb{0};
    bool isSystem{false};
    bool selected{false};
};

// Visualization descriptor attached to benchmarks that operate on graphics/compute
// pipeline data (BVH, scenes, materials, ray scheduling, OMM, Work Graphs, etc.).
// Tier 1: static pre-baked 400x225 thumbnail (assetPath, may contain "{scene}").
// Tier 2: procedural vector schematic drawn via ImDrawList (diagramId).
struct BenchmarkVisualization {
    std::string assetPath;           // e.g. "assets/thumbnails/thumb_{scene}_stage1_bvh.png"
    std::string diagramId;           // e.g. "cone_divergence_45", "traversal_morton_8x4"
    std::string caption;             // One-line explanation shown under the canvas
    float aspectRatio{16.0f / 9.0f}; // Canvas aspect (thumbnails are 16:9)
    int linkedViewportMode{-1};      // -1: none | 0: Scenes & Parity | 1: Pipeline Passes | 2: PBR Materials | 3: Geometry & BVH
    int linkedViewportSubIndex{0};   // Pass / material / geometry index within that mode

    bool hasVisualization() const { return !assetPath.empty() || !diagramId.empty(); }
};

struct BenchmarkItem {
    std::string id;          // Exact C++ engine benchmark name (e.g. "FP16", "RayASBuild", "RayScheduling")
    std::string subcategory; // Subcategory / group name (e.g. "FP16", "BLAS Build & Update", "Directional Shadows")
    std::string name;        // Friendly display name / config name (e.g. "Vector", "Matrix", "BLAS Build (1M Tris)")
    std::string category;    // Component name (e.g. "Compute", "Memory", "Ray Tracing", "Scheduling", "Graphics", "System")
    std::string metricType;  // e.g. TFLOPS, GB/s, GIS/s, MRays/s, MItems/s, us
    std::string description; // Tooltip / detail description (hydrated from IBenchmark metadata)
    bool selected{true};
    bool isSupported{true};
    std::string supportReason;
    std::string limitationCategory;
    int configIndex{-1};
    BenchmarkVisualization viz;   // Optional pipeline visualization for the hover tooltip

    BenchmarkItem() = default;
    BenchmarkItem(std::string id_, std::string subcat_, std::string name_, std::string cat_,
                  std::string metric_, std::string desc_, bool sel_ = true, int cfg_ = -1)
        : id(std::move(id_)), subcategory(std::move(subcat_)), name(std::move(name_)),
          category(std::move(cat_)), metricType(std::move(metric_)), description(std::move(desc_)),
          selected(sel_), configIndex(cfg_) {}
};

struct BenchmarkSubgroup {
    std::string name;        // e.g. "Compute Precision", "BLAS Build & Update"
    std::string component;   // e.g. "Compute", "Memory", "Ray Tracing", "Scheduling", "Graphics", "System"
    std::string engineId;    // e.g. "FP16", "Device Memory Bandwidth", "RayASBuild"
    std::string description;
    std::vector<BenchmarkItem> items;
    bool allSelected{true};
    bool collapsed{false};
};

struct BenchmarkCategory {
    std::string name;        // e.g. "Compute", "Memory", "Ray Tracing", "Scheduling", "Graphics", "Host System"
    std::string description;
    std::vector<BenchmarkSubgroup> subgroups;
    bool allSelected{true};

    size_t totalItemCount() const {
        size_t c = 0;
        for (const auto& sg : subgroups) c += sg.items.size();
        return c;
    }
};

class GuiApp {
public:
    GuiApp();
    ~GuiApp();

    void init(VulkanContext* vulkanContext = nullptr, float uiScale = 1.0f);
    void updateAndRender();
    void processEvents();

    void setUiScale(float scale);
    float getUiScale() const { return m_uiScale; }
    float getBaseScale() const { return m_baseScale; }
    float s(float v) const { return std::round(v * m_uiScale); }

    bool shouldQuit() const { return m_shouldQuit; }
    void requestQuit() { m_shouldQuit = true; }

    const std::vector<BenchmarkCategory>& getCategories() const { return m_categories; }

    void setSelectedDevice(int deviceIndex);
    void setSelectedDevices(const std::vector<int>& deviceIndices);
    void selectAllGpus();
    void setSelectedBackend(const std::string& backend);
    void selectOnlyBenchmark(const std::string& benchmarkId);
    void setScene(const std::string& scene);
    void setRenderResolution(uint32_t width, uint32_t height) { m_renderWidth = width; m_renderHeight = height; }
    void applyAutoRenderResolution();
    void setDumpRenders(bool dump) { m_dumpRenders = dump; }
    void setVerbose(bool verbose) { m_verbose = verbose; }
    void setDebug(bool debug) { m_debug = debug; if (debug) m_verbose = true; }
    uint32_t getRenderWidth() const { return m_renderWidth; }
    uint32_t getRenderHeight() const { return m_renderHeight; }
    bool getDumpRenders() const { return m_dumpRenders; }

    void startBenchmarks();
    void abortBenchmarks();

    struct ScopeTrigger {
        float timestampSec{0.0f};
        std::string label;
    };

    void setInstrumentMode(int mode);
    int getInstrumentMode() const { return m_instrumentMode; }
    void setPalette(BringUpPalette p);
    BringUpPalette getPalette() const { return m_palette; }
    bool waitForPendingInspections(uint32_t timeoutMs = 2500);

    // Unified v2.0 JSON Session Import / Export / Comparison
    bool importResultsFromJson(const std::string& filepath);
    bool loadComparisonFromJson(const std::string& filepath);
    void clearComparisonResults();
    void exportResultsToJson(const std::string& filepath);

    ExecutionState getExecutionState() const { return m_execState; }
    ExecutionState getState() const { return m_execState; }
    const std::vector<ResultData>& getResults() const { return m_allResults; }
    void injectResultForTesting(const ResultData& res) { m_allResults.push_back(res); }

    struct BenchmarkDisplayInfo {
        ResultData primaryResult;
        ResultData baselineResult;
        bool hasResult{false};
        bool isBaseline{false};
        bool hasComparison{false};
        double speedupRatio{1.0};
        double percentDelta{0.0};
        std::string scoreText;
        std::string deltaText;
        ImVec4 deltaColor{0.6f, 0.6f, 0.6f, 1.0f};
        std::string tooltipDetail;
        std::string pipelineExplanation;

        struct SubResult {
            std::string label;
            std::string score;
            std::string delta;
            ImVec4 deltaColor{0.6f, 0.6f, 0.6f, 1.0f};
            std::string details;
            bool isUnsupported{false};
        };
        std::vector<SubResult> subResults;
    };

    BenchmarkDisplayInfo getBenchmarkDisplayInfo(
        const BenchmarkItem& item,
        uint32_t targetDeviceIndex,
        const std::string& sceneOverride = "") const;

private:
    void setupDarkTheme(float scale = 1.0f);
    void initializeBenchmarkCategories();
    void hydrateMetadataFromCatalog();
    void applyVisualizationMetadata();
    void discoverHardware();

    void updateZoomShortcuts();
    void renderZoomToast();

    ImFont* boldFont() const { return m_vulkanContext ? m_vulkanContext->getBoldFont() : nullptr; }
    ImFont* monoFont() const { return m_vulkanContext ? m_vulkanContext->getMonoFont() : nullptr; }
    void pushBoldFont() const { if (ImFont* f = boldFont()) ImGui::PushFont(f, f->LegacySize); }
    void popBoldFont() const { if (boldFont()) ImGui::PopFont(); }
    void pushMonoFont() const { if (ImFont* f = monoFont()) ImGui::PushFont(f, f->LegacySize); }
    void popMonoFont() const { if (monoFont()) ImGui::PopFont(); }

    float m_uiScale{1.0f};
    float m_baseScale{1.0f};
    ImGuiStyle m_baseStyle{};
    bool m_baseStyleInitialized{false};
    ImPlotStyle m_basePlotStyle{};
    bool m_basePlotStyleInitialized{false};
    float m_zoomToastTimer{0.0f};

    void renderTopInstrumentHeader(float width);
    void renderIsaInspectorPanel();
    void renderCommandPromptBar(float width);
    void syncBringUpTheme();

    const BenchmarkItem* getInspectedItem() const;
    void selectInspectedBenchmark(int catIdx, int sgIdx, int itemIdx, bool switchToF1 = false);
    void ensureDefaultInspectedBenchmark();
    std::string makeIsaCacheKey(const BenchmarkItem& item) const;
    void requestIsaInspection(const BenchmarkItem& item, bool forceRefresh = false);
    void joinIsaThread();
    void runSingleBenchmarkItem(const BenchmarkItem& item);
    void stampScopeTrigger(const std::string& label);
    void executeCommand(const std::string& rawCmd);

    // UI Panels (implemented across modular translation units)
    void renderSidebarTelemetry();
    void renderBenchmarkSuitePanel();
    void renderLiveTelemetryDock();
    void renderResultsScorecard();
    void renderRooflinePanel();
    void renderRayTracingViewport();
    void renderSettingsModal();
    void renderJsonSessionModal();

    void processIncomingResults();

    // Benchmark Tooltip Visualizations (GuiDiagramRenderer.cpp)
    void renderBenchmarkTooltip(const BenchmarkItem& item, const std::string& sceneOverride = "");
    void renderProceduralDiagram(const std::string& diagramId, ImVec2 p0, ImVec2 p1) const;
    std::string currentSceneTag() const;

    // Subsystems
    TelemetryWorker m_telemetryWorker;
    bool m_shouldQuit{false};

    // Hardware & Device Selection
    std::vector<SelectableDevice> m_devices;
    std::vector<DeviceProfile> m_deviceProfiles;
    std::string m_selectedBackend{"vulkan"};
    std::vector<ComputeApiSupportInfo> m_apiSupportList;
    uint32_t m_telemetryGpuIndex{0};
    bool m_telemetryDualGpuMode{false};
    void updateTelemetrySelection();

    // Benchmark Suite Definitions
    std::vector<BenchmarkCategory> m_categories;
    std::unordered_map<std::string, bool> m_sceneSubgroupCollapsed;
    int m_suiteCategoryFilter{0}; // 0: All, 1..N: m_categories[idx - 1]
    std::string m_suiteActiveScene{"indoor"};
    bool m_hideUnsupported{false};
    std::string m_lastProbedBackend{""};
    uint32_t m_lastProbedDeviceIndex{0xFFFFFFFF};
    void updateBenchmarkSupport();

    // Execution State
    ExecutionState m_execState{ExecutionState::Idle};
    std::atomic<bool> m_cancelToken{false};
    std::thread m_execThread;
    std::mutex m_resultsMutex;
    std::vector<ResultData> m_incomingResults;
    std::vector<ResultData> m_allResults;
    std::unordered_map<std::string, ResultData> m_latestResults;

    // Loaded Comparison Baseline (from unified CLI/GUI v2.0 JSON)
    std::vector<ResultData> m_comparisonResults;
    std::string m_comparisonSourceLabel;
    char m_jsonPathBuffer[512]{};
    bool m_showJsonSessionModal{false};

    std::string m_currentBenchmarkName;
    std::string m_currentlyRunningTestId;
    ResultData m_currentlyRunningResult;
    bool m_hasCurrentlyRunningResult{false};
    bool matchesItem(const ResultData& r, const BenchmarkItem& itm, uint32_t activeDev) const;
    const ResultData* findComparisonMatch(const ResultData& curRes) const;
    size_t m_completedTasks{0};
    size_t m_totalTasks{0};
    std::chrono::steady_clock::time_point m_benchmarkStartTime;
    double m_elapsedSeconds{0.0};
    std::string m_statusMessage;

    // Settings
    std::string m_scene{"all"};
    uint32_t m_renderWidth{3840};
    uint32_t m_renderHeight{2160};
    uint32_t m_samplesPerPixel{1};
    bool m_dumpRenders{true};
    bool m_verbose{false};
    bool m_debug{false};
    bool m_showSettingsModal{false};
    int m_scorecardSceneFilter{0}; // 0: All, 1: Indoor (Sponza), 2: Forest, 3: Landscape, 4: Showroom

    // Vulkan GUI Context & Texture Management
    VulkanContext* m_vulkanContext{nullptr};
    std::unordered_map<std::string, VulkanContext::TextureResource> m_textureCache;
    VulkanContext::TextureResource getOrLoadTexture(const std::string& relPath);
    void clearTextureCache();
    std::atomic<bool> m_textureCacheNeedsClear{false};
    size_t m_prevTotalSelectedItems{(size_t)-1};

    // Ray Tracing Viewport Configuration & State
    int m_rtViewportMode{0};    // 0: Scenes & Parity, 1: Pipeline Passes, 2: PBR Materials, 3: Geometry & BVH
    int m_rtSceneIndex{0};      // 0: Showroom Studio, 1: Indoor Atrium, 2: Open-World Forest, 3: Outdoor Landscape
    int m_rtParityViewMode{0};  // 0: DGC Render, 1: Megakernel Render, 2: Side-by-Side Parity, 3: Path Tracing (1 vs 16 SPP), 4: 128-Light Direct
    int m_rtPassIndex{0};       // 0: BVH Heatmap, 1: G-Buffer Normals, 2: Shadows, 3: RTAO, 4: Direct PBR, 5: Indirect GI, 6: Final Render
    int m_rtMaterialIndex{0};   // 0: Lineup, 1: Car Paint, 2: Subsurface, 3: Velvet, 4: Glass, 5: Rust, 6: Range
    int m_rtGeometryIndex{0};   // 0: Vehicle Wireframe & BVH, 1: Alpha Cutout Layers, 2: Scene BVH Heatmap

    // Navigation & Notifications
    bool m_switchToScorecard{false};
    bool m_switchToRoofline{false};
    bool m_switchToRtViewport{false};
    std::string m_exportNotificationText;
    float m_exportNotificationTimer{0.0f};
    float m_telemetryTimeWindow{0.0f};
    bool m_telemetryNeedsFit{true};

    // Scorecard & Charts Filters
    int m_activeScorecardFilter{0}; // 0: All, 1: Compute, 2: Memory, 3: Ray Tracing, 4: Scheduling, 5: Graphics, 6: Host System
    int m_activeDeviceScorecardFilter{-1}; // -1: All, -2: Host CPU, 0: GPU 0, 1: GPU 1, etc.
    int m_rooflineDeviceFilter{0};
    int m_chartDomainFilter{-1};

    // Silicon Bring-Up Mode & Palette
    BringUpPalette m_palette{BringUpPalette::AmberP3};
    bool m_themeDirty{true};
    int m_instrumentMode{0}; // 0: F1 ISA & DIE, 1: F2 PERFORMANCE & SCORECARD, 2: F3 RT PARITY
    int m_f2SubMode{0};      // 0: Split, 1: Charts, 2: Scorecard
    bool m_telemetryExpanded{false};
    float m_rtSplitRatio{0.5f};
    int m_rtComparatorPair{0};

    // Inspected ROM Slot & Async ISA Disassembly Cache
    int m_inspectedCategoryIdx{-1};
    int m_inspectedSubgroupIdx{-1};
    int m_inspectedItemIdx{-1};
    bool m_isaHotLoopOnly{true};
    int m_isaExecStageIdx{0};
    char m_isaFilterBuf[128]{};
    char m_suiteSearchBuf[128]{};
    std::mutex m_isaMutex;
    std::thread m_isaThread;
    std::atomic<bool> m_isaLoading{false};
    std::string m_isaActiveKey;
    std::unordered_map<std::string, IsaInspectionResult> m_isaCache;

    // Oscilloscope Triggers & Hardware Command Prompt
    std::vector<ScopeTrigger> m_scopeTriggers;
    char m_cmdBuffer[256]{};
    std::vector<std::string> m_cmdHistory;
    std::string m_cmdStatusText;
    bool m_focusCmdInput{false};
    bool m_singleSlotSelectionRestore{false};
    std::vector<bool> m_savedSelectionSnapshot;
};

} // namespace gpubench::gui
