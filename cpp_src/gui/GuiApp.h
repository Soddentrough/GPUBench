#pragma once

#include "TelemetryWorker.h"
#include "core/RunnerAPI.h"
#include "core/ResultFormatter.h"
#include <imgui.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>

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

struct BenchmarkItem {
    std::string id;          // Exact C++ engine benchmark name (e.g. "FP16", "RayASBuild", "RayScheduling")
    std::string subcategory; // Subcategory / group name (e.g. "FP16", "BLAS Build & Update", "Directional Shadows")
    std::string name;        // Friendly display name / config name (e.g. "Vector", "Matrix", "BLAS Build (1M Tris)")
    std::string category;    // Component name (e.g. "Compute", "Memory", "Ray Tracing", "Graphics", "System")
    std::string metricType;  // e.g. TFLOPS, GB/s, GIS/s, MRays/s
    std::string description; // Tooltip / detail description
    bool selected{true};
    bool isSupported{true};
    std::string supportReason;
    std::string limitationCategory;
};

struct BenchmarkSubgroup {
    std::string name;        // e.g. "FP16", "Bandwidth", "BLAS Build & Update"
    std::string component;   // e.g. "Compute", "Memory", "Ray Tracing", "Graphics", "System"
    std::string engineId;    // e.g. "FP16", "Device Memory Bandwidth", "RayASBuild"
    std::string description;
    std::vector<BenchmarkItem> items;
    bool allSelected{true};
    bool collapsed{false};
};

struct BenchmarkCategory {
    std::string name;        // e.g. "Compute", "Memory", "Ray Tracing", "Graphics", "Host System"
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

    void init();
    void updateAndRender();
    void processEvents();

    bool shouldQuit() const { return m_shouldQuit; }
    void requestQuit() { m_shouldQuit = true; }

    void setSelectedDevice(int deviceIndex);
    void setSelectedDevices(const std::vector<int>& deviceIndices);
    void setSelectedBackend(const std::string& backend);
    void selectOnlyBenchmark(const std::string& benchmarkId);

    void startBenchmarks();
    void abortBenchmarks();

    ExecutionState getExecutionState() const { return m_execState; }
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
        uint32_t targetDeviceIndex) const;

private:
    void setupDarkTheme();
    void initializeBenchmarkCategories();
    void discoverHardware();

    // UI Panels
    void renderLeftSidebar(float width, float height);
    void renderSidebarTelemetry();
    void renderRightWorkspace(float width, float height);
    void renderBenchmarkSuitePanel();
    void renderLiveTelemetryDock();
    void renderResultsScorecard();
    void renderRayTracingViewport();
    void renderSettingsModal();

    void processIncomingResults();
    void exportResultsToJson(const std::string& filepath);

    // Subsystems
    TelemetryWorker m_telemetryWorker;
    bool m_shouldQuit{false};

    // Hardware & Device Selection
    std::vector<SelectableDevice> m_devices;
    std::string m_selectedBackend{"vulkan"};
    uint32_t m_telemetryGpuIndex{0}; // Which GPU to inspect in Telemetry HUD (default GPU 0 Primary)
    bool m_telemetryDualGpuMode{false}; // Dual GPU comparative overlay
    void updateTelemetrySelection();

    // Benchmark Suite Definitions
    std::vector<BenchmarkCategory> m_categories;
    int m_suiteCategoryFilter{0}; // 0: All, 1: Compute, 2: Memory, 3: Ray Tracing, 4: Graphics, 5: Host System
    bool m_hideUnsupported{false}; // Show all tests by default
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

    std::string m_currentBenchmarkName;
    std::string m_currentlyRunningTestId;
    ResultData m_currentlyRunningResult;
    bool m_hasCurrentlyRunningResult{false};
    bool matchesItem(const ResultData& r, const BenchmarkItem& itm, uint32_t activeDev) const;
    size_t m_completedTasks{0};
    size_t m_totalTasks{0};
    std::chrono::steady_clock::time_point m_benchmarkStartTime;
    double m_elapsedSeconds{0.0};
    std::string m_statusMessage;

    // Settings
    std::string m_scene{"all"};
    uint32_t m_renderWidth{1280};
    uint32_t m_renderHeight{720};
    uint32_t m_samplesPerPixel{1};
    bool m_dumpRenders{true};
    bool m_showSettingsModal{false};

    // Parity Split Slider & Viewport Configuration
    float m_paritySplitRatio{0.5f};
    std::string m_selectedRtScene{"Cornell Box (Architectural / GI)"};
    int m_rtBounces{4};
    bool m_showRtRayPaths{true};
    bool m_showRtBvhHeatmap{false};
    bool m_showRtDivergence{false};

    // Navigation & Notifications
    bool m_switchToScorecard{false};
    std::string m_exportNotificationText;
    float m_exportNotificationTimer{0.0f};
    float m_telemetryTimeWindow{60.0f}; // 30s, 60s, 120s

    // Scorecard Filters
    int m_activeScorecardFilter{0}; // 0: All, 1: Compute, 2: Memory, 3: Ray Tracing, 4: Raster, 5: System
    int m_activeDeviceScorecardFilter{-1}; // -1: All, 0: GPU 0, 1: GPU 1, etc.
};

} // namespace gpubench::gui
