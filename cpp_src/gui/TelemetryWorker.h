#pragma once

#include "utils/HardwareTelemetry.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gpubench::gui {

class TelemetryBuffer {
public:
    void push(float val) {
        m_data.push_back(val);
        // Safety cap at 36,000 samples (1 hour at 10Hz) to prevent unbounded memory
        if (m_data.size() > 36000) {
            size_t half = m_data.size() / 2;
            for (size_t i = 0; i < half; ++i) {
                m_data[i] = (m_data[2 * i] + m_data[2 * i + 1]) * 0.5f;
            }
            m_data.resize(half);
        }
    }

    void trimRolling(size_t maxSamples) {
        if (m_data.size() > maxSamples) {
            size_t drop = m_data.size() - maxSamples;
            m_data.erase(m_data.begin(), m_data.begin() + static_cast<std::ptrdiff_t>(drop));
        }
    }

    size_t size() const { return m_data.size(); }
    bool empty() const { return m_data.empty(); }
    size_t offset() const { return 0; }
    const float* data() const { return m_data.data(); }
    float* data() { return m_data.data(); }
    float latest() const { return m_data.empty() ? 0.0f : m_data.back(); }
    float back() const { return latest(); }
    float front() const { return m_data.empty() ? 0.0f : m_data.front(); }

    void clear() { m_data.clear(); }
    void reserve(size_t n) { m_data.reserve(n); }

private:
    std::vector<float> m_data;
};

struct DeviceTelemetrySnapshot {
    std::string name;
    std::string pciBus;
    uint32_t deviceIndex{0};

    // Current instant metrics
    float sclkMhz{0.0f};
    float mclkMhz{0.0f};
    float powerWatts{0.0f};
    float tempEdgeC{0.0f};
    float tempJctC{0.0f};
    float tempMemC{0.0f};
    float fanRpm{0.0f};
    float fanSpeedPct{0.0f};
    float gpuBusyPct{0.0f};
    float memBusyPct{0.0f};
    uint64_t vramUsedBytes{0};
    uint64_t vramTotalBytes{0};
    uint64_t gttUsedBytes{0};
    uint64_t gttTotalBytes{0};

    // Explicit per-sensor availability flags (never display fake 0.0 W / 0 C)
    bool isAvailable{false};
    bool hasSclk{false};
    bool hasMclk{false};
    bool hasPower{false};
    bool hasTempEdge{false};
    bool hasTempJct{false};
    bool hasTempMem{false};
    bool hasFanRpm{false};
    bool hasFanSpeedPct{false};
    bool hasGpuBusy{false};
    bool hasMemBusy{false};
    bool hasVram{false};
    bool hasGtt{false};

    // Histories for ImPlot (rolling live window when idle; full run from t=0 when recording)
    TelemetryBuffer timeHistory;
    TelemetryBuffer sclkHistory;
    TelemetryBuffer mclkHistory;
    TelemetryBuffer powerHistory;
    TelemetryBuffer tempEdgeHistory;
    TelemetryBuffer tempJctHistory;
    TelemetryBuffer tempMemHistory;
    TelemetryBuffer vramUsedMbHistory;
    TelemetryBuffer gpuBusyHistory;
};

class TelemetryWorker {
public:
    TelemetryWorker();
    ~TelemetryWorker();

    void start();
    void stop();
    void refreshDevices();

    // Benchmark Run Session Telemetry
    void startRecording();
    void stopRecording();
    bool isRecording() const { return m_isRecording.load(); }
    double getRunDuration() const;
    void resetHistory();

    uint32_t getDeviceCount() const;
    bool getSnapshot(uint32_t deviceIndex, DeviceTelemetrySnapshot& outSnapshot);
    std::vector<std::string> getDeviceNames() const;

private:
    struct DevicePollState {
        uint32_t deviceIndex{0};
        GpuTelemetryData prevSample{};
        std::chrono::steady_clock::time_point prevSampleTime{};
        bool hasPrevSample{false};
    };

    void workerLoop();
    void discoverDevices();
    void applySample(DevicePollState& state, DeviceTelemetrySnapshot& snapshot,
                     const GpuTelemetryData& hw,
                     std::chrono::steady_clock::time_point now,
                     double timestampSec, bool isRecordingRun);

    std::atomic<bool> m_running{false};
    std::atomic<bool> m_isRecording{false};
    std::atomic<bool> m_hasCompletedRun{false};
    std::thread m_workerThread;
    mutable std::mutex m_mutex;
    std::vector<DevicePollState> m_pollStates;
    std::vector<DeviceTelemetrySnapshot> m_snapshots;
    std::chrono::steady_clock::time_point m_startTime;
    std::chrono::steady_clock::time_point m_idleStartTime;
    std::chrono::steady_clock::time_point m_runStartTime;
    double m_runDuration{0.0};
};

} // namespace gpubench::gui
