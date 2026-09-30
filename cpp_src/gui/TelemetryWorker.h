#pragma once

#include <vector>
#include <string>
#include <array>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstdint>
#include <chrono>

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

    size_t size() const { return m_data.size(); }
    bool empty() const { return m_data.empty(); }
    size_t offset() const { return 0; }
    const float* data() const { return m_data.data(); }
    float* data() { return m_data.data(); }
    float latest() const { return m_data.empty() ? 0.0f : m_data.back(); }
    float back() const { return latest(); }

    void clear() {
        m_data.clear();
    }
    void reserve(size_t n) {
        m_data.reserve(n);
    }

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
    float gpuBusyPct{0.0f};
    float memBusyPct{0.0f};
    uint64_t vramUsedBytes{0};
    uint64_t vramTotalBytes{0};

    // Histories for ImPlot (records full run duration)
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

struct DeviceSysfsPaths {
    uint32_t deviceIndex{0};
    std::string cardName;
    std::string hwmonDir;
    std::string drmDeviceDir;
};

class TelemetryWorker {
public:
    TelemetryWorker();
    ~TelemetryWorker();

    void start();
    void stop();

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
    void workerLoop();
    void discoverDevices();
    void pollDevice(const DeviceSysfsPaths& paths, DeviceTelemetrySnapshot& snapshot, double timestampSec, bool recordHistory);

    std::atomic<bool> m_running{false};
    std::atomic<bool> m_isRecording{false};
    std::thread m_workerThread;
    mutable std::mutex m_mutex;
    std::vector<DeviceSysfsPaths> m_devicePaths;
    std::vector<DeviceTelemetrySnapshot> m_snapshots;
    std::chrono::steady_clock::time_point m_startTime;
    std::chrono::steady_clock::time_point m_runStartTime;
    double m_runDuration{0.0};
};

} // namespace gpubench::gui
