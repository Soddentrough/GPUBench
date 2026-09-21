#pragma once

#include <vector>
#include <string>
#include <array>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstdint>

namespace gpubench::gui {

constexpr size_t TELEMETRY_HISTORY_CAPACITY = 600; // 60 seconds at 10Hz

template<typename T, size_t N = TELEMETRY_HISTORY_CAPACITY>
class CircularBuffer {
public:
    void push(T val) {
        m_data[m_head] = val;
        m_head = (m_head + 1) % N;
        if (m_size < N) m_size++;
    }

    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }
    size_t offset() const { return (m_size < N) ? 0 : m_head; }
    const T* data() const { return m_data.data(); }
    T latest() const {
        if (m_size == 0) return T{};
        size_t idx = (m_head + N - 1) % N;
        return m_data[idx];
    }
    T back() const { return latest(); }

    void clear() {
        m_data.fill(T{});
        m_head = 0;
        m_size = 0;
    }

private:
    std::array<T, N> m_data{};
    size_t m_head{0};
    size_t m_size{0};
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

    // Circular histories for ImPlot
    CircularBuffer<float> timeHistory;
    CircularBuffer<float> sclkHistory;
    CircularBuffer<float> mclkHistory;
    CircularBuffer<float> powerHistory;
    CircularBuffer<float> tempEdgeHistory;
    CircularBuffer<float> tempJctHistory;
    CircularBuffer<float> tempMemHistory;
    CircularBuffer<float> vramUsedMbHistory;
    CircularBuffer<float> gpuBusyHistory;
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

    uint32_t getDeviceCount() const;
    bool getSnapshot(uint32_t deviceIndex, DeviceTelemetrySnapshot& outSnapshot);
    std::vector<std::string> getDeviceNames() const;

private:
    void workerLoop();
    void discoverDevices();
    void pollDevice(const DeviceSysfsPaths& paths, DeviceTelemetrySnapshot& snapshot, double timestampSec);

    std::atomic<bool> m_running{false};
    std::thread m_workerThread;
    mutable std::mutex m_mutex;
    std::vector<DeviceSysfsPaths> m_devicePaths;
    std::vector<DeviceTelemetrySnapshot> m_snapshots;
    std::chrono::steady_clock::time_point m_startTime;
};

} // namespace gpubench::gui
