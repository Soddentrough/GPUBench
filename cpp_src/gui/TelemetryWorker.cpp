#include "TelemetryWorker.h"
#include "core/RunnerAPI.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif

#include <algorithm>
#include <chrono>

namespace gpubench::gui {

TelemetryWorker::TelemetryWorker() {
    auto now = std::chrono::steady_clock::now();
    m_startTime = now;
    m_idleStartTime = now;
    discoverDevices();
}

TelemetryWorker::~TelemetryWorker() {
    stop();
}

void TelemetryWorker::refreshDevices() {
    discoverDevices();
}

void TelemetryWorker::discoverDevices() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_pollStates.clear();
    m_snapshots.clear();

    std::vector<GpuTelemetryData> hwGpus = HardwareTelemetry::queryAllGpus();
    std::vector<DeviceProfile> profiles = GetDeviceProfilesAPI();

    size_t count = std::max(hwGpus.size(), profiles.size());
    auto now = std::chrono::steady_clock::now();

    for (size_t i = 0; i < count; ++i) {
        uint32_t devIdx = static_cast<uint32_t>(i);
        std::string devName;
        std::string pciBus;

        for (const auto& prof : profiles) {
            if (prof.deviceIndex == devIdx && !prof.deviceName.empty()) {
                devName = prof.deviceName;
                break;
            }
        }
        if (i < hwGpus.size()) {
            if (devName.empty() && !hwGpus[i].deviceName.empty()) {
                devName = hwGpus[i].deviceName;
            }
            pciBus = hwGpus[i].pciBusId;
        }
        if (devName.empty()) {
            devName = "GPU " + std::to_string(devIdx);
        }

        DevicePollState pollState;
        pollState.deviceIndex = devIdx;
        if (i < hwGpus.size() && hwGpus[i].isAvailable) {
            pollState.prevSample = hwGpus[i];
            pollState.prevSampleTime = now;
            pollState.hasPrevSample = true;
        }
        m_pollStates.push_back(pollState);

        DeviceTelemetrySnapshot snap;
        snap.deviceIndex = devIdx;
        snap.name = "GPU " + std::to_string(devIdx) + ": " + devName;
        snap.pciBus = pciBus;
        m_snapshots.push_back(std::move(snap));
    }

    // Perform an initial synchronous sample so readouts are populated on frame 0
    for (size_t i = 0; i < m_pollStates.size() && i < m_snapshots.size(); ++i) {
        pollDevice(m_pollStates[i], m_snapshots[i], 0.0, false);
    }
}

void TelemetryWorker::start() {
    if (m_running.load()) return;
    if (m_snapshots.empty()) {
        discoverDevices();
    }
    m_running.store(true);
    m_workerThread = std::thread(&TelemetryWorker::workerLoop, this);
}

void TelemetryWorker::stop() {
    if (!m_running.load()) return;
    m_running.store(false);
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
}

void TelemetryWorker::startRecording() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_runStartTime = std::chrono::steady_clock::now();
    m_runDuration = 0.0;
    m_hasCompletedRun.store(false);
    for (auto& snap : m_snapshots) {
        snap.timeHistory.clear();
        snap.sclkHistory.clear();
        snap.mclkHistory.clear();
        snap.powerHistory.clear();
        snap.tempEdgeHistory.clear();
        snap.tempJctHistory.clear();
        snap.tempMemHistory.clear();
        snap.vramUsedMbHistory.clear();
        snap.gpuBusyHistory.clear();
    }
    m_isRecording.store(true);
}

void TelemetryWorker::stopRecording() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_isRecording.load()) {
        auto now = std::chrono::steady_clock::now();
        m_runDuration = std::chrono::duration<double>(now - m_runStartTime).count();
        m_hasCompletedRun.store(true);
    }
    m_isRecording.store(false);
}

double TelemetryWorker::getRunDuration() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_isRecording.load()) {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - m_runStartTime).count();
    }
    return m_runDuration;
}

void TelemetryWorker::resetHistory() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_runDuration = 0.0;
    m_hasCompletedRun.store(false);
    m_idleStartTime = std::chrono::steady_clock::now();
    for (auto& snap : m_snapshots) {
        snap.timeHistory.clear();
        snap.sclkHistory.clear();
        snap.mclkHistory.clear();
        snap.powerHistory.clear();
        snap.tempEdgeHistory.clear();
        snap.tempJctHistory.clear();
        snap.tempMemHistory.clear();
        snap.vramUsedMbHistory.clear();
        snap.gpuBusyHistory.clear();
    }
}

uint32_t TelemetryWorker::getDeviceCount() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return static_cast<uint32_t>(m_snapshots.size());
}

bool TelemetryWorker::getSnapshot(uint32_t deviceIndex, DeviceTelemetrySnapshot& outSnapshot) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& snap : m_snapshots) {
        if (snap.deviceIndex == deviceIndex) {
            outSnapshot = snap;
            return true;
        }
    }
    if (!m_snapshots.empty()) {
        outSnapshot = m_snapshots.front();
        return true;
    }
    return false;
}

std::vector<std::string> TelemetryWorker::getDeviceNames() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<std::string> names;
    names.reserve(m_snapshots.size());
    for (const auto& snap : m_snapshots) {
        names.push_back(snap.name);
    }
    return names;
}

void TelemetryWorker::pollDevice(DevicePollState& state, DeviceTelemetrySnapshot& snap,
                                 double timestampSec, bool isRecordingRun) {
    auto now = std::chrono::steady_clock::now();
    GpuTelemetryData hw = HardwareTelemetry::queryGpu(state.deviceIndex);

    float powerWatts = 0.0f;
    bool hasPower = false;
    if (state.hasPrevSample && hw.hasEnergyCounter && state.prevSample.hasEnergyCounter) {
        double dtMs = std::chrono::duration<double, std::milli>(now - state.prevSampleTime).count();
        if (dtMs >= 10.0) {
            float energyWatts = HardwareTelemetry::computeWindowPowerWatts(state.prevSample, hw, dtMs);
            if (energyWatts > 0.0f) {
                powerWatts = energyWatts;
                hasPower = true;
            }
        }
    }
    if (!hasPower && hw.hasPower) {
        powerWatts = hw.powerWatts;
        hasPower = true;
    }

    state.prevSample = hw;
    state.prevSampleTime = now;
    state.hasPrevSample = hw.isAvailable;

    uint64_t vramUsed = hw.vramUsedBytes;
    uint64_t vramTotal = hw.vramTotalBytes;
    bool hasVram = hw.hasVram;

#ifdef HAVE_VULKAN
    if (!hasVram || vramTotal == 0) {
        auto sharedVk = GetSharedVulkanContextAPI();
        if (sharedVk && sharedVk->getSelectedDeviceIndex() == state.deviceIndex) {
            auto budget = sharedVk->queryVramBudget();
            if (budget.available && budget.budgetBytes > 0) {
                vramUsed = budget.usageBytes;
                vramTotal = std::max(budget.budgetBytes, sharedVk->getCurrentDeviceInfo().memorySize);
                hasVram = true;
            }
        }
    }
#endif

    snap.isAvailable = hw.isAvailable || hasVram;
    snap.hasSclk = hw.hasCoreClock;
    snap.sclkMhz = hw.hasCoreClock ? static_cast<float>(hw.coreClockMhz) : 0.0f;

    snap.hasMclk = hw.hasMemoryClock;
    snap.mclkMhz = hw.hasMemoryClock ? static_cast<float>(hw.memoryClockMhz) : 0.0f;

    snap.hasPower = hasPower;
    snap.powerWatts = hasPower ? powerWatts : 0.0f;

    snap.hasTempEdge = hw.hasTempEdge;
    snap.tempEdgeC = hw.hasTempEdge ? hw.temperatureC : 0.0f;

    snap.hasTempJct = hw.hasTempJct;
    snap.tempJctC = hw.hasTempJct ? hw.tempJunctionC : 0.0f;

    snap.hasTempMem = hw.hasTempMem;
    snap.tempMemC = hw.hasTempMem ? hw.tempMemoryC : 0.0f;

    snap.hasFanRpm = hw.hasFanRpm;
    snap.fanRpm = hw.hasFanRpm ? static_cast<float>(hw.fanRpm) : 0.0f;

    snap.hasFanSpeedPct = hw.hasFanSpeed;
    snap.fanSpeedPct = hw.hasFanSpeed ? static_cast<float>(hw.fanSpeedPct) : 0.0f;

    snap.hasGpuBusy = hw.hasGpuActivity;
    snap.gpuBusyPct = hw.hasGpuActivity ? static_cast<float>(hw.gpuActivityPct) : 0.0f;

    snap.hasMemBusy = hw.hasMemActivity;
    snap.memBusyPct = hw.hasMemActivity ? static_cast<float>(hw.memActivityPct) : 0.0f;

    snap.hasVram = hasVram;
    snap.vramUsedBytes = hasVram ? vramUsed : 0;
    snap.vramTotalBytes = hasVram ? vramTotal : 0;

    snap.hasGtt = hw.hasGtt;
    snap.gttUsedBytes = hw.hasGtt ? hw.gttUsedBytes : 0;
    snap.gttTotalBytes = hw.hasGtt ? hw.gttTotalBytes : 0;

    // Record history during active benchmark run, OR maintain a live 60s rolling trace when idle
    // (preserving a completed benchmark run's full trace until reset or next run).
    bool pushSample = isRecordingRun || !m_hasCompletedRun.load();
    if (pushSample && snap.isAvailable) {
        snap.timeHistory.push(static_cast<float>(timestampSec));
        snap.sclkHistory.push(snap.sclkMhz);
        snap.mclkHistory.push(snap.mclkMhz);
        snap.powerHistory.push(snap.powerWatts);
        snap.tempEdgeHistory.push(snap.tempEdgeC);
        snap.tempJctHistory.push(snap.tempJctC);
        snap.tempMemHistory.push(snap.tempMemC);
        snap.vramUsedMbHistory.push(static_cast<float>(snap.vramUsedBytes / (1024ULL * 1024ULL)));
        snap.gpuBusyHistory.push(snap.gpuBusyPct);

        if (!isRecordingRun) {
            constexpr size_t kIdleRollingSamples = 600; // 60s at 10Hz
            snap.timeHistory.trimRolling(kIdleRollingSamples);
            snap.sclkHistory.trimRolling(kIdleRollingSamples);
            snap.mclkHistory.trimRolling(kIdleRollingSamples);
            snap.powerHistory.trimRolling(kIdleRollingSamples);
            snap.tempEdgeHistory.trimRolling(kIdleRollingSamples);
            snap.tempJctHistory.trimRolling(kIdleRollingSamples);
            snap.tempMemHistory.trimRolling(kIdleRollingSamples);
            snap.vramUsedMbHistory.trimRolling(kIdleRollingSamples);
            snap.gpuBusyHistory.trimRolling(kIdleRollingSamples);
        }
    }
}

void TelemetryWorker::workerLoop() {
    while (m_running.load()) {
        bool recording = m_isRecording.load();
        auto now = std::chrono::steady_clock::now();
        double timestamp = recording
            ? std::chrono::duration<double>(now - m_runStartTime).count()
            : std::chrono::duration<double>(now - m_idleStartTime).count();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (size_t i = 0; i < m_pollStates.size() && i < m_snapshots.size(); ++i) {
                pollDevice(m_pollStates[i], m_snapshots[i], timestamp, recording);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10Hz sampling
    }
}

} // namespace gpubench::gui
