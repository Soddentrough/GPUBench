#include "TelemetryWorker.h"
#include "core/RunnerAPI.h"
#ifdef HAVE_VULKAN
#include "core/VulkanContext.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#endif

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
        auto now = std::chrono::steady_clock::now();
        GpuTelemetryData hw = HardwareTelemetry::queryGpu(m_pollStates[i].deviceIndex);
        applySample(m_pollStates[i], m_snapshots[i], hw, now, 0.0, false);
    }
}

void TelemetryWorker::start() {
    if (m_running.load()) return;
    if (m_snapshots.empty()) {
        discoverDevices();
    }
    m_running.store(true);
    m_graphWorkPending = true;
    m_workerThread = std::thread(&TelemetryWorker::workerLoop, this);
    m_graphThread = std::thread(&TelemetryWorker::graphPrepLoop, this);
}

void TelemetryWorker::stop() {
    if (!m_running.load()) return;
    m_running.store(false);
    {
        std::lock_guard<std::mutex> lock(m_graphMutex);
        m_graphWorkPending = true;
    }
    m_graphCv.notify_all();
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    if (m_graphThread.joinable()) {
        m_graphThread.join();
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
    notifyGraphThread();
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
    notifyGraphThread();
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

bool TelemetryWorker::getPreparedGraphs(uint32_t deviceIndex,
                                        std::shared_ptr<const PreparedGraphData>& outData) const {
    std::lock_guard<std::mutex> lock(m_graphMutex);
    for (const auto& ptr : m_preparedGraphs) {
        if (ptr && ptr->deviceIndex == deviceIndex) {
            outData = ptr;
            return true;
        }
    }
    if (!m_preparedGraphs.empty() && m_preparedGraphs.front()) {
        outData = m_preparedGraphs.front();
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

void TelemetryWorker::applySample(DevicePollState& state, DeviceTelemetrySnapshot& snap,
                                  const GpuTelemetryData& hw,
                                  std::chrono::steady_clock::time_point now,
                                  double timestampSec, bool isRecordingRun) {

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

void TelemetryWorker::setLowThreadPriority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#elif defined(__linux__)
    // Lower thread scheduling priority so telemetry & graph prep never contend with benchmark timing
    struct sched_param sp{};
    sp.sched_priority = 0;
#ifdef SCHED_IDLE
    if (pthread_setschedparam(pthread_self(), SCHED_IDLE, &sp) != 0) {
#ifdef SCHED_BATCH
        pthread_setschedparam(pthread_self(), SCHED_BATCH, &sp);
#endif
        setpriority(PRIO_PROCESS, 0, 19);
    }
#else
    setpriority(PRIO_PROCESS, 0, 19);
#endif
#endif
}

void TelemetryWorker::notifyGraphThread() {
    {
        std::lock_guard<std::mutex> lock(m_graphMutex);
        m_sampleGeneration++;
        m_graphWorkPending = true;
    }
    m_graphCv.notify_one();
}

void TelemetryWorker::graphPrepLoop() {
    setLowThreadPriority();

    constexpr size_t kMaxPlotPoints = 256;

    while (m_running.load()) {
        {
            std::unique_lock<std::mutex> lk(m_graphMutex);
            m_graphCv.wait(lk, [this]() { return !m_running.load() || m_graphWorkPending; });
            if (!m_running.load()) {
                break;
            }
            m_graphWorkPending = false;
        }

        std::vector<DeviceTelemetrySnapshot> localSnapshots;
        uint64_t gen = 0;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            localSnapshots = m_snapshots;
        }
        {
            std::lock_guard<std::mutex> lock(m_graphMutex);
            gen = m_sampleGeneration;
        }

        std::vector<std::shared_ptr<const PreparedGraphData>> built;
        built.reserve(localSnapshots.size());

        for (const auto& snap : localSnapshots) {
            auto g = std::make_shared<PreparedGraphData>();
            g->deviceIndex = snap.deviceIndex;
            g->generation = gen;
            g->sclkMhz = snap.sclkMhz;
            g->mclkMhz = snap.mclkMhz;
            g->powerWatts = snap.powerWatts;
            g->tempEdgeC = snap.tempEdgeC;
            g->tempJctC = snap.tempJctC;
            g->tempMemC = snap.tempMemC;
            g->vramUsedBytes = snap.vramUsedBytes;
            g->vramTotalBytes = snap.vramTotalBytes;
            g->gpuBusyPct = snap.gpuBusyPct;

            g->hasSclk = snap.hasSclk;
            g->hasMclk = snap.hasMclk;
            g->hasPower = snap.hasPower;
            g->hasTempEdge = snap.hasTempEdge;
            g->hasTempJct = snap.hasTempJct;
            g->hasTempMem = snap.hasTempMem;
            g->hasVram = snap.hasVram;
            g->hasGpuBusy = snap.hasGpuBusy;

            const size_t nPts = snap.timeHistory.size();
            const float* tData = (nPts > 0) ? snap.timeHistory.data() : nullptr;
            g->tMin = (nPts >= 2) ? static_cast<double>(tData[0]) : 0.0;
            g->tMax = (nPts >= 2) ? std::max(g->tMin + 5.0, static_cast<double>(tData[nPts - 1])) : 60.0;
            g->tSpan = std::max(0.5, g->tMax - g->tMin);

            g->sclkCeiling = 3000.0f;
            if (snap.hasSclk && snap.sclkHistory.size() == nPts) {
                const float* d = snap.sclkHistory.data();
                for (size_t i = 0; i < nPts; ++i) g->sclkCeiling = std::max(g->sclkCeiling, d[i] * 1.15f);
            }
            g->pwrCeiling = 360.0f;
            if (snap.hasPower && snap.powerHistory.size() == nPts) {
                const float* d = snap.powerHistory.data();
                for (size_t i = 0; i < nPts; ++i) g->pwrCeiling = std::max(g->pwrCeiling, d[i] * 1.15f);
            }
            g->vramUsedGb = static_cast<double>(snap.vramUsedBytes) / (1024.0 * 1024.0 * 1024.0);
            g->vramTotGb = std::max(0.1, static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0 * 1024.0));
            g->maxVramMbPlot = (snap.vramTotalBytes > 0)
                ? static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0)
                : 16384.0;

            if (nPts > 0) {
                std::vector<size_t> indices;
                if (nPts <= kMaxPlotPoints) {
                    indices.resize(nPts);
                    for (size_t i = 0; i < nPts; ++i) indices[i] = i;
                } else {
                    indices.reserve(kMaxPlotPoints);
                    const double step = static_cast<double>(nPts - 1) / static_cast<double>(kMaxPlotPoints - 1);
                    for (size_t k = 0; k < kMaxPlotPoints - 1; ++k) {
                        indices.push_back(static_cast<size_t>(std::round(static_cast<double>(k) * step)));
                    }
                    indices.push_back(nPts - 1);
                }

                auto copyDecimated = [&](const TelemetryBuffer& src, std::vector<float>& dst) {
                    if (src.size() != nPts) return;
                    dst.reserve(indices.size());
                    const float* raw = src.data();
                    for (size_t idx : indices) dst.push_back(raw[idx]);
                };

                copyDecimated(snap.timeHistory, g->timePts);
                if (snap.hasSclk) copyDecimated(snap.sclkHistory, g->sclkPts);
                if (snap.hasMclk) copyDecimated(snap.mclkHistory, g->mclkPts);
                if (snap.hasPower) copyDecimated(snap.powerHistory, g->powerPts);
                if (snap.hasTempEdge) copyDecimated(snap.tempEdgeHistory, g->tempEdgePts);
                if (snap.hasTempJct) copyDecimated(snap.tempJctHistory, g->tempJctPts);
                if (snap.hasTempMem) copyDecimated(snap.tempMemHistory, g->tempMemPts);
                if (snap.hasVram) copyDecimated(snap.vramUsedMbHistory, g->vramUsedMbPts);
                if (snap.hasGpuBusy) copyDecimated(snap.gpuBusyHistory, g->gpuBusyPts);

                auto buildNorm = [&](const std::vector<float>& ys, float yMin, float yMax,
                                     std::vector<NormalizedGraphPoint>& outNorm) {
                    if (ys.size() != g->timePts.size() || ys.size() < 2) return;
                    outNorm.reserve(ys.size());
                    const float denom = std::max(0.001f, yMax - yMin);
                    for (size_t i = 0; i < ys.size(); ++i) {
                        float tx = static_cast<float>((static_cast<double>(g->timePts[i]) - g->tMin) / g->tSpan);
                        float ty = std::clamp((ys[i] - yMin) / denom, 0.0f, 1.0f);
                        outNorm.push_back(NormalizedGraphPoint{tx, ty});
                    }
                };

                buildNorm(g->sclkPts, 0.0f, g->sclkCeiling, g->sclkNorm);
                buildNorm(g->powerPts, 0.0f, g->pwrCeiling, g->pwrNorm);
                buildNorm(g->gpuBusyPts, 0.0f, 105.0f, g->busyNorm);
                if (snap.hasTempJct) {
                    buildNorm(g->tempJctPts, 20.0f, 105.0f, g->tempNorm);
                } else if (snap.hasTempEdge) {
                    buildNorm(g->tempEdgePts, 20.0f, 105.0f, g->tempNorm);
                }
                const float maxVramMb = std::max(1024.0f, static_cast<float>(g->vramTotGb * 1024.0));
                buildNorm(g->vramUsedMbPts, 0.0f, maxVramMb, g->vramNorm);
            }

            built.push_back(std::move(g));
        }

        {
            std::lock_guard<std::mutex> lock(m_graphMutex);
            m_preparedGraphs.swap(built);
        }
    }
}

void TelemetryWorker::workerLoop() {
    setLowThreadPriority();
    while (m_running.load()) {
        std::vector<uint32_t> deviceIndices;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            deviceIndices.reserve(m_pollStates.size());
            for (const auto& st : m_pollStates) {
                deviceIndices.push_back(st.deviceIndex);
            }
        }

        struct RawSample {
            uint32_t deviceIndex{0};
            std::chrono::steady_clock::time_point sampleTime;
            GpuTelemetryData hw;
        };
        std::vector<RawSample> rawSamples;
        rawSamples.reserve(deviceIndices.size());
        for (uint32_t devIdx : deviceIndices) {
            RawSample sample;
            sample.deviceIndex = devIdx;
            sample.sampleTime = std::chrono::steady_clock::now();
            sample.hw = HardwareTelemetry::queryGpu(devIdx);
            rawSamples.push_back(std::move(sample));
        }

        const bool recording = m_isRecording.load();
        const auto applyNow = std::chrono::steady_clock::now();
        const double timestamp = recording
            ? std::chrono::duration<double>(applyNow - m_runStartTime).count()
            : std::chrono::duration<double>(applyNow - m_idleStartTime).count();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (size_t i = 0; i < m_pollStates.size() && i < m_snapshots.size(); ++i) {
                const uint32_t targetIdx = m_pollStates[i].deviceIndex;
                for (const auto& sample : rawSamples) {
                    if (sample.deviceIndex == targetIdx) {
                        applySample(m_pollStates[i], m_snapshots[i],
                                    sample.hw, sample.sampleTime,
                                    timestamp, recording);
                        break;
                    }
                }
            }
        }

        notifyGraphThread();
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10Hz sampling
    }
}

} // namespace gpubench::gui
