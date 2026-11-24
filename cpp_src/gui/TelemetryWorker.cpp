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

namespace {

void clearHistoryBuffers(DeviceTelemetrySnapshot& snap) {
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

void pushHistoryPoint(DeviceTelemetrySnapshot& snap, float timestampSec) {
    if (!snap.isAvailable) {
        return;
    }
    snap.timeHistory.push(timestampSec);
    snap.sclkHistory.push(snap.sclkMhz);
    snap.mclkHistory.push(snap.mclkMhz);
    snap.powerHistory.push(snap.powerWatts);
    snap.tempEdgeHistory.push(snap.tempEdgeC);
    snap.tempJctHistory.push(snap.tempJctC);
    snap.tempMemHistory.push(snap.tempMemC);
    snap.vramUsedMbHistory.push(static_cast<float>(snap.vramUsedBytes / (1024ULL * 1024ULL)));
    snap.gpuBusyHistory.push(snap.gpuBusyPct);
}

} // namespace

TelemetryWorker::TelemetryWorker() {
    m_startTime = std::chrono::steady_clock::now();
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
        pollState.pciBusId = pciBus;
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
        GpuTelemetryData hw = m_pollStates[i].pciBusId.empty()
            ? HardwareTelemetry::queryGpu(m_pollStates[i].deviceIndex)
            : HardwareTelemetry::queryGpuByPciBusId(m_pollStates[i].pciBusId, m_pollStates[i].deviceIndex);
        applySample(m_pollStates[i], m_snapshots[i], hw, now, 0.0, false);
    }
    m_historyGeneration++;
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
        clearHistoryBuffers(snap);
        pushHistoryPoint(snap, 0.0f);
    }
    m_historyGeneration++;
    m_isRecording.store(true);
    notifyGraphThread();
}

void TelemetryWorker::stopRecording() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_isRecording.load()) {
        auto now = std::chrono::steady_clock::now();
        m_runDuration = std::chrono::duration<double>(now - m_runStartTime).count();
        const float endSec = static_cast<float>(m_runDuration);
        for (auto& snap : m_snapshots) {
            if (snap.timeHistory.empty() || snap.timeHistory.back() < endSec) {
                pushHistoryPoint(snap, endSec);
            }
        }
        m_historyGeneration++;
        m_hasCompletedRun.store(true);
    }
    m_isRecording.store(false);
    notifyGraphThread();
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
    if (m_isRecording.load()) {
        m_runStartTime = std::chrono::steady_clock::now();
    }
    m_runDuration = 0.0;
    m_hasCompletedRun.store(false);
    for (auto& snap : m_snapshots) {
        clearHistoryBuffers(snap);
    }
    m_historyGeneration++;
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

    // Update EMA smoothing for UI display (alpha = 0.25 at 10Hz ~350ms window)
    if (snap.hasSclk) {
        if (snap.sclkSmoothMhz <= 1.0f) {
            snap.sclkSmoothMhz = snap.sclkMhz;
        } else {
            constexpr float kAlpha = 0.25f;
            snap.sclkSmoothMhz = snap.sclkSmoothMhz + kAlpha * (snap.sclkMhz - snap.sclkSmoothMhz);
        }
    } else {
        snap.sclkSmoothMhz = 0.0f;
    }

    if (snap.hasPower) {
        if (snap.powerSmoothWatts <= 0.1f) {
            snap.powerSmoothWatts = snap.powerWatts;
        } else {
            constexpr float kAlpha = 0.25f;
            snap.powerSmoothWatts = snap.powerSmoothWatts + kAlpha * (snap.powerWatts - snap.powerSmoothWatts);
        }
    } else {
        snap.powerSmoothWatts = 0.0f;
    }

    if (snap.hasGpuBusy) {
        if (snap.gpuBusySmoothPct <= 0.1f && snap.gpuBusyPct <= 0.1f) {
            snap.gpuBusySmoothPct = snap.gpuBusyPct;
        } else {
            constexpr float kAlpha = 0.25f;
            snap.gpuBusySmoothPct = snap.gpuBusySmoothPct + kAlpha * (snap.gpuBusyPct - snap.gpuBusySmoothPct);
        }
    } else {
        snap.gpuBusySmoothPct = 0.0f;
    }

    // Record lower-graph history only while an active benchmark run is executing.
    // Instantaneous sensor readouts above continue updating even when idle.
    if (isRecordingRun) {
        pushHistoryPoint(snap, static_cast<float>(timestampSec));
        m_historyGeneration++;
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

    constexpr size_t kMaxScopePoints = 256;
    constexpr size_t kMaxImPlotPoints = 512;
    std::vector<float> scratchStats;
    uint64_t lastBuiltHistoryGen = 0;
    std::vector<std::shared_ptr<const PreparedGraphData>> prevBuilt;

    struct IdleScalarUpdate {
        uint32_t deviceIndex{0};
        float sclkMhz{0.0f};
        float mclkMhz{0.0f};
        float powerWatts{0.0f};
        float tempEdgeC{0.0f};
        float tempJctC{0.0f};
        float tempMemC{0.0f};
        uint64_t vramUsedBytes{0};
        uint64_t vramTotalBytes{0};
        float gpuBusyPct{0.0f};
        bool hasSclk{false};
        bool hasMclk{false};
        bool hasPower{false};
        bool hasTempEdge{false};
        bool hasTempJct{false};
        bool hasTempMem{false};
        bool hasVram{false};
        bool hasGpuBusy{false};
    };

    auto computeChannelStats = [&](const TelemetryBuffer& buf, size_t nPts,
                                   float minFloor, float roundStep,
                                   float& outMin, float& outMax, float& outCeiling) {
        outMin = 0.0f;
        outMax = 0.0f;
        outCeiling = minFloor;
        if (nPts == 0 || buf.size() != nPts) return;

        const float* d = buf.data();
        scratchStats.clear();
        scratchStats.reserve(nPts);
        bool any = false;
        for (size_t i = 0; i < nPts; ++i) {
            const float v = d[i];
            if (!std::isfinite(v)) continue;
            if (!any) {
                outMin = v;
                outMax = v;
                any = true;
            } else {
                outMin = std::min(outMin, v);
                outMax = std::max(outMax, v);
            }
            if (v > 0.0f) {
                scratchStats.push_back(v);
            }
        }
        if (scratchStats.empty()) return;

        float refPeak = outMax;
        if (scratchStats.size() >= 20) {
            const size_t p99Idx = (scratchStats.size() * 99) / 100;
            const size_t clampedIdx = std::min(p99Idx, scratchStats.size() - 1);
            std::nth_element(scratchStats.begin(),
                             scratchStats.begin() + static_cast<std::ptrdiff_t>(clampedIdx),
                             scratchStats.end());
            refPeak = scratchStats[clampedIdx];
        }
        const float target = std::max(minFloor, refPeak * 1.06f);
        outCeiling = (roundStep > 0.0f)
                         ? std::ceil(target / roundStep) * roundStep
                         : target;
    };

    while (m_running.load()) {
        {
            std::unique_lock<std::mutex> lk(m_graphMutex);
            m_graphCv.wait(lk, [this]() { return !m_running.load() || m_graphWorkPending; });
            if (!m_running.load()) {
                break;
            }
            m_graphWorkPending = false;
        }

        uint64_t gen = 0;
        {
            std::lock_guard<std::mutex> lock(m_graphMutex);
            gen = m_sampleGeneration;
        }

        std::vector<DeviceTelemetrySnapshot> localSnapshots;
        std::vector<IdleScalarUpdate> idleScalars;
        uint64_t curHistoryGen = 0;
        bool historyUnchanged = false;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            curHistoryGen = m_historyGeneration;
            historyUnchanged = (curHistoryGen == lastBuiltHistoryGen) &&
                               !prevBuilt.empty() &&
                               (prevBuilt.size() == m_snapshots.size());
            if (historyUnchanged) {
                idleScalars.reserve(m_snapshots.size());
                for (const auto& snap : m_snapshots) {
                    IdleScalarUpdate u;
                    u.deviceIndex = snap.deviceIndex;
                    u.sclkMhz = snap.sclkMhz;
                    u.mclkMhz = snap.mclkMhz;
                    u.powerWatts = snap.powerWatts;
                    u.tempEdgeC = snap.tempEdgeC;
                    u.tempJctC = snap.tempJctC;
                    u.tempMemC = snap.tempMemC;
                    u.vramUsedBytes = snap.vramUsedBytes;
                    u.vramTotalBytes = snap.vramTotalBytes;
                    u.gpuBusyPct = snap.gpuBusyPct;
                    u.hasSclk = snap.hasSclk;
                    u.hasMclk = snap.hasMclk;
                    u.hasPower = snap.hasPower;
                    u.hasTempEdge = snap.hasTempEdge;
                    u.hasTempJct = snap.hasTempJct;
                    u.hasTempMem = snap.hasTempMem;
                    u.hasVram = snap.hasVram;
                    u.hasGpuBusy = snap.hasGpuBusy;
                    idleScalars.push_back(u);
                }
            } else {
                localSnapshots = m_snapshots;
            }
        }

        std::vector<std::shared_ptr<const PreparedGraphData>> built;
        if (historyUnchanged) {
            built.reserve(idleScalars.size());
            for (size_t i = 0; i < idleScalars.size(); ++i) {
                auto g = std::make_shared<PreparedGraphData>(*prevBuilt[i]);
                const IdleScalarUpdate& u = idleScalars[i];
                g->deviceIndex = u.deviceIndex;
                g->generation = gen;
                g->sclkMhz = u.sclkMhz;
                g->mclkMhz = u.mclkMhz;
                g->powerWatts = u.powerWatts;
                g->tempEdgeC = u.tempEdgeC;
                g->tempJctC = u.tempJctC;
                g->tempMemC = u.tempMemC;
                g->vramUsedBytes = u.vramUsedBytes;
                g->vramTotalBytes = u.vramTotalBytes;
                g->gpuBusyPct = u.gpuBusyPct;
                g->hasSclk = u.hasSclk;
                g->hasMclk = u.hasMclk;
                g->hasPower = u.hasPower;
                g->hasTempEdge = u.hasTempEdge;
                g->hasTempJct = u.hasTempJct;
                g->hasTempMem = u.hasTempMem;
                g->hasVram = u.hasVram;
                g->hasGpuBusy = u.hasGpuBusy;
                g->vramUsedGb = static_cast<double>(u.vramUsedBytes) / (1024.0 * 1024.0 * 1024.0);
                g->vramTotGb = std::max(0.1, static_cast<double>(u.vramTotalBytes) / (1024.0 * 1024.0 * 1024.0));
                if (u.hasSclk && u.sclkMhz > 0.0f) {
                    const float liveSclkCeil = std::ceil((u.sclkMhz * 1.10f) / 250.0f) * 250.0f;
                    g->sclkCeiling = std::max(g->sclkCeiling, liveSclkCeil);
                    g->clockCeilingPlot = std::max(g->clockCeilingPlot, g->sclkCeiling);
                }
                if (u.hasPower && u.powerWatts > 0.0f) {
                    const float livePwrCeil = std::ceil((u.powerWatts * 1.15f) / 50.0f) * 50.0f;
                    g->pwrCeiling = std::max(g->pwrCeiling, livePwrCeil);
                }
                built.push_back(std::move(g));
            }
            prevBuilt = built;
            std::lock_guard<std::mutex> lock(m_graphMutex);
            m_preparedGraphs.swap(built);
            continue;
        }
        lastBuiltHistoryGen = curHistoryGen;

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

            if (snap.hasSclk) {
                computeChannelStats(snap.sclkHistory, nPts, 1000.0f, 250.0f,
                                    g->sclkWinMin, g->sclkWinMax, g->sclkCeiling);
                if (snap.sclkMhz > 0.0f) {
                    const float liveSclkCeil = std::ceil((snap.sclkMhz * 1.10f) / 250.0f) * 250.0f;
                    g->sclkCeiling = std::max(g->sclkCeiling, liveSclkCeil);
                }
            } else {
                g->sclkCeiling = 1000.0f;
            }
            float mclkMin = 0.0f, mclkMax = 0.0f, mclkCeiling = 1000.0f;
            if (snap.hasMclk) {
                computeChannelStats(snap.mclkHistory, nPts, 1000.0f, 250.0f,
                                    mclkMin, mclkMax, mclkCeiling);
            }
            g->clockCeilingPlot = std::max(g->sclkCeiling, snap.hasMclk ? mclkCeiling : 1000.0f);

            if (snap.hasPower) {
                computeChannelStats(snap.powerHistory, nPts, 100.0f, 50.0f,
                                    g->pwrWinMin, g->pwrWinMax, g->pwrCeiling);
                if (snap.powerWatts > 0.0f) {
                    const float livePwrCeil = std::ceil((snap.powerWatts * 1.15f) / 50.0f) * 50.0f;
                    g->pwrCeiling = std::max(g->pwrCeiling, livePwrCeil);
                }
            } else {
                g->pwrCeiling = 100.0f;
            }

            if (snap.hasGpuBusy) {
                float unusedCeil = 105.0f;
                computeChannelStats(snap.gpuBusyHistory, nPts, 100.0f, 10.0f,
                                    g->busyWinMin, g->busyWinMax, unusedCeil);
            }

            float tempFloor = 20.0f;
            float tempCeil = 105.0f;
            const TelemetryBuffer& activeTempHist = snap.hasTempJct ? snap.tempJctHistory : snap.tempEdgeHistory;
            if ((snap.hasTempJct || snap.hasTempEdge) && activeTempHist.size() == nPts && nPts > 0) {
                float rawTempCeil = 105.0f;
                computeChannelStats(activeTempHist, nPts, 105.0f, 5.0f,
                                    g->tempWinMin, g->tempWinMax, rawTempCeil);
                if (g->tempWinMin > 0.0f && g->tempWinMin < 20.0f) {
                    tempFloor = std::max(0.0f, std::floor(g->tempWinMin / 5.0f) * 5.0f);
                }
                tempCeil = std::max(105.0f, rawTempCeil);
            }
            g->tempMinScale = tempFloor;
            g->tempMaxScale = tempCeil;

            g->vramUsedGb = static_cast<double>(snap.vramUsedBytes) / (1024.0 * 1024.0 * 1024.0);
            g->vramTotGb = std::max(0.1, static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0 * 1024.0));
            g->maxVramMbPlot = (snap.vramTotalBytes > 0)
                ? static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0)
                : 16384.0;
            if (snap.hasVram && snap.vramUsedMbHistory.size() == nPts && nPts > 0) {
                float vMinMb = 0.0f, vMaxMb = 0.0f, vCeil = 0.0f;
                computeChannelStats(snap.vramUsedMbHistory, nPts, 1024.0f, 256.0f,
                                    vMinMb, vMaxMb, vCeil);
                g->vramWinMinGb = vMinMb / 1024.0f;
                g->vramWinMaxGb = vMaxMb / 1024.0f;
            }

            if (nPts > 0) {
                std::vector<size_t> indices;
                if (nPts <= kMaxImPlotPoints) {
                    indices.resize(nPts);
                    for (size_t i = 0; i < nPts; ++i) indices[i] = i;
                } else {
                    indices.reserve(kMaxImPlotPoints);
                    const double step = static_cast<double>(nPts - 1) / static_cast<double>(kMaxImPlotPoints - 1);
                    for (size_t k = 0; k < kMaxImPlotPoints - 1; ++k) {
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

                // Peak-and-trough preserving bucket envelope for crisp oscilloscope traces
                auto buildNormEnvelope = [&](const TelemetryBuffer& yBuf, float yMin, float yMax,
                                             std::vector<NormalizedGraphPoint>& outNorm) {
                    if (yBuf.size() != nPts || nPts < 2 || !tData) return;
                    const float* ys = yBuf.data();
                    const float denom = std::max(0.001f, yMax - yMin);
                    auto emitPt = [&](size_t idx) {
                        float tx = static_cast<float>((static_cast<double>(tData[idx]) - g->tMin) / g->tSpan);
                        float ty = std::clamp((ys[idx] - yMin) / denom, 0.0f, 1.0f);
                        outNorm.push_back(NormalizedGraphPoint{std::clamp(tx, 0.0f, 1.0f), ty});
                    };

                    if (nPts <= kMaxScopePoints) {
                        outNorm.reserve(nPts);
                        for (size_t i = 0; i < nPts; ++i) emitPt(i);
                        return;
                    }

                    const size_t bucketCount = kMaxScopePoints / 2;
                    outNorm.reserve(kMaxScopePoints);
                    size_t lastEmitted = nPts;
                    for (size_t b = 0; b < bucketCount; ++b) {
                        const size_t bStart = (b * nPts) / bucketCount;
                        const size_t bEnd = std::min(nPts, ((b + 1) * nPts) / bucketCount);
                        if (bStart >= bEnd) continue;
                        size_t minIdx = bStart;
                        size_t maxIdx = bStart;
                        for (size_t i = bStart + 1; i < bEnd; ++i) {
                            if (ys[i] < ys[minIdx]) minIdx = i;
                            if (ys[i] > ys[maxIdx]) maxIdx = i;
                        }
                        if (b == 0 && minIdx != 0 && maxIdx != 0) {
                            emitPt(0);
                            lastEmitted = 0;
                        }
                        if (minIdx == maxIdx) {
                            if (minIdx != lastEmitted) {
                                emitPt(minIdx);
                                lastEmitted = minIdx;
                            }
                        } else if (minIdx < maxIdx) {
                            if (minIdx != lastEmitted) emitPt(minIdx);
                            emitPt(maxIdx);
                            lastEmitted = maxIdx;
                        } else {
                            if (maxIdx != lastEmitted) emitPt(maxIdx);
                            emitPt(minIdx);
                            lastEmitted = minIdx;
                        }
                    }
                    if (lastEmitted != nPts - 1) {
                        emitPt(nPts - 1);
                    }
                };

                if (snap.hasSclk) buildNormEnvelope(snap.sclkHistory, 0.0f, g->sclkCeiling, g->sclkNorm);
                if (snap.hasPower) buildNormEnvelope(snap.powerHistory, 0.0f, g->pwrCeiling, g->pwrNorm);
                if (snap.hasGpuBusy) buildNormEnvelope(snap.gpuBusyHistory, 0.0f, 100.0f, g->busyNorm);
                if (snap.hasTempJct) {
                    buildNormEnvelope(snap.tempJctHistory, g->tempMinScale, g->tempMaxScale, g->tempNorm);
                } else if (snap.hasTempEdge) {
                    buildNormEnvelope(snap.tempEdgeHistory, g->tempMinScale, g->tempMaxScale, g->tempNorm);
                }
                if (snap.hasVram) {
                    const float maxVramMb = std::max(1024.0f, static_cast<float>(g->vramTotGb * 1024.0));
                    buildNormEnvelope(snap.vramUsedMbHistory, 0.0f, maxVramMb, g->vramNorm);
                }
            }

            built.push_back(std::move(g));
        }

        prevBuilt = built;
        {
            std::lock_guard<std::mutex> lock(m_graphMutex);
            m_preparedGraphs.swap(built);
        }
    }
}

void TelemetryWorker::workerLoop() {
    setLowThreadPriority();
    while (m_running.load()) {
        struct PollTarget {
            uint32_t deviceIndex{0};
            std::string pciBusId;
        };
        std::vector<PollTarget> targets;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            targets.reserve(m_pollStates.size());
            for (const auto& st : m_pollStates) {
                targets.push_back({st.deviceIndex, st.pciBusId});
            }
        }

        struct RawSample {
            uint32_t deviceIndex{0};
            std::chrono::steady_clock::time_point sampleTime;
            GpuTelemetryData hw;
        };
        std::vector<RawSample> rawSamples;
        rawSamples.reserve(targets.size());
        for (const auto& tgt : targets) {
            RawSample sample;
            sample.deviceIndex = tgt.deviceIndex;
            sample.sampleTime = std::chrono::steady_clock::now();
            sample.hw = tgt.pciBusId.empty()
                ? HardwareTelemetry::queryGpu(tgt.deviceIndex)
                : HardwareTelemetry::queryGpuByPciBusId(tgt.pciBusId, tgt.deviceIndex);
            rawSamples.push_back(std::move(sample));
        }

        const auto applyNow = std::chrono::steady_clock::now();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const bool recording = m_isRecording.load();
            const double timestamp = recording
                ? std::chrono::duration<double>(applyNow - m_runStartTime).count()
                : 0.0;
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
