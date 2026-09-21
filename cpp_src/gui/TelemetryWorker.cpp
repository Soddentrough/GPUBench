#include "TelemetryWorker.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

namespace gpubench::gui {

static float readSysfsFloat(const std::string& path, float fallback = 0.0f, float scale = 1.0f) {
    if (path.empty()) return fallback;
    std::ifstream file(path);
    if (!file.is_open()) return fallback;
    float val = 0.0f;
    if (file >> val) {
        return val * scale;
    }
    return fallback;
}

static uint64_t readSysfsUint64(const std::string& path, uint64_t fallback = 0) {
    if (path.empty()) return fallback;
    std::ifstream file(path);
    if (!file.is_open()) return fallback;
    uint64_t val = 0;
    if (file >> val) {
        return val;
    }
    return fallback;
}

static std::string readSysfsString(const std::string& path, const std::string& fallback = "") {
    if (path.empty()) return fallback;
    std::ifstream file(path);
    if (!file.is_open()) return fallback;
    std::string line;
    if (std::getline(file, line)) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
            line.pop_back();
        }
        return line;
    }
    return fallback;
}

TelemetryWorker::TelemetryWorker() {
    m_startTime = std::chrono::steady_clock::now();
    discoverDevices();
}

TelemetryWorker::~TelemetryWorker() {
    stop();
}

void TelemetryWorker::discoverDevices() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_devicePaths.clear();
    m_snapshots.clear();

    const std::string drmRoot = "/sys/class/drm";
    if (!fs::exists(drmRoot)) return;

    for (const auto& entry : fs::directory_iterator(drmRoot)) {
        std::string filename = entry.path().filename().string();
        // Match only card0, card1, etc. - skip card0-DP-1, etc.
        if (filename.rfind("card", 0) == 0 && filename.find('-') == std::string::npos) {
            std::string cardIdxStr = filename.substr(4);
            try {
                uint32_t cardIdx = std::stoul(cardIdxStr);
                DeviceSysfsPaths paths;
                paths.deviceIndex = cardIdx;
                paths.cardName = filename;
                paths.drmDeviceDir = entry.path().string() + "/device";

                // Find hwmon dir
                std::string hwmonBase = paths.drmDeviceDir + "/hwmon";
                if (fs::exists(hwmonBase)) {
                    for (const auto& hEntry : fs::directory_iterator(hwmonBase)) {
                        std::string hName = hEntry.path().filename().string();
                        if (hName.rfind("hwmon", 0) == 0) {
                            paths.hwmonDir = hEntry.path().string();
                            break;
                        }
                    }
                }

                DeviceTelemetrySnapshot snap;
                snap.deviceIndex = cardIdx;
                std::string productName = readSysfsString(paths.drmDeviceDir + "/product_name");
                if (productName.empty()) {
                    productName = readSysfsString(paths.hwmonDir + "/name", "AMD Radeon GPU");
                }
                snap.name = "GPU " + std::to_string(cardIdx) + ": " + productName;
                snap.pciBus = readSysfsString(paths.drmDeviceDir + "/uevent");

                m_devicePaths.push_back(paths);
                m_snapshots.push_back(snap);
            } catch (...) {
                continue;
            }
        }
    }

    // Sort paths by canonical PCI path to match Vulkan and ROCm device enumeration
    // (PCI 0000:23:00.0 is GPU 0, PCI 0000:4d:00.0 is GPU 1)
    std::sort(m_devicePaths.begin(), m_devicePaths.end(), [](const DeviceSysfsPaths& a, const DeviceSysfsPaths& b) {
        std::string pciA = fs::exists(a.drmDeviceDir) ? fs::canonical(a.drmDeviceDir).string() : a.drmDeviceDir;
        std::string pciB = fs::exists(b.drmDeviceDir) ? fs::canonical(b.drmDeviceDir).string() : b.drmDeviceDir;
        return pciA < pciB;
    });

    // Rebuild snapshots with matching deviceIndex (0, 1, ...)
    m_snapshots.clear();
    for (size_t i = 0; i < m_devicePaths.size(); ++i) {
        m_devicePaths[i].deviceIndex = static_cast<uint32_t>(i);
        DeviceTelemetrySnapshot snap;
        snap.deviceIndex = static_cast<uint32_t>(i);
        std::string productName = readSysfsString(m_devicePaths[i].drmDeviceDir + "/product_name");
        if (productName.empty()) {
            productName = readSysfsString(m_devicePaths[i].hwmonDir + "/name", "AMD Radeon AI PRO R9700");
        }
        snap.name = "GPU " + std::to_string(i) + ": " + productName;
        try {
            snap.pciBus = fs::canonical(m_devicePaths[i].drmDeviceDir).filename().string();
        } catch (...) {
            snap.pciBus = readSysfsString(m_devicePaths[i].drmDeviceDir + "/uevent");
        }
        m_snapshots.push_back(snap);
    }
}

void TelemetryWorker::start() {
    if (m_running.load()) return;
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
    for (const auto& snap : m_snapshots) {
        names.push_back(snap.name);
    }
    return names;
}

void TelemetryWorker::pollDevice(const DeviceSysfsPaths& paths, DeviceTelemetrySnapshot& snap, double timestampSec) {
    // 1. Clocks: freq1_input is in Hz for AMDGPU hwmon -> divide by 1e6 for MHz
    float sclk = readSysfsFloat(paths.hwmonDir + "/freq1_input", 0.0f, 1e-6f);
    if (sclk <= 0.0f) {
        sclk = readSysfsFloat(paths.drmDeviceDir + "/current_gfxclk", 0.0f);
    }

    float mclk = readSysfsFloat(paths.hwmonDir + "/freq2_input", 0.0f, 1e-6f);
    if (mclk <= 0.0f) {
        mclk = readSysfsFloat(paths.drmDeviceDir + "/current_uclk", 0.0f);
    }

    // 2. Power: power1_average or power1_input in microwatts -> divide by 1e6 for Watts
    float power = readSysfsFloat(paths.hwmonDir + "/power1_average", 0.0f, 1e-6f);
    if (power <= 0.0f) {
        power = readSysfsFloat(paths.hwmonDir + "/power1_input", 0.0f, 1e-6f);
    }

    // 3. Temperatures: millidegrees Celsius -> divide by 1000.0f
    float tempEdge = readSysfsFloat(paths.hwmonDir + "/temp1_input", 0.0f, 0.001f);
    float tempJct = readSysfsFloat(paths.hwmonDir + "/temp2_input", 0.0f, 0.001f);
    float tempMem = readSysfsFloat(paths.hwmonDir + "/temp3_input", 0.0f, 0.001f);

    // 4. Fan RPM
    float fan = readSysfsFloat(paths.hwmonDir + "/fan1_input", 0.0f);

    // 5. Utilization
    float gpuBusy = readSysfsFloat(paths.drmDeviceDir + "/gpu_busy_percent", 0.0f);
    float memBusy = readSysfsFloat(paths.drmDeviceDir + "/mem_busy_percent", 0.0f);

    // 6. VRAM
    uint64_t vramUsed = readSysfsUint64(paths.drmDeviceDir + "/mem_info_vram_used", 0);
    uint64_t vramTotal = readSysfsUint64(paths.drmDeviceDir + "/mem_info_vram_total", 0);

    // Update current snapshot scalars
    snap.sclkMhz = sclk;
    snap.mclkMhz = mclk;
    snap.powerWatts = power;
    snap.tempEdgeC = tempEdge;
    snap.tempJctC = tempJct;
    snap.tempMemC = tempMem;
    snap.fanRpm = fan;
    snap.gpuBusyPct = gpuBusy;
    snap.memBusyPct = memBusy;
    snap.vramUsedBytes = vramUsed;
    snap.vramTotalBytes = vramTotal;

    // Push into circular history
    snap.timeHistory.push(static_cast<float>(timestampSec));
    snap.sclkHistory.push(sclk);
    snap.mclkHistory.push(mclk);
    snap.powerHistory.push(power);
    snap.tempEdgeHistory.push(tempEdge);
    snap.tempJctHistory.push(tempJct);
    snap.tempMemHistory.push(tempMem);
    snap.vramUsedMbHistory.push(static_cast<float>(vramUsed / (1024 * 1024)));
    snap.gpuBusyHistory.push(gpuBusy);
}

void TelemetryWorker::workerLoop() {
    while (m_running.load()) {
        auto now = std::chrono::steady_clock::now();
        double timestamp = std::chrono::duration<double>(now - m_startTime).count();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (size_t i = 0; i < m_devicePaths.size() && i < m_snapshots.size(); ++i) {
                pollDevice(m_devicePaths[i], m_snapshots[i], timestamp);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10Hz sampling
    }
}

} // namespace gpubench::gui
