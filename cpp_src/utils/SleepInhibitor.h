#pragma once

#include <string>
#include <cstdint>
#ifdef __linux__
#include <sys/types.h>
#endif

namespace utils {

/**
 * RAII guard to suppress system idle sleep, display sleep, and screensaver
 * during GPU compute and ray tracing benchmark executions.
 */
class SleepInhibitor {
public:
    explicit SleepInhibitor(const std::string &reason = "GPUBench active computation");
    ~SleepInhibitor();

    SleepInhibitor(const SleepInhibitor &) = delete;
    SleepInhibitor &operator=(const SleepInhibitor &) = delete;
    SleepInhibitor(SleepInhibitor &&other) noexcept;
    SleepInhibitor &operator=(SleepInhibitor &&other) noexcept;

    bool isInhibited() const { return inhibited; }

private:
    void inhibit(const std::string &reason);
    void unInhibit();

    bool inhibited = false;
#ifdef _WIN32
    uint32_t previousState = 0;
#elif defined(__APPLE__)
    uint32_t assertionID = 0;
#elif defined(__linux__)
    pid_t inhibitorPid = -1;
    int inhibitorPipeFd = -1;
#endif
};

} // namespace utils
