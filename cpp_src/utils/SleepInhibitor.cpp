#include "SleepInhibitor.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <CoreFoundation/CoreFoundation.h>
#else
#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif

namespace utils {

SleepInhibitor::SleepInhibitor(const std::string &reason) {
    inhibit(reason);
}

SleepInhibitor::~SleepInhibitor() {
    unInhibit();
}

SleepInhibitor::SleepInhibitor(SleepInhibitor &&other) noexcept
    : inhibited(other.inhibited)
#ifdef _WIN32
    , previousState(other.previousState)
#elif defined(__APPLE__)
    , assertionID(other.assertionID)
#else
    , cookie(other.cookie)
#endif
{
    other.inhibited = false;
#ifdef _WIN32
    other.previousState = 0;
#elif defined(__APPLE__)
    other.assertionID = 0;
#else
    other.cookie = 0;
#endif
}

SleepInhibitor &SleepInhibitor::operator=(SleepInhibitor &&other) noexcept {
    if (this != &other) {
        unInhibit();
        inhibited = other.inhibited;
#ifdef _WIN32
        previousState = other.previousState;
        other.previousState = 0;
#elif defined(__APPLE__)
        assertionID = other.assertionID;
        other.assertionID = 0;
#else
        cookie = other.cookie;
        other.cookie = 0;
#endif
        other.inhibited = false;
    }
    return *this;
}

#ifdef _WIN32
void SleepInhibitor::inhibit(const std::string &/*reason*/) {
    EXECUTION_STATE prev = SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
    if (prev != 0) {
        previousState = static_cast<uint32_t>(prev);
        inhibited = true;
    }
}

void SleepInhibitor::unInhibit() {
    if (inhibited) {
        SetThreadExecutionState(ES_CONTINUOUS);
        inhibited = false;
        previousState = 0;
    }
}

#elif defined(__APPLE__)
void SleepInhibitor::inhibit(const std::string &reason) {
    CFStringRef cfReason = CFStringCreateWithCString(kCFAllocatorDefault, reason.c_str(), kCFStringEncodingUTF8);
    IOReturn res = IOPMAssertionCreateWithName(
        kIOPMAssertionTypePreventUserIdleSystemSleep,
        kIOPMAssertionLevelOn,
        cfReason ? cfReason : CFSTR("GPUBench computation"),
        &assertionID
    );
    if (cfReason) {
        CFRelease(cfReason);
    }
    if (res == kIOReturnSuccess) {
        inhibited = true;
    }
}

void SleepInhibitor::unInhibit() {
    if (inhibited && assertionID != 0) {
        IOPMAssertionRelease(assertionID);
        assertionID = 0;
        inhibited = false;
    }
}

#else // Linux / FreeDesktop
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <spawn.h>

extern char **environ;

void SleepInhibitor::inhibit(const std::string &reason) {
    (void)reason;
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return;
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);

    const char *argv[] = {
        "dbus-send",
        "--print-reply",
        "--dest=org.freedesktop.ScreenSaver",
        "/org/freedesktop/ScreenSaver",
        "org.freedesktop.ScreenSaver.Inhibit",
        "string:GPUBench",
        "string:Active GPU benchmark execution",
        nullptr
    };

    pid_t pid;
    int status = posix_spawnp(&pid, "dbus-send", &actions, nullptr, const_cast<char *const *>(argv), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);

    if (status == 0) {
        FILE *fp = fdopen(pipefd[0], "r");
        if (fp) {
            char buf[256];
            while (fgets(buf, sizeof(buf), fp)) {
                unsigned int c = 0;
                if (sscanf(buf, " uint32 %u", &c) == 1 || sscanf(buf, "uint32 %u", &c) == 1) {
                    cookie = c;
                    inhibited = true;
                    break;
                }
            }
            fclose(fp);
        } else {
            close(pipefd[0]);
        }
        waitpid(pid, nullptr, 0);
    } else {
        close(pipefd[0]);
    }
}

void SleepInhibitor::unInhibit() {
    if (inhibited && cookie > 0) {
        char cookie_arg[64];
        std::snprintf(cookie_arg, sizeof(cookie_arg), "uint32:%u", cookie);

        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
        posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);

        const char *argv[] = {
            "dbus-send",
            "--dest=org.freedesktop.ScreenSaver",
            "/org/freedesktop/ScreenSaver",
            "org.freedesktop.ScreenSaver.UnInhibit",
            cookie_arg,
            nullptr
        };

        pid_t pid;
        if (posix_spawnp(&pid, "dbus-send", &actions, nullptr, const_cast<char *const *>(argv), environ) == 0) {
            waitpid(pid, nullptr, 0);
        }
        posix_spawn_file_actions_destroy(&actions);

        cookie = 0;
        inhibited = false;
    }
}
#endif

} // namespace utils
