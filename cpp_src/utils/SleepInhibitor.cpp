#include "SleepInhibitor.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/pwr_mgt/IOPMLib.h>
#elif defined(__linux__)
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifdef __linux__
extern char **environ;
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
      ,
      previousState(other.previousState)
#elif defined(__APPLE__)
      ,
      assertionID(other.assertionID)
#elif defined(__linux__)
      ,
      inhibitorPid(other.inhibitorPid),
      inhibitorPipeFd(other.inhibitorPipeFd)
#endif
{
  other.inhibited = false;
#ifdef _WIN32
  other.previousState = 0;
#elif defined(__APPLE__)
  other.assertionID = 0;
#elif defined(__linux__)
  other.inhibitorPid = -1;
  other.inhibitorPipeFd = -1;
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
#elif defined(__linux__)
    inhibitorPid = other.inhibitorPid;
    inhibitorPipeFd = other.inhibitorPipeFd;
    other.inhibitorPid = -1;
    other.inhibitorPipeFd = -1;
#endif
    other.inhibited = false;
  }
  return *this;
}

#ifdef _WIN32
void SleepInhibitor::inhibit(const std::string & /*reason*/) {
  EXECUTION_STATE prev = SetThreadExecutionState(
      ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
  if (prev != 0) {
    previousState = static_cast<uint32_t>(prev);
    inhibited = true;
  }
}

void SleepInhibitor::unInhibit() {
  if (inhibited) {
    const EXECUTION_STATE restore =
        (previousState != 0)
            ? static_cast<EXECUTION_STATE>(ES_CONTINUOUS | previousState)
            : ES_CONTINUOUS;
    SetThreadExecutionState(restore);
    inhibited = false;
    previousState = 0;
  }
}

#elif defined(__APPLE__)
void SleepInhibitor::inhibit(const std::string &reason) {
  CFStringRef cfReason = CFStringCreateWithCString(
      kCFAllocatorDefault, reason.c_str(), kCFStringEncodingUTF8);
  IOReturn res = IOPMAssertionCreateWithName(
      kIOPMAssertionTypePreventUserIdleSystemSleep, kIOPMAssertionLevelOn,
      cfReason ? cfReason : CFSTR("GPUBench computation"), &assertionID);
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

#elif defined(__linux__)
void SleepInhibitor::inhibit(const std::string &reason) {
  int pipefd[2] = {-1, -1};
  if (pipe2(pipefd, O_CLOEXEC) != 0) {
    return;
  }

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addclose(&actions, pipefd[1]);
  posix_spawn_file_actions_adddup2(&actions, pipefd[0], STDIN_FILENO);
  posix_spawn_file_actions_addclose(&actions, pipefd[0]);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null",
                                   O_WRONLY, 0);
  posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null",
                                   O_WRONLY, 0);

  const std::string whyStr =
      reason.empty() ? std::string("Active GPU benchmark execution") : reason;
  const char *argv[] = {"systemd-inhibit",
                        "--what=idle:sleep",
                        "--who=GPUBench",
                        "--why",
                        whyStr.c_str(),
                        "--mode=block",
                        "cat",
                        nullptr};

  pid_t pid = -1;
  int spawnStatus = posix_spawnp(&pid, "systemd-inhibit", &actions, nullptr,
                                 const_cast<char *const *>(argv), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(pipefd[0]);

  if (spawnStatus == 0 && pid > 0) {
    inhibitorPid = pid;
    inhibitorPipeFd = pipefd[1];
    inhibited = true;
  } else {
    close(pipefd[1]);
    inhibitorPid = -1;
    inhibitorPipeFd = -1;
    inhibited = false;
  }
}

void SleepInhibitor::unInhibit() {
  if (inhibitorPipeFd >= 0) {
    close(inhibitorPipeFd);
    inhibitorPipeFd = -1;
  }
  if (inhibitorPid > 0) {
    int childStatus = 0;
    while (waitpid(inhibitorPid, &childStatus, 0) < 0 && errno == EINTR) {
    }
    inhibitorPid = -1;
  }
  inhibited = false;
}

#else
void SleepInhibitor::inhibit(const std::string & /*reason*/) {
  inhibited = false;
}

void SleepInhibitor::unInhibit() {
  inhibited = false;
}
#endif

} // namespace utils
