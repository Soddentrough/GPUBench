#include "KernelPath.h"
#include "utils/Config.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <limits.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <limits.h>
#include <mach-o/dyld.h>
#endif

static std::filesystem::path getExecutableDir() {
  std::filesystem::path path;
#if defined(_WIN32)
  char buffer[MAX_PATH];
  if (GetModuleFileNameA(NULL, buffer, MAX_PATH) > 0) {
    path = std::filesystem::path(std::string(buffer));
  }
#elif defined(__linux__)
  char buffer[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
  if (count != -1) {
    path = std::filesystem::path(std::string(buffer, count));
  }
#elif defined(__APPLE__)
  char buffer[PATH_MAX];
  uint32_t size = sizeof(buffer);
  if (_NSGetExecutablePath(buffer, &size) == 0) {
    path = std::filesystem::path(buffer);
  }
#endif

  if (path.empty())
    return std::filesystem::path();
  return path.parent_path();
}

// Kernel directory search order:
//   1. GPUBENCH_KERNEL_PATH environment variable (explicit override)
//   2. Paths relative to the executable (installed layouts):
//      bin/../share/gpubench/kernels, bin/kernels, bin/../kernels
//   3. Compile-time install prefix (GPUBENCH_INSTALL_PREFIX), if defined
//   4. CWD-relative development path ./kernels (running from the build tree)
//   5. CWD-relative share/gpubench/kernels fallbacks
// The CWD dev path intentionally comes AFTER the installed/exe-relative
// paths so an installed binary doesn't accidentally pick up a stray
// ./kernels directory in the user's working directory.
std::string KernelPath::find() {
  std::filesystem::path dev_path = std::filesystem::path("kernels");

  // 1. Check environment variable
  const char *env_path_raw = std::getenv("GPUBENCH_KERNEL_PATH");
  if (env_path_raw) {
    std::filesystem::path env_path(env_path_raw);
    if (std::filesystem::is_directory(env_path)) {
      return env_path.string();
    }
  }

  // 2. Check relative to executable (Robust portable lookup)
  std::filesystem::path exe_dir = getExecutableDir();
  if (!exe_dir.empty()) {
    // Look for standard Linux install structure: bin/../share/gpubench/kernels
    std::filesystem::path exe_share_path =
        exe_dir / std::filesystem::path("..") / std::filesystem::path("share") /
        std::filesystem::path("gpubench") / std::filesystem::path("kernels");
    if (std::filesystem::is_directory(exe_share_path)) {
      return exe_share_path.string();
    }

    // Look for Windows/Portable structure: bin/kernels or bin/../kernels
    std::filesystem::path exe_adj_path =
        exe_dir / std::filesystem::path("kernels");
    if (std::filesystem::is_directory(exe_adj_path)) {
      return exe_adj_path.string();
    }

    std::filesystem::path exe_parent_kernels = exe_dir /
                                               std::filesystem::path("..") /
                                               std::filesystem::path("kernels");
    if (std::filesystem::is_directory(exe_parent_kernels)) {
      return exe_parent_kernels.string();
    }
  }

  // 3. Check installed location from compile-time prefix
#ifdef GPUBENCH_INSTALL_PREFIX
  std::filesystem::path install_path =
      std::filesystem::path(GPUBENCH_INSTALL_PREFIX) /
      std::filesystem::path("share") / std::filesystem::path("gpubench") /
      std::filesystem::path("kernels");
  if (std::filesystem::is_directory(install_path)) {
    return install_path.string();
  }
#endif

  // 4. Check development location (fallback for running from the build tree)
  if (std::filesystem::is_directory(dev_path)) {
    return dev_path.string();
  }

  // 5. Check relative install locations (CWD based - fallback)
  std::filesystem::path share_path = std::filesystem::path("share") /
                                     std::filesystem::path("gpubench") /
                                     std::filesystem::path("kernels");
  if (std::filesystem::is_directory(share_path)) {
    return share_path.string();
  }

  std::filesystem::path rel_share_path =
      std::filesystem::path("..") / std::filesystem::path("share") /
      std::filesystem::path("gpubench") / std::filesystem::path("kernels");
  if (std::filesystem::is_directory(rel_share_path)) {
    return rel_share_path.string();
  }

  // If nothing found, return the development path anyway and let the caller
  // handle the error
  return dev_path.string();
}
