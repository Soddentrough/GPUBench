#include "KernelPath.h"
#include "utils/Config.h"
#include <cstdlib>
#include <filesystem>
#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#endif

// Helper function to check if a directory exists
static bool directoryExists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_directory(path, ec);
}

static std::string getExecutableDir() {
    std::string path;
#if defined(_WIN32)
    char buffer[MAX_PATH];
    if (GetModuleFileNameA(NULL, buffer, MAX_PATH) > 0) {
        path = std::string(buffer);
    }
#elif defined(__linux__)
    char buffer[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
    if (count != -1) {
        path = std::string(buffer, count);
    }
#endif

    if (path.empty()) return "";

    std::filesystem::path p(path);
    return p.parent_path().string();
}

std::string KernelPath::find() {
    // 1. Check development location first
    std::string dev_path = "kernels";
    if (directoryExists(dev_path)) {
        return dev_path;
    }

    // 2. Check environment variable
    const char* env_path = std::getenv("GPUBENCH_KERNEL_PATH");
    if (env_path && directoryExists(env_path)) {
        return env_path;
    }
    
    // 3. Check relative to executable (Robust portable lookup)
    std::string exe_dir = getExecutableDir();
    if (!exe_dir.empty()) {
        // Look for standard Linux install structure: bin/../share/gpubench/kernels
        // resolves to: <install_prefix>/share/gpubench/kernels
        std::string exe_share_path = exe_dir + "/../share/gpubench/kernels";
        if (directoryExists(exe_share_path)) {
            return exe_share_path;
        }
        
        // Look for Windows/Portable structure: bin/kernels or bin/../kernels
        std::string exe_adj_path = exe_dir + "/kernels";
        if (directoryExists(exe_adj_path)) {
            return exe_adj_path;
        }
    }

    // 4. Check relative install locations (CWD based - fallback)
    std::string share_path = "share/gpubench/kernels";
    if (directoryExists(share_path)) {
        return share_path;
    }

    std::string rel_share_path = "../share/gpubench/kernels";
    if (directoryExists(rel_share_path)) {
        return rel_share_path;
    }

    // 5. Check installed location
    std::string install_path = std::string(GPUBENCH_INSTALL_PREFIX) + "/share/gpubench/kernels";
    if (directoryExists(install_path)) {
        return install_path;
    }
    
    // If nothing found, return the development path anyway and let the caller handle the error
    std::cerr << "Warning: Could not find kernel directory. Searched:" << std::endl;
    std::cerr << "  - Development location: " << dev_path << std::endl;
    if (env_path) {
        std::cerr << "  - GPUBENCH_KERNEL_PATH: " << env_path << std::endl;
    }
    if (!exe_dir.empty()) {
        std::cerr << "  - Executable relative: " << exe_dir << "/../share/gpubench/kernels" << std::endl;
    }
    std::cerr << "  - Relative share: " << share_path << std::endl;
    std::cerr << "  - Relative ../share: " << rel_share_path << std::endl;
    std::cerr << "  - Install location: " << install_path << std::endl;
    
    return dev_path;
}
