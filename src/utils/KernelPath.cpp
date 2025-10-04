#include "KernelPath.h"
#include "utils/Config.h"
#include <cstdlib>
#include <sys/stat.h>
#include <iostream>

// Helper function to check if a directory exists
static bool directoryExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return (info.st_mode & S_IFDIR) != 0;
}

std::string KernelPath::find() {
    // 1. Check environment variable
    const char* env_path = std::getenv("GPUBENCH_KERNEL_PATH");
    if (env_path && directoryExists(env_path)) {
        return env_path;
    }
    
    // 2. Check installed location
    std::string install_path = std::string(GPUBENCH_INSTALL_PREFIX) + "/share/gpubench/kernels";
    if (directoryExists(install_path)) {
        return install_path;
    }
    
    // 3. Fall back to development location
    std::string dev_path = "kernels";
    if (directoryExists(dev_path)) {
        return dev_path;
    }
    
    // If nothing found, return the development path anyway and let the caller handle the error
    std::cerr << "Warning: Could not find kernel directory. Searched:" << std::endl;
    if (env_path) {
        std::cerr << "  - GPUBENCH_KERNEL_PATH: " << env_path << std::endl;
    }
    std::cerr << "  - Install location: " << install_path << std::endl;
    std::cerr << "  - Development location: " << dev_path << std::endl;
    
    return dev_path;
}
