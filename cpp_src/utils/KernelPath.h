#ifndef KERNELPATH_H
#define KERNELPATH_H

#include <string>

class KernelPath {
public:
    // Find the kernel directory by searching in this order:
    // 1. Environment variable GPUBENCH_KERNEL_PATH (if set)
    // 2. Installed location (CMAKE_INSTALL_PREFIX/share/gpubench/kernels)
    // 3. Development fallback (./kernels)
    static std::string find();
};

#endif // KERNELPATH_H
