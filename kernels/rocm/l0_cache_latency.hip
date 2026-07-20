#include <hip/hip_runtime.h>

// Kernel to measure L0 cache latency.
// A single thread performs 1,000,000 dependent pointer-chase reads over a
// small buffer that fits in L0, mirroring the Vulkan cache_latency.comp.
// The host measures wall time and divides by the step count.
extern "C" __global__ void run_benchmark(uint* buffer) {
    uint index = 0;

    for (uint i = 0; i < 1000000; ++i) {
        index = buffer[index];
    }

    // Prevent dead-code elimination (never taken in practice)
    if (index == 0xDEADBEEF) {
        buffer[0] = index;
    }
}
