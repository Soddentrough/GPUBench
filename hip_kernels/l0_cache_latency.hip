#include <hip/hip_runtime.h>

// Kernel to measure L0 cache latency.
// A single thread performs dependent pointer-chase reads over a small buffer
// that fits in L0, mirroring the Vulkan cache_latency.comp.
// CacheBench.cpp binds 2 buffers on ROCm: (buffer, pcBuffer).
extern "C" __global__ void l0_cache_latency(unsigned int *buffer,
                                            const unsigned int *pc) {
    unsigned int index = 0;
    unsigned int iterations = pc ? pc[2] : 1000000u;

    for (unsigned int i = 0; i < iterations; ++i) {
        index = buffer[index];
    }

    // Prevent dead-code elimination (never taken in practice)
    if (index == 0xDEADBEEFu) {
        buffer[0] = index;
    }
}

extern "C" __global__ void run_benchmark(unsigned int *buffer,
                                         const unsigned int *pc) {
    unsigned int index = 0;
    unsigned int iterations = pc ? pc[2] : 1000000u;

    for (unsigned int i = 0; i < iterations; ++i) {
        index = buffer[index];
    }

    if (index == 0xDEADBEEFu) {
        buffer[0] = index;
    }
}
