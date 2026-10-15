#include <hip/hip_runtime.h>

// Kernel to measure L0 cache latency.
// Performs dependent pointer-chase reads over a small buffer that fits in L0,
// corresponding to Vulkan shaders/l0_cache_latency.comp (parameterized via pc[2] iterations).
// CacheBench.cpp binds 2 buffers on ROCm: (buffer, pcBuffer).
extern "C" __global__ void l0_cache_latency(unsigned int *buffer,
                                            const unsigned int *pc) {
    unsigned int index = threadIdx.x;
    unsigned int iterations = pc ? pc[2] : 1000000u;

    #pragma nounroll
    for (unsigned int i = 0; i < iterations; ++i) {
        index = buffer[index];
    }

    // Prevent dead-code elimination: store terminal index to unchased word 1
    buffer[1] = index;
}

extern "C" __global__ void run_benchmark(unsigned int *buffer,
                                         const unsigned int *pc) {
    unsigned int index = threadIdx.x;
    unsigned int iterations = pc ? pc[2] : 1000000u;

    #pragma nounroll
    for (unsigned int i = 0; i < iterations; ++i) {
        index = buffer[index];
    }

    buffer[1] = index;
}
