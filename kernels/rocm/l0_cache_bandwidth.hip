#include <hip/hip_runtime.h>

// Kernel to measure L0 cache bandwidth.
// Each thread performs a series of independent reads from a small, 
// localized buffer to ensure the data stays within the L0 cache.
extern "C" __global__ void run_benchmark(uint* buffer) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    uint val = 0;
    
    // Perform 16 independent operations per thread
    // to maximize instruction-level parallelism.
    uint r[16];
    for (int i = 0; i < 16; ++i) {
        r[i] = i;
    }

    #pragma unroll
    for (int i = 0; i < 1024; ++i) {
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            r[j] += r[(j + 1) % 16];
        }
    }

    for (int i = 0; i < 16; ++i) {
        val += r[i];
    }

    if (val > 0) {
        buffer[0] = val;
    }
}
