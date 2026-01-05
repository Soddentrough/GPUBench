#include <hip/hip_runtime.h>

// Kernel to measure L0 cache latency.
// A single thread performs a chain of 16 dependent reads 
// to measure the latency of the L0 cache.
extern "C" __global__ void run_benchmark(uint* buffer) {
    uint val = 0;
    
    // 16 dependent operations
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        val = (val + 1) & 15;
        val = (val + 2) & 15;
        val = (val + 3) & 15;
        val = (val + 4) & 15;
        val = (val + 5) & 15;
        val = (val + 6) & 15;
        val = (val + 7) & 15;
        val = (val + 8) & 15;
        val = (val + 9) & 15;
        val = (val + 10) & 15;
        val = (val + 11) & 15;
        val = (val + 12) & 15;
        val = (val + 13) & 15;
        val = (val + 14) & 15;
        val = (val + 15) & 15;
        val = (val + 16) & 15;
    }

    if (val > 0) {
        buffer[0] = val;
    }
}
