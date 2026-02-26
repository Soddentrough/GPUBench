#include <hip/hip_runtime.h>

// Kernel to measure cache latency.
// A single thread performs a chain of dependent reads.
extern "C" __global__ void run_benchmark(uint* buffer, uint* pc) {
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];

    uint val = 0;

    #pragma unroll
    for (int i = 0; i < 1024; ++i) {
        val = buffer[val];
    }

    if (val > 0) {
        buffer[0] = val;
    }
}
