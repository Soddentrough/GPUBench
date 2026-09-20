#include <hip/hip_runtime.h>

// Kernel to measure cache latency.
// A single thread performs a chain of dependent reads.
extern "C" __global__ void run_benchmark(uint* buffer, uint* pc) {
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];

    uint val = 0;

    #pragma unroll
    for (uint i = 0; i < iterations; ++i) {
        val = buffer[val];
    }

    if (stride == 0xFFFFFFFF) { buffer[1] = mask; }

    if (val > 0) {
        buffer[0] = val;
    }
}
