#include <hip/hip_runtime.h>

// Kernel to measure cache bandwidth.
// Each thread reads from a buffer in a coalesced manner.
extern "C" __global__ void run_benchmark(uint* buffer) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    uint val = 0;

    #pragma unroll
    for (int i = 0; i < 1024; ++i) {
        val += buffer[(id * 16 + (i % 16)) & 6143];
    }

    if (val > 0) {
        buffer[id & 6143] = val;
    }
}
