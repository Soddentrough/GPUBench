// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(double* data) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    const double c1 = 0.5;
    const double c2 = 0.25;
    double val = data[index];
    // Loop count reduced from 65536 → 2048 (32×) to keep dispatch time <50ms.
    for (int i = 0; i < 2048; ++i) {
        val = val * c1 + c2;
    }
    data[index] = val;
}
