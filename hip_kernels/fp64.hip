// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(double* data) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    double add = (double)(index) * 0.00000001;
    double mult = 1.000001 + add;

    // Use 8 independent double accumulators to hide ALU pipeline latency (matching Vulkan fp64.comp)
    double val0 = data[index] + 1.0;
    double val1 = val0 + 0.1;
    double val2 = val0 + 0.2;
    double val3 = val0 + 0.3;
    double val4 = val0 + 0.4;
    double val5 = val0 + 0.5;
    double val6 = val0 + 0.6;
    double val7 = val0 + 0.7;

    // 256 iters * 8 FMAs * 2 ops = 4096 FP64 ops
    #pragma unroll 4
    for (int i = 0; i < 256; ++i) {
        val0 = fma(val0, mult, val1);
        val1 = fma(val1, mult, val2);
        val2 = fma(val2, mult, val3);
        val3 = fma(val3, mult, val4);
        val4 = fma(val4, mult, val5);
        val5 = fma(val5, mult, val6);
        val6 = fma(val6, mult, val7);
        val7 = fma(val7, mult, val0);
    }
    data[index] = val0 + val1 + val2 + val3 + val4 + val5 + val6 + val7;
}
