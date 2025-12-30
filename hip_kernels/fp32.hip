// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

// Helper for float4 fma
__device__ inline float4 fmaf4(float4 a, float4 b, float4 c) {
    return make_float4(
        fmaf(a.x, b.x, c.x),
        fmaf(a.y, b.y, c.y),
        fmaf(a.z, b.z, c.z),
        fmaf(a.w, b.w, c.w)
    );
}

extern "C" __global__ void run_benchmark(float* data, float multiplier, uint32_t num_elements) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    // Use 8 independent accumulators to hide FMA latency (4-5 cycles)
    float4 acc0 = make_float4(data[index], data[index], data[index], data[index]);
    float4 acc1 = make_float4(0.1f, 0.1f, 0.1f, 0.1f);
    float4 acc2 = make_float4(0.2f, 0.2f, 0.2f, 0.2f);
    float4 acc3 = make_float4(0.3f, 0.3f, 0.3f, 0.3f);
    float4 acc4 = make_float4(0.4f, 0.4f, 0.4f, 0.4f);
    float4 acc5 = make_float4(0.5f, 0.5f, 0.5f, 0.5f);
    float4 acc6 = make_float4(0.6f, 0.6f, 0.6f, 0.6f);
    float4 acc7 = make_float4(0.7f, 0.7f, 0.7f, 0.7f);
    
    float4 m = make_float4(multiplier, multiplier, multiplier, multiplier);
    float4 add = make_float4(0.0001f, 0.0001f, 0.0001f, 0.0001f);

    // Host expects 256 ops / iter.
    // 8 accumulators * 4 calls each = 32 calls.
    // 32 calls * 4 components * 2 ops (FMA) = 256 ops.
    // Perfect match.

    #pragma unroll 4
    for (int i = 0; i < 16384; ++i) {
        acc0 = fmaf4(acc0, m, add);
        acc1 = fmaf4(acc1, m, add);
        acc2 = fmaf4(acc2, m, add);
        acc3 = fmaf4(acc3, m, add);
        acc4 = fmaf4(acc4, m, add);
        acc5 = fmaf4(acc5, m, add);
        acc6 = fmaf4(acc6, m, add);
        acc7 = fmaf4(acc7, m, add);

        acc0 = fmaf4(acc0, m, add);
        acc1 = fmaf4(acc1, m, add);
        acc2 = fmaf4(acc2, m, add);
        acc3 = fmaf4(acc3, m, add);
        acc4 = fmaf4(acc4, m, add);
        acc5 = fmaf4(acc5, m, add);
        acc6 = fmaf4(acc6, m, add);
        acc7 = fmaf4(acc7, m, add);

        acc0 = fmaf4(acc0, m, add);
        acc1 = fmaf4(acc1, m, add);
        acc2 = fmaf4(acc2, m, add);
        acc3 = fmaf4(acc3, m, add);
        acc4 = fmaf4(acc4, m, add);
        acc5 = fmaf4(acc5, m, add);
        acc6 = fmaf4(acc6, m, add);
        acc7 = fmaf4(acc7, m, add);

        acc0 = fmaf4(acc0, m, add);
        acc1 = fmaf4(acc1, m, add);
        acc2 = fmaf4(acc2, m, add);
        acc3 = fmaf4(acc3, m, add);
        acc4 = fmaf4(acc4, m, add);
        acc5 = fmaf4(acc5, m, add);
        acc6 = fmaf4(acc6, m, add);
        acc7 = fmaf4(acc7, m, add);
    }
    
    // Reduce results
    float4 res = acc0;
    res = make_float4(res.x+acc1.x, res.y+acc1.y, res.z+acc1.z, res.w+acc1.w);
    res = make_float4(res.x+acc2.x, res.y+acc2.y, res.z+acc2.z, res.w+acc2.w);
    res = make_float4(res.x+acc3.x, res.y+acc3.y, res.z+acc3.z, res.w+acc3.w);
    res = make_float4(res.x+acc4.x, res.y+acc4.y, res.z+acc4.z, res.w+acc4.w);
    res = make_float4(res.x+acc5.x, res.y+acc5.y, res.z+acc5.z, res.w+acc5.w);
    res = make_float4(res.x+acc6.x, res.y+acc6.y, res.z+acc6.z, res.w+acc6.w);
    res = make_float4(res.x+acc7.x, res.y+acc7.y, res.z+acc7.z, res.w+acc7.w);

    data[index] = res.x + res.y + res.z + res.w;
}
