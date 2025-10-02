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

extern "C" __global__ void run_benchmark(float* data) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    // Work with multiple accumulators to avoid dependency chains
    float scalar_val = data[index];
    float4 val1 = make_float4(scalar_val, scalar_val, scalar_val, scalar_val);
    float4 val2 = make_float4(0.1f, 0.2f, 0.3f, 0.4f);
    float4 val3 = make_float4(0.5f, 0.6f, 0.7f, 0.8f);
    float4 val4 = make_float4(0.9f, 1.0f, 1.1f, 1.2f);
    const float4 fma_val = make_float4(1.0001f, 1.0001f, 1.0001f, 1.0001f);
    
    // Each iteration performs 4 vec4 FMAs = 4 * 4 * 2 = 32 FP32 ops
    #pragma unroll
    for (int i = 0; i < 16384; ++i) {
        val1 = fmaf4(val1, fma_val, val2);
        val2 = fmaf4(val2, fma_val, val3);
        val3 = fmaf4(val3, fma_val, val4);
        val4 = fmaf4(val4, fma_val, val1);
    }
    
    // Prevent optimization away
    data[index] = val1.x + val2.y + val3.z + val4.w;
}
