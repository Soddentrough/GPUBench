#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float4* data, uint* pc) {
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];

    uint globalId = blockIdx.x * blockDim.x + threadIdx.x;
    uint idx = (globalId * stride) & mask;

    float4 sum0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum5 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum6 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum7 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll 4
    for (uint i = 0; i < iterations; i++) {
        uint base = (idx + i * stride * 8) & mask;
        
        float4 v0 = data[(base + 0 * stride) & mask];
        float4 v1 = data[(base + 1 * stride) & mask];
        float4 v2 = data[(base + 2 * stride) & mask];
        float4 v3 = data[(base + 3 * stride) & mask];
        float4 v4 = data[(base + 4 * stride) & mask];
        float4 v5 = data[(base + 5 * stride) & mask];
        float4 v6 = data[(base + 6 * stride) & mask];
        float4 v7 = data[(base + 7 * stride) & mask];

        sum0.x += v0.x; sum0.y += v0.y; sum0.z += v0.z; sum0.w += v0.w;
        sum1.x += v1.x; sum1.y += v1.y; sum1.z += v1.z; sum1.w += v1.w;
        sum2.x += v2.x; sum2.y += v2.y; sum2.z += v2.z; sum2.w += v2.w;
        sum3.x += v3.x; sum3.y += v3.y; sum3.z += v3.z; sum3.w += v3.w;
        sum4.x += v4.x; sum4.y += v4.y; sum4.z += v4.z; sum4.w += v4.w;
        sum5.x += v5.x; sum5.y += v5.y; sum5.z += v5.z; sum5.w += v5.w;
        sum6.x += v6.x; sum6.y += v6.y; sum6.z += v6.z; sum6.w += v6.w;
        sum7.x += v7.x; sum7.y += v7.y; sum7.z += v7.z; sum7.w += v7.w;
    }

    float final_x = sum0.x + sum1.x + sum2.x + sum3.x + sum4.x + sum5.x + sum6.x + sum7.x;
    float final_y = sum0.y + sum1.y + sum2.y + sum3.y + sum4.y + sum5.y + sum6.y + sum7.y;
    float final_z = sum0.z + sum1.z + sum2.z + sum3.z + sum4.z + sum5.z + sum6.z + sum7.z;
    float final_w = sum0.w + sum1.w + sum2.w + sum3.w + sum4.w + sum5.w + sum6.w + sum7.w;

    if (stride == 0xFFFFFFFF) {
        data[globalId & mask] = make_float4(final_x, final_y, final_z, final_w);
    }
}
