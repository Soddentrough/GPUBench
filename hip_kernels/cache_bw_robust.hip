#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128) void run_benchmark(
    const float4 *__restrict__ inBuf,
    float *__restrict__ outBuf,
    const uint32_t *__restrict__ pc) {
    const uint32_t lid = threadIdx.x;
    const uint32_t wid = blockIdx.x;
    const uint32_t gid = wid * 128u + lid;

    const uint32_t stepVec4 = pc ? pc[0] : 1024u;
    const uint32_t n = pc ? pc[1] : 1024u;
    const uint32_t iterations = pc ? pc[2] : 2000u;

    const uint32_t numBlocks = (n >> 10u) > 0u ? (n >> 10u) : 1u;
    uint32_t blockBase = ((wid * 1009u) % numBlocks) << 10u;
    const uint32_t lane = lid;

    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma nounroll
    for (uint32_t i = 0u; i < iterations; ++i) {
        const uint32_t p0 = blockBase + lane;
        const float4 v0 = inBuf[p0];
        const float4 v1 = inBuf[p0 + 128u];
        const float4 v2 = inBuf[p0 + 256u];
        const float4 v3 = inBuf[p0 + 384u];
        const float4 v4 = inBuf[p0 + 512u];
        const float4 v5 = inBuf[p0 + 640u];
        const float4 v6 = inBuf[p0 + 768u];
        const float4 v7 = inBuf[p0 + 896u];

        acc0.x += (v0.x + v1.x) + (v2.x + v3.x);
        acc0.y += (v0.y + v1.y) + (v2.y + v3.y);
        acc0.z += (v0.z + v1.z) + (v2.z + v3.z);
        acc0.w += (v0.w + v1.w) + (v2.w + v3.w);

        acc1.x += (v4.x + v5.x) + (v6.x + v7.x);
        acc1.y += (v4.y + v5.y) + (v6.y + v7.y);
        acc1.z += (v4.z + v5.z) + (v6.z + v7.z);
        acc1.w += (v4.w + v5.w) + (v6.w + v7.w);

        blockBase += stepVec4;
        if (blockBase >= n) {
            blockBase -= n;
        }
    }

    const float sumX = acc0.x + acc1.x;
    const float sumY = acc0.y + acc1.y;
    const float sumZ = acc0.z + acc1.z;
    const float sumW = acc0.w + acc1.w;
    outBuf[gid & 65535u] = (sumX + sumY) + (sumZ + sumW);
}
