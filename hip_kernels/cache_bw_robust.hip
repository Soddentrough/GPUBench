#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128) void run_benchmark(
    const float4 *__restrict__ inBuf,
    float *__restrict__ outBuf,
    const uint32_t *__restrict__ pc) {
    if (!pc) {
        return;
    }
    const uint32_t lid = threadIdx.x;
    const uint32_t wid = blockIdx.x;
    const uint32_t gid = wid * 128u + lid;

    const uint32_t stepVec4 = pc[0];
    const uint32_t n = pc[1];
    const uint32_t iterations = pc[2];

    const uint32_t numBlocks = (n >> 10u) > 0u ? (n >> 10u) : 1u;
    uint32_t blockBase = ((wid * 1009u) % numBlocks) << 10u;
    const float4 *__restrict__ blockPtr = inBuf + static_cast<size_t>(blockBase);
    const size_t stepSize = static_cast<size_t>(stepVec4);
    const size_t nSize = static_cast<size_t>(n);
    const size_t lane = static_cast<size_t>(lid);

    float4 acc0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma nounroll
    for (uint32_t i = 0u; i < iterations; ++i) {
        const float4 *__restrict__ p = blockPtr + lane;
        const float4 v0 = p[0];
        const float4 v1 = p[128];
        const float4 v2 = p[256];
        const float4 v3 = p[384];
        const float4 v4 = p[512];
        const float4 v5 = p[640];
        const float4 v6 = p[768];
        const float4 v7 = p[896];

        acc0.x += (v0.x + v1.x) + (v2.x + v3.x);
        acc0.y += (v0.y + v1.y) + (v2.y + v3.y);
        acc0.z += (v0.z + v1.z) + (v2.z + v3.z);
        acc0.w += (v0.w + v1.w) + (v2.w + v3.w);

        acc1.x += (v4.x + v5.x) + (v6.x + v7.x);
        acc1.y += (v4.y + v5.y) + (v6.y + v7.y);
        acc1.z += (v4.z + v5.z) + (v6.z + v7.z);
        acc1.w += (v4.w + v5.w) + (v6.w + v7.w);

        blockBase += stepVec4;
        blockPtr += stepSize;
        if (blockBase >= n) {
            blockBase -= n;
            blockPtr -= nSize;
        }
    }

    const float sumX = acc0.x + acc1.x;
    const float sumY = acc0.y + acc1.y;
    const float sumZ = acc0.z + acc1.z;
    const float sumW = acc0.w + acc1.w;
    outBuf[gid & 65535u] = (sumX + sumY) + (sumZ + sumW);
}
