#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ __launch_bounds__(256) void run_benchmark(
    float4 *outBuf, uint32_t count, uint32_t iterations, float seed) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) {
        return;
    }
    float4 acc = make_float4(
        seed + static_cast<float>(gid & 255u) * 0.001953125f,
        seed * 0.75f + 0.125f,
        seed * 0.5f + 0.25f,
        1.0f);
    const float4 mul = make_float4(0.99993896f, 0.99996948f, 0.99990845f, 0.99998474f);
    const float4 add = make_float4(0.00012207f, 0.00006104f, 0.00018311f, 0.00003052f);

    #pragma unroll 1
    for (uint32_t i = 0u; i < iterations; ++i) {
        acc.x = fmaf(acc.x, mul.x, add.x);
        acc.y = fmaf(acc.y, mul.y, add.y);
        acc.z = fmaf(acc.z, mul.z, add.z);
        acc.w = fmaf(acc.w, mul.w, add.w);

        float tx = acc.y, ty = acc.z, tz = acc.w, tw = acc.x;
        acc.x = fmaf(tx, mul.x, add.x);
        acc.y = fmaf(ty, mul.y, add.y);
        acc.z = fmaf(tz, mul.z, add.z);
        acc.w = fmaf(tw, mul.w, add.w);

        tx = acc.z; ty = acc.w; tz = acc.x; tw = acc.y;
        acc.x = fmaf(tx, mul.x, add.x);
        acc.y = fmaf(ty, mul.y, add.y);
        acc.z = fmaf(tz, mul.z, add.z);
        acc.w = fmaf(tw, mul.w, add.w);

        tx = acc.w; ty = acc.x; tz = acc.y; tw = acc.z;
        acc.x = fmaf(tx, mul.x, add.x);
        acc.y = fmaf(ty, mul.y, add.y);
        acc.z = fmaf(tz, mul.z, add.z);
        acc.w = fmaf(tw, mul.w, add.w);
    }
    outBuf[gid] = acc;
}
