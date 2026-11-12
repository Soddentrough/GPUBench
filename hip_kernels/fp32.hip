// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

// Helper for float4 fmac (acc = fmaf(m, acc, acc)) with scalar SGPR m in src0 and dst == addend for VOPD v_dual_fmac_f32 pairing
__device__ inline float4 fmac4_self(float m, float4 acc) {
    return make_float4(
        fmaf(m, acc.x, acc.x),
        fmaf(m, acc.y, acc.y),
        fmaf(m, acc.z, acc.z),
        fmaf(m, acc.w, acc.w)
    );
}

__device__ inline float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

#define INIT_ACC(i) make_float4(seed + 0.04f * (4 * (i) + 1), seed + 0.04f * (4 * (i) + 2), seed + 0.04f * (4 * (i) + 3), seed + 0.04f * (4 * (i) + 4))

extern "C" __global__ void run_benchmark(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    // 8 independent float4 accumulators (32 scalar FMACs in flight, ~34-40 VGPRs) matching shaders/fp32.comp:
    // avoids >128 VGPR occupancy collapse and preserves dst == addend + SGPR src0 for RDNA 3/4 v_dual_fmac_f32.
    float seed = data[index & 0x1FFFu] * 0.0001f;
    float4 a0 = INIT_ACC(0);
    float4 a1 = INIT_ACC(1);
    float4 a2 = INIT_ACC(2);
    float4 a3 = INIT_ACC(3);
    float4 a4 = INIT_ACC(4);
    float4 a5 = INIT_ACC(5);
    float4 a6 = INIT_ACC(6);
    float4 a7 = INIT_ACC(7);

    // Pre-scaled on host (0.999f * 0.000025f) so m stays directly in a kernarg SGPR without VALU scaling.
    float m = multiplier;

    // 8 float4 FMACs × 4 components × 2 ops = 64 FP32 ops/iter × 65536 iters = 4,194,304 ops/thread.
    #pragma unroll 4
    for (int i = 0; i < 65536; ++i) {
        a0 = fmac4_self(m, a0);
        a1 = fmac4_self(m, a1);
        a2 = fmac4_self(m, a2);
        a3 = fmac4_self(m, a3);
        a4 = fmac4_self(m, a4);
        a5 = fmac4_self(m, a5);
        a6 = fmac4_self(m, a6);
        a7 = fmac4_self(m, a7);
    }

    float4 g0 = add4(add4(a0, a1), add4(a2, a3));
    float4 g1 = add4(add4(a4, a5), add4(a6, a7));
    float4 sum = add4(g0, g1);

    data[index] = sum.x + sum.y + sum.z + sum.w;
}
