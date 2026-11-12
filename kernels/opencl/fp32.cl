// Requires OpenCL 1.2+

#define INIT_ACC(i) (seed + 0.04f * (float4)((float)(4 * (i) + 1), (float)(4 * (i) + 2), (float)(4 * (i) + 3), (float)(4 * (i) + 4)))

__kernel void run_benchmark(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    // Load initial seed from memory so the compiler cannot dead-code eliminate
    float in_val = data[index & 0x1FFFu];
    float4 seed = (float4)(in_val * 0.0001f);

    float4 a0 = INIT_ACC(0);
    float4 a1 = INIT_ACC(1);
    float4 a2 = INIT_ACC(2);
    float4 a3 = INIT_ACC(3);
    float4 a4 = INIT_ACC(4);
    float4 a5 = INIT_ACC(5);
    float4 a6 = INIT_ACC(6);
    float4 a7 = INIT_ACC(7);

    // Pre-scaled on host (0.999f * 0.000025f) so multiplier stays directly in a scalar register
    // without VALU scaling, preserving dst == addend for v_dual_fmac_f32 (~34-40 VGPRs).
    float4 m = (float4)(multiplier);

    // 8 float4 FMAs × 4 components × 2 ops = 64 FP32 ops/iter × 65536 iters = 4,194,304 ops/thread
    for (int i = 0; i < 65536; ++i) {
        a0 = fma(m, a0, a0);
        a1 = fma(m, a1, a1);
        a2 = fma(m, a2, a2);
        a3 = fma(m, a3, a3);
        a4 = fma(m, a4, a4);
        a5 = fma(m, a5, a5);
        a6 = fma(m, a6, a6);
        a7 = fma(m, a7, a7);
    }

    float4 sum = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
    data[index] = sum.x + sum.y + sum.z + sum.w;
}
