// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

// Helper for float4 fmac (acc = fmaf(a, b, acc)) for VOPD dual-issue pairing
__device__ inline float4 fmac4(float4 acc, float4 a, float4 b) {
    return make_float4(
        fmaf(a.x, b.x, acc.x),
        fmaf(a.y, b.y, acc.y),
        fmaf(a.z, b.z, acc.z),
        fmaf(a.w, b.w, acc.w)
    );
}

extern "C" __global__ void run_benchmark(float* data, float multiplier, uint32_t num_elements) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    // Use 32 independent float4 accumulators for high ILP and dual-issue ALU saturation (128 scalar FMAs in flight)
    float seed   = data[index & 0x1FFFu] * 0.0001f;
    float4 val0  = make_float4(seed + 0.1f, seed + 0.2f, seed + 0.3f, seed + 0.4f);
    float4 val1  = make_float4(seed + 0.2f, seed + 0.3f, seed + 0.4f, seed + 0.5f);
    float4 val2  = make_float4(seed + 0.3f, seed + 0.4f, seed + 0.5f, seed + 0.6f);
    float4 val3  = make_float4(seed + 0.4f, seed + 0.5f, seed + 0.6f, seed + 0.7f);
    float4 val4  = make_float4(seed + 0.5f, seed + 0.6f, seed + 0.7f, seed + 0.8f);
    float4 val5  = make_float4(seed + 0.6f, seed + 0.7f, seed + 0.8f, seed + 0.9f);
    float4 val6  = make_float4(seed + 0.7f, seed + 0.8f, seed + 0.9f, seed + 1.0f);
    float4 val7  = make_float4(seed + 0.8f, seed + 0.9f, seed + 1.0f, seed + 1.1f);
    float4 val8  = make_float4(seed + 0.9f, seed + 1.0f, seed + 1.1f, seed + 1.2f);
    float4 val9  = make_float4(seed + 1.0f, seed + 1.1f, seed + 1.2f, seed + 1.3f);
    float4 val10 = make_float4(seed + 1.1f, seed + 1.2f, seed + 1.3f, seed + 1.4f);
    float4 val11 = make_float4(seed + 1.2f, seed + 1.3f, seed + 1.4f, seed + 1.5f);
    float4 val12 = make_float4(seed + 1.3f, seed + 1.4f, seed + 1.5f, seed + 1.6f);
    float4 val13 = make_float4(seed + 1.4f, seed + 1.5f, seed + 1.6f, seed + 1.7f);
    float4 val14 = make_float4(seed + 1.5f, seed + 1.6f, seed + 1.7f, seed + 1.8f);
    float4 val15 = make_float4(seed + 1.6f, seed + 1.7f, seed + 1.8f, seed + 1.9f);
    float4 val16 = make_float4(seed + 1.7f, seed + 1.8f, seed + 1.9f, seed + 2.0f);
    float4 val17 = make_float4(seed + 1.8f, seed + 1.9f, seed + 2.0f, seed + 2.1f);
    float4 val18 = make_float4(seed + 1.9f, seed + 2.0f, seed + 2.1f, seed + 2.2f);
    float4 val19 = make_float4(seed + 2.0f, seed + 2.1f, seed + 2.2f, seed + 2.3f);
    float4 val20 = make_float4(seed + 2.1f, seed + 2.2f, seed + 2.3f, seed + 2.4f);
    float4 val21 = make_float4(seed + 2.2f, seed + 2.3f, seed + 2.4f, seed + 2.5f);
    float4 val22 = make_float4(seed + 2.3f, seed + 2.4f, seed + 2.5f, seed + 2.6f);
    float4 val23 = make_float4(seed + 2.4f, seed + 2.5f, seed + 2.6f, seed + 2.7f);
    float4 val24 = make_float4(seed + 2.5f, seed + 2.6f, seed + 2.7f, seed + 2.8f);
    float4 val25 = make_float4(seed + 2.6f, seed + 2.7f, seed + 2.8f, seed + 2.9f);
    float4 val26 = make_float4(seed + 2.7f, seed + 2.8f, seed + 2.9f, seed + 3.0f);
    float4 val27 = make_float4(seed + 2.8f, seed + 2.9f, seed + 3.0f, seed + 3.1f);
    float4 val28 = make_float4(seed + 2.9f, seed + 3.0f, seed + 3.1f, seed + 3.2f);
    float4 val29 = make_float4(seed + 3.0f, seed + 3.1f, seed + 3.2f, seed + 3.3f);
    float4 val30 = make_float4(seed + 3.1f, seed + 3.2f, seed + 3.3f, seed + 3.4f);
    float4 val31 = make_float4(seed + 3.2f, seed + 3.3f, seed + 3.4f, seed + 3.5f);
    
    // Spectral radius (1 + m)^16384 stays bounded in [0.2, 5.2] with |m| <= 1e-4,
    // avoiding IEEE-754 saturation to +inf while preserving dst == addend for v_dual_fmac_f32
    float scaled_m = multiplier * 0.0001f;
    float4 m = make_float4(scaled_m, scaled_m, scaled_m, scaled_m);

    // Each iteration performs 32 float4 FMACs = 32 * 4 * 2 = 256 FP32 ops.
    #pragma unroll 4
    for (int i = 0; i < 16384; ++i) {
        val0  = fmac4(val0,  m, val1);
        val1  = fmac4(val1,  m, val2);
        val2  = fmac4(val2,  m, val3);
        val3  = fmac4(val3,  m, val4);
        val4  = fmac4(val4,  m, val5);
        val5  = fmac4(val5,  m, val6);
        val6  = fmac4(val6,  m, val7);
        val7  = fmac4(val7,  m, val8);
        val8  = fmac4(val8,  m, val9);
        val9  = fmac4(val9,  m, val10);
        val10 = fmac4(val10, m, val11);
        val11 = fmac4(val11, m, val12);
        val12 = fmac4(val12, m, val13);
        val13 = fmac4(val13, m, val14);
        val14 = fmac4(val14, m, val15);
        val15 = fmac4(val15, m, val16);
        val16 = fmac4(val16, m, val17);
        val17 = fmac4(val17, m, val18);
        val18 = fmac4(val18, m, val19);
        val19 = fmac4(val19, m, val20);
        val20 = fmac4(val20, m, val21);
        val21 = fmac4(val21, m, val22);
        val22 = fmac4(val22, m, val23);
        val23 = fmac4(val23, m, val24);
        val24 = fmac4(val24, m, val25);
        val25 = fmac4(val25, m, val26);
        val26 = fmac4(val26, m, val27);
        val27 = fmac4(val27, m, val28);
        val28 = fmac4(val28, m, val29);
        val29 = fmac4(val29, m, val30);
        val30 = fmac4(val30, m, val31);
        val31 = fmac4(val31, m, val0);
    }
    
    // Reduce results to prevent dead-code elimination
    float4 res1 = make_float4(val0.x + val1.y, val2.z + val3.w, val4.x + val5.y, val6.z + val7.w);
    float4 res2 = make_float4(val8.x + val9.y, val10.z + val11.w, val12.x + val13.y, val14.z + val15.w);
    float4 res3 = make_float4(val16.x + val17.y, val18.z + val19.w, val20.x + val21.y, val22.z + val23.w);
    float4 res4 = make_float4(val24.x + val25.y, val26.z + val27.w, val28.x + val29.y, val30.z + val31.w);

    float4 res = make_float4(res1.x + res2.x + res3.x + res4.x,
                             res1.y + res2.y + res3.y + res4.y,
                             res1.z + res2.z + res3.z + res4.z,
                             res1.w + res2.w + res3.w + res4.w);

    data[index] = res.x + res.y + res.z + res.w;
}
