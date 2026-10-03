// Requires OpenCL 1.2+
// Config 0: Standard FP32 (Single-Issue Baseline)
__kernel void run_dual_issue_ilp4(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = data[index & 0x1FFFu] * 0.0001f;
    float val0 = base;
    float val1 = base + 0.1f;

    // 131072 iterations * 2 scalar FMAs = 131072 * 4 = 524,288 FP32 operations
    for (int i = 0; i < 131072; ++i) {
        val0 = fma(val1, multiplier, val0);
        val1 = fma(val0, multiplier, val1);
    }

    data[index] = val0 + val1;
}

// Config 1: Dual-Issue FP32 (Partial Co-Issue)
__kernel void run_dual_issue_ilp8(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = data[index & 0x1FFFu] * 0.0001f;
    float val0 = base;
    float val1 = base + 0.1f;
    float val2 = base + 0.2f;
    float val3 = base + 0.3f;

    // 131072 iterations * 4 scalar FMAs = 131072 * 8 = 1,048,576 FP32 operations
    for (int i = 0; i < 131072; ++i) {
        val0 = fma(val1, multiplier, val0);
        val1 = fma(val0, multiplier, val1);
        val2 = fma(val3, multiplier, val2);
        val3 = fma(val2, multiplier, val3);
    }

    data[index] = val0 + val1 + val2 + val3;
}

// Config 2: Dual-Issue FP32 (FP32+FP32)
__kernel void run_dual_issue_ilp16(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = data[index & 0x1FFFu] * 0.0001f;
    float val0 = base;
    float val1 = base + 0.1f;
    float val2 = base + 0.2f;
    float val3 = base + 0.3f;
    float val4 = base + 0.4f;
    float val5 = base + 0.5f;
    float val6 = base + 0.6f;
    float val7 = base + 0.7f;

    // 131072 iterations * 8 scalar FMAs = 131072 * 16 = 2,097,152 FP32 operations
    for (int i = 0; i < 131072; ++i) {
        val0 = fma(val1, multiplier, val0);
        val2 = fma(val3, multiplier, val2);
        val4 = fma(val5, multiplier, val4);
        val6 = fma(val7, multiplier, val6);

        val1 = fma(val0, multiplier, val1);
        val3 = fma(val2, multiplier, val3);
        val5 = fma(val4, multiplier, val5);
        val7 = fma(val6, multiplier, val7);
    }

    data[index] = (val0 + val1) + (val2 + val3) + (val4 + val5) + (val6 + val7);
}

// Config 6: Dual-Issue Mixed (FP32+INT32)
__kernel void run_dual_issue_mixed(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    uint ubase = (uint)(in_val * 1000.0f) + index;
    float4 seed = (float4)(in_val * 0.0001f);

    float4 f0 = seed + (float4)(0.01f, 0.02f, 0.03f, 0.04f);
    float4 f1 = seed + (float4)(0.05f, 0.06f, 0.07f, 0.08f);
    float4 f2 = seed + (float4)(0.09f, 0.10f, 0.11f, 0.12f);
    float4 f3 = seed + (float4)(0.13f, 0.14f, 0.15f, 0.16f);
    float4 f4 = seed + (float4)(0.17f, 0.18f, 0.19f, 0.20f);
    float4 f5 = seed + (float4)(0.21f, 0.22f, 0.23f, 0.24f);
    float4 f6 = seed + (float4)(0.25f, 0.26f, 0.27f, 0.28f);
    float4 f7 = seed + (float4)(0.29f, 0.30f, 0.31f, 0.32f);

    float4 mf = (float4)(multiplier);
    float4 cf0 = (float4)(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 cf1 = (float4)(0.0005f, 0.0006f, 0.0007f, 0.0008f);
    float4 cf2 = (float4)(0.0009f, 0.0010f, 0.0011f, 0.0012f);
    float4 cf3 = (float4)(0.0013f, 0.0014f, 0.0015f, 0.0016f);
    float4 cf4 = (float4)(0.0017f, 0.0018f, 0.0019f, 0.0020f);
    float4 cf5 = (float4)(0.0021f, 0.0022f, 0.0023f, 0.0024f);
    float4 cf6 = (float4)(0.0025f, 0.0026f, 0.0027f, 0.0028f);
    float4 cf7 = (float4)(0.0029f, 0.0030f, 0.0031f, 0.0032f);

    uint4 u0 = (uint4)(ubase);
    uint4 u1 = (uint4)(ubase + 1u);
    uint4 u2 = (uint4)(ubase + 2u);
    uint4 u3 = (uint4)(ubase + 3u);
    uint4 u4 = (uint4)(ubase + 4u);
    uint4 u5 = (uint4)(ubase + 5u);
    uint4 u6 = (uint4)(ubase + 6u);
    uint4 u7 = (uint4)(ubase + 7u);

    uint4 mu = (uint4)((uint)multiplier);
    if (mu.x == 0u) mu = (uint4)(1u);

    uint4 cu0 = (uint4)(13u, 17u, 19u, 23u);
    uint4 cu1 = (uint4)(29u, 31u, 37u, 41u);
    uint4 cu2 = (uint4)(43u, 47u, 53u, 59u);
    uint4 cu3 = (uint4)(61u, 67u, 71u, 73u);
    uint4 cu4 = (uint4)(79u, 83u, 89u, 97u);
    uint4 cu5 = (uint4)(101u, 103u, 107u, 109u);
    uint4 cu6 = (uint4)(113u, 127u, 131u, 137u);
    uint4 cu7 = (uint4)(139u, 149u, 151u, 157u);

    for (int i = 0; i < 16384; ++i) {
        f0 = fma(f0, mf, cf0);
        u0 = (u0 + cu0) + 1u;

        f1 = fma(f1, mf, cf1);
        u1 = (u1 + cu1) + 1u;

        f2 = fma(f2, mf, cf2);
        u2 = (u2 + cu2) + 1u;

        f3 = fma(f3, mf, cf3);
        u3 = (u3 + cu3) + 1u;

        f4 = fma(f4, mf, cf4);
        u4 = (u4 + cu4) + 1u;

        f5 = fma(f5, mf, cf5);
        u5 = (u5 + cu5) + 1u;

        f6 = fma(f6, mf, cf6);
        u6 = (u6 + cu6) + 1u;

        f7 = fma(f7, mf, cf7);
        u7 = (u7 + cu7) + 1u;
    }

    float4 fres = (f0 + f1) + (f2 + f3) + (f4 + f5) + (f6 + f7);
    uint4 ures = (u0 + u1) + (u2 + u3) + (u4 + u5) + (u6 + u7);

    data[index] = fres.x + fres.y + fres.z + fres.w + (float)(ures.x ^ ures.y ^ ures.z ^ ures.w);
}

// Config 3: Standard INT32 (Single-Issue Baseline)
__kernel void run_dual_issue_int32_4(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    uint ubase = (uint)(in_val * 1000.0f) + index;

    uint4 u0  = (uint4)(ubase);
    uint4 u1  = (uint4)(ubase + 1u);
    uint4 u2  = (uint4)(ubase + 2u);
    uint4 u3  = (uint4)(ubase + 3u);

    uint4 cu0  = (uint4)(13u, 17u, 19u, 23u);
    uint4 cu1  = (uint4)(29u, 31u, 37u, 41u);
    uint4 cu2  = (uint4)(43u, 47u, 53u, 59u);
    uint4 cu3  = (uint4)(61u, 67u, 71u, 73u);

    uint4 mu = (uint4)((uint)multiplier);
    if (mu.x == 0u) mu = (uint4)(1u);

    // 16384 iterations * 4 uint4 full-rate INT32 ALU ops (add + xor = 2 ops each) = 16384 * 32 = 524,288 INT32 ops
    for (int i = 0; i < 16384; ++i) {
        u0  = (u0  + cu0) ^ mu;
        u1  = (u1  + cu1) ^ mu;
        u2  = (u2  + cu2) ^ mu;
        u3  = (u3  + cu3) ^ mu;
    }

    uint4 ures = (u0 + u1) + (u2 + u3);
    data[index] = (float)(ures.x ^ ures.y ^ ures.z ^ ures.w);
}

// Config 4: Dual-Issue INT32 (Partial Co-Issue)
__kernel void run_dual_issue_int32_8(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    uint ubase = (uint)(in_val * 1000.0f) + index;

    uint4 u0  = (uint4)(ubase);
    uint4 u1  = (uint4)(ubase + 1u);
    uint4 u2  = (uint4)(ubase + 2u);
    uint4 u3  = (uint4)(ubase + 3u);

    uint4 cu0  = (uint4)(13u, 17u, 19u, 23u);
    uint4 cu1  = (uint4)(29u, 31u, 37u, 41u);
    uint4 cu2  = (uint4)(43u, 47u, 53u, 59u);
    uint4 cu3  = (uint4)(61u, 67u, 71u, 73u);

    uint4 mu = (uint4)((uint)multiplier);
    if (mu.x == 0u) mu = (uint4)(1u);

    // 32768 iterations * 4 uint4 full-rate INT32 ALU ops (add + xor = 2 ops each) = 32768 * 32 = 1,048,576 INT32 ops
    for (int i = 0; i < 32768; ++i) {
        u0  = (u0  + cu0) ^ mu;
        u1  = (u1  + cu1) ^ mu;
        u2  = (u2  + cu2) ^ mu;
        u3  = (u3  + cu3) ^ mu;
    }

    uint4 ures = (u0 + u1) + (u2 + u3);
    data[index] = (float)(ures.x ^ ures.y ^ ures.z ^ ures.w);
}

// Config 5: Dual-Issue INT32 (INT32+INT32)
__kernel void run_dual_issue_int32(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    uint ubase = (uint)(in_val * 1000.0f) + index;

    uint4 u0  = (uint4)(ubase);
    uint4 u1  = (uint4)(ubase + 1u);
    uint4 u2  = (uint4)(ubase + 2u);
    uint4 u3  = (uint4)(ubase + 3u);

    uint4 cu0  = (uint4)(13u, 17u, 19u, 23u);
    uint4 cu1  = (uint4)(29u, 31u, 37u, 41u);
    uint4 cu2  = (uint4)(43u, 47u, 53u, 59u);
    uint4 cu3  = (uint4)(61u, 67u, 71u, 73u);

    uint4 mu = (uint4)((uint)multiplier);
    if (mu.x == 0u) mu = (uint4)(1u);

    // 65536 iterations * 4 uint4 full-rate INT32 ALU ops (add + xor = 2 ops each) = 65536 * 32 = 2,097,152 INT32 ops
    for (int i = 0; i < 65536; ++i) {
        u0  = (u0  + cu0) ^ mu;
        u1  = (u1  + cu1) ^ mu;
        u2  = (u2  + cu2) ^ mu;
        u3  = (u3  + cu3) ^ mu;
    }

    uint4 ures = (u0 + u1) + (u2 + u3);
    data[index] = (float)(ures.x ^ ures.y ^ ures.z ^ ures.w);
}
