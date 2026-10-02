// Requires OpenCL 1.2+
// GPUBench Dual-Issue & Concurrency Benchmark Suite

__kernel void run_dual_issue_ilp1(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    float4 val = (float4)(in_val * 0.0001f);
    float4 m = (float4)(multiplier);
    float4 c = (float4)(0.0001f, 0.0002f, 0.0003f, 0.0004f);

    for (int i = 0; i < 32768; ++i) {
        val = fma(val, m, c);
    }

    data[index] = val.x + val.y + val.z + val.w;
}

__kernel void run_dual_issue_ilp4(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    float4 seed = (float4)(in_val * 0.0001f);
    float4 val0 = seed + (float4)(0.01f, 0.02f, 0.03f, 0.04f);
    float4 val1 = seed + (float4)(0.05f, 0.06f, 0.07f, 0.08f);
    float4 val2 = seed + (float4)(0.09f, 0.10f, 0.11f, 0.12f);
    float4 val3 = seed + (float4)(0.13f, 0.14f, 0.15f, 0.16f);

    float4 m = (float4)(multiplier);
    float4 c0 = (float4)(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 c1 = (float4)(0.0005f, 0.0006f, 0.0007f, 0.0008f);
    float4 c2 = (float4)(0.0009f, 0.0010f, 0.0011f, 0.0012f);
    float4 c3 = (float4)(0.0013f, 0.0014f, 0.0015f, 0.0016f);

    for (int i = 0; i < 16384; ++i) {
        val0 = fma(val0, m, c0);
        val1 = fma(val1, m, c1);
        val2 = fma(val2, m, c2);
        val3 = fma(val3, m, c3);
    }

    float4 res = val0 + val1 + val2 + val3;
    data[index] = res.x + res.y + res.z + res.w;
}

__kernel void run_dual_issue_ilp8(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    float4 seed = (float4)(in_val * 0.0001f);
    float4 val0 = seed + (float4)(0.01f, 0.02f, 0.03f, 0.04f);
    float4 val1 = seed + (float4)(0.05f, 0.06f, 0.07f, 0.08f);
    float4 val2 = seed + (float4)(0.09f, 0.10f, 0.11f, 0.12f);
    float4 val3 = seed + (float4)(0.13f, 0.14f, 0.15f, 0.16f);
    float4 val4 = seed + (float4)(0.17f, 0.18f, 0.19f, 0.20f);
    float4 val5 = seed + (float4)(0.21f, 0.22f, 0.23f, 0.24f);
    float4 val6 = seed + (float4)(0.25f, 0.26f, 0.27f, 0.28f);
    float4 val7 = seed + (float4)(0.29f, 0.30f, 0.31f, 0.32f);

    float4 m = (float4)(multiplier);
    float4 c0 = (float4)(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 c1 = (float4)(0.0005f, 0.0006f, 0.0007f, 0.0008f);
    float4 c2 = (float4)(0.0009f, 0.0010f, 0.0011f, 0.0012f);
    float4 c3 = (float4)(0.0013f, 0.0014f, 0.0015f, 0.0016f);
    float4 c4 = (float4)(0.0017f, 0.0018f, 0.0019f, 0.0020f);
    float4 c5 = (float4)(0.0021f, 0.0022f, 0.0023f, 0.0024f);
    float4 c6 = (float4)(0.0025f, 0.0026f, 0.0027f, 0.0028f);
    float4 c7 = (float4)(0.0029f, 0.0030f, 0.0031f, 0.0032f);

    for (int i = 0; i < 16384; ++i) {
        val0 = fma(val0, m, c0);
        val1 = fma(val1, m, c1);
        val2 = fma(val2, m, c2);
        val3 = fma(val3, m, c3);
        val4 = fma(val4, m, c4);
        val5 = fma(val5, m, c5);
        val6 = fma(val6, m, c6);
        val7 = fma(val7, m, c7);
    }

    float4 res = (val0 + val1) + (val2 + val3) + (val4 + val5) + (val6 + val7);
    data[index] = res.x + res.y + res.z + res.w;
}

__kernel void run_dual_issue_ilp16(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    float4 seed = (float4)(in_val * 0.0001f);
    float4 val0  = seed + (float4)(0.01f, 0.02f, 0.03f, 0.04f);
    float4 val1  = seed + (float4)(0.05f, 0.06f, 0.07f, 0.08f);
    float4 val2  = seed + (float4)(0.09f, 0.10f, 0.11f, 0.12f);
    float4 val3  = seed + (float4)(0.13f, 0.14f, 0.15f, 0.16f);
    float4 val4  = seed + (float4)(0.17f, 0.18f, 0.19f, 0.20f);
    float4 val5  = seed + (float4)(0.21f, 0.22f, 0.23f, 0.24f);
    float4 val6  = seed + (float4)(0.25f, 0.26f, 0.27f, 0.28f);
    float4 val7  = seed + (float4)(0.29f, 0.30f, 0.31f, 0.32f);
    float4 val8  = seed + (float4)(0.33f, 0.34f, 0.35f, 0.36f);
    float4 val9  = seed + (float4)(0.37f, 0.38f, 0.39f, 0.40f);
    float4 val10 = seed + (float4)(0.41f, 0.42f, 0.43f, 0.44f);
    float4 val11 = seed + (float4)(0.45f, 0.46f, 0.47f, 0.48f);
    float4 val12 = seed + (float4)(0.49f, 0.50f, 0.51f, 0.52f);
    float4 val13 = seed + (float4)(0.53f, 0.54f, 0.55f, 0.56f);
    float4 val14 = seed + (float4)(0.57f, 0.58f, 0.59f, 0.60f);
    float4 val15 = seed + (float4)(0.61f, 0.62f, 0.63f, 0.64f);

    float4 m = (float4)(multiplier);
    float4 c0  = (float4)(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 c1  = (float4)(0.0005f, 0.0006f, 0.0007f, 0.0008f);
    float4 c2  = (float4)(0.0009f, 0.0010f, 0.0011f, 0.0012f);
    float4 c3  = (float4)(0.0013f, 0.0014f, 0.0015f, 0.0016f);
    float4 c4  = (float4)(0.0017f, 0.0018f, 0.0019f, 0.0020f);
    float4 c5  = (float4)(0.0021f, 0.0022f, 0.0023f, 0.0024f);
    float4 c6  = (float4)(0.0025f, 0.0026f, 0.0027f, 0.0028f);
    float4 c7  = (float4)(0.0029f, 0.0030f, 0.0031f, 0.0032f);
    float4 c8  = (float4)(0.0033f, 0.0034f, 0.0035f, 0.0036f);
    float4 c9  = (float4)(0.0037f, 0.0038f, 0.0039f, 0.0040f);
    float4 c10 = (float4)(0.0041f, 0.0042f, 0.0043f, 0.0044f);
    float4 c11 = (float4)(0.0045f, 0.0046f, 0.0047f, 0.0048f);
    float4 c12 = (float4)(0.0049f, 0.0050f, 0.51f, 0.0052f);
    float4 c13 = (float4)(0.0053f, 0.0054f, 0.0055f, 0.0056f);
    float4 c14 = (float4)(0.0057f, 0.0058f, 0.0059f, 0.0060f);
    float4 c15 = (float4)(0.0061f, 0.0062f, 0.0063f, 0.0064f);

    for (int i = 0; i < 16384; ++i) {
        val0  = fma(val0,  m, c0);
        val1  = fma(val1,  m, c1);
        val2  = fma(val2,  m, c2);
        val3  = fma(val3,  m, c3);
        val4  = fma(val4,  m, c4);
        val5  = fma(val5,  m, c5);
        val6  = fma(val6,  m, c6);
        val7  = fma(val7,  m, c7);
        val8  = fma(val8,  m, c8);
        val9  = fma(val9,  m, c9);
        val10 = fma(val10, m, c10);
        val11 = fma(val11, m, c11);
        val12 = fma(val12, m, c12);
        val13 = fma(val13, m, c13);
        val14 = fma(val14, m, c14);
        val15 = fma(val15, m, c15);
    }

    float4 s0 = (val0 + val1) + (val2 + val3);
    float4 s1 = (val4 + val5) + (val6 + val7);
    float4 s2 = (val8 + val9) + (val10 + val11);
    float4 s3 = (val12 + val13) + (val14 + val15);
    float4 res = (s0 + s1) + (s2 + s3);

    data[index] = res.x + res.y + res.z + res.w;
}

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
        u0 = (u0 * mu) + cu0;

        f1 = fma(f1, mf, cf1);
        u1 = (u1 * mu) + cu1;

        f2 = fma(f2, mf, cf2);
        u2 = (u2 * mu) + cu2;

        f3 = fma(f3, mf, cf3);
        u3 = (u3 * mu) + cu3;

        f4 = fma(f4, mf, cf4);
        u4 = (u4 * mu) + cu4;

        f5 = fma(f5, mf, cf5);
        u5 = (u5 * mu) + cu5;

        f6 = fma(f6, mf, cf6);
        u6 = (u6 * mu) + cu6;

        f7 = fma(f7, mf, cf7);
        u7 = (u7 * mu) + cu7;
    }

    float4 fres = (f0 + f1) + (f2 + f3) + (f4 + f5) + (f6 + f7);
    uint4 ures = (u0 + u1) + (u2 + u3) + (u4 + u5) + (u6 + u7);

    data[index] = fres.x + fres.y + fres.z + fres.w + (float)(ures.x ^ ures.y ^ ures.z ^ ures.w);
}

__kernel void run_dual_issue_int32(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float in_val = data[index & 0x1FFFu];
    uint ubase = (uint)(in_val * 1000.0f) + index;

    uint4 u0  = (uint4)(ubase);
    uint4 u1  = (uint4)(ubase + 1u);
    uint4 u2  = (uint4)(ubase + 2u);
    uint4 u3  = (uint4)(ubase + 3u);
    uint4 u4  = (uint4)(ubase + 4u);
    uint4 u5  = (uint4)(ubase + 5u);
    uint4 u6  = (uint4)(ubase + 6u);
    uint4 u7  = (uint4)(ubase + 7u);
    uint4 u8  = (uint4)(ubase + 8u);
    uint4 u9  = (uint4)(ubase + 9u);
    uint4 u10 = (uint4)(ubase + 10u);
    uint4 u11 = (uint4)(ubase + 11u);
    uint4 u12 = (uint4)(ubase + 12u);
    uint4 u13 = (uint4)(ubase + 13u);
    uint4 u14 = (uint4)(ubase + 14u);
    uint4 u15 = (uint4)(ubase + 15u);

    uint4 mu = (uint4)((uint)multiplier);
    if (mu.x == 0u) mu = (uint4)(1u);

    uint4 cu0  = (uint4)(13u, 17u, 19u, 23u);
    uint4 cu1  = (uint4)(29u, 31u, 37u, 41u);
    uint4 cu2  = (uint4)(43u, 47u, 53u, 59u);
    uint4 cu3  = (uint4)(61u, 67u, 71u, 73u);
    uint4 cu4  = (uint4)(79u, 83u, 89u, 97u);
    uint4 cu5  = (uint4)(101u, 103u, 107u, 109u);
    uint4 cu6  = (uint4)(113u, 127u, 131u, 137u);
    uint4 cu7  = (uint4)(139u, 149u, 151u, 157u);
    uint4 cu8  = (uint4)(163u, 167u, 173u, 179u);
    uint4 cu9  = (uint4)(181u, 191u, 193u, 197u);
    uint4 cu10 = (uint4)(199u, 211u, 223u, 227u);
    uint4 cu11 = (uint4)(229u, 233u, 239u, 241u);
    uint4 cu12 = (uint4)(251u, 257u, 263u, 269u);
    uint4 cu13 = (uint4)(271u, 277u, 281u, 283u);
    uint4 cu14 = (uint4)(293u, 307u, 311u, 313u);
    uint4 cu15 = (uint4)(317u, 331u, 337u, 347u);

    for (int i = 0; i < 16384; ++i) {
        u0  = (u0  * mu) + cu0;
        u1  = (u1  * mu) + cu1;
        u2  = (u2  * mu) + cu2;
        u3  = (u3  * mu) + cu3;
        u4  = (u4  * mu) + cu4;
        u5  = (u5  * mu) + cu5;
        u6  = (u6  * mu) + cu6;
        u7  = (u7  * mu) + cu7;
        u8  = (u8  * mu) + cu8;
        u9  = (u9  * mu) + cu9;
        u10 = (u10 * mu) + cu10;
        u11 = (u11 * mu) + cu11;
        u12 = (u12 * mu) + cu12;
        u13 = (u13 * mu) + cu13;
        u14 = (u14 * mu) + cu14;
        u15 = (u15 * mu) + cu15;
    }

    uint4 s0 = (u0 + u1) + (u2 + u3);
    uint4 s1 = (u4 + u5) + (u6 + u7);
    uint4 s2 = (u8 + u9) + (u10 + u11);
    uint4 s3 = (u12 + u13) + (u14 + u15);
    uint4 ures = (s0 + s1) + (s2 + s3);

    data[index] = (float)(ures.x ^ ures.y ^ ures.z ^ ures.w);
}
