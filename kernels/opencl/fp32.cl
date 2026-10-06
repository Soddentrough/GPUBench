// Requires OpenCL 1.2+

__kernel void run_benchmark(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    // Load initial seed from memory to ensure compiler cannot dead-code eliminate
    float in_val = data[index & 0x1FFFu];
    float4 seed = (float4)(in_val * 0.0001f);

    float4 val1  = seed + (float4)(0.01f, 0.02f, 0.03f, 0.04f);
    float4 val2  = seed + (float4)(0.05f, 0.06f, 0.07f, 0.08f);
    float4 val3  = seed + (float4)(0.09f, 0.10f, 0.11f, 0.12f);
    float4 val4  = seed + (float4)(0.13f, 0.14f, 0.15f, 0.16f);
    float4 val5  = seed + (float4)(0.17f, 0.18f, 0.19f, 0.20f);
    float4 val6  = seed + (float4)(0.21f, 0.22f, 0.23f, 0.24f);
    float4 val7  = seed + (float4)(0.25f, 0.26f, 0.27f, 0.28f);
    float4 val8  = seed + (float4)(0.29f, 0.30f, 0.31f, 0.32f);
    float4 val9  = seed + (float4)(0.33f, 0.34f, 0.35f, 0.36f);
    float4 val10 = seed + (float4)(0.37f, 0.38f, 0.39f, 0.40f);
    float4 val11 = seed + (float4)(0.41f, 0.42f, 0.43f, 0.44f);
    float4 val12 = seed + (float4)(0.45f, 0.46f, 0.47f, 0.48f);
    float4 val13 = seed + (float4)(0.49f, 0.50f, 0.51f, 0.52f);
    float4 val14 = seed + (float4)(0.53f, 0.54f, 0.55f, 0.56f);
    float4 val15 = seed + (float4)(0.57f, 0.58f, 0.59f, 0.60f);
    float4 val16 = seed + (float4)(0.61f, 0.62f, 0.63f, 0.64f);
    float4 val17 = seed + (float4)(0.65f, 0.66f, 0.67f, 0.68f);
    float4 val18 = seed + (float4)(0.69f, 0.70f, 0.71f, 0.72f);
    float4 val19 = seed + (float4)(0.73f, 0.74f, 0.75f, 0.76f);
    float4 val20 = seed + (float4)(0.77f, 0.78f, 0.79f, 0.80f);
    float4 val21 = seed + (float4)(0.81f, 0.82f, 0.83f, 0.84f);
    float4 val22 = seed + (float4)(0.85f, 0.86f, 0.87f, 0.88f);
    float4 val23 = seed + (float4)(0.89f, 0.90f, 0.91f, 0.92f);
    float4 val24 = seed + (float4)(0.93f, 0.94f, 0.95f, 0.96f);
    float4 val25 = seed + (float4)(0.97f, 0.98f, 0.99f, 1.00f);
    float4 val26 = seed + (float4)(1.01f, 1.02f, 1.03f, 1.04f);
    float4 val27 = seed + (float4)(1.05f, 1.06f, 1.07f, 1.08f);
    float4 val28 = seed + (float4)(1.09f, 1.10f, 1.11f, 1.12f);
    float4 val29 = seed + (float4)(1.13f, 1.14f, 1.15f, 1.16f);
    float4 val30 = seed + (float4)(1.17f, 1.18f, 1.19f, 1.20f);
    float4 val31 = seed + (float4)(1.21f, 1.22f, 1.23f, 1.24f);
    float4 val32 = seed + (float4)(1.25f, 1.26f, 1.27f, 1.28f);

    float4 m = (float4)(multiplier);

    // 32 vec4 FMAs × 4 components × 2 ops = 256 FP32 ops per iteration.
    // 16384 iters * 256 ops = 4,194,304 ops per thread (matching Vulkan and ROCm)
    for (int i = 0; i < 16384; ++i) {
        val1  = fma(m, val2,  val1);
        val2  = fma(m, val3,  val2);
        val3  = fma(m, val4,  val3);
        val4  = fma(m, val5,  val4);
        val5  = fma(m, val6,  val5);
        val6  = fma(m, val7,  val6);
        val7  = fma(m, val8,  val7);
        val8  = fma(m, val9,  val8);
        val9  = fma(m, val10, val9);
        val10 = fma(m, val11, val10);
        val11 = fma(m, val12, val11);
        val12 = fma(m, val13, val12);
        val13 = fma(m, val14, val13);
        val14 = fma(m, val15, val14);
        val15 = fma(m, val16, val15);
        val16 = fma(m, val17, val16);
        val17 = fma(m, val18, val17);
        val18 = fma(m, val19, val18);
        val19 = fma(m, val20, val19);
        val20 = fma(m, val21, val20);
        val21 = fma(m, val22, val21);
        val22 = fma(m, val23, val22);
        val23 = fma(m, val24, val23);
        val24 = fma(m, val25, val24);
        val25 = fma(m, val26, val25);
        val26 = fma(m, val27, val26);
        val27 = fma(m, val28, val27);
        val28 = fma(m, val29, val28);
        val29 = fma(m, val30, val29);
        val30 = fma(m, val31, val30);
        val31 = fma(m, val32, val31);
        val32 = fma(m, val1,  val32);
    }

    float4 sum = ((val1 + val2) + (val3 + val4)) + ((val5 + val6) + (val7 + val8)) +
                 ((val9 + val10) + (val11 + val12)) + ((val13 + val14) + (val15 + val16)) +
                 ((val17 + val18) + (val19 + val20)) + ((val21 + val22) + (val23 + val24)) +
                 ((val25 + val26) + (val27 + val28)) + ((val29 + val30) + (val31 + val32));

    data[index] = sum.x + sum.y + sum.z + sum.w;
}
