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

    float4 c = (float4)(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 m = (float4)(multiplier);

    // 32 vec4 FMAs × 4 components × 2 ops = 256 FP32 ops per iteration.
    for (int i = 0; i < 16384; ++i) {
        val1  = fma(val1,  m, c);
        val2  = fma(val2,  m, c);
        val3  = fma(val3,  m, c);
        val4  = fma(val4,  m, c);
        val5  = fma(val5,  m, c);
        val6  = fma(val6,  m, c);
        val7  = fma(val7,  m, c);
        val8  = fma(val8,  m, c);
        val9  = fma(val9,  m, c);
        val10 = fma(val10, m, c);
        val11 = fma(val11, m, c);
        val12 = fma(val12, m, c);
        val13 = fma(val13, m, c);
        val14 = fma(val14, m, c);
        val15 = fma(val15, m, c);
        val16 = fma(val16, m, c);
        val17 = fma(val17, m, c);
        val18 = fma(val18, m, c);
        val19 = fma(val19, m, c);
        val20 = fma(val20, m, c);
        val21 = fma(val21, m, c);
        val22 = fma(val22, m, c);
        val23 = fma(val23, m, c);
        val24 = fma(val24, m, c);
        val25 = fma(val25, m, c);
        val26 = fma(val26, m, c);
        val27 = fma(val27, m, c);
        val28 = fma(val28, m, c);
        val29 = fma(val29, m, c);
        val30 = fma(val30, m, c);
        val31 = fma(val31, m, c);
        val32 = fma(val32, m, c);
    }

    data[index] = val1.x + val2.y + val3.z + val4.w + val5.x + val6.y + val7.z + val8.w +
                  val9.x + val10.y + val11.z + val12.w + val13.x + val14.y + val15.z + val16.w +
                  val17.x + val18.y + val19.z + val20.w + val21.x + val22.y + val23.z + val24.w +
                  val25.x + val26.y + val27.z + val28.w + val29.x + val30.y + val31.z + val32.w;
}

