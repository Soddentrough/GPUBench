// Requires OpenCL 1.2+

__kernel void run_benchmark(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float4 val1  = (float4)(data[index]);
    float4 val2  = (float4)(0.10f, 0.11f, 0.12f, 0.13f);
    float4 val3  = (float4)(0.20f, 0.21f, 0.22f, 0.23f);
    float4 val4  = (float4)(0.30f, 0.31f, 0.32f, 0.33f);
    float4 val5  = (float4)(0.40f, 0.41f, 0.42f, 0.43f);
    float4 val6  = (float4)(0.50f, 0.51f, 0.52f, 0.53f);
    float4 val7  = (float4)(0.60f, 0.61f, 0.62f, 0.63f);
    float4 val8  = (float4)(0.70f, 0.71f, 0.72f, 0.73f);
    float4 val9  = (float4)(0.80f, 0.81f, 0.82f, 0.83f);
    float4 val10 = (float4)(0.90f, 0.91f, 0.92f, 0.93f);
    float4 val11 = (float4)(1.00f, 1.01f, 1.02f, 1.03f);
    float4 val12 = (float4)(1.10f, 1.11f, 1.12f, 1.13f);
    float4 val13 = (float4)(1.20f, 1.21f, 1.22f, 1.23f);
    float4 val14 = (float4)(1.30f, 1.31f, 1.32f, 1.33f);
    float4 val15 = (float4)(1.40f, 1.41f, 1.42f, 1.43f);
    float4 val16 = (float4)(1.50f, 1.51f, 1.52f, 1.53f);
    float4 val17 = (float4)(1.60f, 1.61f, 1.62f, 1.63f);
    float4 val18 = (float4)(1.70f, 1.71f, 1.72f, 1.73f);
    float4 val19 = (float4)(1.80f, 1.81f, 1.82f, 1.83f);
    float4 val20 = (float4)(1.90f, 1.91f, 1.92f, 1.93f);
    float4 val21 = (float4)(2.00f, 2.01f, 2.02f, 2.03f);
    float4 val22 = (float4)(2.10f, 2.11f, 2.12f, 2.13f);
    float4 val23 = (float4)(2.20f, 2.21f, 2.22f, 2.23f);
    float4 val24 = (float4)(2.30f, 2.31f, 2.32f, 2.33f);
    float4 val25 = (float4)(2.40f, 2.41f, 2.42f, 2.43f);
    float4 val26 = (float4)(2.50f, 2.51f, 2.52f, 2.53f);
    float4 val27 = (float4)(2.60f, 2.61f, 2.62f, 2.63f);
    float4 val28 = (float4)(2.70f, 2.71f, 2.72f, 2.73f);
    float4 val29 = (float4)(2.80f, 2.81f, 2.82f, 2.83f);
    float4 val30 = (float4)(2.90f, 2.91f, 2.92f, 2.93f);
    float4 val31 = (float4)(3.00f, 3.01f, 3.02f, 3.03f);
    float4 val32 = (float4)(3.10f, 3.11f, 3.12f, 3.13f);

    float4 m = (float4)(multiplier);

    // 32 vec4 FMAs × 4 components × 2 ops = 256 FP32 ops per iteration.
    for (int i = 0; i < 16384; ++i) {
        val1  = fma(val1,  m, val2);
        val2  = fma(val2,  m, val3);
        val3  = fma(val3,  m, val4);
        val4  = fma(val4,  m, val5);
        val5  = fma(val5,  m, val6);
        val6  = fma(val6,  m, val7);
        val7  = fma(val7,  m, val8);
        val8  = fma(val8,  m, val9);
        val9  = fma(val9,  m, val10);
        val10 = fma(val10, m, val11);
        val11 = fma(val11, m, val12);
        val12 = fma(val12, m, val13);
        val13 = fma(val13, m, val14);
        val14 = fma(val14, m, val15);
        val15 = fma(val15, m, val16);
        val16 = fma(val16, m, val17);
        val17 = fma(val17, m, val18);
        val18 = fma(val18, m, val19);
        val19 = fma(val19, m, val20);
        val20 = fma(val20, m, val21);
        val21 = fma(val21, m, val22);
        val22 = fma(val22, m, val23);
        val23 = fma(val23, m, val24);
        val24 = fma(val24, m, val25);
        val25 = fma(val25, m, val26);
        val26 = fma(val26, m, val27);
        val27 = fma(val27, m, val28);
        val28 = fma(val28, m, val29);
        val29 = fma(val29, m, val30);
        val30 = fma(val30, m, val31);
        val31 = fma(val31, m, val32);
        val32 = fma(val32, m, val1);
    }

    data[index] = val1.x + val2.y + val3.z + val4.w + val5.x + val6.y + val7.z + val8.w +
                  val9.x + val10.y + val11.z + val12.w + val13.x + val14.y + val15.z + val16.w +
                  val17.x + val18.y + val19.z + val20.w + val21.x + val22.y + val23.z + val24.w +
                  val25.x + val26.y + val27.z + val28.w + val29.x + val30.y + val31.z + val32.w;
}

