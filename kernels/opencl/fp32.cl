// Requires OpenCL 1.2+

__kernel void run_benchmark(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    // Work with multiple accumulators to avoid dependency chains
    float4 val1 = (float4)(data[index]);
    float4 val2 = (float4)(0.1f, 0.2f, 0.3f, 0.4f);
    float4 val3 = (float4)(0.5f, 0.6f, 0.7f, 0.8f);
    float4 val4 = (float4)(0.9f, 1.0f, 1.1f, 1.2f);
    
    // Each iteration performs 32 vec4 FMAs = 32 * 4 * 2 = 256 FP32 ops
    for (int i = 0; i < 16384; ++i) {
        val1 = fma(val1, (float4)(1.0001f), val2);
        val2 = fma(val2, (float4)(1.0001f), val3);
        val3 = fma(val3, (float4)(1.0001f), val4);
        val4 = fma(val4, (float4)(1.0001f), val1);

        val1 = fma(val1, (float4)(1.0001f), val2);
        val2 = fma(val2, (float4)(1.0001f), val3);
        val3 = fma(val3, (float4)(1.0001f), val4);
        val4 = fma(val4, (float4)(1.0001f), val1);

        val1 = fma(val1, (float4)(1.0001f), val2);
        val2 = fma(val2, (float4)(1.0001f), val3);
        val3 = fma(val3, (float4)(1.0001f), val4);
        val4 = fma(val4, (float4)(1.0001f), val1);

        val1 = fma(val1, (float4)(1.0001f), val2);
        val2 = fma(val2, (float4)(1.0001f), val3);
        val3 = fma(val3, (float4)(1.0001f), val4);
        val4 = fma(val4, (float4)(1.0001f), val1);

        val1 = fma(val1, (float4)(1.0001f), val2);
        val2 = fma(val2, (float4)(1.0001f), val3);
        val3 = fma(val3, (float4)(1.0001f), val4);
        val4 = fma(val4, (float4)(1.0001f), val1);

        val1 = fma(val1, (float4)(1.0001f), val2);
        val2 = fma(val2, (float4)(1.0001f), val3);
        val3 = fma(val3, (float4)(1.0001f), val4);
        val4 = fma(val4, (float4)(1.0001f), val1);

        val1 = fma(val1, (float4)(1.0001f), val2);
        val2 = fma(val2, (float4)(1.0001f), val3);
        val3 = fma(val3, (float4)(1.0001f), val4);
        val4 = fma(val4, (float4)(1.0001f), val1);

        val1 = fma(val1, (float4)(1.0001f), val2);
        val2 = fma(val2, (float4)(1.0001f), val3);
        val3 = fma(val3, (float4)(1.0001f), val4);
        val4 = fma(val4, (float4)(1.0001f), val1);
    }
    
    // Prevent optimization away
    data[index] = val1.x + val2.y + val3.z + val4.w;
}
