// Requires OpenCL 1.2+

__kernel void run_benchmark(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    // 4 fully independent accumulators — no cross-reads between them.
    // This matches the ROCm design that achieves spec throughput.
    // 4 vec4 FMAs × 4 components × 2 ops = 32 FP32 ops per iteration.
    float4 m   = (float4)(multiplier);
    float4 v0  = (float4)(data[index]);
    float4 v1  = (float4)(0.10f, 0.11f, 0.12f, 0.13f);
    float4 v2  = (float4)(0.20f, 0.21f, 0.22f, 0.23f);
    float4 v3  = (float4)(0.30f, 0.31f, 0.32f, 0.33f);

    for (int i = 0; i < 16384; ++i) {
        v0 = fma(v0, m, (float4)(0.001f));
        v1 = fma(v1, m, (float4)(0.002f));
        v2 = fma(v2, m, (float4)(0.003f));
        v3 = fma(v3, m, (float4)(0.004f));
    }

    data[index] = v0.x + v1.y + v2.z + v3.w;
}
