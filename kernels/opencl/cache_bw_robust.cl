__kernel void run_benchmark(__global float4* data, __global uint* pc) {
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];

    uint globalId = get_global_id(0);
    uint idx = (globalId * stride) & mask;
    
    float4 sum0 = (float4)(0.0f);
    float4 sum1 = (float4)(0.0f);
    float4 sum2 = (float4)(0.0f);
    float4 sum3 = (float4)(0.0f);
    float4 sum4 = (float4)(0.0f);
    float4 sum5 = (float4)(0.0f);
    float4 sum6 = (float4)(0.0f);
    float4 sum7 = (float4)(0.0f);

    for (uint i = 0; i < iterations; i++) {
        uint base = (idx + i * stride * 8) & mask;
        
        sum0 += data[(base + 0 * stride) & mask];
        sum1 += data[(base + 1 * stride) & mask];
        sum2 += data[(base + 2 * stride) & mask];
        sum3 += data[(base + 3 * stride) & mask];
        sum4 += data[(base + 4 * stride) & mask];
        sum5 += data[(base + 5 * stride) & mask];
        sum6 += data[(base + 6 * stride) & mask];
        sum7 += data[(base + 7 * stride) & mask];
    }

    float4 final_sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

    // Force the compiler to keep the loop by adding a condition it can't evaluate statically.
    // final_sum.x will almost never equal mask + 0.5f, but the compiler doesn't know.
    if (final_sum.x == (float)mask + 0.5f) {
        data[globalId & mask] = final_sum;
    }
}
