__kernel void run_benchmark(__global float4* data, __global uint* pc) {
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];

    uint workgroupOffset = get_group_id(0) * 2;
    uint localId = get_local_id(0);
    
    uint baseIndex = workgroupOffset + (localId % 2);
    
    float4 sum = (float4)(0.0f);
    
    for (int iter = 0; iter < 1024; iter++) {
        float4 v0 = data[baseIndex];
        sum += v0;
    }
    
    data[baseIndex] = sum;
}
