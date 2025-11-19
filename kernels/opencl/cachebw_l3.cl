__kernel void run_benchmark(__global volatile float4* data) {
    uint workgroupOffset = get_group_id(0) * 8192;
    uint localId = get_local_id(0);
    
    uint baseIndex = (workgroupOffset + (localId * 32)) & 0xFFFFF;
    
    float4 sum = (float4)(0.0f);
    
    for (int iter = 0; iter < 200; iter++) {
        for (int i = 0; i < 32; i++) {
            float4 v = data[baseIndex + i];
            sum += v;
        }
    }
    
    data[baseIndex] = sum;
}
