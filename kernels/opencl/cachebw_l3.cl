__kernel void run_benchmark(__global float4* data, __global uint* pc) {
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];

    uint workgroupOffset = get_group_id(0) * 8192;
    uint localId = get_local_id(0);
    
    uint baseIndex = workgroupOffset + (localId * 32);
    
    float4 sum = (float4)(0.0f);
    
    for (int iter = 0; iter < 200; iter++) {
        for (int i = 0; i < 32; i++) {
            float4 v = data[(baseIndex + i) & 0xFFFFF];
            sum += v;
        }
    }
    
    data[baseIndex & 0xFFFFF] = sum;
}
