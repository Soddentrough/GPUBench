// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void compute(float4* data, uint iterations) {
    uint workgroupOffset = blockIdx.x * 2;
    uint localId = threadIdx.x;
    
    uint baseIndex = workgroupOffset + (localId % 2);
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int iter = 0; iter < iterations; iter++) {
        float4 v0 = data[baseIndex];
        sum.x += v0.x;
        sum.y += v0.y;
        sum.z += v0.z;
        sum.w += v0.w;
    }
    
    data[baseIndex] = sum;
}
