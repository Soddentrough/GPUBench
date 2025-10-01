// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void compute(float4* data, uint iterations) {
    uint workgroupOffset = blockIdx.x * 8192;
    uint localId = threadIdx.x;
    
    uint baseIndex = workgroupOffset + (localId * 32);
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 32; i++) {
            float4 v = data[baseIndex + i];
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
    }
    
    data[baseIndex] = sum;
}
