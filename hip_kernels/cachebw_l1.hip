// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float4* data) {
    uint workgroupOffset = blockIdx.x * 2;
    uint localId = threadIdx.x;
    
    // The buffer has 1536 float4s. Mask is 1535 (0x5FF).
    uint baseIndex = (workgroupOffset + (localId % 2)) & 1535;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    #pragma unroll
    for (int iter = 0; iter < 1024; iter++) {
        float4 v0 = data[baseIndex];
        sum.x += v0.x;
        sum.y += v0.y;
        sum.z += v0.z;
        sum.w += v0.w;
    }
    
    data[baseIndex] = sum;
}
