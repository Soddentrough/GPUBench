// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float4* data) {
    uint workgroupOffset = blockIdx.x * 8192;
    uint localId = threadIdx.x;
    
    // The buffer has 1M float4s. Mask is 1048575 (0xFFFFF).
    uint baseIndex = workgroupOffset + (localId * 32);
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    #pragma unroll
    for (int iter = 0; iter < 200; iter++) {
        for (int i = 0; i < 32; i++) {
            float4 v = data[(baseIndex + i) & 1048575];
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
    }
    
    data[baseIndex & 1048575] = sum;
}
