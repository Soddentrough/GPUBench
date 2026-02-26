// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float4* data, uint* pc) {
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];

    uint globalId = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int iter = 0; iter < 200; iter++) {
        for (int i = 0; i < 32; i++) {
            // Strided access: globalId + (i + iter) * 65537
            int idx = (globalId + (i + iter) * 65537) & mask;
            
            // Standard load
            float4 v = data[idx];
            
            // Compiler barrier to prevent optimization (hoisting/DCE)
            asm volatile("" ::: "memory");
            
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
    }
    
    data[globalId & mask] = sum;
}
