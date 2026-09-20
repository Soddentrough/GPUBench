// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Matrix benchmarking using RDNA3+ WMMA
    // RDNA3/4 features __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32
    // A and B are bf16, C and D are fp32
    
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    using float8 = __attribute__((ext_vector_type(8))) float;
    using short16 = __attribute__((ext_vector_type(16))) short; // bf16 is 16-bit
    
    short16 a;
    short16 b;
    float8 c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Initialize with dummy values
    for (int i = 0; i < 16; ++i) {
        a[i] = 0x3DCC; // ~0.1
        b[i] = 0x3E4C; // ~0.2
    }

    #pragma unroll 4
    for(int i = 0; i < 32768; i++) {
        c = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    }
    
    data[idx] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#else
    // Disabled for RDNA4 (gfx1200/1201) due to ROCm 7.1.1 backend crash.
    data[idx] = 0.0f;
#endif
}
