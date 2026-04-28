// Requires ROCm 6.4+
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void run_benchmark(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Matrix benchmarking using RDNA3+ WMMA
    // RDNA3 features __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
    // A and B are fp16, C and D are fp32

    // Emulate 16x16x16 matrix math if we don't explicitly use the intrinsic,
    // but the intention is that compiler maps fma to wmma if organized properly, 
    // or we use the intrinsic directly if on gfx1100+
    
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    // RDNA3 WMMA Intrinsic
    // Note: Disabled for RDNA4 (gfx1200/1201) due to ROCm 7.1.1 backend crash.
    using float8 = __attribute__((ext_vector_type(8))) float;
    using half16 = __attribute__((ext_vector_type(16))) _Float16;
    
    half16 a;
    half16 b;
    float8 c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Initialize with dummy values
    for (int i = 0; i < 16; ++i) {
        a[i] = (_Float16)0.1f;
        b[i] = (_Float16)0.2f;
    }

    #pragma unroll 4
    for(int i = 0; i < 32768; i++) {
        c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
    
    data[idx] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#else
    // Fallback if not RDNA3/4
    float acc = 1.0f;
    for(int i=0; i<32768; i++) {
        acc += 0.001f;
    }
    data[idx] = acc;
#endif
}
