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
    
#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef _Float16 half8_v __attribute__((vector_size(16)));
    typedef float float8_v __attribute__((vector_size(32)));
    
    half8_v a = {(_Float16)0.1f, (_Float16)0.1f, (_Float16)0.1f, (_Float16)0.1f,
                 (_Float16)0.1f, (_Float16)0.1f, (_Float16)0.1f, (_Float16)0.1f};
    half8_v b = {(_Float16)0.2f, (_Float16)0.2f, (_Float16)0.2f, (_Float16)0.2f,
                 (_Float16)0.2f, (_Float16)0.2f, (_Float16)0.2f, (_Float16)0.2f};
    
    float8_v c0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c1 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c2 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c3 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c4 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c5 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c6 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c7 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c7);
    }
    
    float8_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    data[idx % 524288] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    // RDNA3 WMMA Intrinsic
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
    
    data[idx % 524288] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#else
    data[idx % 524288] = 0.0f;
#endif
}
