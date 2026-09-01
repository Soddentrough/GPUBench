#include <hip/hip_runtime.h>

// Experimental GFX12 FP8 Matrix Kernel
// Trying to use RDNA3/4 WMMA intrinsic.
// Note: RDNA3 (gfx11) does not support fp8 wmma. RDNA4 (gfx12) does.
// The intrinsic name is a guess based on LLVM conventions.

extern "C" __global__ void run_benchmark(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 16x16x16 Matrix Multiply
    // Registers for A (16x16 fp8), B (16x16 fp8), C (16x16 fp32) -> D (16x16 fp32)
    // RDNA WMMA uses vector registers.
    // A: 16x16 fp8 elements. 256 elements. Packed into VGPRs?
    // On RDNA3, wmma_f16 takes vector args.
    
    // Using a placeholder loop if intrinsic fails, but we want to fail compilation if intrinsic is missing
    // to prove the point.
    
// Speculative intrinsic implementation for RDNA4 (gfx12)
// Disabled due to ROCm 7.1.1 backend failure: "Cannot select: intrinsic %llvm.amdgcn.wmma.f32.16x16x16.f16"
#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef int int2_v __attribute__((vector_size(8)));
    typedef float float8_v __attribute__((vector_size(32)));

    int2_v a = {0x3C3C3C3C, 0x3C3C3C3C};
    int2_v b = {0x3C3C3C3C, 0x3C3C3C3C};

    float8_v c0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c1 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c2 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c3 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c4 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c5 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c6 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_v c7 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c7);
    }

    float8_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    int safe_idx = idx % 524288;
    data[safe_idx] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#else
    data[idx % 524288] = 0.0f;
#endif
}
