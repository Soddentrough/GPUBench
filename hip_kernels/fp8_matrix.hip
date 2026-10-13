#include <hip/hip_runtime.h>

// Experimental GFX12 FP8 Matrix Kernel
// Trying to use RDNA3/4 WMMA intrinsic.
// Note: RDNA3 (gfx11) does not support fp8 wmma. RDNA4 (gfx12) does.
// The intrinsic name is a guess based on LLVM conventions.

extern "C" __global__ void run_benchmark(float* data) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x & 31u;

#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef int int2_v __attribute__((vector_size(8)));
    typedef float float8_v __attribute__((vector_size(32)));

    const int2_v* iptr = reinterpret_cast<const int2_v*>(data);
    int2_v a = iptr[lane];
    int2_v b = iptr[lane + 32u];

    float seed = static_cast<float>((a[0] & 0xFF) + 1) * 0.001f + static_cast<float>(lane + 1u) * 0.0001f;
    float8_v c0 = {seed * 1.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c1 = {seed * 2.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c2 = {seed * 3.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c3 = {seed * 4.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c4 = {seed * 5.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c5 = {seed * 6.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c6 = {seed * 7.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c7 = {seed * 8.0f, seed, seed, seed, seed, seed, seed, seed};

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
    uint32_t safe_idx = idx % 524288u;
    data[safe_idx] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#else
    data[idx % 524288u] = 0.0f;
#endif
}
