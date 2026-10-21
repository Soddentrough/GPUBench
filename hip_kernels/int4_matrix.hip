// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(int* data) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = threadIdx.x & 31u;

#if defined(__gfx1200__) || defined(__gfx1201__)
    using int8 = __attribute__((ext_vector_type(8))) int;

    const int* inptr = data + 1048576u;
    int a = inptr[lane & 31u];
    int b = inptr[(lane + 16u) & 31u];
    int seed = inptr[lane & 31u] & 0x7;

    int8 c0 = {seed + 1, 0, 0, 0, 0, 0, 0, 0};
    int8 c1 = {seed + 2, 0, 0, 0, 0, 0, 0, 0};
    int8 c2 = {seed + 3, 0, 0, 0, 0, 0, 0, 0};
    int8 c3 = {seed + 4, 0, 0, 0, 0, 0, 0, 0};
    int8 c4 = {seed + 5, 0, 0, 0, 0, 0, 0, 0};
    int8 c5 = {seed + 6, 0, 0, 0, 0, 0, 0, 0};
    int8 c6 = {seed + 7, 0, 0, 0, 0, 0, 0, 0};
    int8 c7 = {seed + 8, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll 4
    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c3, false);
        c4 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c4, false);
        c5 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c5, false);
        c6 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c6, false);
        c7 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c7, false);
    }

    int8 sum = (c0 + c1) + (c2 + c3) + (c4 + c5) + (c6 + c7);
    data[idx % 524288u] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    using int8 = __attribute__((ext_vector_type(8))) int;
    using int2 = __attribute__((ext_vector_type(2))) int;

    const int* inptr = data + 1048576u;
    int2 a = {inptr[(lane * 2u) & 63u], inptr[(lane * 2u + 1u) & 63u]};
    int2 b = {inptr[(lane * 2u + 32u) & 63u], inptr[(lane * 2u + 33u) & 63u]};
    int seed = inptr[lane & 31u] & 0x7;

    int8 c0 = {seed + 1, 0, 0, 0, 0, 0, 0, 0};
    int8 c1 = {seed + 2, 0, 0, 0, 0, 0, 0, 0};
    int8 c2 = {seed + 3, 0, 0, 0, 0, 0, 0, 0};
    int8 c3 = {seed + 4, 0, 0, 0, 0, 0, 0, 0};
    int8 c4 = {seed + 5, 0, 0, 0, 0, 0, 0, 0};
    int8 c5 = {seed + 6, 0, 0, 0, 0, 0, 0, 0};
    int8 c6 = {seed + 7, 0, 0, 0, 0, 0, 0, 0};
    int8 c7 = {seed + 8, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll 4
    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c3, false);
        c4 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c4, false);
        c5 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c5, false);
        c6 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c6, false);
        c7 = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c7, false);
    }

    int8 sum = (c0 + c1) + (c2 + c3) + (c4 + c5) + (c6 + c7);
    data[idx % 524288u] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#else
    data[idx % 524288u] = 0;
#endif
}
