// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(const int* in_data, int* out_data) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x & 31u;

#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef int int2_v __attribute__((vector_size(8)));
    typedef int int8_v __attribute__((vector_size(32)));

    const int2_v* iptr = reinterpret_cast<const int2_v*>(in_data);
    int2_v a = iptr[lane];
    int2_v b = iptr[lane + 32u];

    int seed = (a[0] & 0xFF) + static_cast<int>(lane + 1u);
    int8_v c0 = {seed * 1, seed, seed, seed, seed, seed, seed, seed};
    int8_v c1 = {seed * 2, seed, seed, seed, seed, seed, seed, seed};
    int8_v c2 = {seed * 3, seed, seed, seed, seed, seed, seed, seed};
    int8_v c3 = {seed * 4, seed, seed, seed, seed, seed, seed, seed};
    int8_v c4 = {seed * 5, seed, seed, seed, seed, seed, seed, seed};
    int8_v c5 = {seed * 6, seed, seed, seed, seed, seed, seed, seed};
    int8_v c6 = {seed * 7, seed, seed, seed, seed, seed, seed, seed};
    int8_v c7 = {seed * 8, seed, seed, seed, seed, seed, seed, seed};

    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c3, false);
        c4 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c4, false);
        c5 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c5, false);
        c6 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c6, false);
        c7 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c7, false);
    }

    int8_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out_data[idx % 524288u] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    using int4_v = __attribute__((ext_vector_type(4))) int;
    using int8_v = __attribute__((ext_vector_type(8))) int;

    const int4_v* iptr = reinterpret_cast<const int4_v*>(in_data);
    int4_v a = iptr[lane];
    int4_v b = iptr[lane + 32u];

    int seed = (a[0] & 0xFF) + static_cast<int>(lane + 1u);
    int8_v c0 = {seed * 1, seed, seed, seed, seed, seed, seed, seed};
    int8_v c1 = {seed * 2, seed, seed, seed, seed, seed, seed, seed};
    int8_v c2 = {seed * 3, seed, seed, seed, seed, seed, seed, seed};
    int8_v c3 = {seed * 4, seed, seed, seed, seed, seed, seed, seed};
    int8_v c4 = {seed * 5, seed, seed, seed, seed, seed, seed, seed};
    int8_v c5 = {seed * 6, seed, seed, seed, seed, seed, seed, seed};
    int8_v c6 = {seed * 7, seed, seed, seed, seed, seed, seed, seed};
    int8_v c7 = {seed * 8, seed, seed, seed, seed, seed, seed, seed};

    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c0, false);
        c1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c1, false);
        c2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c2, false);
        c3 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c3, false);
        c4 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c4, false);
        c5 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c5, false);
        c6 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c6, false);
        c7 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c7, false);
    }

    int8_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out_data[idx % 524288u] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#else
    out_data[idx % 524288u] = 0;
#endif
}
