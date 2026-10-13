// Requires ROCm 6.4+
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void run_benchmark(float* data) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x & 31u;

#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef _Float16 half8_v __attribute__((vector_size(16)));
    typedef float float8_v __attribute__((vector_size(32)));

    const half8_v* hptr = reinterpret_cast<const half8_v*>(data) + 262144u;
    half8_v a = hptr[lane];
    half8_v b = hptr[lane + 32u];

    float seed = static_cast<float>(a[0]) * 0.01f + static_cast<float>(lane + 1u) * 0.001f;
    float8_v c0 = {seed * 1.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c1 = {seed * 2.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c2 = {seed * 3.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c3 = {seed * 4.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c4 = {seed * 5.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c5 = {seed * 6.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c6 = {seed * 7.0f, seed, seed, seed, seed, seed, seed, seed};
    float8_v c7 = {seed * 8.0f, seed, seed, seed, seed, seed, seed, seed};

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
    data[idx % 65536u] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    using float8 = __attribute__((ext_vector_type(8))) float;
    using half16 = __attribute__((ext_vector_type(16))) _Float16;

    const half16* hptr = reinterpret_cast<const half16*>(data) + 131072u;
    half16 a = hptr[lane];
    half16 b = hptr[lane + 32u];

    float seed = static_cast<float>(a[0]) * 0.01f + static_cast<float>(lane + 1u) * 0.001f;
    float8 c0 = {seed * 1.0f, seed, seed, seed, seed, seed, seed, seed};
    float8 c1 = {seed * 2.0f, seed, seed, seed, seed, seed, seed, seed};
    float8 c2 = {seed * 3.0f, seed, seed, seed, seed, seed, seed, seed};
    float8 c3 = {seed * 4.0f, seed, seed, seed, seed, seed, seed, seed};
    float8 c4 = {seed * 5.0f, seed, seed, seed, seed, seed, seed, seed};
    float8 c5 = {seed * 6.0f, seed, seed, seed, seed, seed, seed, seed};
    float8 c6 = {seed * 7.0f, seed, seed, seed, seed, seed, seed, seed};
    float8 c7 = {seed * 8.0f, seed, seed, seed, seed, seed, seed, seed};

    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c7);
    }

    float8 c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    data[idx % 65536u] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#else
    data[idx % 524288u] = 0.0f;
#endif
}
