// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float* data) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = threadIdx.x & 31u;

#if defined(__gfx1200__) || defined(__gfx1201__)
    using float8 = __attribute__((ext_vector_type(8))) float;
    using short8 = __attribute__((ext_vector_type(8))) short;

    const short* sdata = reinterpret_cast<const short*>(data);
    short8 a;
    short8 b;
    for (int i = 0; i < 8; ++i) {
        a[i] = sdata[(lane * 8u + i) & 1023u];
        b[i] = sdata[(lane * 8u + i + 256u) & 1023u];
    }

    float seed = static_cast<float>(sdata[lane & 31u]) * (1.0f / 32768.0f);
    float8 c0 = {seed * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c1 = {seed * 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c2 = {seed * 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c3 = {seed * 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c4 = {seed * 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c5 = {seed * 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c6 = {seed * 7.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c7 = {seed * 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll 4
    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c7);
    }

    float8 sum = (c0 + c1) + (c2 + c3) + (c4 + c5) + (c6 + c7);
    data[idx % 524288u] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    using float8 = __attribute__((ext_vector_type(8))) float;
    using short16 = __attribute__((ext_vector_type(16))) short;

    const short* sdata = reinterpret_cast<const short*>(data);
    short16 a;
    short16 b;
    for (int i = 0; i < 16; ++i) {
        a[i] = sdata[(lane * 16u + i) & 1023u];
        b[i] = sdata[(lane * 16u + i + 256u) & 1023u];
    }

    float seed = static_cast<float>(sdata[lane & 31u]) * (1.0f / 32768.0f);
    float8 c0 = {seed * 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c1 = {seed * 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c2 = {seed * 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c3 = {seed * 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c4 = {seed * 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c5 = {seed * 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c6 = {seed * 7.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8 c7 = {seed * 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll 4
    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c0);
        c1 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c1);
        c2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c2);
        c3 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c3);
        c4 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c4);
        c5 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c5);
        c6 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c6);
        c7 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c7);
    }

    float8 sum = (c0 + c1) + (c2 + c3) + (c4 + c5) + (c6 + c7);
    data[idx % 524288u] = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#else
    data[idx % 524288u] = 0.0f;
#endif
}
