#include <hip/hip_runtime.h>

// RDNA 4 (gfx1200 / gfx1201) FP8 Wave32 16x16x16 WMMA Matrix Benchmark.
// Performs 4096 iterations * 8 independent 16x16x16 WMMA accumulators per wave
// (32768 matrix operations * 8192 FLOPs = 268,435,456 FP8 ops per Wave32).

extern "C" __global__ void run_benchmark(const float* in_data, float* out_data) {
    uint32_t wg_id = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t idx = wg_id * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x & 31u;

#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef int int2_v __attribute__((vector_size(8)));
    typedef float float8_v __attribute__((vector_size(32)));

    uint32_t wg_offset = (wg_id & 8191u) * 32u;
    const int2_v* iptr = reinterpret_cast<const int2_v*>(in_data) + wg_offset;
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
    out_data[safe_idx] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
    using float4_v = __attribute__((ext_vector_type(4))) float;

    uint32_t lane64 = threadIdx.x & 63u;
    uint32_t wg_offset = (wg_id & 4095u) * 64u;
    const long long* lptr = reinterpret_cast<const long long*>(in_data) + wg_offset;
    long long a = lptr[lane64];
    long long b = lptr[(lane64 + 32u) & 63u];

    float seed = static_cast<float>((a & 0xFF) + 1) * 0.001f + static_cast<float>((lane64 & 31u) + 1u) * 0.0001f;
    float4_v c0 = {seed * 1.0f, seed, seed, seed};
    float4_v c1 = {seed * 2.0f, seed, seed, seed};
    float4_v c2 = {seed * 3.0f, seed, seed, seed};
    float4_v c3 = {seed * 4.0f, seed, seed, seed};
    float4_v c4 = {seed * 5.0f, seed, seed, seed};
    float4_v c5 = {seed * 6.0f, seed, seed, seed};
    float4_v c6 = {seed * 7.0f, seed, seed, seed};
    float4_v c7 = {seed * 8.0f, seed, seed, seed};

    // 16x16x32 MFMA executes 16,384 FP8 ops (2x a 16x16x16 tile), so 2048 * 8 = 268,435,456 ops/WG
    for (int i = 0; i < 2048; i++) {
        c0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c0, 0, 0, 0);
        c1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c1, 0, 0, 0);
        c2 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c2, 0, 0, 0);
        c3 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c3, 0, 0, 0);
        c4 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c4, 0, 0, 0);
        c5 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c5, 0, 0, 0);
        c6 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c6, 0, 0, 0);
        c7 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, c7, 0, 0, 0);
    }

    float4_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out_data[idx % 524288u] = c[0] + c[1] + c[2] + c[3];
#else
    out_data[idx % 524288u] = 0.0f;
#endif
}
