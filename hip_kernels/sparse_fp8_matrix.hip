#include <hip/hip_runtime.h>

// Hardware 2:4 Structured Sparse FP8 Matrix Benchmark (RDNA 4 v_swmmac / CDNA 3+ v_smfmac).
// RDNA 4 16x16x32 sparse FP8 executes 16,384 effective FLOPs * 8 * 4096 = 536,870,912 FLOPs/wave.
// CDNA 3+ 16x16x64 sparse FP8 executes 32,768 effective FLOPs * 8 * 2048 = 536,870,912 FLOPs/wave.

extern "C" __global__ void run_benchmark(const float* in_data, float* out_data) {
    uint32_t wg_id = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t idx = wg_id * blockDim.x + threadIdx.x;

#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef int int2_v __attribute__((vector_size(8)));
    typedef int int4_v __attribute__((vector_size(16)));
    typedef float float8_v __attribute__((vector_size(32)));

    uint32_t lane = threadIdx.x & 31u;
    uint32_t wg_offset = (wg_id & 4095u) * 32u;
    const int2_v* aptr = reinterpret_cast<const int2_v*>(in_data) + wg_offset;
    const int4_v* bptr = reinterpret_cast<const int4_v*>(in_data) + wg_offset;
    int2_v a = aptr[lane];
    int4_v b = bptr[lane + 32u];
    int sp_idx = static_cast<int>(0x88888888u);

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
        c0 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c0, sp_idx);
        c1 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c1, sp_idx);
        c2 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c2, sp_idx);
        c3 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c3, sp_idx);
        c4 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c4, sp_idx);
        c5 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c5, sp_idx);
        c6 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c6, sp_idx);
        c7 = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c7, sp_idx);
    }

    float8_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out_data[idx % 524288u] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#elif defined(__gfx942__) || defined(__gfx950__)
    typedef int int2_v __attribute__((vector_size(8)));
    typedef int int4_v __attribute__((vector_size(16)));
    typedef float float4_v __attribute__((vector_size(16)));

    uint32_t lane64 = threadIdx.x & 63u;
    uint32_t wg_offset = (wg_id & 2047u) * 64u;
    const int2_v* aptr = reinterpret_cast<const int2_v*>(in_data) + wg_offset;
    const int4_v* bptr = reinterpret_cast<const int4_v*>(in_data) + wg_offset;
    int2_v a = aptr[lane64];
    int4_v b = bptr[(lane64 + 32u) & 63u];
    int sp_idx = static_cast<int>(0x88888888u);

    float seed = static_cast<float>((a[0] & 0xFF) + 1) * 0.001f + static_cast<float>((lane64 & 31u) + 1u) * 0.0001f;
    float4_v c0 = {seed * 1.0f, seed, seed, seed};
    float4_v c1 = {seed * 2.0f, seed, seed, seed};
    float4_v c2 = {seed * 3.0f, seed, seed, seed};
    float4_v c3 = {seed * 4.0f, seed, seed, seed};
    float4_v c4 = {seed * 5.0f, seed, seed, seed};
    float4_v c5 = {seed * 6.0f, seed, seed, seed};
    float4_v c6 = {seed * 7.0f, seed, seed, seed};
    float4_v c7 = {seed * 8.0f, seed, seed, seed};

    for (int i = 0; i < 2048; i++) {
        c0 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c0, sp_idx, 0, 0);
        c1 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c1, sp_idx, 0, 0);
        c2 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c2, sp_idx, 0, 0);
        c3 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c3, sp_idx, 0, 0);
        c4 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c4, sp_idx, 0, 0);
        c5 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c5, sp_idx, 0, 0);
        c6 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c6, sp_idx, 0, 0);
        c7 = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, c7, sp_idx, 0, 0);
    }

    float4_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out_data[idx % 524288u] = c[0] + c[1] + c[2] + c[3];
#else
    out_data[idx % 524288u] = 0.0f;
#endif
}
