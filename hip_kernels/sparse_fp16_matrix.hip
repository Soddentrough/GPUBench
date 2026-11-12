#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Hardware 2:4 Structured Sparse FP16 Matrix Benchmark (RDNA 4 v_swmmac / CDNA 3+ v_smfmac).
// Each 16x16x32 sparse tile executes 2 * 16 * 16 * 32 = 16,384 effective FP16 FLOPs.
// 4096 iterations * 8 accumulators * 16,384 FLOPs = 536,870,912 effective FP16 FLOPs per wave.

extern "C" __global__ void run_benchmark(const float* in_data, float* out_data) {
    uint32_t wg_id = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t idx = wg_id * blockDim.x + threadIdx.x;

#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef _Float16 half8_v __attribute__((vector_size(16)));
    typedef _Float16 half16_v __attribute__((vector_size(32)));
    typedef float float8_v __attribute__((vector_size(32)));

    uint32_t lane = threadIdx.x & 31u;
    uint32_t wg_offset = (wg_id & 511u) * 32u;
    const half8_v* aptr = reinterpret_cast<const half8_v*>(in_data) + wg_offset;
    const half16_v* bptr = reinterpret_cast<const half16_v*>(in_data) + wg_offset;
    half8_v a = aptr[lane];
    half16_v b = bptr[lane + 32u];
    // Valid 2:4 sparsity index mask: selects elements (0, 2) in every 4-tuple (0b10001000... = 0x88888888)
    int sp_idx = static_cast<int>(0x88888888u);

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
        c0 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c0, sp_idx);
        c1 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c1, sp_idx);
        c2 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c2, sp_idx);
        c3 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c3, sp_idx);
        c4 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c4, sp_idx);
        c5 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c5, sp_idx);
        c6 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c6, sp_idx);
        c7 = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c7, sp_idx);
    }

    float8_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out_data[idx % 524288u] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
    typedef _Float16 half4_v __attribute__((vector_size(8)));
    typedef _Float16 half8_v __attribute__((vector_size(16)));
    typedef float float4_v __attribute__((vector_size(16)));

    uint32_t lane64 = threadIdx.x & 63u;
    uint32_t wg_offset = (wg_id & 511u) * 64u;
    const half4_v* aptr = reinterpret_cast<const half4_v*>(in_data) + wg_offset;
    const half8_v* bptr = reinterpret_cast<const half8_v*>(in_data) + wg_offset;
    half4_v a = aptr[lane64];
    half8_v b = bptr[(lane64 + 32u) & 63u];
    int sp_idx = static_cast<int>(0x88888888u);

    float seed = static_cast<float>(a[0]) * 0.01f + static_cast<float>((lane64 & 31u) + 1u) * 0.001f;
    float4_v c0 = {seed * 1.0f, seed, seed, seed};
    float4_v c1 = {seed * 2.0f, seed, seed, seed};
    float4_v c2 = {seed * 3.0f, seed, seed, seed};
    float4_v c3 = {seed * 4.0f, seed, seed, seed};
    float4_v c4 = {seed * 5.0f, seed, seed, seed};
    float4_v c5 = {seed * 6.0f, seed, seed, seed};
    float4_v c6 = {seed * 7.0f, seed, seed, seed};
    float4_v c7 = {seed * 8.0f, seed, seed, seed};

    for (int i = 0; i < 4096; i++) {
        c0 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c0, sp_idx, 0, 0);
        c1 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c1, sp_idx, 0, 0);
        c2 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c2, sp_idx, 0, 0);
        c3 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c3, sp_idx, 0, 0);
        c4 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c4, sp_idx, 0, 0);
        c5 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c5, sp_idx, 0, 0);
        c6 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c6, sp_idx, 0, 0);
        c7 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, c7, sp_idx, 0, 0);
    }

    float4_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out_data[idx % 524288u] = c[0] + c[1] + c[2] + c[3];
#else
    out_data[idx % 524288u] = 0.0f;
#endif
}
