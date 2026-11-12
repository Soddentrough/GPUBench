#include <hip/hip_runtime.h>

// AMD CDNA 4 (gfx950) OCP Microscaling MXFP4 (E2M1) 16x16x128 MFMA Matrix Benchmark.
// Each 16x16x128 MFMA performs 2 * 16 * 16 * 128 = 65,536 ops (= 8 * 8,192 ops).
// 512 iterations * 8 independent accumulators = 4,096 MFMAs * 65,536 ops = 268,435,456 ops per Wave64.
// Dispatched across (32768, 2, 1) workgroups to match Vulkan cooperative matrix total work.

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

extern "C" __global__ void run_benchmark(const uint32_t* in_data, float* out_data) {
    uint32_t wg_id = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t idx = wg_id * blockDim.x + threadIdx.x;

#if defined(__gfx950__) && __has_builtin(__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
    typedef int int8_v __attribute__((vector_size(32)));
    typedef float float4_v __attribute__((vector_size(16)));

    uint32_t lane = threadIdx.x & 63u;
    uint32_t wg_offset = (wg_id & 4095u) * 64u;
    const int8_v* iptr = reinterpret_cast<const int8_v*>(in_data) + wg_offset;
    int8_v a = iptr[lane];
    int8_v b = iptr[(lane + 1u) & 63u];

    float seed = static_cast<float>((a[0] & 0xFF) + 1) * 0.001f + static_cast<float>(lane + 1u) * 0.0001f;
    float4_v c0 = {seed * 1.0f, seed, seed, seed};
    float4_v c1 = {seed * 2.0f, seed, seed, seed};
    float4_v c2 = {seed * 3.0f, seed, seed, seed};
    float4_v c3 = {seed * 4.0f, seed, seed, seed};
    float4_v c4 = {seed * 5.0f, seed, seed, seed};
    float4_v c5 = {seed * 6.0f, seed, seed, seed};
    float4_v c6 = {seed * 7.0f, seed, seed, seed};
    float4_v c7 = {seed * 8.0f, seed, seed, seed};

    // Format 4 = OCP FP4 E2M1; UE8M0 scale = 0x7F (2^0 = 1.0)
    for (int i = 0; i < 512; ++i) {
        c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c0, 4, 4, 0, 0x7Fu, 0, 0x7Fu);
        c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c1, 4, 4, 0, 0x7Fu, 0, 0x7Fu);
        c2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c2, 4, 4, 0, 0x7Fu, 0, 0x7Fu);
        c3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c3, 4, 4, 0, 0x7Fu, 0, 0x7Fu);
        c4 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c4, 4, 4, 0, 0x7Fu, 0, 0x7Fu);
        c5 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c5, 4, 4, 0, 0x7Fu, 0, 0x7Fu);
        c6 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c6, 4, 4, 0, 0x7Fu, 0, 0x7Fu);
        c7 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c7, 4, 4, 0, 0x7Fu, 0, 0x7Fu);
    }

    float4_v c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out_data[idx % 524288u] = c[0] + c[1] + c[2] + c[3];
#else
    (void)in_data;
    out_data[idx % 524288u] = 0.0f;
#endif
}
