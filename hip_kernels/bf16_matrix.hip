// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    unsigned int u = static_cast<unsigned int>(bits) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ unsigned short f32_to_bf16_bits(float v) {
    unsigned int u = __float_as_uint(v);
    // Match Vulkan's NaN preservation + truncation to BF16
    if (v != v) {
        return 0xFFFFu;
    }
    return static_cast<unsigned short>(u >> 16);
}

extern "C" __global__ void run_benchmark(float* data) {
    unsigned int lane = threadIdx.x & 31u;
    unsigned int wgOffset = (blockIdx.x & 1023u) * 256u;
    unsigned int loadOffset = 2097152u + wgOffset;

    const unsigned short* bf16_data = reinterpret_cast<const unsigned short*>(data);
    unsigned short* bf16_out = reinterpret_cast<unsigned short*>(data);

#if defined(__gfx1200__) || defined(__gfx1201__)
    using float8 = __attribute__((ext_vector_type(8))) float;
    using short8 = __attribute__((ext_vector_type(8))) short;

    // Load 8 BF16 elements (128-bit) per lane for matA and matB, matching coopMatLoad
    const short8* a_ptr = reinterpret_cast<const short8*>(bf16_data + loadOffset + lane * 8u);
    const short8* b_ptr = reinterpret_cast<const short8*>(bf16_data + loadOffset + ((lane + 16u) & 31u) * 8u);
    short8 a = *a_ptr;
    short8 b = *b_ptr;

    float seed = bf16_bits_to_f32(bf16_data[loadOffset]);
    float s1 = seed * 1.0f;
    float s2 = seed * 2.0f;
    float s3 = seed * 3.0f;
    float s4 = seed * 4.0f;
    float s5 = seed * 5.0f;
    float s6 = seed * 6.0f;
    float s7 = seed * 7.0f;
    float s8 = seed * 8.0f;
    float8 c0 = {s1, s1, s1, s1, s1, s1, s1, s1};
    float8 c1 = {s2, s2, s2, s2, s2, s2, s2, s2};
    float8 c2 = {s3, s3, s3, s3, s3, s3, s3, s3};
    float8 c3 = {s4, s4, s4, s4, s4, s4, s4, s4};
    float8 c4 = {s5, s5, s5, s5, s5, s5, s5, s5};
    float8 c5 = {s6, s6, s6, s6, s6, s6, s6, s6};
    float8 c6 = {s7, s7, s7, s7, s7, s7, s7, s7};
    float8 c7 = {s8, s8, s8, s8, s8, s8, s8, s8};

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

    c0 = c0 + c1;
    c2 = c2 + c3;
    c4 = c4 + c5;
    c6 = c6 + c7;
    c0 = c0 + c2;
    c4 = c4 + c6;
    c0 = c0 + c4;

    unsigned int row_col = (lane & 15u) + ((lane >> 4u) * 128u);
    unsigned int outBase = wgOffset + row_col;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        bf16_out[outBase + j * 16u] = f32_to_bf16_bits(c0[j]);
    }
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    using float8 = __attribute__((ext_vector_type(8))) float;
    using short16 = __attribute__((ext_vector_type(16))) short;

    // On GFX11, Wave32 WMMA takes short16 (replicated 8x16 BF16 sub-tiles per half-wave)
    const short16* a_ptr = reinterpret_cast<const short16*>(bf16_data + loadOffset + (lane & 15u) * 16u);
    const short16* b_ptr = reinterpret_cast<const short16*>(bf16_data + loadOffset + ((lane + 8u) & 15u) * 16u);
    short16 a = *a_ptr;
    short16 b = *b_ptr;

    float seed = bf16_bits_to_f32(bf16_data[loadOffset]);
    float s1 = seed * 1.0f;
    float s2 = seed * 2.0f;
    float s3 = seed * 3.0f;
    float s4 = seed * 4.0f;
    float s5 = seed * 5.0f;
    float s6 = seed * 6.0f;
    float s7 = seed * 7.0f;
    float s8 = seed * 8.0f;
    float8 c0 = {s1, s1, s1, s1, s1, s1, s1, s1};
    float8 c1 = {s2, s2, s2, s2, s2, s2, s2, s2};
    float8 c2 = {s3, s3, s3, s3, s3, s3, s3, s3};
    float8 c3 = {s4, s4, s4, s4, s4, s4, s4, s4};
    float8 c4 = {s5, s5, s5, s5, s5, s5, s5, s5};
    float8 c5 = {s6, s6, s6, s6, s6, s6, s6, s6};
    float8 c6 = {s7, s7, s7, s7, s7, s7, s7, s7};
    float8 c7 = {s8, s8, s8, s8, s8, s8, s8, s8};

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

    c0 = c0 + c1;
    c2 = c2 + c3;
    c4 = c4 + c5;
    c6 = c6 + c7;
    c0 = c0 + c2;
    c4 = c4 + c6;
    c0 = c0 + c4;

    unsigned int row_col = (lane & 15u) + ((lane >> 4u) * 128u);
    unsigned int outBase = wgOffset + row_col;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        bf16_out[outBase + j * 16u] = f32_to_bf16_bits(c0[j]);
    }
#else
    bf16_out[wgOffset + lane] = 0u;
#endif
}

