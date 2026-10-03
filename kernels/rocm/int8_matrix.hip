// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(int* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

#if defined(__gfx1200__) || defined(__gfx1201__)
    typedef int int2_v __attribute__((vector_size(8)));
    typedef int int8_v __attribute__((vector_size(32)));

    int2_v a = {0x01010101, 0x01010101};
    int2_v b = {0x02020202, 0x02020202};

    int8_v c0 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c1 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c2 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c3 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c4 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c5 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c6 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c7 = {0, 0, 0, 0, 0, 0, 0, 0};

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
    data[idx % 524288] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    using int4_v = __attribute__((ext_vector_type(4))) int;
    using int8_v = __attribute__((ext_vector_type(8))) int;

    int4_v a = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    int4_v b = {0x02020202, 0x02020202, 0x02020202, 0x02020202};

    int8_v c0 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c1 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c2 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c3 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c4 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c5 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c6 = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_v c7 = {0, 0, 0, 0, 0, 0, 0, 0};

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
    data[idx % 524288] = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7];
#else
#error "WMMA INT8 is not supported on this target architecture"
#endif
}
