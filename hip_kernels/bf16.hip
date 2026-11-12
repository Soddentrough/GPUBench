// Requires ROCm 6.4+
// BF16 vector benchmark using hardware packed BF16 dot2 instructions
// (v_dot2_f32_bf16 on RDNA 3 / RDNA 4).
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using short2_v = __attribute__((ext_vector_type(2))) short;

#define INIT(i) \
  v0[i] = {static_cast<short>(s0 + 2 * (i)), static_cast<short>(s1 + 2 * (i))}; \
  v1[i] = {static_cast<short>(s0 + 2 * (i) + 1), static_cast<short>(s1 + 2 * (i) + 1)}; \
  acc[i] = 0.5f;

#define STEP(i) \
  acc[i] = __builtin_amdgcn_fdot2_f32_bf16( \
      v1[i], m1, __builtin_amdgcn_fdot2_f32_bf16( \
          v0[i], m1, __builtin_amdgcn_fdot2_f32_bf16( \
              v1[i], m0, __builtin_amdgcn_fdot2_f32_bf16(v0[i], m0, acc[i], false), false), false), false);

#define APPLY4(OP, i)  OP(i) OP((i)+1) OP((i)+2) OP((i)+3)
#define APPLY16(OP)    APPLY4(OP, 0) APPLY4(OP, 4) APPLY4(OP, 8) APPLY4(OP, 12)

#define GROUP(i) ((acc[(i)] + acc[(i)+1]) + (acc[(i)+2] + acc[(i)+3]))

extern "C" __global__ void run_benchmark(unsigned int *data) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

#if defined(__gfx950__) || \
    defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
  unsigned int raw = data[524288u + (idx % 1024u)];
  short s0 = static_cast<short>((raw & 0xFFFFu) + 1u);
  short s1 = static_cast<short>(((raw >> 16) & 0xFFFFu) + 2u);
  unsigned int rawM = __builtin_amdgcn_readfirstlane(raw);

  short2_v v0[16];
  short2_v v1[16];
  float acc[16];
  APPLY16(INIT)

  // Each iteration invokes 64 __builtin_amdgcn_fdot2_f32_bf16 calls (16 chains x 4 dot2 = 256 BF16 ops),
  // matching Vulkan shaders/bf16.comp (32768 iterations x 256 ops/iter) while accumulating in FP32.
  for (int i = 0; i < 32768; ++i) {
    short2_v m0 = {static_cast<short>(rawM & 0xFFFFu), static_cast<short>(rawM >> 16u)};
    unsigned int rawM1 = rawM ^ 0x00020001u;
    short2_v m1 = {static_cast<short>(rawM1 & 0xFFFFu), static_cast<short>(rawM1 >> 16u)};
    APPLY16(STEP)
    rawM ^= 1u;
  }

  float total = (GROUP(0) + GROUP(4)) + (GROUP(8) + GROUP(12));
  data[idx] = __float_as_uint(total);
#elif defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  // CDNA 2/3 (gfx90a / gfx94x) lack VALU dot12-insts (v_dot2_f32_bf16) and execute BF16 via Wave64 MFMA.
  // Each v_mfma_f32_4x4x4bf16_1k executes 16 blocks x (2*4*4*4) = 2048 ops/wave = 32 ops/lane;
  // 8 accumulators x 32 ops = 256 BF16 ops/lane/iter across 32,768 iterations.
  using float4_v = __attribute__((ext_vector_type(4))) float;
  using short4_v = __attribute__((ext_vector_type(4))) short;

  unsigned int raw = data[524288u + (idx % 1024u)];
  short b0 = static_cast<short>(raw & 0xFFFFu);
  short b1 = static_cast<short>((raw >> 16u) & 0xFFFFu);
  short4_v a = {b0, b1, b0, b1};
  short4_v b = {b1, b0, b1, b0};

  float4_v c0 = {0.5f, 0.5f, 0.5f, 0.5f};
  float4_v c1 = {1.0f, 1.0f, 1.0f, 1.0f};
  float4_v c2 = {1.5f, 1.5f, 1.5f, 1.5f};
  float4_v c3 = {2.0f, 2.0f, 2.0f, 2.0f};
  float4_v c4 = {2.5f, 2.5f, 2.5f, 2.5f};
  float4_v c5 = {3.0f, 3.0f, 3.0f, 3.0f};
  float4_v c6 = {3.5f, 3.5f, 3.5f, 3.5f};
  float4_v c7 = {4.0f, 4.0f, 4.0f, 4.0f};

  for (int i = 0; i < 32768; ++i) {
    c0 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c0, 0, 0, 0);
    c1 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c1, 0, 0, 0);
    c2 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c2, 0, 0, 0);
    c3 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c3, 0, 0, 0);
    c4 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c4, 0, 0, 0);
    c5 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c5, 0, 0, 0);
    c6 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c6, 0, 0, 0);
    c7 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c7, 0, 0, 0);
  }

  float4_v c = (c0 + c1) + (c2 + c3) + (c4 + c5) + (c6 + c7);
  float total = (c[0] + c[1]) + (c[2] + c[3]);
  data[idx] = __float_as_uint(total);
#else
  data[idx] = 0u;
#endif
}

