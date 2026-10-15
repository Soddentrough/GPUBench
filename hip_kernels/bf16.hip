// Requires ROCm 6.4+
// BF16 vector benchmark using hardware packed BF16 dot2 instructions
// (v_dot2_f32_bf16 on RDNA 3 / RDNA 4).
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using short2_v = __attribute__((ext_vector_type(2))) short;

extern "C" __global__ void run_benchmark(unsigned int *data) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
  unsigned int raw = data[idx % 1024u];
  short s0 = static_cast<short>((raw & 0xFFFFu) + 1u);
  short s1 = static_cast<short>(((raw >> 16) & 0xFFFFu) + 2u);
  short2_v base = {s0, s1};

  short2_v val1  = base;
  short2_v val2  = {static_cast<short>(s0 + 1),  static_cast<short>(s1 + 1)};
  short2_v val3  = {static_cast<short>(s0 + 2),  static_cast<short>(s1 + 2)};
  short2_v val4  = {static_cast<short>(s0 + 3),  static_cast<short>(s1 + 3)};
  short2_v val5  = {static_cast<short>(s0 + 4),  static_cast<short>(s1 + 4)};
  short2_v val6  = {static_cast<short>(s0 + 5),  static_cast<short>(s1 + 5)};
  short2_v val7  = {static_cast<short>(s0 + 6),  static_cast<short>(s1 + 6)};
  short2_v val8  = {static_cast<short>(s0 + 7),  static_cast<short>(s1 + 7)};
  short2_v val9  = {static_cast<short>(s0 + 8),  static_cast<short>(s1 + 8)};
  short2_v val10 = {static_cast<short>(s0 + 9),  static_cast<short>(s1 + 9)};
  short2_v val11 = {static_cast<short>(s0 + 10), static_cast<short>(s1 + 10)};
  short2_v val12 = {static_cast<short>(s0 + 11), static_cast<short>(s1 + 11)};
  short2_v val13 = {static_cast<short>(s0 + 12), static_cast<short>(s1 + 12)};
  short2_v val14 = {static_cast<short>(s0 + 13), static_cast<short>(s1 + 13)};
  short2_v val15 = {static_cast<short>(s0 + 14), static_cast<short>(s1 + 14)};
  short2_v val16 = {static_cast<short>(s0 + 15), static_cast<short>(s1 + 15)};
  short2_v val17 = {static_cast<short>(s0 + 16), static_cast<short>(s1 + 16)};
  short2_v val18 = {static_cast<short>(s0 + 17), static_cast<short>(s1 + 17)};
  short2_v val19 = {static_cast<short>(s0 + 18), static_cast<short>(s1 + 18)};
  short2_v val20 = {static_cast<short>(s0 + 19), static_cast<short>(s1 + 19)};
  short2_v val21 = {static_cast<short>(s0 + 20), static_cast<short>(s1 + 20)};
  short2_v val22 = {static_cast<short>(s0 + 21), static_cast<short>(s1 + 21)};
  short2_v val23 = {static_cast<short>(s0 + 22), static_cast<short>(s1 + 22)};
  short2_v val24 = {static_cast<short>(s0 + 23), static_cast<short>(s1 + 23)};
  short2_v val25 = {static_cast<short>(s0 + 24), static_cast<short>(s1 + 24)};
  short2_v val26 = {static_cast<short>(s0 + 25), static_cast<short>(s1 + 25)};
  short2_v val27 = {static_cast<short>(s0 + 26), static_cast<short>(s1 + 26)};
  short2_v val28 = {static_cast<short>(s0 + 27), static_cast<short>(s1 + 27)};
  short2_v val29 = {static_cast<short>(s0 + 28), static_cast<short>(s1 + 28)};
  short2_v val30 = {static_cast<short>(s0 + 29), static_cast<short>(s1 + 29)};
  short2_v val31 = {static_cast<short>(s0 + 30), static_cast<short>(s1 + 30)};
  short2_v val32 = {static_cast<short>(s0 + 31), static_cast<short>(s1 + 31)};

  // -0.248046875 in BF16 is 0xBE7E (-0.248046875f)
  short2_v m = {static_cast<short>(0xBE7Eu), static_cast<short>(0xBE7Eu)};

  float acc1 = 0.5f, acc2 = 0.5f, acc3 = 0.5f, acc4 = 0.5f;
  float acc5 = 0.5f, acc6 = 0.5f, acc7 = 0.5f, acc8 = 0.5f;
  float acc9 = 0.5f, acc10 = 0.5f, acc11 = 0.5f, acc12 = 0.5f;
  float acc13 = 0.5f, acc14 = 0.5f, acc15 = 0.5f, acc16 = 0.5f;
  float acc17 = 0.5f, acc18 = 0.5f, acc19 = 0.5f, acc20 = 0.5f;
  float acc21 = 0.5f, acc22 = 0.5f, acc23 = 0.5f, acc24 = 0.5f;
  float acc25 = 0.5f, acc26 = 0.5f, acc27 = 0.5f, acc28 = 0.5f;
  float acc29 = 0.5f, acc30 = 0.5f, acc31 = 0.5f, acc32 = 0.5f;

  // Each iteration performs 64 v_dot2_f32_bf16 instructions (32 pairs x 2 dot2 = 256 BF16 ops)
  // matching Vulkan shaders/bf16.comp (32768 iterations x 256 ops/iter).
  for (int i = 0; i < 32768; ++i) {
    acc1  = __builtin_amdgcn_fdot2_f32_bf16(val1,  m, __builtin_amdgcn_fdot2_f32_bf16(val1,  m, acc1,  false), false);
    acc2  = __builtin_amdgcn_fdot2_f32_bf16(val2,  m, __builtin_amdgcn_fdot2_f32_bf16(val2,  m, acc2,  false), false);
    acc3  = __builtin_amdgcn_fdot2_f32_bf16(val3,  m, __builtin_amdgcn_fdot2_f32_bf16(val3,  m, acc3,  false), false);
    acc4  = __builtin_amdgcn_fdot2_f32_bf16(val4,  m, __builtin_amdgcn_fdot2_f32_bf16(val4,  m, acc4,  false), false);
    acc5  = __builtin_amdgcn_fdot2_f32_bf16(val5,  m, __builtin_amdgcn_fdot2_f32_bf16(val5,  m, acc5,  false), false);
    acc6  = __builtin_amdgcn_fdot2_f32_bf16(val6,  m, __builtin_amdgcn_fdot2_f32_bf16(val6,  m, acc6,  false), false);
    acc7  = __builtin_amdgcn_fdot2_f32_bf16(val7,  m, __builtin_amdgcn_fdot2_f32_bf16(val7,  m, acc7,  false), false);
    acc8  = __builtin_amdgcn_fdot2_f32_bf16(val8,  m, __builtin_amdgcn_fdot2_f32_bf16(val8,  m, acc8,  false), false);
    acc9  = __builtin_amdgcn_fdot2_f32_bf16(val9,  m, __builtin_amdgcn_fdot2_f32_bf16(val9,  m, acc9,  false), false);
    acc10 = __builtin_amdgcn_fdot2_f32_bf16(val10, m, __builtin_amdgcn_fdot2_f32_bf16(val10, m, acc10, false), false);
    acc11 = __builtin_amdgcn_fdot2_f32_bf16(val11, m, __builtin_amdgcn_fdot2_f32_bf16(val11, m, acc11, false), false);
    acc12 = __builtin_amdgcn_fdot2_f32_bf16(val12, m, __builtin_amdgcn_fdot2_f32_bf16(val12, m, acc12, false), false);
    acc13 = __builtin_amdgcn_fdot2_f32_bf16(val13, m, __builtin_amdgcn_fdot2_f32_bf16(val13, m, acc13, false), false);
    acc14 = __builtin_amdgcn_fdot2_f32_bf16(val14, m, __builtin_amdgcn_fdot2_f32_bf16(val14, m, acc14, false), false);
    acc15 = __builtin_amdgcn_fdot2_f32_bf16(val15, m, __builtin_amdgcn_fdot2_f32_bf16(val15, m, acc15, false), false);
    acc16 = __builtin_amdgcn_fdot2_f32_bf16(val16, m, __builtin_amdgcn_fdot2_f32_bf16(val16, m, acc16, false), false);
    acc17 = __builtin_amdgcn_fdot2_f32_bf16(val17, m, __builtin_amdgcn_fdot2_f32_bf16(val17, m, acc17, false), false);
    acc18 = __builtin_amdgcn_fdot2_f32_bf16(val18, m, __builtin_amdgcn_fdot2_f32_bf16(val18, m, acc18, false), false);
    acc19 = __builtin_amdgcn_fdot2_f32_bf16(val19, m, __builtin_amdgcn_fdot2_f32_bf16(val19, m, acc19, false), false);
    acc20 = __builtin_amdgcn_fdot2_f32_bf16(val20, m, __builtin_amdgcn_fdot2_f32_bf16(val20, m, acc20, false), false);
    acc21 = __builtin_amdgcn_fdot2_f32_bf16(val21, m, __builtin_amdgcn_fdot2_f32_bf16(val21, m, acc21, false), false);
    acc22 = __builtin_amdgcn_fdot2_f32_bf16(val22, m, __builtin_amdgcn_fdot2_f32_bf16(val22, m, acc22, false), false);
    acc23 = __builtin_amdgcn_fdot2_f32_bf16(val23, m, __builtin_amdgcn_fdot2_f32_bf16(val23, m, acc23, false), false);
    acc24 = __builtin_amdgcn_fdot2_f32_bf16(val24, m, __builtin_amdgcn_fdot2_f32_bf16(val24, m, acc24, false), false);
    acc25 = __builtin_amdgcn_fdot2_f32_bf16(val25, m, __builtin_amdgcn_fdot2_f32_bf16(val25, m, acc25, false), false);
    acc26 = __builtin_amdgcn_fdot2_f32_bf16(val26, m, __builtin_amdgcn_fdot2_f32_bf16(val26, m, acc26, false), false);
    acc27 = __builtin_amdgcn_fdot2_f32_bf16(val27, m, __builtin_amdgcn_fdot2_f32_bf16(val27, m, acc27, false), false);
    acc28 = __builtin_amdgcn_fdot2_f32_bf16(val28, m, __builtin_amdgcn_fdot2_f32_bf16(val28, m, acc28, false), false);
    acc29 = __builtin_amdgcn_fdot2_f32_bf16(val29, m, __builtin_amdgcn_fdot2_f32_bf16(val29, m, acc29, false), false);
    acc30 = __builtin_amdgcn_fdot2_f32_bf16(val30, m, __builtin_amdgcn_fdot2_f32_bf16(val30, m, acc30, false), false);
    acc31 = __builtin_amdgcn_fdot2_f32_bf16(val31, m, __builtin_amdgcn_fdot2_f32_bf16(val31, m, acc31, false), false);
    acc32 = __builtin_amdgcn_fdot2_f32_bf16(val32, m, __builtin_amdgcn_fdot2_f32_bf16(val32, m, acc32, false), false);
  }

  float total = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 +
                acc9 + acc10 + acc11 + acc12 + acc13 + acc14 + acc15 + acc16 +
                acc17 + acc18 + acc19 + acc20 + acc21 + acc22 + acc23 + acc24 +
                acc25 + acc26 + acc27 + acc28 + acc29 + acc30 + acc31 + acc32;
  data[idx] = __float_as_uint(total);
#else
  data[idx] = 0u;
#endif
}
