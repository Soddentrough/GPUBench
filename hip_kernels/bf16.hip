// Requires ROCm 6.4+
// BF16 vector benchmark using hardware packed BF16 dot2 instructions
// (v_dot2_f32_bf16 on RDNA 3 / RDNA 4).
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using short2_v = __attribute__((ext_vector_type(2))) short;

#define INIT(i) \
  v[i] = {static_cast<short>(s0 + (i)), static_cast<short>(s1 + (i))}; \
  acc[i] = 0.5f;

#define STEP(i) \
  acc[i] = __builtin_amdgcn_fdot2_f32_bf16( \
      v[i], m, __builtin_amdgcn_fdot2_f32_bf16(v[i], m, acc[i], false), false);

#define APPLY4(OP, i)  OP(i) OP((i)+1) OP((i)+2) OP((i)+3)
#define APPLY16(OP, i) APPLY4(OP, i) APPLY4(OP, (i)+4) \
                       APPLY4(OP, (i)+8) APPLY4(OP, (i)+12)
#define APPLY32(OP)    APPLY16(OP, 0) APPLY16(OP, 16)

#define GROUP(i) ((acc[(i)] + acc[(i)+1]) + (acc[(i)+2] + acc[(i)+3]))

extern "C" __global__ void run_benchmark(unsigned int *data) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
  unsigned int raw = data[524288u + (idx % 1024u)];
  short s0 = static_cast<short>((raw & 0xFFFFu) + 1u);
  short s1 = static_cast<short>(((raw >> 16) & 0xFFFFu) + 2u);

  short2_v v[32];
  float acc[32];
  APPLY32(INIT)

  // -0.248046875 in BF16 is 0xBE7E
  short2_v m = {static_cast<short>(0xBE7Eu), static_cast<short>(0xBE7Eu)};

  // Each iteration invokes 64 __builtin_amdgcn_fdot2_f32_bf16 calls (32 pairs x 2 dot2 = 256 BF16 ops,
  // which hipcc dual-issues primarily as v_dual_dot2acc_f32_bf16 on RDNA 3/4), matching the iteration
  // and operation count of Vulkan shaders/bf16.comp (32768 iterations x 256 ops/iter) while accumulating in FP32.
  for (int i = 0; i < 32768; ++i) {
    APPLY32(STEP)
  }

  float total = GROUP(0)  + GROUP(4)  + GROUP(8)  + GROUP(12)
              + GROUP(16) + GROUP(20) + GROUP(24) + GROUP(28);
  data[idx] = __float_as_uint(total);
#else
  data[idx] = 0u;
#endif
}
