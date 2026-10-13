#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

extern "C" __global__ void run_benchmark(hip_bfloat16 *data) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  hip_bfloat16 seed = data[idx % 1024u];
  hip_bfloat16 val1  = seed + hip_bfloat16(0.01f);
  hip_bfloat16 val2  = seed + hip_bfloat16(0.02f);
  hip_bfloat16 val3  = seed + hip_bfloat16(0.03f);
  hip_bfloat16 val4  = seed + hip_bfloat16(0.04f);
  hip_bfloat16 val5  = seed + hip_bfloat16(0.05f);
  hip_bfloat16 val6  = seed + hip_bfloat16(0.06f);
  hip_bfloat16 val7  = seed + hip_bfloat16(0.07f);
  hip_bfloat16 val8  = seed + hip_bfloat16(0.08f);
  hip_bfloat16 val9  = seed + hip_bfloat16(0.09f);
  hip_bfloat16 val10 = seed + hip_bfloat16(0.10f);
  hip_bfloat16 val11 = seed + hip_bfloat16(0.11f);
  hip_bfloat16 val12 = seed + hip_bfloat16(0.12f);
  hip_bfloat16 val13 = seed + hip_bfloat16(0.13f);
  hip_bfloat16 val14 = seed + hip_bfloat16(0.14f);
  hip_bfloat16 val15 = seed + hip_bfloat16(0.15f);
  hip_bfloat16 val16 = seed + hip_bfloat16(0.16f);
  hip_bfloat16 val17 = seed + hip_bfloat16(0.17f);
  hip_bfloat16 val18 = seed + hip_bfloat16(0.18f);
  hip_bfloat16 val19 = seed + hip_bfloat16(0.19f);
  hip_bfloat16 val20 = seed + hip_bfloat16(0.20f);
  hip_bfloat16 val21 = seed + hip_bfloat16(0.21f);
  hip_bfloat16 val22 = seed + hip_bfloat16(0.22f);
  hip_bfloat16 val23 = seed + hip_bfloat16(0.23f);
  hip_bfloat16 val24 = seed + hip_bfloat16(0.24f);
  hip_bfloat16 val25 = seed + hip_bfloat16(0.25f);
  hip_bfloat16 val26 = seed + hip_bfloat16(0.26f);
  hip_bfloat16 val27 = seed + hip_bfloat16(0.27f);
  hip_bfloat16 val28 = seed + hip_bfloat16(0.28f);
  hip_bfloat16 val29 = seed + hip_bfloat16(0.29f);
  hip_bfloat16 val30 = seed + hip_bfloat16(0.30f);
  hip_bfloat16 val31 = seed + hip_bfloat16(0.31f);
  hip_bfloat16 val32 = seed + hip_bfloat16(0.32f);

  hip_bfloat16 m = hip_bfloat16(-0.9921875f);
  hip_bfloat16 c = hip_bfloat16(0.5f);

  for (int i = 0; i < 4096; ++i) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    val1  = val1  * m + c;
    val2  = val2  * m + c;
    val3  = val3  * m + c;
    val4  = val4  * m + c;
    val5  = val5  * m + c;
    val6  = val6  * m + c;
    val7  = val7  * m + c;
    val8  = val8  * m + c;
    val9  = val9  * m + c;
    val10 = val10 * m + c;
    val11 = val11 * m + c;
    val12 = val12 * m + c;
    val13 = val13 * m + c;
    val14 = val14 * m + c;
    val15 = val15 * m + c;
    val16 = val16 * m + c;
    val17 = val17 * m + c;
    val18 = val18 * m + c;
    val19 = val19 * m + c;
    val20 = val20 * m + c;
    val21 = val21 * m + c;
    val22 = val22 * m + c;
    val23 = val23 * m + c;
    val24 = val24 * m + c;
    val25 = val25 * m + c;
    val26 = val26 * m + c;
    val27 = val27 * m + c;
    val28 = val28 * m + c;
    val29 = val29 * m + c;
    val30 = val30 * m + c;
    val31 = val31 * m + c;
    val32 = val32 * m + c;
#endif
  }

  data[idx] = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 +
              val9 + val10 + val11 + val12 + val13 + val14 + val15 + val16 +
              val17 + val18 + val19 + val20 + val21 + val22 + val23 + val24 +
              val25 + val26 + val27 + val28 + val29 + val30 + val31 + val32;
}
