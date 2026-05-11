#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

extern "C" __global__ void run_benchmark(hip_bfloat16 *data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  hip_bfloat16 val1 = data[idx % 1024];
  hip_bfloat16 val2 = hip_bfloat16(0.2f);
  hip_bfloat16 val3 = hip_bfloat16(0.3f);
  hip_bfloat16 val4 = hip_bfloat16(0.4f);
  hip_bfloat16 val5 = hip_bfloat16(0.5f);
  hip_bfloat16 val6 = hip_bfloat16(0.6f);
  hip_bfloat16 val7 = hip_bfloat16(0.7f);
  hip_bfloat16 val8 = hip_bfloat16(0.8f);
  hip_bfloat16 val9 = hip_bfloat16(0.9f);
  hip_bfloat16 val10 = hip_bfloat16(1.0f);
  hip_bfloat16 val11 = hip_bfloat16(1.1f);
  hip_bfloat16 val12 = hip_bfloat16(1.2f);
  hip_bfloat16 val13 = hip_bfloat16(1.3f);
  hip_bfloat16 val14 = hip_bfloat16(1.4f);
  hip_bfloat16 val15 = hip_bfloat16(1.5f);
  hip_bfloat16 val16 = hip_bfloat16(1.6f);
  hip_bfloat16 val17 = hip_bfloat16(1.7f);
  hip_bfloat16 val18 = hip_bfloat16(1.8f);
  hip_bfloat16 val19 = hip_bfloat16(1.9f);
  hip_bfloat16 val20 = hip_bfloat16(2.0f);
  hip_bfloat16 val21 = hip_bfloat16(2.1f);
  hip_bfloat16 val22 = hip_bfloat16(2.2f);
  hip_bfloat16 val23 = hip_bfloat16(2.3f);
  hip_bfloat16 val24 = hip_bfloat16(2.4f);
  hip_bfloat16 val25 = hip_bfloat16(2.5f);
  hip_bfloat16 val26 = hip_bfloat16(2.6f);
  hip_bfloat16 val27 = hip_bfloat16(2.7f);
  hip_bfloat16 val28 = hip_bfloat16(2.8f);
  hip_bfloat16 val29 = hip_bfloat16(2.9f);
  hip_bfloat16 val30 = hip_bfloat16(3.0f);
  hip_bfloat16 val31 = hip_bfloat16(3.1f);
  hip_bfloat16 val32 = hip_bfloat16(3.2f);

  hip_bfloat16 m = hip_bfloat16(1.0001f);

  for (int i = 0; i < 16384; ++i) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    val1 = val1 * m + val2;
    val2 = val2 * m + val3;
    val3 = val3 * m + val4;
    val4 = val4 * m + val5;
    val5 = val5 * m + val6;
    val6 = val6 * m + val7;
    val7 = val7 * m + val8;
    val8 = val8 * m + val9;
    val9 = val9 * m + val10;
    val10 = val10 * m + val11;
    val11 = val11 * m + val12;
    val12 = val12 * m + val13;
    val13 = val13 * m + val14;
    val14 = val14 * m + val15;
    val15 = val15 * m + val16;
    val16 = val16 * m + val17;
    val17 = val17 * m + val18;
    val18 = val18 * m + val19;
    val19 = val19 * m + val20;
    val20 = val20 * m + val21;
    val21 = val21 * m + val22;
    val22 = val22 * m + val23;
    val23 = val23 * m + val24;
    val24 = val24 * m + val25;
    val25 = val25 * m + val26;
    val26 = val26 * m + val27;
    val27 = val27 * m + val28;
    val28 = val28 * m + val29;
    val29 = val29 * m + val30;
    val30 = val30 * m + val31;
    val31 = val31 * m + val32;
    val32 = val32 * m + val1;
#endif
  }

  data[idx] = val1;
}
