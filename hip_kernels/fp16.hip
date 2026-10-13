// Requires ROCm 6.4+
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void run_benchmark(half* data) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    half2 base = reinterpret_cast<half2*>(data)[index % 1024u];
    half2 val1  = base;
    half2 val2  = __hadd2(base, __half2half2(0.01f));
    half2 val3  = __hadd2(base, __half2half2(0.02f));
    half2 val4  = __hadd2(base, __half2half2(0.03f));
    half2 val5  = __hadd2(base, __half2half2(0.04f));
    half2 val6  = __hadd2(base, __half2half2(0.05f));
    half2 val7  = __hadd2(base, __half2half2(0.06f));
    half2 val8  = __hadd2(base, __half2half2(0.07f));
    half2 val9  = __hadd2(base, __half2half2(0.08f));
    half2 val10 = __hadd2(base, __half2half2(0.09f));
    half2 val11 = __hadd2(base, __half2half2(0.10f));
    half2 val12 = __hadd2(base, __half2half2(0.11f));
    half2 val13 = __hadd2(base, __half2half2(0.12f));
    half2 val14 = __hadd2(base, __half2half2(0.13f));
    half2 val15 = __hadd2(base, __half2half2(0.14f));
    half2 val16 = __hadd2(base, __half2half2(0.15f));
    half2 val17 = __hadd2(base, __half2half2(0.16f));
    half2 val18 = __hadd2(base, __half2half2(0.17f));
    half2 val19 = __hadd2(base, __half2half2(0.18f));
    half2 val20 = __hadd2(base, __half2half2(0.19f));
    half2 val21 = __hadd2(base, __half2half2(0.20f));
    half2 val22 = __hadd2(base, __half2half2(0.21f));
    half2 val23 = __hadd2(base, __half2half2(0.22f));
    half2 val24 = __hadd2(base, __half2half2(0.23f));
    half2 val25 = __hadd2(base, __half2half2(0.24f));
    half2 val26 = __hadd2(base, __half2half2(0.25f));
    half2 val27 = __hadd2(base, __half2half2(0.26f));
    half2 val28 = __hadd2(base, __half2half2(0.27f));
    half2 val29 = __hadd2(base, __half2half2(0.28f));
    half2 val30 = __hadd2(base, __half2half2(0.29f));
    half2 val31 = __hadd2(base, __half2half2(0.30f));
    half2 val32 = __hadd2(base, __half2half2(0.31f));

    half2 mult = __half2half2(-0.9990234375f);
    half2 addend = __half2half2(0.5f);

    // 16384 iterations * 32 half2 FMAs = 16384 * 128 FP16 ops per thread
    for (int i = 0; i < 16384; ++i) {
        val1  = __hfma2(val1,  mult, addend);
        val2  = __hfma2(val2,  mult, addend);
        val3  = __hfma2(val3,  mult, addend);
        val4  = __hfma2(val4,  mult, addend);
        val5  = __hfma2(val5,  mult, addend);
        val6  = __hfma2(val6,  mult, addend);
        val7  = __hfma2(val7,  mult, addend);
        val8  = __hfma2(val8,  mult, addend);
        val9  = __hfma2(val9,  mult, addend);
        val10 = __hfma2(val10, mult, addend);
        val11 = __hfma2(val11, mult, addend);
        val12 = __hfma2(val12, mult, addend);
        val13 = __hfma2(val13, mult, addend);
        val14 = __hfma2(val14, mult, addend);
        val15 = __hfma2(val15, mult, addend);
        val16 = __hfma2(val16, mult, addend);
        val17 = __hfma2(val17, mult, addend);
        val18 = __hfma2(val18, mult, addend);
        val19 = __hfma2(val19, mult, addend);
        val20 = __hfma2(val20, mult, addend);
        val21 = __hfma2(val21, mult, addend);
        val22 = __hfma2(val22, mult, addend);
        val23 = __hfma2(val23, mult, addend);
        val24 = __hfma2(val24, mult, addend);
        val25 = __hfma2(val25, mult, addend);
        val26 = __hfma2(val26, mult, addend);
        val27 = __hfma2(val27, mult, addend);
        val28 = __hfma2(val28, mult, addend);
        val29 = __hfma2(val29, mult, addend);
        val30 = __hfma2(val30, mult, addend);
        val31 = __hfma2(val31, mult, addend);
        val32 = __hfma2(val32, mult, addend);
    }

    half2 s0 = __hadd2(__hadd2(__hadd2(val1, val2), __hadd2(val3, val4)),
                       __hadd2(__hadd2(val5, val6), __hadd2(val7, val8)));
    half2 s1 = __hadd2(__hadd2(__hadd2(val9, val10), __hadd2(val11, val12)),
                       __hadd2(__hadd2(val13, val14), __hadd2(val15, val16)));
    half2 s2 = __hadd2(__hadd2(__hadd2(val17, val18), __hadd2(val19, val20)),
                       __hadd2(__hadd2(val21, val22), __hadd2(val23, val24)));
    half2 s3 = __hadd2(__hadd2(__hadd2(val25, val26), __hadd2(val27, val28)),
                       __hadd2(__hadd2(val29, val30), __hadd2(val31, val32)));
    half2 result = __hadd2(__hadd2(s0, s1), __hadd2(s2, s3));
    reinterpret_cast<half2*>(data)[index] = result;
}
