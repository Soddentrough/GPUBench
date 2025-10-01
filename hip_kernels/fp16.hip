// Requires ROCm 6.4+
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void compute(half* data) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    half2 val1 = reinterpret_cast<half2*>(data)[index];
    half2 val2 = __half2half2(0.1f);
    half2 val3 = __half2half2(0.3f);
    half2 val4 = __half2half2(0.5f);
    half2 val5 = __half2half2(0.7f);
    half2 val6 = __half2half2(0.9f);
    half2 val7 = __half2half2(1.1f);
    half2 val8 = __half2half2(1.3f);
    
    for (int i = 0; i < 16384; ++i) {
        val1 = __hfma2(val1, __half2half2(1.0001f), val2);
        val2 = __hfma2(val2, __half2half2(1.0001f), val3);
        val3 = __hfma2(val3, __half2half2(1.0001f), val4);
        val4 = __hfma2(val4, __half2half2(1.0001f), val5);
        val5 = __hfma2(val5, __half2half2(1.0001f), val6);
        val6 = __hfma2(val6, __half2half2(1.0001f), val7);
        val7 = __hfma2(val7, __half2half2(1.0001f), val8);
        val8 = __hfma2(val8, __half2half2(1.0001f), val1);
    }
    
    half2 result = __hadd2(__hadd2(__hadd2(val1, val2), __hadd2(val3, val4)), __hadd2(__hadd2(val5, val6), __hadd2(val7, val8)));
    reinterpret_cast<half2*>(data)[index] = result;
}
