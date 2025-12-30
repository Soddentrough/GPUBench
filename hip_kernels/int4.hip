#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void run_benchmark(
    const int8_t* a,
    const int8_t* b,
    int32_t* c,
    uint32_t iterations) 
{
    #if defined(__gfx1201__)
    // __builtin_amdgcn_wmma_i32_16x16x16_iu4_iu4
    #endif
}
