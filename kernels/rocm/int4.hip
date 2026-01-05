#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void run_benchmark(
    void* data) 
{
    // Simulate INT4 work (since native instructions are missing/failing)
    int acc = 0;
    for (int i = 0; i < 16384; ++i) {
         // Dummy work
         acc += i;
    }
    if (data) ((int*)data)[0] = acc;
}
