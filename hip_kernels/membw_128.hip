#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float4* inputDataRaw, float4* outputDataRaw, uint mode, uint bufferSize) {
    // The compiler cannot prove 16-byte alignment of the raw pointers and
    // splits every float4 access into b96+b32 pairs (half VMEM throughput).
    // Assert alignment so float4 loads/stores compile to single b128 ops.
    const float4* __restrict__ inputData =
        (const float4*)__builtin_assume_aligned(inputDataRaw, 16);
    float4* __restrict__ outputData =
        (float4*)__builtin_assume_aligned(outputDataRaw, 16);
    // mode == 0: Read
    // mode == 1: Write
    // mode == 2: Read/Write
    
    uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    
    const uint num_float4 = bufferSize / sizeof(float4);
    const uint buffer_mask = num_float4 - 1;

    float4 accumulator = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            uint idx = (thread_id + (i * 32 + j) * num_threads) & buffer_mask;
            if (mode == 1) { // Write
                outputData[idx] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            } else if (mode == 0) { // Read
                float4 d = inputData[idx];
                accumulator.x += d.x;
                accumulator.y += d.y;
                accumulator.z += d.z;
                accumulator.w += d.w;
            } else { // Read/Write
                outputData[idx] = inputData[idx];
            }
        }
    }
    
    // Prevent compiler from optimizing away reads (branch never taken, but compiler can't prove it)
    if (accumulator.x > 1e30f) {
        outputData[0] = accumulator;
    }
}
