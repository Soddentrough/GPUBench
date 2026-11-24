#include <hip/hip_runtime.h>

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

extern "C" __global__ __launch_bounds__(256) void run_benchmark(
    float4* inputDataRaw, float4* outputDataRaw, uint mode, uint bufferSize, uint outerIters) {
    const float4* __restrict__ inputData =
        (const float4*)__builtin_assume_aligned(inputDataRaw, 16);
    float4* __restrict__ outputData =
        (float4*)__builtin_assume_aligned(outputDataRaw, 16);

    uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;

    const uint num_float4 = bufferSize / sizeof(float4);
    const uint buffer_mask = num_float4 - 1u;
    const uint iters = (outerIters > 0u) ? outerIters : 32u;

    float4 accumulator = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (mode == 1u) { // Write
        #pragma unroll 1
        for (uint i = 0u; i < iters; ++i) {
            uint base = (i * 32u) * num_threads + thread_id;
            #pragma unroll
            for (uint k = 0u; k < 32u; ++k) {
                outputData[(base + k * num_threads) & buffer_mask] =
                    make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            }
        }
    } else if (mode == 0u) { // Read
        #pragma unroll 1
        for (uint i = 0u; i < iters; ++i) {
            uint base = (i * 32u) * num_threads + thread_id;

            float4 data0  = inputData[(base +  0u * num_threads) & buffer_mask];
            float4 data1  = inputData[(base +  1u * num_threads) & buffer_mask];
            float4 data2  = inputData[(base +  2u * num_threads) & buffer_mask];
            float4 data3  = inputData[(base +  3u * num_threads) & buffer_mask];
            float4 data4  = inputData[(base +  4u * num_threads) & buffer_mask];
            float4 data5  = inputData[(base +  5u * num_threads) & buffer_mask];
            float4 data6  = inputData[(base +  6u * num_threads) & buffer_mask];
            float4 data7  = inputData[(base +  7u * num_threads) & buffer_mask];
            float4 data8  = inputData[(base +  8u * num_threads) & buffer_mask];
            float4 data9  = inputData[(base +  9u * num_threads) & buffer_mask];
            float4 data10 = inputData[(base + 10u * num_threads) & buffer_mask];
            float4 data11 = inputData[(base + 11u * num_threads) & buffer_mask];
            float4 data12 = inputData[(base + 12u * num_threads) & buffer_mask];
            float4 data13 = inputData[(base + 13u * num_threads) & buffer_mask];
            float4 data14 = inputData[(base + 14u * num_threads) & buffer_mask];
            float4 data15 = inputData[(base + 15u * num_threads) & buffer_mask];
            float4 data16 = inputData[(base + 16u * num_threads) & buffer_mask];
            float4 data17 = inputData[(base + 17u * num_threads) & buffer_mask];
            float4 data18 = inputData[(base + 18u * num_threads) & buffer_mask];
            float4 data19 = inputData[(base + 19u * num_threads) & buffer_mask];
            float4 data20 = inputData[(base + 20u * num_threads) & buffer_mask];
            float4 data21 = inputData[(base + 21u * num_threads) & buffer_mask];
            float4 data22 = inputData[(base + 22u * num_threads) & buffer_mask];
            float4 data23 = inputData[(base + 23u * num_threads) & buffer_mask];
            float4 data24 = inputData[(base + 24u * num_threads) & buffer_mask];
            float4 data25 = inputData[(base + 25u * num_threads) & buffer_mask];
            float4 data26 = inputData[(base + 26u * num_threads) & buffer_mask];
            float4 data27 = inputData[(base + 27u * num_threads) & buffer_mask];
            float4 data28 = inputData[(base + 28u * num_threads) & buffer_mask];
            float4 data29 = inputData[(base + 29u * num_threads) & buffer_mask];
            float4 data30 = inputData[(base + 30u * num_threads) & buffer_mask];
            float4 data31 = inputData[(base + 31u * num_threads) & buffer_mask];

            float4 s0 = add4(add4(add4(data0, data1), add4(data2, data3)),
                             add4(add4(data4, data5), add4(data6, data7)));
            float4 s1 = add4(add4(add4(data8, data9), add4(data10, data11)),
                             add4(add4(data12, data13), add4(data14, data15)));
            float4 s2 = add4(add4(add4(data16, data17), add4(data18, data19)),
                             add4(add4(data20, data21), add4(data22, data23)));
            float4 s3 = add4(add4(add4(data24, data25), add4(data26, data27)),
                             add4(add4(data28, data29), add4(data30, data31)));
            accumulator = add4(accumulator, add4(add4(s0, s1), add4(s2, s3)));
        }
        // Genuine writeback: unconditionally commit accumulated loads to global memory in read mode
        outputData[thread_id] = accumulator;
    } else { // Read/Write
        #pragma unroll 1
        for (uint i = 0u; i < iters; ++i) {
            uint base = (i * 32u) * num_threads + thread_id;
            #pragma unroll
            for (uint k = 0u; k < 32u; ++k) {
                uint idx = (base + k * num_threads) & buffer_mask;
                outputData[idx] = inputData[idx];
            }
        }
    }
}
