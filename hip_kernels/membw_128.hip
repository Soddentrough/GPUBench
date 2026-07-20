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
    
    // Calculate buffer mask based on actual buffer size
    // Each chunk is 32 float4s (512 bytes)
    const uint buffer_num_chunks = bufferSize / 512;
    const uint buffer_mask = buffer_num_chunks - 1;

    uint chunk_index = thread_id;
    float4 accumulator = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        uint current_chunk = chunk_index & buffer_mask;
        uint baseIndex = current_chunk * 32;

        float4 data0 = inputData[baseIndex + 0];
        float4 data1 = inputData[baseIndex + 1];
        float4 data2 = inputData[baseIndex + 2];
        float4 data3 = inputData[baseIndex + 3];
        float4 data4 = inputData[baseIndex + 4];
        float4 data5 = inputData[baseIndex + 5];
        float4 data6 = inputData[baseIndex + 6];
        float4 data7 = inputData[baseIndex + 7];
        float4 data8 = inputData[baseIndex + 8];
        float4 data9 = inputData[baseIndex + 9];
        float4 data10 = inputData[baseIndex + 10];
        float4 data11 = inputData[baseIndex + 11];
        float4 data12 = inputData[baseIndex + 12];
        float4 data13 = inputData[baseIndex + 13];
        float4 data14 = inputData[baseIndex + 14];
        float4 data15 = inputData[baseIndex + 15];
        float4 data16 = inputData[baseIndex + 16];
        float4 data17 = inputData[baseIndex + 17];
        float4 data18 = inputData[baseIndex + 18];
        float4 data19 = inputData[baseIndex + 19];
        float4 data20 = inputData[baseIndex + 20];
        float4 data21 = inputData[baseIndex + 21];
        float4 data22 = inputData[baseIndex + 22];
        float4 data23 = inputData[baseIndex + 23];
        float4 data24 = inputData[baseIndex + 24];
        float4 data25 = inputData[baseIndex + 25];
        float4 data26 = inputData[baseIndex + 26];
        float4 data27 = inputData[baseIndex + 27];
        float4 data28 = inputData[baseIndex + 28];
        float4 data29 = inputData[baseIndex + 29];
        float4 data30 = inputData[baseIndex + 30];
        float4 data31 = inputData[baseIndex + 31];
        
        if (mode == 0) { // Read
            // Accumulate FULL vectors to prevent optimization. (Accumulating
            // only .x lanes lets LLVM narrow the 128-bit loads into b96+b32
            // pairs, halving measured bandwidth.)
            accumulator += data0 + data1 + data2 + data3 + data4 + data5 + data6 + data7;
            accumulator += data8 + data9 + data10 + data11 + data12 + data13 + data14 + data15;
            accumulator += data16 + data17 + data18 + data19 + data20 + data21 + data22 + data23;
            accumulator += data24 + data25 + data26 + data27 + data28 + data29 + data30 + data31;
        } else if (mode == 1) { // Write
            outputData[baseIndex + 0] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 1] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 2] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 3] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 4] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 5] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 6] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 7] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 8] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 9] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 10] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 11] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 12] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 13] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 14] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 15] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 16] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 17] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 18] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 19] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 20] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 21] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 22] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 23] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 24] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 25] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 26] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 27] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 28] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 29] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 30] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            outputData[baseIndex + 31] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        } else { // Read/Write
            outputData[baseIndex + 0] = data0;
            outputData[baseIndex + 1] = data1;
            outputData[baseIndex + 2] = data2;
            outputData[baseIndex + 3] = data3;
            outputData[baseIndex + 4] = data4;
            outputData[baseIndex + 5] = data5;
            outputData[baseIndex + 6] = data6;
            outputData[baseIndex + 7] = data7;
            outputData[baseIndex + 8] = data8;
            outputData[baseIndex + 9] = data9;
            outputData[baseIndex + 10] = data10;
            outputData[baseIndex + 11] = data11;
            outputData[baseIndex + 12] = data12;
            outputData[baseIndex + 13] = data13;
            outputData[baseIndex + 14] = data14;
            outputData[baseIndex + 15] = data15;
            outputData[baseIndex + 16] = data16;
            outputData[baseIndex + 17] = data17;
            outputData[baseIndex + 18] = data18;
            outputData[baseIndex + 19] = data19;
            outputData[baseIndex + 20] = data20;
            outputData[baseIndex + 21] = data21;
            outputData[baseIndex + 22] = data22;
            outputData[baseIndex + 23] = data23;
            outputData[baseIndex + 24] = data24;
            outputData[baseIndex + 25] = data25;
            outputData[baseIndex + 26] = data26;
            outputData[baseIndex + 27] = data27;
            outputData[baseIndex + 28] = data28;
            outputData[baseIndex + 29] = data29;
            outputData[baseIndex + 30] = data30;
            outputData[baseIndex + 31] = data31;
        }

        chunk_index += num_threads;
    }
    
    // Prevent compiler from optimizing away reads (branch never taken, but compiler can't prove it)
    if (accumulator.x > 1e30f) {
        outputData[0] = accumulator;
    }
}
