#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float4* inputData, float4* outputData) {
    uint thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint num_threads = gridDim.x * blockDim.x;
    
    // Buffer size is 1GB, which is 64M float4s.
    // Each thread processes a chunk of 16 float4s.
    // Total chunks in buffer is 4M.
    const uint buffer_num_chunks = 4 * 1024 * 1024;
    const uint buffer_mask = buffer_num_chunks - 1;

    uint chunk_index = thread_id;

    #pragma unroll
    for (int i = 0; i < 1024; ++i) {
        uint current_chunk = chunk_index & buffer_mask;
        uint baseIndex = current_chunk * 16;

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

        chunk_index += num_threads;
    }
}
