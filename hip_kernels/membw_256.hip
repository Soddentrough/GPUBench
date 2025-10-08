#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(float4* inputData, float4* outputData) {
    uint global_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint global_size = gridDim.x * blockDim.x;

    uint baseIndex = global_id * 32;
    uint stride = global_size * 32;
    uint buffer_mask = (4UL * 1024 * 1024 * 1024) / 16 - 1;

    #pragma unroll
    for (int i = 0; i < 1024; ++i) {
        uint currentIndex = (baseIndex + i * stride) & buffer_mask;

        float4 data0 = inputData[currentIndex + 0];
        float4 data1 = inputData[currentIndex + 1];
        float4 data2 = inputData[currentIndex + 2];
        float4 data3 = inputData[currentIndex + 3];
        float4 data4 = inputData[currentIndex + 4];
        float4 data5 = inputData[currentIndex + 5];
        float4 data6 = inputData[currentIndex + 6];
        float4 data7 = inputData[currentIndex + 7];
        float4 data8 = inputData[currentIndex + 8];
        float4 data9 = inputData[currentIndex + 9];
        float4 data10 = inputData[currentIndex + 10];
        float4 data11 = inputData[currentIndex + 11];
        float4 data12 = inputData[currentIndex + 12];
        float4 data13 = inputData[currentIndex + 13];
        float4 data14 = inputData[currentIndex + 14];
        float4 data15 = inputData[currentIndex + 15];
        float4 data16 = inputData[currentIndex + 16];
        float4 data17 = inputData[currentIndex + 17];
        float4 data18 = inputData[currentIndex + 18];
        float4 data19 = inputData[currentIndex + 19];
        float4 data20 = inputData[currentIndex + 20];
        float4 data21 = inputData[currentIndex + 21];
        float4 data22 = inputData[currentIndex + 22];
        float4 data23 = inputData[currentIndex + 23];
        float4 data24 = inputData[currentIndex + 24];
        float4 data25 = inputData[currentIndex + 25];
        float4 data26 = inputData[currentIndex + 26];
        float4 data27 = inputData[currentIndex + 27];
        float4 data28 = inputData[currentIndex + 28];
        float4 data29 = inputData[currentIndex + 29];
        float4 data30 = inputData[currentIndex + 30];
        float4 data31 = inputData[currentIndex + 31];
        
        outputData[currentIndex + 0] = data0 * 2.0f - 1.0f;
        outputData[currentIndex + 1] = data1 * 2.0f - 1.0f;
        outputData[currentIndex + 2] = data2 * 2.0f - 1.0f;
        outputData[currentIndex + 3] = data3 * 2.0f - 1.0f;
        outputData[currentIndex + 4] = data4 * 2.0f - 1.0f;
        outputData[currentIndex + 5] = data5 * 2.0f - 1.0f;
        outputData[currentIndex + 6] = data6 * 2.0f - 1.0f;
        outputData[currentIndex + 7] = data7 * 2.0f - 1.0f;
        outputData[currentIndex + 8] = data8 * 2.0f - 1.0f;
        outputData[currentIndex + 9] = data9 * 2.0f - 1.0f;
        outputData[currentIndex + 10] = data10 * 2.0f - 1.0f;
        outputData[currentIndex + 11] = data11 * 2.0f - 1.0f;
        outputData[currentIndex + 12] = data12 * 2.0f - 1.0f;
        outputData[currentIndex + 13] = data13 * 2.0f - 1.0f;
        outputData[currentIndex + 14] = data14 * 2.0f - 1.0f;
        outputData[currentIndex + 15] = data15 * 2.0f - 1.0f;
        outputData[currentIndex + 16] = data16 * 2.0f - 1.0f;
        outputData[currentIndex + 17] = data17 * 2.0f - 1.0f;
        outputData[currentIndex + 18] = data18 * 2.0f - 1.0f;
        outputData[currentIndex + 19] = data19 * 2.0f - 1.0f;
        outputData[currentIndex + 20] = data20 * 2.0f - 1.0f;
        outputData[currentIndex + 21] = data21 * 2.0f - 1.0f;
        outputData[currentIndex + 22] = data22 * 2.0f - 1.0f;
        outputData[currentIndex + 23] = data23 * 2.0f - 1.0f;
        outputData[currentIndex + 24] = data24 * 2.0f - 1.0f;
        outputData[currentIndex + 25] = data25 * 2.0f - 1.0f;
        outputData[currentIndex + 26] = data26 * 2.0f - 1.0f;
        outputData[currentIndex + 27] = data27 * 2.0f - 1.0f;
        outputData[currentIndex + 28] = data28 * 2.0f - 1.0f;
        outputData[currentIndex + 29] = data29 * 2.0f - 1.0f;
        outputData[currentIndex + 30] = data30 * 2.0f - 1.0f;
        outputData[currentIndex + 31] = data31 * 2.0f - 1.0f;
    }
}
