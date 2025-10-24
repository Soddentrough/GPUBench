__kernel void run_benchmark(__global float4* inputData, __global float4* outputData, uint mode) {
    uint baseIndex = get_global_id(0) * 32;
    uint stride = get_global_size(0) * 32;
    uint buffer_mask = (256 * 1024 * 1024) / 16 - 1;

    float4 data[32];

    for (int i = 0; i < 1024; ++i) {
        uint currentIndex = (baseIndex + i * stride) & buffer_mask;

        // Mode 0: Read, Mode 2: ReadWrite
        if (mode == 0 || mode == 2) {
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                data[j] = inputData[currentIndex + j];
            }
        }

        // Mode 1: Write
        if (mode == 1) {
            float4 val = (float4)((float)currentIndex, 1.0f, 2.0f, 3.0f);
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                data[j] = val;
            }
        }
        
        // Mode 1: Write, Mode 2: ReadWrite
        if (mode == 1 || mode == 2) {
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                outputData[currentIndex + j] = data[j];
            }
        }
    }
}
