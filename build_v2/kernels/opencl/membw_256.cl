__kernel void run_benchmark(__global float4* inputData, __global float4* outputData, uint mode, uint bufferSize) {
    uint baseIndex = get_global_id(0) * 32;
    uint stride = get_global_size(0) * 32;
    uint buffer_mask = (bufferSize / 16) - 1;

    float4 data[32];
    float4 accumulator = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = 0; i < 32; ++i) {
        uint currentIndex = (baseIndex + i * stride) & buffer_mask;

        // Mode 0: Read, Mode 2: ReadWrite
        if (mode == 0 || mode == 2) {
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                data[j] = inputData[(currentIndex + j) & buffer_mask];
                accumulator += data[j];
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
                outputData[(currentIndex + j) & buffer_mask] = data[j];
            }
        }
    }
    
    // Prevent compiler from optimizing away reads (branch never taken, but compiler can't prove it)
    if (accumulator.x > 1e30f) {
        outputData[0] = accumulator;
    }
}
