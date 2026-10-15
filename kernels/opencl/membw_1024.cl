__kernel void run_benchmark(__global const float4* restrict inputData,
                           __global float4* restrict outputData,
                           uint mode,
                           uint bufferSize) {
    uint thread_id = get_global_id(0);
    uint num_threads = get_global_size(0);

    uint num_float4 = bufferSize / sizeof(float4);
    uint buffer_mask = num_float4 - 1;

    float4 accumulator = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    if (mode == 1) { // Write
        for (int i = 0; i < 32; ++i) {
            uint base = (i * 32) * num_threads + thread_id;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                outputData[(base + j * num_threads) & buffer_mask] = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
            }
        }
    } else if (mode == 0) { // Read
        for (int i = 0; i < 32; ++i) {
            uint base = (i * 32) * num_threads + thread_id;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                accumulator += inputData[(base + j * num_threads) & buffer_mask];
            }
        }
    } else { // Read/Write
        for (int i = 0; i < 32; ++i) {
            uint base = (i * 32) * num_threads + thread_id;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                uint idx = (base + j * num_threads) & buffer_mask;
                outputData[idx] = inputData[idx];
            }
        }
    }

    // Genuine writeback: unconditionally commit accumulated loads to global memory in read mode
    if (mode == 0) {
        outputData[thread_id] = accumulator;
    }
}
