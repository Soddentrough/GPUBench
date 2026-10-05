__kernel void run_benchmark(__global const float4* restrict inputData,
                           __global float4* restrict outputData,
                           uint mode,
                           uint bufferSize) {
    uint thread_id = get_global_id(0);
    uint num_threads = get_global_size(0);

    uint num_float4 = bufferSize / sizeof(float4);
    uint buffer_mask = num_float4 - 1;

    float4 accumulator = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            uint idx = (thread_id + (i * 32 + j) * num_threads) & buffer_mask;
            if (mode == 1) { // Write
                outputData[idx] = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
            } else if (mode == 0) { // Read
                accumulator += inputData[idx];
            } else { // Read/Write
                outputData[idx] = inputData[idx];
            }
        }
    }

    // Prevent compiler from optimizing away reads
    if (accumulator.x > 1e30f) {
        outputData[0] = accumulator;
    }
}
