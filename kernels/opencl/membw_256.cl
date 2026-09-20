__kernel void run_benchmark(__global const float4* restrict inputData,
                           __global float4* restrict outputData,
                           uint mode,
                           uint bufferSize) {
    uint thread_id = get_global_id(0);
    uint num_threads = get_global_size(0);

    uint buffer_num_chunks = bufferSize / 512;
    uint buffer_mask = buffer_num_chunks - 1;

    uint chunk_index = thread_id;
    float4 accumulator = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        uint current_chunk = chunk_index & buffer_mask;
        uint baseIndex = current_chunk * 32;

        if (mode == 1) { // Write
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                outputData[baseIndex + j] = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
            }
        } else {
            float4 data[32];
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                data[j] = inputData[baseIndex + j];
            }

            if (mode == 0) { // Read
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    accumulator += data[j];
                }
            } else { // Read/Write
                #pragma unroll
                for (int j = 0; j < 32; ++j) {
                    outputData[baseIndex + j] = data[j];
                }
            }
        }

        chunk_index += num_threads;
    }

    // Prevent compiler from optimizing away reads
    if (accumulator.x > 1e30f) {
        outputData[0] = accumulator;
    }
}
