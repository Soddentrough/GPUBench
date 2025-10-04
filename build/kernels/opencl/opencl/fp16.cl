// Requires OpenCL 1.2+
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void run_benchmark(__global half* data) {
    uint index = get_global_id(0);

    // Work with multiple packed accumulators to avoid dependency chains
    // Using half2 for 2x throughput
    half2 val1 = vload2(index, data);
    half2 val2 = (half2)(0.1h, 0.2h);
    half2 val3 = (half2)(0.3h, 0.4h);
    half2 val4 = (half2)(0.5h, 0.6h);
    half2 val5 = (half2)(0.7h, 0.8h);
    half2 val6 = (half2)(0.9h, 1.0h);
    half2 val7 = (half2)(1.1h, 1.2h);
    half2 val8 = (half2)(1.3h, 1.4h);
    
    // Each iteration performs 8 half2 FMAs = 8 * 2 * 2 = 32 FP16 ops
    for (int i = 0; i < 16384; ++i) {
        val1 = fma(val1, (half2)(1.0001h), val2);
        val2 = fma(val2, (half2)(1.0001h), val3);
        val3 = fma(val3, (half2)(1.0001h), val4);
        val4 = fma(val4, (half2)(1.0001h), val5);
        val5 = fma(val5, (half2)(1.0001h), val6);
        val6 = fma(val6, (half2)(1.0001h), val7);
        val7 = fma(val7, (half2)(1.0001h), val8);
        val8 = fma(val8, (half2)(1.0001h), val1);
    }
    
    // Prevent optimization away
    half2 result = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8;
    vstore2(result, index, data);
}
