// Requires OpenCL 1.2+
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void run_benchmark(__global half* data) {
    uint index = get_global_id(0);

    // Emulating FP4 with FP16 packed values (half4 = 8 FP4-equivalent values)
    // Using multiple accumulators to avoid dependency chains
    half4 val1 = vload4(index, data);
    half4 val2 = (half4)(0.1h, 0.2h, 0.3h, 0.4h);
    half4 val3 = (half4)(0.5h, 0.6h, 0.7h, 0.8h);
    half4 val4 = (half4)(0.9h, 1.0h, 1.1h, 1.2h);
    half4 val5 = (half4)(1.3h, 1.4h, 1.5h, 1.6h);
    half4 val6 = (half4)(1.7h, 1.8h, 1.9h, 2.0h);
    half4 val7 = (half4)(2.1h, 2.2h, 2.3h, 2.4h);
    half4 val8 = (half4)(2.5h, 2.6h, 2.7h, 2.8h);
    half4 val9 = (half4)(0.1h, 0.2h, 0.3h, 0.4h);
    half4 val10 = (half4)(0.5h, 0.6h, 0.7h, 0.8h);
    half4 val11 = (half4)(0.9h, 1.0h, 1.1h, 1.2h);
    half4 val12 = (half4)(1.3h, 1.4h, 1.5h, 1.6h);
    half4 val13 = (half4)(1.7h, 1.8h, 1.9h, 2.0h);
    half4 val14 = (half4)(2.1h, 2.2h, 2.3h, 2.4h);
    half4 val15 = (half4)(2.5h, 2.6h, 2.7h, 2.8h);
    half4 val16 = (half4)(2.5h, 2.6h, 2.7h, 2.8h);
    
    // Each iteration performs 16 half4 FMAs = 16 * 4 * 2 = 128 FP4-equivalent ops
    // This emulates what FP4 workloads would look like with higher throughput
    for (int i = 0; i < 16384; ++i) {
        val1 = fma(val1, (half4)(1.0001h), val2);
        val2 = fma(val2, (half4)(1.0001h), val3);
        val3 = fma(val3, (half4)(1.0001h), val4);
        val4 = fma(val4, (half4)(1.0001h), val5);
        val5 = fma(val5, (half4)(1.0001h), val6);
        val6 = fma(val6, (half4)(1.0001h), val7);
        val7 = fma(val7, (half4)(1.0001h), val8);
        val8 = fma(val8, (half4)(1.0001h), val9);
        val9 = fma(val9, (half4)(1.0001h), val10);
        val10 = fma(val10, (half4)(1.0001h), val11);
        val11 = fma(val11, (half4)(1.0001h), val12);
        val12 = fma(val12, (half4)(1.0001h), val13);
        val13 = fma(val13, (half4)(1.0001h), val14);
        val14 = fma(val14, (half4)(1.0001h), val15);
        val15 = fma(val15, (half4)(1.0001h), val16);
        val16 = fma(val16, (half4)(1.0001h), val1);
    }
    
    // Prevent optimization away
    half4 result = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 + val9 + val10 + val11 + val12 + val13 + val14 + val15 + val16;
    vstore4(result, index, data);
}
