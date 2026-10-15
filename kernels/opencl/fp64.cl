// Requires OpenCL 1.2+
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void run_benchmark(__global double* data) {
    uint index = get_global_id(0);

    double seed = data[index & 0x1FFFu] * 0.0001;
    double m = 0.00001 + (double)(index & 63u) * 0.00000001;

    // Use 8 independent double accumulators to hide ALU pipeline latency (matching Vulkan fp64.comp)
    double val0 = seed + 1.0;
    double val1 = val0 + 0.1;
    double val2 = val0 + 0.2;
    double val3 = val0 + 0.3;
    double val4 = val0 + 0.4;
    double val5 = val0 + 0.5;
    double val6 = val0 + 0.6;
    double val7 = val0 + 0.7;

    // 256 iters * 8 FMAs * 2 ops = 4096 FP64 ops
    for (int i = 0; i < 256; ++i) {
        val0 = fma(m, val1, val0);
        val1 = fma(m, val2, val1);
        val2 = fma(m, val3, val2);
        val3 = fma(m, val4, val3);
        val4 = fma(m, val5, val4);
        val5 = fma(m, val6, val5);
        val6 = fma(m, val7, val6);
        val7 = fma(m, val0, val7);
    }
    data[index] = val0 + val1 + val2 + val3 + val4 + val5 + val6 + val7;
}

