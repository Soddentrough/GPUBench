// Requires OpenCL 1.2+
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void run_benchmark(__global double* data) {
    uint index = get_global_id(0);

    double add = (double)(index) * 0.00000001;
    double mult = 1.000001 + add;
    double val = data[index] + 1.0;

    for (int i = 0; i < 65536; ++i) {
        val = fma(val, mult, 1.0);
    }
    data[index] = val;
}

