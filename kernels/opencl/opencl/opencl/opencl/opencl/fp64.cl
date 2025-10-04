// Requires OpenCL 1.2+
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void run_benchmark(__global double* data) {
    uint index = get_global_id(0);

    const double c1 = 0.5;
    const double c2 = 0.25;
    double val = data[index];
    for (int i = 0; i < 65536; ++i) {
        val = val * c1 + c2;
    }
    data[index] = val;
}
