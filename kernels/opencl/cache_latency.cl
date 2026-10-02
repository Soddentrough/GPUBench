__kernel void run_benchmark(__global uint* data, __global uint* pc) {
    uint tid = get_local_id(0);
    uint iterations = pc ? pc[2] : 1000000;
    uint val = tid;

    for (uint i = 0; i < iterations; ++i) {
        val = data[val + tid];
    }

    if (val == 0xDEADBEEF) {
        data[0] = val;
    }
}
