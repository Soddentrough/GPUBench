__kernel void run_benchmark(__global uint* data, __global uint* pc) {
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];

    uint index = 0;
    for (uint i = 0; i < iterations; ++i) {
        index = data[index];
    }
    if (stride == 0xFFFFFFFF) { data[1] = mask; }
    data[0] = index;
}
