__kernel void run_benchmark(__global uint* data) {
    uint index = 0;
    for (int i = 0; i < 1024; ++i) {
        index = data[index];
    }
    data[0] = index;
}
