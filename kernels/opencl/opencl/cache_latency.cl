__kernel void cl_compute(__global uint* data) {
    uint index = 0;
    for (int i = 0; i < 1024; ++i) {
        index = data[index];
    }
    data[0] = index;
}
