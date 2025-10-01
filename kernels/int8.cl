// Requires OpenCL 1.2+

__kernel void compute(__global char* data) {
    uint index = get_global_id(0);

    // Use packed char4 for 4x throughput
    char4 val1 = vload4(index, data);
    char4 val2 = (char4)(1, 2, 3, 4);
    char4 val3 = (char4)(5, 6, 7, 8);
    char4 val4 = (char4)(9, 10, 11, 12);
    char4 val5 = (char4)(13, 14, 15, 16);
    char4 val6 = (char4)(17, 18, 19, 20);
    char4 val7 = (char4)(21, 22, 23, 24);
    char4 val8 = (char4)(25, 26, 27, 28);
    
    // Each iteration performs 8 char4 multiply-adds = 8 * 4 * 2 = 64 INT8 ops
    for (int i = 0; i < 16384; ++i) {
        val1 = val1 * (char4)(3) + val2;
        val2 = val2 * (char4)(3) + val3;
        val3 = val3 * (char4)(3) + val4;
        val4 = val4 * (char4)(3) + val5;
        val5 = val5 * (char4)(3) + val6;
        val6 = val6 * (char4)(3) + val7;
        val7 = val7 * (char4)(3) + val8;
        val8 = val8 * (char4)(3) + val1;
    }
    
    char4 result = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8;
    vstore4(result, index, data);
}
