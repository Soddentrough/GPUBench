// Requires OpenCL 1.2+

__kernel void run_benchmark(__global char* data) {
    uint index = get_global_id(0);

    // Use packed char4 for 4x throughput
    // Emulate 4-bit operations (values kept in 0-15 range)
    char4 val1 = vload4(index, data) & (char4)(0x0F);
    char4 val2 = (char4)(1, 2, 3, 4);
    char4 val3 = (char4)(5, 6, 7, 8);
    char4 val4 = (char4)(9, 10, 11, 12);
    char4 val5 = (char4)(13, 14, 15, 1);
    char4 val6 = (char4)(2, 3, 4, 5);
    char4 val7 = (char4)(6, 7, 8, 9);
    char4 val8 = (char4)(10, 11, 12, 13);
    char4 val9 = (char4)(14, 15, 1, 2);
    char4 val10 = (char4)(3, 4, 5, 6);
    char4 val11 = (char4)(7, 8, 9, 10);
    char4 val12 = (char4)(11, 12, 13, 14);
    
    // Each iteration performs 12 char4 multiply-adds = 12 * 4 * 2 = 96 INT4 ops
    // (counting 2 ops per 4-bit value packed in each char4 component)
    for (int i = 0; i < 16384; ++i) {
        val1 = (val1 * (char4)(3) + val2) & (char4)(0x0F);
        val2 = (val2 * (char4)(3) + val3) & (char4)(0x0F);
        val3 = (val3 * (char4)(3) + val4) & (char4)(0x0F);
        val4 = (val4 * (char4)(3) + val5) & (char4)(0x0F);
        val5 = (val5 * (char4)(3) + val6) & (char4)(0x0F);
        val6 = (val6 * (char4)(3) + val7) & (char4)(0x0F);
        val7 = (val7 * (char4)(3) + val8) & (char4)(0x0F);
        val8 = (val8 * (char4)(3) + val9) & (char4)(0x0F);
        val9 = (val9 * (char4)(3) + val10) & (char4)(0x0F);
        val10 = (val10 * (char4)(3) + val11) & (char4)(0x0F);
        val11 = (val11 * (char4)(3) + val12) & (char4)(0x0F);
        val12 = (val12 * (char4)(3) + val1) & (char4)(0x0F);
    }
    
    char4 result = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 + val9 + val10 + val11 + val12;
    vstore4(result, index, data);
}
