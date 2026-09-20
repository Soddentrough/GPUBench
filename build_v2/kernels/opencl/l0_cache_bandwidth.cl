__kernel void run_benchmark(__global uint* data) {
    uint r0 = get_global_id(0) * 1, r1 = get_global_id(0) * 2, r2 = get_global_id(0) * 3, r3 = get_global_id(0) * 4;
    uint r4 = get_global_id(0) * 5, r5 = get_global_id(0) * 6, r6 = get_global_id(0) * 7, r7 = get_global_id(0) * 8;
    uint r8 = get_global_id(0) * 9, r9 = get_global_id(0) * 10, r10 = get_global_id(0) * 11, r11 = get_global_id(0) * 12;
    uint r12 = get_global_id(0) * 13, r13 = get_global_id(0) * 14, r14 = get_global_id(0) * 15, r15 = get_global_id(0) * 16;

    uint start_val = data[0];

    for (int i = 0; i < 1024; ++i) {
        r0 += r1 + start_val;
        r1 += r2 + start_val;
        r2 += r3 + start_val;
        r3 += r4 + start_val;
        r4 += r5 + start_val;
        r5 += r6 + start_val;
        r6 += r7 + start_val;
        r7 += r8 + start_val;
        r8 += r9 + start_val;
        r9 += r10 + start_val;
        r10 += r11 + start_val;
        r11 += r12 + start_val;
        r12 += r13 + start_val;
        r13 += r14 + start_val;
        r14 += r15 + start_val;
        r15 += r0 + start_val;
    }

    data[0] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15;
}
