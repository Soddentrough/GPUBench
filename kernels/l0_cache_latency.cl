__kernel void cl_compute(__global uint* data) {
    uint r0 = 0, r1 = 1, r2 = 2, r3 = 3, r4 = 4, r5 = 5, r6 = 6, r7 = 7;
    uint r8 = 8, r9 = 9, r10 = 10, r11 = 11, r12 = 12, r13 = 13, r14 = 14, r15 = 15;

    uint start_val = data[0];

    for (int i = 0; i < 128; ++i) {
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
