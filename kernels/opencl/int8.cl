// Requires OpenCL 1.2+
// INT8 vector benchmark using native AMDGPU v_dot4 assembly.

inline int sdot4_asm(int a, int b, int c) {
    int dst;
    __asm__ volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(dst) : "v"(a), "v"(b), "v"(c));
    return dst;
}

__kernel void run_benchmark(__global char4* data) {
    uint index = get_global_id(0);

    // Dynamic load — compiler cannot constant-fold loop body.
    char4 a = data[index & 0x1FFFu];

    // 8 constant weight vectors (different per accumulator prevents merging).
    char4 w0 = (char4)( 1,  2,  3,  4);
    char4 w1 = (char4)( 5,  6,  7,  8);
    char4 w2 = (char4)( 9, 10, 11, 12);
    char4 w3 = (char4)(13, 14, 15, 16);
    char4 w4 = (char4)(17, 18, 19, 20);
    char4 w5 = (char4)(21, 22, 23, 24);
    char4 w6 = (char4)(25, 26, 27, 28);
    char4 w7 = (char4)(29, 30, 31, 32);

    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    for (int i = 0; i < 16384; ++i) {
        // ai varies each iteration to prevent loop hoisting.
        char4 ai = a + (char4)((char)i);

        acc0 = sdot4_asm(as_int(ai), as_int(w0), acc0);
        acc1 = sdot4_asm(as_int(ai), as_int(w1), acc1);
        acc2 = sdot4_asm(as_int(ai), as_int(w2), acc2);
        acc3 = sdot4_asm(as_int(ai), as_int(w3), acc3);
        acc4 = sdot4_asm(as_int(ai), as_int(w4), acc4);
        acc5 = sdot4_asm(as_int(ai), as_int(w5), acc5);
        acc6 = sdot4_asm(as_int(ai), as_int(w6), acc6);
        acc7 = sdot4_asm(as_int(ai), as_int(w7), acc7);
    }

    int total = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    char4 result = (char4)((char)total);
    data[index] = result;
}
