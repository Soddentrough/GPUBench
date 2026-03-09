// Requires OpenCL 1.2+
// INT8 vector benchmark using dot4 accumulation.
// OpenCL's dot(char4, char4) → int maps to V_DOT4_I32_IU8 on AMD RDNA4.
// 8 independent int32 accumulators × 8 INT8 ops per dot4 = 64 INT8 ops/iter.

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

        acc0 += dot(convert_int4(ai), convert_int4(w0));
        acc1 += dot(convert_int4(ai), convert_int4(w1));
        acc2 += dot(convert_int4(ai), convert_int4(w2));
        acc3 += dot(convert_int4(ai), convert_int4(w3));
        acc4 += dot(convert_int4(ai), convert_int4(w4));
        acc5 += dot(convert_int4(ai), convert_int4(w5));
        acc6 += dot(convert_int4(ai), convert_int4(w6));
        acc7 += dot(convert_int4(ai), convert_int4(w7));
    }

    int total = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    char4 result = (char4)((char)total);
    data[index] = result;
}
