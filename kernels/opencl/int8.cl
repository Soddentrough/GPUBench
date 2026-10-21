// Requires OpenCL 1.2+
// Cross-vendor INT8 vector benchmark with optional native ISA acceleration.

inline int sdot4_packed(int a_packed, int b_packed, int c) {
#if defined(__AMDGPU__) && defined(__IMAGE_SUPPORT__)
    // On AMD GPUs with inline assembly support, use native dot4 instruction
    int dst;
    __asm__ volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(dst) : "v"(a_packed), "v"(b_packed), "v"(c));
    return dst;
#else
    // Standard OpenCL C portable dot product
    char4 a = as_char4(a_packed);
    char4 b = as_char4(b_packed);
    return c + ((int)a.x * (int)b.x + (int)a.y * (int)b.y + (int)a.z * (int)b.z + (int)a.w * (int)b.w);
#endif
}

inline int pack_char4(char4 v) {
    return ((int)v.x & 0xFF) | (((int)v.y & 0xFF) << 8) | (((int)v.z & 0xFF) << 16) | (((int)v.w & 0xFF) << 24);
}

__kernel void run_benchmark(__global char4* data) {
    uint index = get_global_id(0);

    // Dynamic load — compiler cannot constant-fold loop body.
    char4 a = data[index & 0x1FFFu];

    // 8 constant packed int32 weight vectors
    const int w0 = ( 1 & 0xFF) | (( 2 & 0xFF) << 8) | (( 3 & 0xFF) << 16) | (( 4 & 0xFF) << 24);
    const int w1 = ( 5 & 0xFF) | (( 6 & 0xFF) << 8) | (( 7 & 0xFF) << 16) | (( 8 & 0xFF) << 24);
    const int w2 = ( 9 & 0xFF) | ((10 & 0xFF) << 8) | ((11 & 0xFF) << 16) | ((12 & 0xFF) << 24);
    const int w3 = (13 & 0xFF) | ((14 & 0xFF) << 8) | ((15 & 0xFF) << 16) | ((16 & 0xFF) << 24);
    const int w4 = (17 & 0xFF) | ((18 & 0xFF) << 8) | ((19 & 0xFF) << 16) | ((20 & 0xFF) << 24);
    const int w5 = (21 & 0xFF) | ((22 & 0xFF) << 8) | ((23 & 0xFF) << 16) | ((24 & 0xFF) << 24);
    const int w6 = (25 & 0xFF) | ((26 & 0xFF) << 8) | ((27 & 0xFF) << 16) | ((28 & 0xFF) << 24);
    const int w7 = (29 & 0xFF) | ((30 & 0xFF) << 8) | ((31 & 0xFF) << 16) | ((32 & 0xFF) << 24);

    int base_packed = pack_char4(a);
    int acc0 = base_packed ^ 0x01010101;
    int acc1 = base_packed ^ 0x02020202;
    int acc2 = base_packed ^ 0x03030303;
    int acc3 = base_packed ^ 0x04040404;
    int acc4 = base_packed ^ 0x05050505;
    int acc5 = base_packed ^ 0x06060606;
    int acc6 = base_packed ^ 0x07070707;
    int acc7 = base_packed ^ 0x08080808;

    for (int i = 0; i < 16384; ++i) {
        acc0 = sdot4_packed(acc0, w0, acc0);
        acc1 = sdot4_packed(acc1, w1, acc1);
        acc2 = sdot4_packed(acc2, w2, acc2);
        acc3 = sdot4_packed(acc3, w3, acc3);
        acc4 = sdot4_packed(acc4, w4, acc4);
        acc5 = sdot4_packed(acc5, w5, acc5);
        acc6 = sdot4_packed(acc6, w6, acc6);
        acc7 = sdot4_packed(acc7, w7, acc7);
    }

    int total = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    char4 result = (char4)((char)((total >> 16) | 1));
    data[index] = result;
}
