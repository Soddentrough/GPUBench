// Requires ROCm 6.4+
// INT8 vector benchmark using the dot4 MAC pattern.
// AMDGPU LLVM auto-maps the "4 sign-extended int8 multiply-add" pattern to V_DOT4_I32_IU8.
// Uses __attribute__((target("dot1-insts"))) on the kernel to enable the feature.
// 8 independent int32 accumulators × 8 INT8 ops per dot4 = 64 INT8 ops per iteration.
#include <hip/hip_runtime.h>

// Helper: signed int8 dot product of 4 pairs into int32 + accumulate.
// LLVM lowers this to V_DOT4_I32_IU8 when targeting GFX1201.
__device__ __forceinline__
int sdot4(char4 a, char4 b, int c) {
    return (int)a.x * (int)b.x
         + (int)a.y * (int)b.y
         + (int)a.z * (int)b.z
         + (int)a.w * (int)b.w
         + c;
}

extern "C" __global__ void run_benchmark(char4* data) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    // Dynamic load — compiler cannot constant-fold loop body.
    char4 a = data[index & 0x1FFFu];

    const char4 w0 = make_char4( 1,  2,  3,  4);
    const char4 w1 = make_char4( 5,  6,  7,  8);
    const char4 w2 = make_char4( 9, 10, 11, 12);
    const char4 w3 = make_char4(13, 14, 15, 16);
    const char4 w4 = make_char4(17, 18, 19, 20);
    const char4 w5 = make_char4(21, 22, 23, 24);
    const char4 w6 = make_char4(25, 26, 27, 28);
    const char4 w7 = make_char4(29, 30, 31, 32);

    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;

    #pragma unroll 4
    for (int i = 0; i < 512; ++i) {
        // ai varies each iteration to prevent loop hoisting.
        char4 ai = make_char4(a.x + (char)i, a.y + (char)i,
                              a.z + (char)i, a.w + (char)i);

        acc0 = sdot4(ai, w0, acc0);
        acc1 = sdot4(ai, w1, acc1);
        acc2 = sdot4(ai, w2, acc2);
        acc3 = sdot4(ai, w3, acc3);
        acc4 = sdot4(ai, w4, acc4);
        acc5 = sdot4(ai, w5, acc5);
        acc6 = sdot4(ai, w6, acc6);
        acc7 = sdot4(ai, w7, acc7);
    }

    int total = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
    data[index] = make_char4((char)total, (char)(total >> 8),
                             (char)(total >> 16), (char)(total >> 24));
}
