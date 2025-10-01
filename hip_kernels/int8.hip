// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void compute(char4* data) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    char4 val1 = data[index];
    char4 val2 = make_char4(1, 2, 3, 4);
    char4 val3 = make_char4(5, 6, 7, 8);
    char4 val4 = make_char4(9, 10, 11, 12);
    char4 val5 = make_char4(13, 14, 15, 16);
    char4 val6 = make_char4(17, 18, 19, 20);
    char4 val7 = make_char4(21, 22, 23, 24);
    char4 val8 = make_char4(25, 26, 27, 28);
    
    for (int i = 0; i < 16384; ++i) {
        val1 = make_char4(val1.x * 3 + val2.x, val1.y * 3 + val2.y, val1.z * 3 + val2.z, val1.w * 3 + val2.w);
        val2 = make_char4(val2.x * 3 + val3.x, val2.y * 3 + val3.y, val2.z * 3 + val3.z, val2.w * 3 + val3.w);
        val3 = make_char4(val3.x * 3 + val4.x, val3.y * 3 + val4.y, val3.z * 3 + val4.z, val3.w * 3 + val4.w);
        val4 = make_char4(val4.x * 3 + val5.x, val4.y * 3 + val5.y, val4.z * 3 + val5.z, val4.w * 3 + val5.w);
        val5 = make_char4(val5.x * 3 + val6.x, val5.y * 3 + val6.y, val5.z * 3 + val6.z, val5.w * 3 + val6.w);
        val6 = make_char4(val6.x * 3 + val7.x, val6.y * 3 + val7.y, val6.z * 3 + val7.z, val6.w * 3 + val7.w);
        val7 = make_char4(val7.x * 3 + val8.x, val7.y * 3 + val8.y, val7.z * 3 + val8.z, val7.w * 3 + val8.w);
        val8 = make_char4(val8.x * 3 + val1.x, val8.y * 3 + val1.y, val8.z * 3 + val1.z, val8.w * 3 + val1.w);
    }
    
    data[index] = make_char4(val1.x + val2.x + val3.x + val4.x + val5.x + val6.x + val7.x + val8.x,
                             val1.y + val2.y + val3.y + val4.y + val5.y + val6.y + val7.y + val8.y,
                             val1.z + val2.z + val3.z + val4.z + val5.z + val6.z + val7.z + val8.z,
                             val1.w + val2.w + val3.w + val4.w + val5.w + val6.w + val7.w + val8.w);
}
