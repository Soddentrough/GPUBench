// Requires ROCm 6.4+
#include <hip/hip_runtime.h>

extern "C" __global__ void compute(char4* data) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    char4 val1 = data[index];
    char4 val2 = make_char4(1, 2, 3, 4);
    char4 val3 = make_char4(5, 6, 7, 8);
    char4 val4 = make_char4(9, 10, 11, 12);
    char4 val5 = make_char4(13, 14, 15, 1);
    char4 val6 = make_char4(2, 3, 4, 5);
    char4 val7 = make_char4(6, 7, 8, 9);
    char4 val8 = make_char4(10, 11, 12, 13);
    char4 val9 = make_char4(14, 15, 1, 2);
    char4 val10 = make_char4(3, 4, 5, 6);
    char4 val11 = make_char4(7, 8, 9, 10);
    char4 val12 = make_char4(11, 12, 13, 14);
    
    for (int i = 0; i < 16384; ++i) {
        val1 = make_char4((val1.x * 3 + val2.x) & 0x0F, (val1.y * 3 + val2.y) & 0x0F, (val1.z * 3 + val2.z) & 0x0F, (val1.w * 3 + val2.w) & 0x0F);
        val2 = make_char4((val2.x * 3 + val3.x) & 0x0F, (val2.y * 3 + val3.y) & 0x0F, (val2.z * 3 + val3.z) & 0x0F, (val2.w * 3 + val3.w) & 0x0F);
        val3 = make_char4((val3.x * 3 + val4.x) & 0x0F, (val3.y * 3 + val4.y) & 0x0F, (val3.z * 3 + val4.z) & 0x0F, (val3.w * 3 + val4.w) & 0x0F);
        val4 = make_char4((val4.x * 3 + val5.x) & 0x0F, (val4.y * 3 + val5.y) & 0x0F, (val4.z * 3 + val5.z) & 0x0F, (val4.w * 3 + val5.w) & 0x0F);
        val5 = make_char4((val5.x * 3 + val6.x) & 0x0F, (val5.y * 3 + val6.y) & 0x0F, (val5.z * 3 + val6.z) & 0x0F, (val5.w * 3 + val6.w) & 0x0F);
        val6 = make_char4((val6.x * 3 + val7.x) & 0x0F, (val6.y * 3 + val7.y) & 0x0F, (val6.z * 3 + val7.z) & 0x0F, (val6.w * 3 + val7.w) & 0x0F);
        val7 = make_char4((val7.x * 3 + val8.x) & 0x0F, (val7.y * 3 + val8.y) & 0x0F, (val7.z * 3 + val8.z) & 0x0F, (val7.w * 3 + val8.w) & 0x0F);
        val8 = make_char4((val8.x * 3 + val9.x) & 0x0F, (val8.y * 3 + val9.y) & 0x0F, (val8.z * 3 + val9.z) & 0x0F, (val8.w * 3 + val9.w) & 0x0F);
        val9 = make_char4((val9.x * 3 + val10.x) & 0x0F, (val9.y * 3 + val10.y) & 0x0F, (val9.z * 3 + val10.z) & 0x0F, (val9.w * 3 + val10.w) & 0x0F);
        val10 = make_char4((val10.x * 3 + val11.x) & 0x0F, (val10.y * 3 + val11.y) & 0x0F, (val10.z * 3 + val11.z) & 0x0F, (val10.w * 3 + val11.w) & 0x0F);
        val11 = make_char4((val11.x * 3 + val12.x) & 0x0F, (val11.y * 3 + val12.y) & 0x0F, (val11.z * 3 + val12.z) & 0x0F, (val11.w * 3 + val12.w) & 0x0F);
        val12 = make_char4((val12.x * 3 + val1.x) & 0x0F, (val12.y * 3 + val1.y) & 0x0F, (val12.z * 3 + val1.z) & 0x0F, (val12.w * 3 + val1.w) & 0x0F);
    }
    
    data[index] = make_char4(val1.x + val2.x + val3.x + val4.x + val5.x + val6.x + val7.x + val8.x + val9.x + val10.x + val11.x + val12.x,
                             val1.y + val2.y + val3.y + val4.y + val5.y + val6.y + val7.y + val8.y + val9.y + val10.y + val11.y + val12.y,
                             val1.z + val2.z + val3.z + val4.z + val5.z + val6.z + val7.z + val8.z + val9.z + val10.z + val11.z + val12.z,
                             val1.w + val2.w + val3.w + val4.w + val5.w + val6.w + val7.w + val8.w + val9.w + val10.w + val11.w + val12.w);
}
