// Requires OpenCL 1.2+
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void run_benchmark(__global half* data) {
    uint index = get_global_id(0);

    half2 val1  = vload2(index, data);
    half2 val2  = (half2)(0.1h, 0.2h);
    half2 val3  = (half2)(0.3h, 0.4h);
    half2 val4  = (half2)(0.5h, 0.6h);
    half2 val5  = (half2)(0.7h, 0.8h);
    half2 val6  = (half2)(0.9h, 1.0h);
    half2 val7  = (half2)(1.1h, 1.2h);
    half2 val8  = (half2)(1.3h, 1.4h);
    half2 val9  = (half2)(1.5h, 1.6h);
    half2 val10 = (half2)(1.7h, 1.8h);
    half2 val11 = (half2)(1.9h, 2.0h);
    half2 val12 = (half2)(2.1h, 2.2h);
    half2 val13 = (half2)(2.3h, 2.4h);
    half2 val14 = (half2)(2.5h, 2.6h);
    half2 val15 = (half2)(2.7h, 2.8h);
    half2 val16 = (half2)(2.9h, 3.0h);
    half2 val17 = (half2)(3.1h, 3.2h);
    half2 val18 = (half2)(3.3h, 3.4h);
    half2 val19 = (half2)(3.5h, 3.6h);
    half2 val20 = (half2)(3.7h, 3.8h);
    half2 val21 = (half2)(3.9h, 4.0h);
    half2 val22 = (half2)(4.1h, 4.2h);
    half2 val23 = (half2)(4.3h, 4.4h);
    half2 val24 = (half2)(4.5h, 4.6h);
    half2 val25 = (half2)(4.7h, 4.8h);
    half2 val26 = (half2)(4.9h, 5.0h);
    half2 val27 = (half2)(5.1h, 5.2h);
    half2 val28 = (half2)(5.3h, 5.4h);
    half2 val29 = (half2)(5.5h, 5.6h);
    half2 val30 = (half2)(5.7h, 5.8h);
    half2 val31 = (half2)(5.9h, 6.0h);
    half2 val32 = (half2)(6.1h, 6.2h);

    half2 m = (half2)(1.0001h);

    // 32 half2 FMAs × 2 components × 2 ops = 128 ops per iteration.
    for (int i = 0; i < 16384; ++i) {
        val1  = fma(val1,  m, val2);
        val2  = fma(val2,  m, val3);
        val3  = fma(val3,  m, val4);
        val4  = fma(val4,  m, val5);
        val5  = fma(val5,  m, val6);
        val6  = fma(val6,  m, val7);
        val7  = fma(val7,  m, val8);
        val8  = fma(val8,  m, val9);
        val9  = fma(val9,  m, val10);
        val10 = fma(val10, m, val11);
        val11 = fma(val11, m, val12);
        val12 = fma(val12, m, val13);
        val13 = fma(val13, m, val14);
        val14 = fma(val14, m, val15);
        val15 = fma(val15, m, val16);
        val16 = fma(val16, m, val17);
        val17 = fma(val17, m, val18);
        val18 = fma(val18, m, val19);
        val19 = fma(val19, m, val20);
        val20 = fma(val20, m, val21);
        val21 = fma(val21, m, val22);
        val22 = fma(val22, m, val23);
        val23 = fma(val23, m, val24);
        val24 = fma(val24, m, val25);
        val25 = fma(val25, m, val26);
        val26 = fma(val26, m, val27);
        val27 = fma(val27, m, val28);
        val28 = fma(val28, m, val29);
        val29 = fma(val29, m, val30);
        val30 = fma(val30, m, val31);
        val31 = fma(val31, m, val32);
        val32 = fma(val32, m, val1);
    }
    
    half2 result = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 +
                  val9 + val10 + val11 + val12 + val13 + val14 + val15 + val16 +
                  val17 + val18 + val19 + val20 + val21 + val22 + val23 + val24 +
                  val25 + val26 + val27 + val28 + val29 + val30 + val31 + val32;
    vstore2(result, index, data);
}

