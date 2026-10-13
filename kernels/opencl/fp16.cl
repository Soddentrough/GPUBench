// Requires OpenCL 1.2+
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void run_benchmark(__global half* data) {
    uint index = get_global_id(0);

    // 32 half2 accumulators for high ILP and hardware saturation.
    half2 base  = vload2(index % 1024u, data);
    half2 val1  = base;
    half2 val2  = base + (half2)(0.01h, 0.02h);
    half2 val3  = base + (half2)(0.03h, 0.04h);
    half2 val4  = base + (half2)(0.05h, 0.06h);
    half2 val5  = base + (half2)(0.07h, 0.08h);
    half2 val6  = base + (half2)(0.09h, 0.10h);
    half2 val7  = base + (half2)(0.11h, 0.12h);
    half2 val8  = base + (half2)(0.13h, 0.14h);
    half2 val9  = base + (half2)(0.15h, 0.16h);
    half2 val10 = base + (half2)(0.17h, 0.18h);
    half2 val11 = base + (half2)(0.19h, 0.20h);
    half2 val12 = base + (half2)(0.21h, 0.22h);
    half2 val13 = base + (half2)(0.23h, 0.24h);
    half2 val14 = base + (half2)(0.25h, 0.26h);
    half2 val15 = base + (half2)(0.27h, 0.28h);
    half2 val16 = base + (half2)(0.29h, 0.30h);
    half2 val17 = base + (half2)(0.31h, 0.32h);
    half2 val18 = base + (half2)(0.33h, 0.34h);
    half2 val19 = base + (half2)(0.35h, 0.36h);
    half2 val20 = base + (half2)(0.37h, 0.38h);
    half2 val21 = base + (half2)(0.39h, 0.40h);
    half2 val22 = base + (half2)(0.41h, 0.42h);
    half2 val23 = base + (half2)(0.43h, 0.44h);
    half2 val24 = base + (half2)(0.45h, 0.46h);
    half2 val25 = base + (half2)(0.47h, 0.48h);
    half2 val26 = base + (half2)(0.49h, 0.50h);
    half2 val27 = base + (half2)(0.51h, 0.52h);
    half2 val28 = base + (half2)(0.53h, 0.54h);
    half2 val29 = base + (half2)(0.55h, 0.56h);
    half2 val30 = base + (half2)(0.57h, 0.58h);
    half2 val31 = base + (half2)(0.59h, 0.60h);
    half2 val32 = base + (half2)(0.61h, 0.62h);

    half2 m = (half2)(-0.9990234375h);
    half2 c = (half2)(0.5h);

    // Each iteration performs 32 half2 FMAs = 32 * 2 * 2 = 128 FP16 ops
    for (int i = 0; i < 16384; ++i) {
        val1  = fma(val1,  m, c);
        val2  = fma(val2,  m, c);
        val3  = fma(val3,  m, c);
        val4  = fma(val4,  m, c);
        val5  = fma(val5,  m, c);
        val6  = fma(val6,  m, c);
        val7  = fma(val7,  m, c);
        val8  = fma(val8,  m, c);
        val9  = fma(val9,  m, c);
        val10 = fma(val10, m, c);
        val11 = fma(val11, m, c);
        val12 = fma(val12, m, c);
        val13 = fma(val13, m, c);
        val14 = fma(val14, m, c);
        val15 = fma(val15, m, c);
        val16 = fma(val16, m, c);
        val17 = fma(val17, m, c);
        val18 = fma(val18, m, c);
        val19 = fma(val19, m, c);
        val20 = fma(val20, m, c);
        val21 = fma(val21, m, c);
        val22 = fma(val22, m, c);
        val23 = fma(val23, m, c);
        val24 = fma(val24, m, c);
        val25 = fma(val25, m, c);
        val26 = fma(val26, m, c);
        val27 = fma(val27, m, c);
        val28 = fma(val28, m, c);
        val29 = fma(val29, m, c);
        val30 = fma(val30, m, c);
        val31 = fma(val31, m, c);
        val32 = fma(val32, m, c);
    }
    
    // Prevent optimization away
    half2 result = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 +
                  val9 + val10 + val11 + val12 + val13 + val14 + val15 + val16 +
                  val17 + val18 + val19 + val20 + val21 + val22 + val23 + val24 +
                  val25 + val26 + val27 + val28 + val29 + val30 + val31 + val32;
    vstore2(result, index, data);
}

