// Requires OpenCL 1.2+

// Config 0: Standard FP32 (Single-Issue Baseline)
__kernel void run_dual_issue_ilp4(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = as_float((as_uint(data[index]) & 0x007FFFFFu) | 0x3F800000u);
    float val0 = base;
    float val1 = base + 0.1f;
    float m = multiplier * 0.00001f;

    // 131072 iterations * 2 scalar FMAs = 131072 * 4 = 524,288 FP32 operations
    for (int i = 0; i < 131072; ++i) {
        val0 = fma(val0, m, val1);
        val1 = fma(val1, m, val0);
    }

    data[index] = val0 + val1;
}

// Config 1: Dual-Issue FP32 (Partial Co-Issue - 8 chains)
__kernel void run_dual_issue_ilp8(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = as_float((as_uint(data[index]) & 0x007FFFFFu) | 0x3F800000u);
    float val0 = base;
    float val1 = base + 0.1f;
    float val2 = base + 0.2f;
    float val3 = base + 0.3f;
    float val4 = base + 0.4f;
    float val5 = base + 0.5f;
    float val6 = base + 0.6f;
    float val7 = base + 0.7f;

    float m = multiplier * 0.00001f;

    // 65536 iterations * 8 scalar FMAs = 65536 * 16 = 1,048,576 FP32 operations
    for (int i = 0; i < 65536; ++i) {
        val0 = fma(val0, m, val0);
        val1 = fma(val1, m, val1);
        val2 = fma(val2, m, val2);
        val3 = fma(val3, m, val3);
        val4 = fma(val4, m, val4);
        val5 = fma(val5, m, val5);
        val6 = fma(val6, m, val6);
        val7 = fma(val7, m, val7);
    }

    data[index] = (val0 + val1) + (val2 + val3) + (val4 + val5) + (val6 + val7);
}

// Config 2: Dual-Issue FP32 (Peak Co-Issue - 16 chains)
__kernel void run_dual_issue_ilp16(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = as_float((as_uint(data[index]) & 0x007FFFFFu) | 0x3F800000u);
    float val0  = base;
    float val1  = base + 0.05f;
    float val2  = base + 0.10f;
    float val3  = base + 0.15f;
    float val4  = base + 0.20f;
    float val5  = base + 0.25f;
    float val6  = base + 0.30f;
    float val7  = base + 0.35f;
    float val8  = base + 0.40f;
    float val9  = base + 0.45f;
    float val10 = base + 0.50f;
    float val11 = base + 0.55f;
    float val12 = base + 0.60f;
    float val13 = base + 0.65f;
    float val14 = base + 0.70f;
    float val15 = base + 0.75f;

    float m = multiplier * 0.00001f;

    // 65536 iterations * 16 scalar FMAs = 65536 * 32 = 2,097,152 FP32 operations
    for (int i = 0; i < 65536; ++i) {
        val0  = fma(val0,  m, val0);
        val1  = fma(val1,  m, val1);
        val2  = fma(val2,  m, val2);
        val3  = fma(val3,  m, val3);
        val4  = fma(val4,  m, val4);
        val5  = fma(val5,  m, val5);
        val6  = fma(val6,  m, val6);
        val7  = fma(val7,  m, val7);
        val8  = fma(val8,  m, val8);
        val9  = fma(val9,  m, val9);
        val10 = fma(val10, m, val10);
        val11 = fma(val11, m, val11);
        val12 = fma(val12, m, val12);
        val13 = fma(val13, m, val13);
        val14 = fma(val14, m, val14);
        val15 = fma(val15, m, val15);
    }

    data[index] = ((val0 + val1) + (val2 + val3)) +
                  ((val4 + val5) + (val6 + val7)) +
                  ((val8 + val9) + (val10 + val11)) +
                  ((val12 + val13) + (val14 + val15));
}

// Config 3: Standard INT32 (Single-Issue Baseline)
__kernel void run_dual_issue_int32_4(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    uint ubase = (uint)(data[index] * 1000.0f) + index;
    uint val0 = ubase;
    uint val1 = ubase + 1u;

    uint mu = ((uint)(multiplier * 1000.0f)) | 1u;
    uint mask = (num_elements ^ 0x7FFF7FFFu) | 1u;

    // 131072 iterations * 2 scalar INT32 chains (add + and = 2 ops each) = 131072 * 4 = 524,288 INT32 ops
    for (int i = 0; i < 131072; ++i) {
        val0 = (val1 + mu) & mask;
        val1 = (val0 + mu) & mask;
    }

    data[index] = (float)(((val0 + val1) & 0x00FFFFFFu) | 1u);
}

// Config 4: Dual-Issue INT32 (Partial Co-Issue - 8 chains)
__kernel void run_dual_issue_int32_8(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    uint ubase = (uint)(data[index] * 1000.0f) + index;
    uint u0 = ubase;
    uint u1 = ubase + 1u;
    uint u2 = ubase + 2u;
    uint u3 = ubase + 3u;
    uint u4 = ubase + 4u;
    uint u5 = ubase + 5u;
    uint u6 = ubase + 6u;
    uint u7 = ubase + 7u;

    uint mu = ((uint)(multiplier * 1000.0f)) | 1u;
    uint mask = (num_elements ^ 0x7FFF7FFFu) | 1u;

    // 65536 iterations * 8 scalar INT32 chains (add + and = 2 ops each) = 65536 * 16 = 1,048,576 INT32 ops
    for (int i = 0; i < 65536; ++i) {
        u0 = (u0 + mu) & mask;
        u1 = (u1 + mu) & mask;
        u2 = (u2 + mu) & mask;
        u3 = (u3 + mu) & mask;
        u4 = (u4 + mu) & mask;
        u5 = (u5 + mu) & mask;
        u6 = (u6 + mu) & mask;
        u7 = (u7 + mu) & mask;
    }

    uint ures = (u0 + u1) ^ (u2 + u3) ^ (u4 + u5) ^ (u6 + u7);
    data[index] = (float)((ures & 0x00FFFFFFu) | 1u);
}

// Config 5: Dual-Issue INT32 (Peak Co-Issue - 16 chains)
__kernel void run_dual_issue_int32(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    uint ubase = (uint)(data[index] * 1000.0f) + index;
    uint u0  = ubase;
    uint u1  = ubase + 1u;
    uint u2  = ubase + 2u;
    uint u3  = ubase + 3u;
    uint u4  = ubase + 4u;
    uint u5  = ubase + 5u;
    uint u6  = ubase + 6u;
    uint u7  = ubase + 7u;
    uint u8  = ubase + 8u;
    uint u9  = ubase + 9u;
    uint u10 = ubase + 10u;
    uint u11 = ubase + 11u;
    uint u12 = ubase + 12u;
    uint u13 = ubase + 13u;
    uint u14 = ubase + 14u;
    uint u15 = ubase + 15u;

    uint mu = ((uint)(multiplier * 1000.0f)) | 1u;
    uint mask = (num_elements ^ 0x7FFF7FFFu) | 1u;

    // 65536 iterations * 16 scalar INT32 chains (add + and = 2 ops each) = 65536 * 32 = 2,097,152 INT32 ops
    for (int i = 0; i < 65536; ++i) {
        u0  = (u0  + mu) & mask;
        u1  = (u1  + mu) & mask;
        u2  = (u2  + mu) & mask;
        u3  = (u3  + mu) & mask;
        u4  = (u4  + mu) & mask;
        u5  = (u5  + mu) & mask;
        u6  = (u6  + mu) & mask;
        u7  = (u7  + mu) & mask;
        u8  = (u8  + mu) & mask;
        u9  = (u9  + mu) & mask;
        u10 = (u10 + mu) & mask;
        u11 = (u11 + mu) & mask;
        u12 = (u12 + mu) & mask;
        u13 = (u13 + mu) & mask;
        u14 = (u14 + mu) & mask;
        u15 = (u15 + mu) & mask;
    }

    uint ures = ((u0 + u1) ^ (u2 + u3)) + ((u4 + u5) ^ (u6 + u7)) +
                ((u8 + u9) ^ (u10 + u11)) + ((u12 + u13) ^ (u14 + u15));
    data[index] = (float)((ures & 0x00FFFFFFu) | 1u);
}

// Config 6: Dual-Issue Mixed (FP32+INT32, 50/50 Dual-Issue)
__kernel void run_dual_issue_mixed(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = as_float((as_uint(data[index]) & 0x007FFFFFu) | 0x3F800000u);
    uint ubase = (uint)(base * 1000.0f) + index;

    // 8 independent FP32 scalar accumulator chains
    float f0 = base;
    float f1 = base + 0.1f;
    float f2 = base + 0.2f;
    float f3 = base + 0.3f;
    float f4 = base + 0.4f;
    float f5 = base + 0.5f;
    float f6 = base + 0.6f;
    float f7 = base + 0.7f;

    float mf = multiplier * 0.00001f;

    // 8 independent INT32 scalar accumulator chains
    uint u0 = ubase;
    uint u1 = ubase + 1u;
    uint u2 = ubase + 2u;
    uint u3 = ubase + 3u;
    uint u4 = ubase + 4u;
    uint u5 = ubase + 5u;
    uint u6 = ubase + 6u;
    uint u7 = ubase + 7u;

    uint mu = ((uint)(multiplier * 1000.0f)) | 1u;
    uint mask = (num_elements ^ 0x7FFF7FFFu) | 1u;

    // 65536 iterations * (8 scalar FMAs + 8 scalar INT add+and)
    // = 65536 * (16 FP32 ops + 16 INT32 ops) = 65536 * 32 = 2,097,152 operations
    for (int i = 0; i < 65536; ++i) {
        f0 = fma(f0, mf, f0);
        u0 = (u0 + mu) & mask;

        f1 = fma(f1, mf, f1);
        u1 = (u1 + mu) & mask;

        f2 = fma(f2, mf, f2);
        u2 = (u2 + mu) & mask;

        f3 = fma(f3, mf, f3);
        u3 = (u3 + mu) & mask;

        f4 = fma(f4, mf, f4);
        u4 = (u4 + mu) & mask;

        f5 = fma(f5, mf, f5);
        u5 = (u5 + mu) & mask;

        f6 = fma(f6, mf, f6);
        u6 = (u6 + mu) & mask;

        f7 = fma(f7, mf, f7);
        u7 = (u7 + mu) & mask;
    }

    float fres = (f0 + f1) + (f2 + f3) + (f4 + f5) + (f6 + f7);
    uint ures = ((u0 + u1) ^ (u2 + u3) ^ (u4 + u5) ^ (u6 + u7)) & 0x00FFFFFFu;

    data[index] = fres + (float)(ures | 1u);
}
