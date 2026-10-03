// Requires OpenCL 1.2+

// Config 0: Standard FP32 (Single-Issue Baseline)
__kernel void run_dual_issue_ilp4(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = data[index];
    float val0 = base;
    float val1 = base + 0.1f;

    // 131072 iterations * 2 scalar FMAs = 131072 * 4 = 524,288 FP32 operations
    for (int i = 0; i < 131072; ++i) {
        val0 = fma(val0, multiplier, val1);
        val1 = fma(val1, multiplier, val0);
    }

    data[index] = val0 + val1;
}

// Config 1: Dual-Issue FP32 (Partial Co-Issue - 8 chains)
__kernel void run_dual_issue_ilp8(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = data[index];
    float val0 = base;
    float val1 = base + 0.1f;
    float val2 = base + 0.2f;
    float val3 = base + 0.3f;
    float val4 = base + 0.4f;
    float val5 = base + 0.5f;
    float val6 = base + 0.6f;
    float val7 = base + 0.7f;

    float c0 = 0.0001f;
    float c1 = 0.0002f;
    float c2 = 0.0003f;
    float c3 = 0.0004f;
    float c4 = 0.0005f;
    float c5 = 0.0006f;
    float c6 = 0.0007f;
    float c7 = 0.0008f;

    // 65536 iterations * 8 scalar FMAs = 65536 * 16 = 1,048,576 FP32 operations
    for (int i = 0; i < 65536; ++i) {
        val0 = fma(val0, multiplier, c0);
        val1 = fma(val1, multiplier, c1);
        val2 = fma(val2, multiplier, c2);
        val3 = fma(val3, multiplier, c3);
        val4 = fma(val4, multiplier, c4);
        val5 = fma(val5, multiplier, c5);
        val6 = fma(val6, multiplier, c6);
        val7 = fma(val7, multiplier, c7);
    }

    data[index] = (val0 + val1) + (val2 + val3) + (val4 + val5) + (val6 + val7);
}

// Config 2: Dual-Issue FP32 (Peak Co-Issue - 16 chains)
__kernel void run_dual_issue_ilp16(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = data[index];
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

    float c0  = 0.0001f;
    float c1  = 0.0002f;
    float c2  = 0.0003f;
    float c3  = 0.0004f;
    float c4  = 0.0005f;
    float c5  = 0.0006f;
    float c6  = 0.0007f;
    float c7  = 0.0008f;
    float c8  = 0.0009f;
    float c9  = 0.0010f;
    float c10 = 0.0011f;
    float c11 = 0.0012f;
    float c12 = 0.0013f;
    float c13 = 0.0014f;
    float c14 = 0.0015f;
    float c15 = 0.0016f;

    // 65536 iterations * 16 scalar FMAs = 65536 * 32 = 2,097,152 FP32 operations
    for (int i = 0; i < 65536; ++i) {
        val0  = fma(val0,  multiplier, c0);
        val1  = fma(val1,  multiplier, c1);
        val2  = fma(val2,  multiplier, c2);
        val3  = fma(val3,  multiplier, c3);
        val4  = fma(val4,  multiplier, c4);
        val5  = fma(val5,  multiplier, c5);
        val6  = fma(val6,  multiplier, c6);
        val7  = fma(val7,  multiplier, c7);
        val8  = fma(val8,  multiplier, c8);
        val9  = fma(val9,  multiplier, c9);
        val10 = fma(val10, multiplier, c10);
        val11 = fma(val11, multiplier, c11);
        val12 = fma(val12, multiplier, c12);
        val13 = fma(val13, multiplier, c13);
        val14 = fma(val14, multiplier, c14);
        val15 = fma(val15, multiplier, c15);
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

    uint mu = (uint)multiplier;
    if (mu == 0u) mu = 1u;

    uint cu0 = 13u;
    uint cu1 = 29u;

    // 131072 iterations * 2 scalar INT32 ops (add + xor = 2 ops each) = 131072 * 4 = 524,288 INT32 ops
    for (int i = 0; i < 131072; ++i) {
        val0 = (val1 + cu0) ^ mu;
        val1 = (val0 + cu1) ^ mu;
    }

    data[index] = (float)(val0 ^ val1);
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

    uint mu = (uint)multiplier;
    if (mu == 0u) mu = 1u;

    uint cu0 = 13u;
    uint cu1 = 29u;
    uint cu2 = 43u;
    uint cu3 = 61u;
    uint cu4 = 79u;
    uint cu5 = 101u;
    uint cu6 = 113u;
    uint cu7 = 139u;

    // 65536 iterations * 8 scalar INT32 ops (add + xor = 2 ops each) = 65536 * 16 = 1,048,576 INT32 ops
    for (int i = 0; i < 65536; ++i) {
        u0 = (u0 + cu0) ^ mu;
        u1 = (u1 + cu1) ^ mu;
        u2 = (u2 + cu2) ^ mu;
        u3 = (u3 + cu3) ^ mu;
        u4 = (u4 + cu4) ^ mu;
        u5 = (u5 + cu5) ^ mu;
        u6 = (u6 + cu6) ^ mu;
        u7 = (u7 + cu7) ^ mu;
    }

    data[index] = (float)(u0 ^ u1 ^ u2 ^ u3 ^ u4 ^ u5 ^ u6 ^ u7);
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

    uint mu = (uint)multiplier;
    if (mu == 0u) mu = 1u;

    uint cu0  = 13u;
    uint cu1  = 29u;
    uint cu2  = 43u;
    uint cu3  = 61u;
    uint cu4  = 79u;
    uint cu5  = 101u;
    uint cu6  = 113u;
    uint cu7  = 139u;
    uint cu8  = 163u;
    uint cu9  = 181u;
    uint cu10 = 199u;
    uint cu11 = 229u;
    uint cu12 = 251u;
    uint cu13 = 271u;
    uint cu14 = 293u;
    uint cu15 = 317u;

    // 65536 iterations * 16 scalar INT32 ops (add + xor = 2 ops each) = 65536 * 32 = 2,097,152 INT32 ops
    for (int i = 0; i < 65536; ++i) {
        u0  = (u0  + cu0)  ^ mu;
        u1  = (u1  + cu1)  ^ mu;
        u2  = (u2  + cu2)  ^ mu;
        u3  = (u3  + cu3)  ^ mu;
        u4  = (u4  + cu4)  ^ mu;
        u5  = (u5  + cu5)  ^ mu;
        u6  = (u6  + cu6)  ^ mu;
        u7  = (u7  + cu7)  ^ mu;
        u8  = (u8  + cu8)  ^ mu;
        u9  = (u9  + cu9)  ^ mu;
        u10 = (u10 + cu10) ^ mu;
        u11 = (u11 + cu11) ^ mu;
        u12 = (u12 + cu12) ^ mu;
        u13 = (u13 + cu13) ^ mu;
        u14 = (u14 + cu14) ^ mu;
        u15 = (u15 + cu15) ^ mu;
    }

    data[index] = (float)(
        u0 ^ u1 ^ u2 ^ u3 ^ u4 ^ u5 ^ u6 ^ u7 ^
        u8 ^ u9 ^ u10 ^ u11 ^ u12 ^ u13 ^ u14 ^ u15
    );
}

// Config 6: Dual-Issue Mixed (FP32+INT32, 50/50 Dual-Issue)
__kernel void run_dual_issue_mixed(__global float* data, float multiplier, uint num_elements) {
    uint index = get_global_id(0);
    if (index >= num_elements) return;

    float base = data[index];
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

    float cf0 = 0.0001f;
    float cf1 = 0.0002f;
    float cf2 = 0.0003f;
    float cf3 = 0.0004f;
    float cf4 = 0.0005f;
    float cf5 = 0.0006f;
    float cf6 = 0.0007f;
    float cf7 = 0.0008f;

    // 8 independent INT32 scalar accumulator chains
    uint u0 = ubase;
    uint u1 = ubase + 1u;
    uint u2 = ubase + 2u;
    uint u3 = ubase + 3u;
    uint u4 = ubase + 4u;
    uint u5 = ubase + 5u;
    uint u6 = ubase + 6u;
    uint u7 = ubase + 7u;

    uint mu = (uint)multiplier;
    if (mu == 0u) mu = 1u;

    uint cu0 = 13u;
    uint cu1 = 29u;
    uint cu2 = 43u;
    uint cu3 = 61u;
    uint cu4 = 79u;
    uint cu5 = 101u;
    uint cu6 = 113u;
    uint cu7 = 139u;

    // 65536 iterations * (8 scalar FMAs + 8 scalar INT add+xor)
    // = 65536 * (16 FP32 ops + 16 INT32 ops) = 65536 * 32 = 2,097,152 operations
    for (int i = 0; i < 65536; ++i) {
        f0 = fma(f0, multiplier, cf0);
        u0 = (u0 + cu0) ^ mu;

        f1 = fma(f1, multiplier, cf1);
        u1 = (u1 + cu1) ^ mu;

        f2 = fma(f2, multiplier, cf2);
        u2 = (u2 + cu2) ^ mu;

        f3 = fma(f3, multiplier, cf3);
        u3 = (u3 + cu3) ^ mu;

        f4 = fma(f4, multiplier, cf4);
        u4 = (u4 + cu4) ^ mu;

        f5 = fma(f5, multiplier, cf5);
        u5 = (u5 + cu5) ^ mu;

        f6 = fma(f6, multiplier, cf6);
        u6 = (u6 + cu6) ^ mu;

        f7 = fma(f7, multiplier, cf7);
        u7 = (u7 + cu7) ^ mu;
    }

    float fres = (f0 + f1) + (f2 + f3) + (f4 + f5) + (f6 + f7);
    uint ures = u0 ^ u1 ^ u2 ^ u3 ^ u4 ^ u5 ^ u6 ^ u7;

    data[index] = fres + (float)ures;
}
