// Requires ROCm 6.4+
#include <hip/hip_runtime.h>
#include <cstdint>

// Config 0: Standard FP32 (Single-Issue Baseline)
extern "C" __global__ void run_dual_issue_ilp4(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    float base = data[index];
    float val0 = base;
    float val1 = base + 0.1f;

    // 131072 iterations * 2 scalar FMAs = 131072 * 4 = 524,288 FP32 operations
    for (int i = 0; i < 131072; ++i) {
        val0 = fmaf(val0, multiplier, val1);
        val1 = fmaf(val1, multiplier, val0);
    }

    data[index] = val0 + val1;
}

// Config 1: Dual-Issue FP32 (Partial Co-Issue - 8 chains)
extern "C" __global__ void run_dual_issue_ilp8(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
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
        val0 = fmaf(val0, multiplier, c0);
        val1 = fmaf(val1, multiplier, c1);
        val2 = fmaf(val2, multiplier, c2);
        val3 = fmaf(val3, multiplier, c3);
        val4 = fmaf(val4, multiplier, c4);
        val5 = fmaf(val5, multiplier, c5);
        val6 = fmaf(val6, multiplier, c6);
        val7 = fmaf(val7, multiplier, c7);
    }

    data[index] = (val0 + val1) + (val2 + val3) + (val4 + val5) + (val6 + val7);
}

// Config 2: Dual-Issue FP32 (Peak Co-Issue - 16 chains)
extern "C" __global__ void run_dual_issue_ilp16(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
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
        val0  = fmaf(val0,  multiplier, c0);
        val1  = fmaf(val1,  multiplier, c1);
        val2  = fmaf(val2,  multiplier, c2);
        val3  = fmaf(val3,  multiplier, c3);
        val4  = fmaf(val4,  multiplier, c4);
        val5  = fmaf(val5,  multiplier, c5);
        val6  = fmaf(val6,  multiplier, c6);
        val7  = fmaf(val7,  multiplier, c7);
        val8  = fmaf(val8,  multiplier, c8);
        val9  = fmaf(val9,  multiplier, c9);
        val10 = fmaf(val10, multiplier, c10);
        val11 = fmaf(val11, multiplier, c11);
        val12 = fmaf(val12, multiplier, c12);
        val13 = fmaf(val13, multiplier, c13);
        val14 = fmaf(val14, multiplier, c14);
        val15 = fmaf(val15, multiplier, c15);
    }

    data[index] = ((val0 + val1) + (val2 + val3)) +
                  ((val4 + val5) + (val6 + val7)) +
                  ((val8 + val9) + (val10 + val11)) +
                  ((val12 + val13) + (val14 + val15));
}

// Config 3: Standard INT32 (Single-Issue Baseline)
extern "C" __global__ void run_dual_issue_int32_4(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    uint32_t ubase = static_cast<uint32_t>(data[index] * 1000.0f) + index;
    uint32_t val0 = ubase;
    uint32_t val1 = ubase + 1u;

    uint32_t mu = static_cast<uint32_t>(multiplier);
    if (mu == 0u) mu = 1u;

    uint32_t cu0 = 13u;
    uint32_t cu1 = 29u;

    // 131072 iterations * 2 scalar INT32 ops (add + xor = 2 ops each) = 131072 * 4 = 524,288 INT32 ops
    for (int i = 0; i < 131072; ++i) {
        val0 = (val1 + cu0) ^ mu;
        val1 = (val0 + cu1) ^ mu;
    }

    data[index] = static_cast<float>(val0 ^ val1);
}

// Config 4: Dual-Issue INT32 (Partial Co-Issue - 8 chains)
extern "C" __global__ void run_dual_issue_int32_8(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    uint32_t ubase = static_cast<uint32_t>(data[index] * 1000.0f) + index;
    uint32_t u0 = ubase;
    uint32_t u1 = ubase + 1u;
    uint32_t u2 = ubase + 2u;
    uint32_t u3 = ubase + 3u;
    uint32_t u4 = ubase + 4u;
    uint32_t u5 = ubase + 5u;
    uint32_t u6 = ubase + 6u;
    uint32_t u7 = ubase + 7u;

    uint32_t mu = static_cast<uint32_t>(multiplier);
    if (mu == 0u) mu = 1u;

    uint32_t cu0 = 13u;
    uint32_t cu1 = 29u;
    uint32_t cu2 = 43u;
    uint32_t cu3 = 61u;
    uint32_t cu4 = 79u;
    uint32_t cu5 = 101u;
    uint32_t cu6 = 113u;
    uint32_t cu7 = 139u;

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

    data[index] = static_cast<float>(u0 ^ u1 ^ u2 ^ u3 ^ u4 ^ u5 ^ u6 ^ u7);
}

// Config 5: Dual-Issue INT32 (Peak Co-Issue - 16 chains)
extern "C" __global__ void run_dual_issue_int32(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    uint32_t ubase = static_cast<uint32_t>(data[index] * 1000.0f) + index;
    uint32_t u0  = ubase;
    uint32_t u1  = ubase + 1u;
    uint32_t u2  = ubase + 2u;
    uint32_t u3  = ubase + 3u;
    uint32_t u4  = ubase + 4u;
    uint32_t u5  = ubase + 5u;
    uint32_t u6  = ubase + 6u;
    uint32_t u7  = ubase + 7u;
    uint32_t u8  = ubase + 8u;
    uint32_t u9  = ubase + 9u;
    uint32_t u10 = ubase + 10u;
    uint32_t u11 = ubase + 11u;
    uint32_t u12 = ubase + 12u;
    uint32_t u13 = ubase + 13u;
    uint32_t u14 = ubase + 14u;
    uint32_t u15 = ubase + 15u;

    uint32_t mu = static_cast<uint32_t>(multiplier);
    if (mu == 0u) mu = 1u;

    uint32_t cu0  = 13u;
    uint32_t cu1  = 29u;
    uint32_t cu2  = 43u;
    uint32_t cu3  = 61u;
    uint32_t cu4  = 79u;
    uint32_t cu5  = 101u;
    uint32_t cu6  = 113u;
    uint32_t cu7  = 139u;
    uint32_t cu8  = 163u;
    uint32_t cu9  = 181u;
    uint32_t cu10 = 199u;
    uint32_t cu11 = 229u;
    uint32_t cu12 = 251u;
    uint32_t cu13 = 271u;
    uint32_t cu14 = 293u;
    uint32_t cu15 = 317u;

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

    data[index] = static_cast<float>(
        u0 ^ u1 ^ u2 ^ u3 ^ u4 ^ u5 ^ u6 ^ u7 ^
        u8 ^ u9 ^ u10 ^ u11 ^ u12 ^ u13 ^ u14 ^ u15
    );
}

// Config 6: Dual-Issue Mixed (FP32+INT32, 50/50 Dual-Issue)
extern "C" __global__ void run_dual_issue_mixed(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    float base = data[index];
    uint32_t ubase = static_cast<uint32_t>(base * 1000.0f) + index;

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
    uint32_t u0 = ubase;
    uint32_t u1 = ubase + 1u;
    uint32_t u2 = ubase + 2u;
    uint32_t u3 = ubase + 3u;
    uint32_t u4 = ubase + 4u;
    uint32_t u5 = ubase + 5u;
    uint32_t u6 = ubase + 6u;
    uint32_t u7 = ubase + 7u;

    uint32_t mu = static_cast<uint32_t>(multiplier);
    if (mu == 0u) mu = 1u;

    uint32_t cu0 = 13u;
    uint32_t cu1 = 29u;
    uint32_t cu2 = 43u;
    uint32_t cu3 = 61u;
    uint32_t cu4 = 79u;
    uint32_t cu5 = 101u;
    uint32_t cu6 = 113u;
    uint32_t cu7 = 139u;

    // 65536 iterations * (8 scalar FMAs + 8 scalar INT add+xor)
    // = 65536 * (16 FP32 ops + 16 INT32 ops) = 65536 * 32 = 2,097,152 operations
    for (int i = 0; i < 65536; ++i) {
        f0 = fmaf(f0, multiplier, cf0);
        u0 = (u0 + cu0) ^ mu;

        f1 = fmaf(f1, multiplier, cf1);
        u1 = (u1 + cu1) ^ mu;

        f2 = fmaf(f2, multiplier, cf2);
        u2 = (u2 + cu2) ^ mu;

        f3 = fmaf(f3, multiplier, cf3);
        u3 = (u3 + cu3) ^ mu;

        f4 = fmaf(f4, multiplier, cf4);
        u4 = (u4 + cu4) ^ mu;

        f5 = fmaf(f5, multiplier, cf5);
        u5 = (u5 + cu5) ^ mu;

        f6 = fmaf(f6, multiplier, cf6);
        u6 = (u6 + cu6) ^ mu;

        f7 = fmaf(f7, multiplier, cf7);
        u7 = (u7 + cu7) ^ mu;
    }

    float fres = (f0 + f1) + (f2 + f3) + (f4 + f5) + (f6 + f7);
    uint32_t ures = u0 ^ u1 ^ u2 ^ u3 ^ u4 ^ u5 ^ u6 ^ u7;

    data[index] = fres + static_cast<float>(ures);
}
