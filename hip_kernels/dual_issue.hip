// Requires ROCm 6.4+
#include <hip/hip_runtime.h>
#include <cstdint>

// Helper for float4 fmac (acc = fmaf(a, b, acc))
__device__ inline float4 fmac4(float4 acc, float4 a, float4 b) {
    return make_float4(
        fmaf(a.x, b.x, acc.x),
        fmaf(a.y, b.y, acc.y),
        fmaf(a.z, b.z, acc.z),
        fmaf(a.w, b.w, acc.w)
    );
}

// Config 0: ILP-1 (Strictly Serialized Baseline)
extern "C" __global__ void run_dual_issue_ilp1(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    float4 val = make_float4(data[index], data[index], data[index], data[index]);
    float4 m = make_float4(multiplier, multiplier, multiplier, multiplier);
    float4 c = make_float4(0.0001f, 0.0002f, 0.0003f, 0.0004f);

    for (int i = 0; i < 32768; ++i) {
        val = fmac4(c, m, val);
    }

    data[index] = val.x + val.y + val.z + val.w;
}

// Config 1: ILP-4 (Latency-Bound Single-Issue)
extern "C" __global__ void run_dual_issue_ilp4(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    float4 val0 = make_float4(data[index], data[index], data[index], data[index]);
    float4 val1 = make_float4(0.1f, 0.2f, 0.3f, 0.4f);
    float4 val2 = make_float4(0.5f, 0.6f, 0.7f, 0.8f);
    float4 val3 = make_float4(0.9f, 1.0f, 1.1f, 1.2f);

    float4 m = make_float4(multiplier, multiplier, multiplier, multiplier);
    float4 c0 = make_float4(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 c1 = make_float4(0.0005f, 0.0006f, 0.0007f, 0.0008f);
    float4 c2 = make_float4(0.0009f, 0.0010f, 0.0011f, 0.0012f);
    float4 c3 = make_float4(0.0013f, 0.0014f, 0.0015f, 0.0016f);

    for (int i = 0; i < 16384; ++i) {
        val0 = fmac4(c0, m, val0);
        val1 = fmac4(c1, m, val1);
        val2 = fmac4(c2, m, val2);
        val3 = fmac4(c3, m, val3);
    }

    float4 res = make_float4(
        val0.x + val1.x + val2.x + val3.x,
        val0.y + val1.y + val2.y + val3.y,
        val0.z + val1.z + val2.z + val3.z,
        val0.w + val1.w + val2.w + val3.w
    );
    data[index] = res.x + res.y + res.z + res.w;
}

// Config 2: ILP-8 (Dual-Issue Transition Threshold)
extern "C" __global__ void run_dual_issue_ilp8(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    float4 val0 = make_float4(data[index], data[index], data[index], data[index]);
    float4 val1 = make_float4(0.1f, 0.2f, 0.3f, 0.4f);
    float4 val2 = make_float4(0.5f, 0.6f, 0.7f, 0.8f);
    float4 val3 = make_float4(0.9f, 1.0f, 1.1f, 1.2f);
    float4 val4 = make_float4(1.3f, 1.4f, 1.5f, 1.6f);
    float4 val5 = make_float4(1.7f, 1.8f, 1.9f, 2.0f);
    float4 val6 = make_float4(2.1f, 2.2f, 2.3f, 2.4f);
    float4 val7 = make_float4(2.5f, 2.6f, 2.7f, 2.8f);

    float4 m = make_float4(multiplier, multiplier, multiplier, multiplier);
    float4 c0 = make_float4(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 c1 = make_float4(0.0005f, 0.0006f, 0.0007f, 0.0008f);
    float4 c2 = make_float4(0.0009f, 0.0010f, 0.0011f, 0.0012f);
    float4 c3 = make_float4(0.0013f, 0.0014f, 0.0015f, 0.0016f);
    float4 c4 = make_float4(0.0017f, 0.0018f, 0.0019f, 0.0020f);
    float4 c5 = make_float4(0.0021f, 0.0022f, 0.0023f, 0.0024f);
    float4 c6 = make_float4(0.0025f, 0.0026f, 0.0027f, 0.0028f);
    float4 c7 = make_float4(0.0029f, 0.0030f, 0.0031f, 0.0032f);

    for (int i = 0; i < 16384; ++i) {
        val0 = fmac4(c0, m, val0);
        val1 = fmac4(c1, m, val1);
        val2 = fmac4(c2, m, val2);
        val3 = fmac4(c3, m, val3);
        val4 = fmac4(c4, m, val4);
        val5 = fmac4(c5, m, val5);
        val6 = fmac4(c6, m, val6);
        val7 = fmac4(c7, m, val7);
    }

    data[index] = val0.x + val1.y + val2.z + val3.w + val4.x + val5.y + val6.z + val7.w;
}

// Config 3: ILP-16 (Peak Dual-Issue Saturated FP32)
extern "C" __global__ void run_dual_issue_ilp16(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    float4 val0  = make_float4(data[index], data[index], data[index], data[index]);
    float4 val1  = make_float4(0.1f, 0.2f, 0.3f, 0.4f);
    float4 val2  = make_float4(0.5f, 0.6f, 0.7f, 0.8f);
    float4 val3  = make_float4(0.9f, 1.0f, 1.1f, 1.2f);
    float4 val4  = make_float4(1.3f, 1.4f, 1.5f, 1.6f);
    float4 val5  = make_float4(1.7f, 1.8f, 1.9f, 2.0f);
    float4 val6  = make_float4(2.1f, 2.2f, 2.3f, 2.4f);
    float4 val7  = make_float4(2.5f, 2.6f, 2.7f, 2.8f);
    float4 val8  = make_float4(2.9f, 3.0f, 3.1f, 3.2f);
    float4 val9  = make_float4(3.3f, 3.4f, 3.5f, 3.6f);
    float4 val10 = make_float4(3.7f, 3.8f, 3.9f, 4.0f);
    float4 val11 = make_float4(4.1f, 4.2f, 4.3f, 4.4f);
    float4 val12 = make_float4(4.5f, 4.6f, 4.7f, 4.8f);
    float4 val13 = make_float4(4.9f, 5.0f, 5.1f, 5.2f);
    float4 val14 = make_float4(5.3f, 5.4f, 5.5f, 5.6f);
    float4 val15 = make_float4(5.7f, 5.8f, 5.9f, 6.0f);

    float4 m = make_float4(multiplier, multiplier, multiplier, multiplier);
    float4 c0  = make_float4(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 c1  = make_float4(0.0005f, 0.0006f, 0.0007f, 0.0008f);
    float4 c2  = make_float4(0.0009f, 0.0010f, 0.0011f, 0.0012f);
    float4 c3  = make_float4(0.0013f, 0.0014f, 0.0015f, 0.0016f);
    float4 c4  = make_float4(0.0017f, 0.0018f, 0.0019f, 0.0020f);
    float4 c5  = make_float4(0.0021f, 0.0022f, 0.0023f, 0.0024f);
    float4 c6  = make_float4(0.0025f, 0.0026f, 0.0027f, 0.0028f);
    float4 c7  = make_float4(0.0029f, 0.0030f, 0.0031f, 0.0032f);
    float4 c8  = make_float4(0.0033f, 0.0034f, 0.0035f, 0.0036f);
    float4 c9  = make_float4(0.0037f, 0.0038f, 0.0039f, 0.0040f);
    float4 c10 = make_float4(0.0041f, 0.0042f, 0.0043f, 0.0044f);
    float4 c11 = make_float4(0.0045f, 0.0046f, 0.0047f, 0.0048f);
    float4 c12 = make_float4(0.0049f, 0.0050f, 0.0051f, 0.0052f);
    float4 c13 = make_float4(0.0053f, 0.0054f, 0.0055f, 0.0056f);
    float4 c14 = make_float4(0.0057f, 0.0058f, 0.0059f, 0.0060f);
    float4 c15 = make_float4(0.0061f, 0.0062f, 0.0063f, 0.0064f);

    for (int i = 0; i < 16384; ++i) {
        val0  = fmac4(c0,  m, val0);
        val1  = fmac4(c1,  m, val1);
        val2  = fmac4(c2,  m, val2);
        val3  = fmac4(c3,  m, val3);
        val4  = fmac4(c4,  m, val4);
        val5  = fmac4(c5,  m, val5);
        val6  = fmac4(c6,  m, val6);
        val7  = fmac4(c7,  m, val7);
        val8  = fmac4(c8,  m, val8);
        val9  = fmac4(c9,  m, val9);
        val10 = fmac4(c10, m, val10);
        val11 = fmac4(c11, m, val11);
        val12 = fmac4(c12, m, val12);
        val13 = fmac4(c13, m, val13);
        val14 = fmac4(c14, m, val14);
        val15 = fmac4(c15, m, val15);
    }

    data[index] = val0.x + val1.y + val2.z + val3.w + val4.x + val5.y + val6.z + val7.w +
                  val8.x + val9.y + val10.z + val11.w + val12.x + val13.y + val14.z + val15.w;
}

// Config 4: Concurrent FP32 + INT32
extern "C" __global__ void run_dual_issue_mixed(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    float4 f0 = make_float4(data[index], data[index], data[index], data[index]);
    float4 f1 = make_float4(0.1f, 0.2f, 0.3f, 0.4f);
    float4 f2 = make_float4(0.5f, 0.6f, 0.7f, 0.8f);
    float4 f3 = make_float4(0.9f, 1.0f, 1.1f, 1.2f);
    float4 f4 = make_float4(1.3f, 1.4f, 1.5f, 1.6f);
    float4 f5 = make_float4(1.7f, 1.8f, 1.9f, 2.0f);
    float4 f6 = make_float4(2.1f, 2.2f, 2.3f, 2.4f);
    float4 f7 = make_float4(2.5f, 2.6f, 2.7f, 2.8f);

    float4 mf = make_float4(multiplier, multiplier, multiplier, multiplier);
    float4 cf0 = make_float4(0.0001f, 0.0002f, 0.0003f, 0.0004f);
    float4 cf1 = make_float4(0.0005f, 0.0006f, 0.0007f, 0.0008f);
    float4 cf2 = make_float4(0.0009f, 0.0010f, 0.0011f, 0.0012f);
    float4 cf3 = make_float4(0.0013f, 0.0014f, 0.0015f, 0.0016f);
    float4 cf4 = make_float4(0.0017f, 0.0018f, 0.0019f, 0.0020f);
    float4 cf5 = make_float4(0.0021f, 0.0022f, 0.0023f, 0.0024f);
    float4 cf6 = make_float4(0.0025f, 0.0026f, 0.0027f, 0.0028f);
    float4 cf7 = make_float4(0.0029f, 0.0030f, 0.0031f, 0.0032f);

    uint32_t ubase = static_cast<uint32_t>(data[index] * 1000.0f) + index;
    uint4 u0 = make_uint4(ubase, ubase + 1, ubase + 2, ubase + 3);
    uint4 u1 = make_uint4(ubase + 4, ubase + 5, ubase + 6, ubase + 7);
    uint4 u2 = make_uint4(ubase + 8, ubase + 9, ubase + 10, ubase + 11);
    uint4 u3 = make_uint4(ubase + 12, ubase + 13, ubase + 14, ubase + 15);
    uint4 u4 = make_uint4(ubase + 16, ubase + 17, ubase + 18, ubase + 19);
    uint4 u5 = make_uint4(ubase + 20, ubase + 21, ubase + 22, ubase + 23);
    uint4 u6 = make_uint4(ubase + 24, ubase + 25, ubase + 26, ubase + 27);
    uint4 u7 = make_uint4(ubase + 28, ubase + 29, ubase + 30, ubase + 31);

    uint32_t umult = static_cast<uint32_t>(multiplier);
    if (umult == 0) umult = 1;

    for (int i = 0; i < 16384; ++i) {
        f0 = fmac4(cf0, mf, f0);
        u0.x = u0.x * umult + 13u; u0.y = u0.y * umult + 17u; u0.z = u0.z * umult + 19u; u0.w = u0.w * umult + 23u;

        f1 = fmac4(cf1, mf, f1);
        u1.x = u1.x * umult + 29u; u1.y = u1.y * umult + 31u; u1.z = u1.z * umult + 37u; u1.w = u1.w * umult + 41u;

        f2 = fmac4(cf2, mf, f2);
        u2.x = u2.x * umult + 43u; u2.y = u2.y * umult + 47u; u2.z = u2.z * umult + 53u; u2.w = u2.w * umult + 59u;

        f3 = fmac4(cf3, mf, f3);
        u3.x = u3.x * umult + 61u; u3.y = u3.y * umult + 67u; u3.z = u3.z * umult + 71u; u3.w = u3.w * umult + 73u;

        f4 = fmac4(cf4, mf, f4);
        u4.x = u4.x * umult + 79u; u4.y = u4.y * umult + 83u; u4.z = u4.z * umult + 89u; u4.w = u4.w * umult + 97u;

        f5 = fmac4(cf5, mf, f5);
        u5.x = u5.x * umult + 101u; u5.y = u5.y * umult + 103u; u5.z = u5.z * umult + 107u; u5.w = u5.w * umult + 109u;

        f6 = fmac4(cf6, mf, f6);
        u6.x = u6.x * umult + 113u; u6.y = u6.y * umult + 127u; u6.z = u6.z * umult + 131u; u6.w = u6.w * umult + 137u;

        f7 = fmac4(cf7, mf, f7);
        u7.x = u7.x * umult + 139u; u7.y = u7.y * umult + 149u; u7.z = u7.z * umult + 151u; u7.w = u7.w * umult + 157u;
    }

    data[index] = f0.x + f1.y + f2.z + f3.w + f4.x + f5.y + f6.z + f7.w +
                  static_cast<float>(u0.x ^ u1.y ^ u2.z ^ u3.w ^ u4.x ^ u5.y ^ u6.z ^ u7.w);
}

// Standard INT32 (4 independent chains: single-issue saturated integer baseline)
extern "C" __global__ void run_dual_issue_int32_4(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    uint32_t ubase = static_cast<uint32_t>(data[index] * 1000.0f) + index;
    uint4 u0  = make_uint4(ubase, ubase + 1, ubase + 2, ubase + 3);
    uint4 u1  = make_uint4(ubase + 4, ubase + 5, ubase + 6, ubase + 7);
    uint4 u2  = make_uint4(ubase + 8, ubase + 9, ubase + 10, ubase + 11);
    uint4 u3  = make_uint4(ubase + 12, ubase + 13, ubase + 14, ubase + 15);

    uint32_t umult = static_cast<uint32_t>(multiplier);
    if (umult == 0) umult = 1;

    for (int i = 0; i < 16384; ++i) {
        u0.x  = u0.x  * umult + 13u;  u0.y  = u0.y  * umult + 17u;  u0.z  = u0.z  * umult + 19u;  u0.w  = u0.w  * umult + 23u;
        u1.x  = u1.x  * umult + 29u;  u1.y  = u1.y  * umult + 31u;  u1.z  = u1.z  * umult + 37u;  u1.w  = u1.w  * umult + 41u;
        u2.x  = u2.x  * umult + 43u;  u2.y  = u2.y  * umult + 47u;  u2.z  = u2.z  * umult + 53u;  u2.w  = u2.w  * umult + 59u;
        u3.x  = u3.x  * umult + 61u;  u3.y  = u3.y  * umult + 67u;  u3.z  = u3.z  * umult + 71u;  u3.w  = u3.w  * umult + 73u;
    }

    data[index] = static_cast<float>(u0.x ^ u1.y ^ u2.z ^ u3.w);
}

// Dual-Issue INT32 Partial Co-Issue (8 independent chains)
extern "C" __global__ void run_dual_issue_int32_8(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    uint32_t ubase = static_cast<uint32_t>(data[index] * 1000.0f) + index;
    uint4 u0  = make_uint4(ubase, ubase + 1, ubase + 2, ubase + 3);
    uint4 u1  = make_uint4(ubase + 4, ubase + 5, ubase + 6, ubase + 7);
    uint4 u2  = make_uint4(ubase + 8, ubase + 9, ubase + 10, ubase + 11);
    uint4 u3  = make_uint4(ubase + 12, ubase + 13, ubase + 14, ubase + 15);
    uint4 u4  = make_uint4(ubase + 16, ubase + 17, ubase + 18, ubase + 19);
    uint4 u5  = make_uint4(ubase + 20, ubase + 21, ubase + 22, ubase + 23);
    uint4 u6  = make_uint4(ubase + 24, ubase + 25, ubase + 26, ubase + 27);
    uint4 u7  = make_uint4(ubase + 28, ubase + 29, ubase + 30, ubase + 31);

    uint32_t umult = static_cast<uint32_t>(multiplier);
    if (umult == 0) umult = 1;

    for (int i = 0; i < 16384; ++i) {
        u0.x  = u0.x  * umult + 13u;  u0.y  = u0.y  * umult + 17u;  u0.z  = u0.z  * umult + 19u;  u0.w  = u0.w  * umult + 23u;
        u1.x  = u1.x  * umult + 29u;  u1.y  = u1.y  * umult + 31u;  u1.z  = u1.z  * umult + 37u;  u1.w  = u1.w  * umult + 41u;
        u2.x  = u2.x  * umult + 43u;  u2.y  = u2.y  * umult + 47u;  u2.z  = u2.z  * umult + 53u;  u2.w  = u2.w  * umult + 59u;
        u3.x  = u3.x  * umult + 61u;  u3.y  = u3.y  * umult + 67u;  u3.z  = u3.z  * umult + 71u;  u3.w  = u3.w  * umult + 73u;
        u4.x  = u4.x  * umult + 79u;  u4.y  = u4.y  * umult + 83u;  u4.z  = u4.z  * umult + 89u;  u4.w  = u4.w  * umult + 97u;
        u5.x  = u5.x  * umult + 101u; u5.y  = u5.y  * umult + 103u; u5.z  = u5.z  * umult + 107u; u5.w  = u5.w  * umult + 109u;
        u6.x  = u6.x  * umult + 113u; u6.y  = u6.y  * umult + 127u; u6.z  = u6.z  * umult + 131u; u6.w  = u6.w  * umult + 137u;
        u7.x  = u7.x  * umult + 139u; u7.y  = u7.y  * umult + 149u; u7.z  = u7.z  * umult + 151u; u7.w  = u7.w  * umult + 157u;
    }

    data[index] = static_cast<float>(u0.x ^ u1.y ^ u2.z ^ u3.w ^ u4.x ^ u5.y ^ u6.z ^ u7.w);
}

// Config 5: Dual-Issue INT32 (INT32+INT32) - 16 independent chains
extern "C" __global__ void run_dual_issue_int32(float* data, float multiplier, uint32_t num_elements) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elements) return;

    uint32_t ubase = static_cast<uint32_t>(data[index] * 1000.0f) + index;
    uint4 u0  = make_uint4(ubase, ubase + 1, ubase + 2, ubase + 3);
    uint4 u1  = make_uint4(ubase + 4, ubase + 5, ubase + 6, ubase + 7);
    uint4 u2  = make_uint4(ubase + 8, ubase + 9, ubase + 10, ubase + 11);
    uint4 u3  = make_uint4(ubase + 12, ubase + 13, ubase + 14, ubase + 15);
    uint4 u4  = make_uint4(ubase + 16, ubase + 17, ubase + 18, ubase + 19);
    uint4 u5  = make_uint4(ubase + 20, ubase + 21, ubase + 22, ubase + 23);
    uint4 u6  = make_uint4(ubase + 24, ubase + 25, ubase + 26, ubase + 27);
    uint4 u7  = make_uint4(ubase + 28, ubase + 29, ubase + 30, ubase + 31);
    uint4 u8  = make_uint4(ubase + 32, ubase + 33, ubase + 34, ubase + 35);
    uint4 u9  = make_uint4(ubase + 36, ubase + 37, ubase + 38, ubase + 39);
    uint4 u10 = make_uint4(ubase + 40, ubase + 41, ubase + 42, ubase + 43);
    uint4 u11 = make_uint4(ubase + 44, ubase + 45, ubase + 46, ubase + 47);
    uint4 u12 = make_uint4(ubase + 48, ubase + 49, ubase + 50, ubase + 51);
    uint4 u13 = make_uint4(ubase + 52, ubase + 53, ubase + 54, ubase + 55);
    uint4 u14 = make_uint4(ubase + 56, ubase + 57, ubase + 58, ubase + 59);
    uint4 u15 = make_uint4(ubase + 60, ubase + 61, ubase + 62, ubase + 63);

    uint32_t umult = static_cast<uint32_t>(multiplier);
    if (umult == 0) umult = 1;

    for (int i = 0; i < 16384; ++i) {
        u0.x  = u0.x  * umult + 13u;  u0.y  = u0.y  * umult + 17u;  u0.z  = u0.z  * umult + 19u;  u0.w  = u0.w  * umult + 23u;
        u1.x  = u1.x  * umult + 29u;  u1.y  = u1.y  * umult + 31u;  u1.z  = u1.z  * umult + 37u;  u1.w  = u1.w  * umult + 41u;
        u2.x  = u2.x  * umult + 43u;  u2.y  = u2.y  * umult + 47u;  u2.z  = u2.z  * umult + 53u;  u2.w  = u2.w  * umult + 59u;
        u3.x  = u3.x  * umult + 61u;  u3.y  = u3.y  * umult + 67u;  u3.z  = u3.z  * umult + 71u;  u3.w  = u3.w  * umult + 73u;
        u4.x  = u4.x  * umult + 79u;  u4.y  = u4.y  * umult + 83u;  u4.z  = u4.z  * umult + 89u;  u4.w  = u4.w  * umult + 97u;
        u5.x  = u5.x  * umult + 101u; u5.y  = u5.y  * umult + 103u; u5.z  = u5.z  * umult + 107u; u5.w  = u5.w  * umult + 109u;
        u6.x  = u6.x  * umult + 113u; u6.y  = u6.y  * umult + 127u; u6.z  = u6.z  * umult + 131u; u6.w  = u6.w  * umult + 137u;
        u7.x  = u7.x  * umult + 139u; u7.y  = u7.y  * umult + 149u; u7.z  = u7.z  * umult + 151u; u7.w  = u7.w  * umult + 157u;
        u8.x  = u8.x  * umult + 163u; u8.y  = u8.y  * umult + 167u; u8.z  = u8.z  * umult + 173u; u8.w  = u8.w  * umult + 179u;
        u9.x  = u9.x  * umult + 181u; u9.y  = u9.y  * umult + 191u; u9.z  = u9.z  * umult + 193u; u9.w  = u9.w  * umult + 197u;
        u10.x = u10.x * umult + 199u; u10.y = u10.y * umult + 211u; u10.z = u10.z * umult + 223u; u10.w = u10.w * umult + 227u;
        u11.x = u11.x * umult + 229u; u11.y = u11.y * umult + 233u; u11.z = u11.z * umult + 239u; u11.w = u11.w * umult + 241u;
        u12.x = u12.x * umult + 251u; u12.y = u12.y * umult + 257u; u12.z = u12.z * umult + 263u; u12.w = u12.w * umult + 269u;
        u13.x = u13.x * umult + 271u; u13.y = u13.y * umult + 277u; u13.z = u13.z * umult + 281u; u13.w = u13.w * umult + 283u;
        u14.x = u14.x * umult + 293u; u14.y = u14.y * umult + 307u; u14.z = u14.z * umult + 311u; u14.w = u14.w * umult + 313u;
        u15.x = u15.x * umult + 317u; u15.y = u15.y * umult + 331u; u15.z = u15.z * umult + 337u; u15.w = u15.w * umult + 347u;
    }

    data[index] = static_cast<float>(
        u0.x ^ u1.y ^ u2.z ^ u3.w ^ u4.x ^ u5.y ^ u6.z ^ u7.w ^
        u8.x ^ u9.y ^ u10.z ^ u11.w ^ u12.x ^ u13.y ^ u14.z ^ u15.w
    );
}
