#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ __launch_bounds__(64) void run_benchmark(
    const uint32_t *__restrict__ args,
    const uint4 *__restrict__ payloads,
    float *__restrict__ results) {
    const uint32_t itemCount = args[3];
    if (itemCount == 0u) {
        return;
    }

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t idx = gid; idx < itemCount; idx += stride) {
        const uint4 p = payloads[idx];
        float v = static_cast<float>(p.y & 0xFFFFu) * 0.0001220703125f +
                  static_cast<float>(p.z) * 0.125f;
        for (uint32_t i = 0u; i < 32u; ++i) {
            v = fmaf(v, 0.99951171875f, 0.0009765625f);
        }
        results[p.x] = v;
    }
}
