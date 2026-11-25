#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ __launch_bounds__(256) void run_benchmark(
    const uint32_t *__restrict__ inputItems,
    float *__restrict__ results,
    uint32_t selectivity,
    uint32_t totalItems) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= totalItems) {
        return;
    }

    const uint32_t itemId = inputItems[gid];
    const uint32_t hash = itemId * 2654435761u;
    if ((hash % 100u) >= selectivity) {
        return;
    }

    float val = static_cast<float>(itemId) * 0.001f;
    for (int i = 0; i < 256; ++i) {
        val = sinf(val) * 1.05f + cosf(val * 0.5f);
    }
    results[gid] = val;
}
