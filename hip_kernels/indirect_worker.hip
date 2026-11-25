#include <hip/hip_runtime.h>
#include <cstdint>

struct GpuIndirectCommand {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t activeCount;
    uint32_t workgroupsFinished;
};

extern "C" __global__ __launch_bounds__(256) void run_benchmark(
    const uint32_t *__restrict__ activeWorklist,
    float *__restrict__ results,
    const GpuIndirectCommand *__restrict__ cmd) {
    const uint32_t total = cmd->activeCount;
    if (total == 0u) {
        return;
    }

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;

    for (uint32_t idx = gid; idx < total; idx += stride) {
        const uint32_t itemId = activeWorklist[idx];
        float val = static_cast<float>(itemId) * 0.001f;
        for (int i = 0; i < 256; ++i) {
            val = sinf(val) * 1.05f + cosf(val * 0.5f);
        }
        results[idx] = val;
    }
}
