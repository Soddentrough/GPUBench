#include <hip/hip_runtime.h>
#include <cstdint>

struct GpuIndirectCommand {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t activeCount;
    uint32_t workgroupsFinished;
};

extern "C" __global__ __launch_bounds__(1) void run_benchmark(GpuIndirectCommand *cmd) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        cmd->x = 0u;
        cmd->y = 0u;
        cmd->z = 0u;
        cmd->activeCount = 0u;
        cmd->workgroupsFinished = 0u;
    }
}
