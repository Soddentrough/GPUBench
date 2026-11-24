#include <hip/hip_runtime.h>
#include <cstdint>

struct GpuIndirectCommand {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t activeCount;
    uint32_t workgroupsFinished;
};

extern "C" __global__ __launch_bounds__(64) void run_benchmark(
    const uint32_t *__restrict__ inputItems,
    uint32_t *__restrict__ activeWorklist,
    GpuIndirectCommand *__restrict__ cmd,
    uint32_t selectivity,
    uint32_t totalItems,
    uint32_t totalWorkgroups) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    bool activeItem = false;
    uint32_t itemId = gid;

    if (gid < totalItems) {
        itemId = inputItems[gid];
        const uint32_t hash = itemId * 2654435761u;
        activeItem = (hash % 100u) < selectivity;
    }

    const unsigned long long ballot = __ballot(activeItem ? 1 : 0);
    if (ballot != 0ULL) {
        const uint32_t lane = threadIdx.x & static_cast<uint32_t>(warpSize - 1);
        const uint32_t subgroupActive = static_cast<uint32_t>(__popcll(ballot));
        const uint64_t belowMask = (1ULL << lane) - 1ULL;
        const uint32_t laneOffset = static_cast<uint32_t>(__popcll(ballot & belowMask));
        const int firstLane = __ffsll(static_cast<long long>(ballot)) - 1;

        uint32_t baseSlot = 0u;
        if (static_cast<int>(lane) == firstLane) {
            baseSlot = atomicAdd(&cmd->activeCount, subgroupActive);
        }
        baseSlot = __shfl(baseSlot, firstLane);

        if (activeItem) {
            activeWorklist[baseSlot + laneOffset] = itemId;
        }
    }

    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0u) {
        const uint32_t finished = atomicAdd(&cmd->workgroupsFinished, 1u);
        if (finished == totalWorkgroups - 1u) {
            __threadfence();
            const uint32_t totalActive = atomicAdd(&cmd->activeCount, 0u);
            cmd->x = (totalActive + 63u) / 64u;
            cmd->y = (totalActive > 0u) ? 1u : 0u;
            cmd->z = (totalActive > 0u) ? 1u : 0u;
            __threadfence();
        }
    }
}
