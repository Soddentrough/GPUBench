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
    const uint32_t *__restrict__ inputItems,
    uint32_t *__restrict__ activeWorklist,
    GpuIndirectCommand *__restrict__ cmd,
    uint32_t selectivity,
    uint32_t totalItems,
    uint32_t totalWorkgroups) {
    __shared__ uint32_t sWarpCount[8];
    __shared__ uint32_t sWarpOffset[8];
    __shared__ uint32_t sWgBase;

    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    bool activeItem = false;
    uint32_t itemId = gid;

    if (gid < totalItems) {
        itemId = inputItems[gid];
        const uint32_t hash = itemId * 2654435761u;
        activeItem = (hash % 100u) < selectivity;
    }

    const uint32_t lane = threadIdx.x & static_cast<uint32_t>(warpSize - 1);
    const uint32_t warpId = threadIdx.x / static_cast<uint32_t>(warpSize);
    const uint32_t numWarps = blockDim.x / static_cast<uint32_t>(warpSize);

    const unsigned long long ballot = __ballot(activeItem ? 1 : 0);
    const uint32_t subgroupActive = static_cast<uint32_t>(__popcll(ballot));
    const uint64_t belowMask = (1ULL << lane) - 1ULL;
    const uint32_t laneOffset = static_cast<uint32_t>(__popcll(ballot & belowMask));

    if (lane == 0u) {
        sWarpCount[warpId] = subgroupActive;
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        uint32_t wgTotal = 0u;
        for (uint32_t w = 0u; w < numWarps; ++w) {
            sWarpOffset[w] = wgTotal;
            wgTotal += sWarpCount[w];
        }
        sWgBase = (wgTotal > 0u) ? atomicAdd(&cmd->activeCount, wgTotal) : 0u;
    }
    __syncthreads();

    if (activeItem) {
        activeWorklist[sWgBase + sWarpOffset[warpId] + laneOffset] = itemId;
    }

    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0u) {
        const uint32_t finished = atomicAdd(&cmd->workgroupsFinished, 1u);
        if (finished == totalWorkgroups - 1u) {
            __threadfence();
            const uint32_t totalActive = atomicAdd(&cmd->activeCount, 0u);
            cmd->x = (totalActive + 255u) / 256u;
            cmd->y = (totalActive > 0u) ? 1u : 0u;
            cmd->z = (totalActive > 0u) ? 1u : 0u;
            __threadfence();
        }
    }
}
