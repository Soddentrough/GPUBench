#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ __launch_bounds__(256) void run_benchmark(
    uint32_t *__restrict__ args,
    uint4 *__restrict__ payloads,
    uint32_t totalItems,
    uint32_t mode) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (mode == 0u) {
        if (gid == 0u) {
            args[0] = 0u;
            args[1] = 1u;
            args[2] = 1u;
            args[3] = 0u;
        }
        return;
    }

    if (gid < totalItems) {
        uint32_t hash = (gid * 747796405u + 2891336453u);
        hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
        hash = (hash >> 22u) ^ hash;
        if ((hash & 1u) == 0u) {
            const uint32_t slot = atomicAdd(&args[3], 1u);
            payloads[slot] = make_uint4(gid, hash | 1u, (gid & 255u) + 1u, 0x5A5AA5A5u);
            if ((slot & 63u) == 0u) {
                atomicAdd(&args[0], 1u);
            }
        }
    }
}
