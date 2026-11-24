#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void run_benchmark(uint32_t *mailbox,
                                         uint32_t *resultBuf,
                                         uint32_t mode,
                                         uint32_t count,
                                         uint32_t seed) {
    if (mode == 0u) {
        if (blockIdx.x != 0u || threadIdx.x != 0u) {
            return;
        }
        // Signal ready (state = 1) so host starts ping-pong loop
        __hip_atomic_store(&mailbox[0], 1u, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
        uint32_t completed = 0u;
        for (uint32_t step = 0u; step < count; ++step) {
            uint32_t expectedFromCpu = step * 2u + 2u;
            uint32_t guard = 0u;
            while (__hip_atomic_load(&mailbox[0], __ATOMIC_ACQUIRE,
                                     __HIP_MEMORY_SCOPE_SYSTEM) !=
                   expectedFromCpu) {
                if (++guard > 20000000u) {
                    resultBuf[0] = completed;
                    return;
                }
            }
            uint32_t replyFromGpu = expectedFromCpu + 1u;
            __hip_atomic_store(&mailbox[0], replyFromGpu, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_SYSTEM);
            completed++;
        }
        resultBuf[0] = completed;
    } else {
        uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= count) {
            return;
        }
        uint32_t val = mailbox[gid];
        uint32_t expected = seed ^ gid;
        if (val != expected) {
            atomicAdd(&resultBuf[0], 1u);
        } else if (gid == 0u) {
            resultBuf[1] = val;
        }
    }
}
