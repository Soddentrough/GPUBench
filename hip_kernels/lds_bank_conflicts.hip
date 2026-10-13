#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(unsigned int *results,
                                         unsigned int stride,
                                         unsigned int iters) {
  __shared__ unsigned int lds[1024];

  unsigned int tid = threadIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int i = tid; i < 1024u; i += 32u) {
    lds[i] = 32u;
  }
  __syncthreads();

  unsigned int base_addr = (tid * stride) & 1023u;
  unsigned int acc0 = 0u;
  unsigned int acc1 = 0u;
  unsigned int acc2 = 0u;
  unsigned int acc3 = 0u;
  unsigned int acc4 = 0u;
  unsigned int acc5 = 0u;
  unsigned int acc6 = 0u;
  unsigned int acc7 = 0u;

  for (unsigned int i = 0u; i < iters; i++) {
    acc0 += lds[(base_addr + 0u * stride) & 1023u];
    acc1 += lds[(base_addr + 1u * stride) & 1023u];
    acc2 += lds[(base_addr + 2u * stride) & 1023u];
    acc3 += lds[(base_addr + 3u * stride) & 1023u];
    acc4 += lds[(base_addr + 4u * stride) & 1023u];
    acc5 += lds[(base_addr + 5u * stride) & 1023u];
    acc6 += lds[(base_addr + 6u * stride) & 1023u];
    acc7 += lds[(base_addr + 7u * stride) & 1023u];
    base_addr = (base_addr + acc0) & 1023u;
  }

  unsigned int total = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
  if (total == 0xDEADBEEFu) {
    results[gid] = total;
  }
}
