#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(unsigned int *results,
                                         unsigned int stride,
                                         unsigned int iters) {
  __shared__ unsigned int lds[1280];

  unsigned int tid = threadIdx.x;
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

  for (unsigned int i = tid; i < 1280u; i += 32u) {
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
    acc0 += lds[base_addr + 0u];
    acc1 += lds[base_addr + 32u];
    acc2 += lds[base_addr + 64u];
    acc3 += lds[base_addr + 96u];
    acc4 += lds[base_addr + 128u];
    acc5 += lds[base_addr + 160u];
    acc6 += lds[base_addr + 192u];
    acc7 += lds[base_addr + 224u];
    base_addr = (base_addr + acc0) & 1023u;
  }

  unsigned int total = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
  results[gid] = total;
}
