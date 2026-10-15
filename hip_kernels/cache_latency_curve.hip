#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(unsigned int *data,
                                         unsigned int iterations,
                                         unsigned int startIndex) {
  unsigned int index = startIndex + threadIdx.x;

  #pragma nounroll
  for (unsigned int i = 0u; i < iterations; i++) {
    index = data[index];
  }

  data[startIndex + 1] = index;
}
