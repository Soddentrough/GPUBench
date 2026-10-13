#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(unsigned int *data,
                                         unsigned int iterations,
                                         unsigned int startIndex) {
  unsigned int index = startIndex;

  for (unsigned int i = 0u; i < iterations; i++) {
    index = data[index];
  }

  if (index == 0xDEADBEEFu) {
    data[0] = index;
  }
}
