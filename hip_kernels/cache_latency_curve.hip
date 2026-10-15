#include <hip/hip_runtime.h>

extern "C" __global__ void run_benchmark(unsigned int *data,
                                         unsigned int iterations,
                                         unsigned int startIndex) {
  unsigned int index = startIndex;
  unsigned int i = 0u;

  do {
    index = data[index];
    i++;
  } while (i < iterations);

  data[startIndex + 1] = index;
}
