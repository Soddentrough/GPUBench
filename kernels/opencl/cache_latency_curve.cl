__kernel void run_benchmark(__global uint *data,
                            uint iterations,
                            uint startIndex) {
  uint index = startIndex;
  uint i = 0u;

  do {
    index = data[index];
    i++;
  } while (i < iterations);

  data[startIndex + 1] = index;
}
