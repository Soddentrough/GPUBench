__kernel void run_benchmark(__global uint *data,
                            uint iterations,
                            uint startIndex) {
  uint index = startIndex;

  for (uint i = 0u; i < iterations; i++) {
    index = data[index];
  }

  if (index == 0xDEADBEEFu) {
    data[0] = index;
  }
}
