__kernel void run_benchmark(__global uint *data,
                            uint iterations,
                            uint startIndex) {
  uint index = startIndex + (uint)get_local_id(0);

  #pragma unroll 1
  for (uint i = 0u; i < iterations; i++) {
    index = data[index];
  }

  data[startIndex + 1] = index;
}
