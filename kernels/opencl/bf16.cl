__kernel void run_benchmark(__global float *data) {
  int idx = get_global_id(0);
  float val = 0.0f;
  for (int i = 0; i < 16384; i++) {
    val += 0.0001f;
  }
  data[idx] = val;
}
