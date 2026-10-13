__kernel void run_benchmark(__global uint *results,
                            uint stride,
                            uint iters) {
  __local uint lds[1024];

  uint tid = get_local_id(0);
  uint gid = get_global_id(0);

  for (uint i = tid; i < 1024u; i += 32u) {
    lds[i] = 32u;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint base_addr = (tid * stride) & 1023u;
  uint acc0 = 0u;
  uint acc1 = 0u;
  uint acc2 = 0u;
  uint acc3 = 0u;
  uint acc4 = 0u;
  uint acc5 = 0u;
  uint acc6 = 0u;
  uint acc7 = 0u;

  for (uint i = 0u; i < iters; i++) {
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

  uint total = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
  if (total == 0xDEADBEEFu) {
    results[gid] = total;
  }
}
