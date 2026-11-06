__kernel void run_benchmark(__global uint *results,
                            uint stride,
                            uint iters) {
  __local uint lds[1280];

  uint tid = get_local_id(0);
  uint gid = get_global_id(0);

  // Initialize local memory with 32u (32 banks) so adding acc0 to base_addr
  // keeps the loop-carried dependency live while preserving (tid * stride) mod 32.
  for (uint i = tid; i < 1280u; i += 32u) {
    lds[i] = 32u;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // 8 dword loads (32 B/iter) at constant +32-dword (+128 B) bank-preserving offsets,
  // allowing the compiler to fold offsets into 4 dual-address ds_load_2addr_b32 instructions.
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

  uint total = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
  results[gid] = total;
}
