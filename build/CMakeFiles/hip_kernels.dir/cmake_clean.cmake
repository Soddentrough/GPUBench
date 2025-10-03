file(REMOVE_RECURSE
  "CMakeFiles/hip_kernels"
  "kernels/rocm/cache_bandwidth.o"
  "kernels/rocm/cache_latency.o"
  "kernels/rocm/cachebw_l1.o"
  "kernels/rocm/cachebw_l2.o"
  "kernels/rocm/cachebw_l3.o"
  "kernels/rocm/fp16.o"
  "kernels/rocm/fp32.o"
  "kernels/rocm/fp4_emulated.o"
  "kernels/rocm/fp64.o"
  "kernels/rocm/fp8.o"
  "kernels/rocm/fp8_emulated.o"
  "kernels/rocm/int4.o"
  "kernels/rocm/int8.o"
  "kernels/rocm/l0_cache_bandwidth.o"
  "kernels/rocm/l0_cache_latency.o"
  "kernels/rocm/membw_1024.o"
  "kernels/rocm/membw_128.o"
  "kernels/rocm/membw_256.o"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/hip_kernels.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
