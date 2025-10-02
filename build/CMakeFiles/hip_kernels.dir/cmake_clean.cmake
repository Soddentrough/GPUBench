file(REMOVE_RECURSE
  "CMakeFiles/hip_kernels"
  "hip_kernels/cache_bandwidth.o"
  "hip_kernels/cache_latency.o"
  "hip_kernels/cachebw_l1.o"
  "hip_kernels/cachebw_l2.o"
  "hip_kernels/cachebw_l3.o"
  "hip_kernels/fp16.o"
  "hip_kernels/fp32.o"
  "hip_kernels/fp4_emulated.o"
  "hip_kernels/fp64.o"
  "hip_kernels/fp8.o"
  "hip_kernels/fp8_emulated.o"
  "hip_kernels/int4.o"
  "hip_kernels/int8.o"
  "hip_kernels/l0_cache_bandwidth.o"
  "hip_kernels/l0_cache_latency.o"
  "hip_kernels/membw_1024.o"
  "hip_kernels/membw_128.o"
  "hip_kernels/membw_256.o"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/hip_kernels.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
