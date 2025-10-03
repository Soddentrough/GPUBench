file(REMOVE_RECURSE
  "CMakeFiles/shaders"
  "kernels/vulkan/cache_bandwidth.spv"
  "kernels/vulkan/cache_latency.spv"
  "kernels/vulkan/cachebw_l1.spv"
  "kernels/vulkan/cachebw_l2.spv"
  "kernels/vulkan/cachebw_l3.spv"
  "kernels/vulkan/fp16.spv"
  "kernels/vulkan/fp32.spv"
  "kernels/vulkan/fp4_emulated.spv"
  "kernels/vulkan/fp4_native.spv"
  "kernels/vulkan/fp64.spv"
  "kernels/vulkan/fp8_emulated.spv"
  "kernels/vulkan/fp8_native.spv"
  "kernels/vulkan/int4.spv"
  "kernels/vulkan/int8.spv"
  "kernels/vulkan/l0_cache.spv"
  "kernels/vulkan/l0_cache_bandwidth.spv"
  "kernels/vulkan/l0_cache_latency.spv"
  "kernels/vulkan/membw.spv"
  "kernels/vulkan/membw_1024.spv"
  "kernels/vulkan/membw_128.spv"
  "kernels/vulkan/membw_256.spv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/shaders.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
