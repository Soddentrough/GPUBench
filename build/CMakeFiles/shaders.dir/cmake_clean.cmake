file(REMOVE_RECURSE
  "CMakeFiles/shaders"
  "cachebw_l1.spv"
  "cachebw_l2.spv"
  "cachebw_l3.spv"
  "fp16.spv"
  "fp32.spv"
  "fp64.spv"
  "fp8.spv"
  "int4.spv"
  "int8.spv"
  "membw.spv"
  "membw_1024.spv"
  "membw_128.spv"
  "membw_256.spv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/shaders.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
