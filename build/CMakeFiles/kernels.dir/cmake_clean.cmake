file(REMOVE_RECURSE
  "CMakeFiles/kernels"
  "kernels/fp16.cl"
  "kernels/fp32.cl"
  "kernels/fp4.cl"
  "kernels/fp64.cl"
  "kernels/fp8.cl"
  "kernels/int4.cl"
  "kernels/int8.cl"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/kernels.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
