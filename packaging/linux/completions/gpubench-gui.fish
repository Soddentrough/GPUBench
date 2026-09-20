# fish completion for gpubench-gui

complete -c gpubench-gui -s h -l help -d "Print help message and exit"
complete -c gpubench-gui -s v -l version -d "Display version information and exit"
complete -c gpubench-gui -s b -l benchmark -l benchmarks -d "Benchmarks to run" -r -a "fp32 fp64 fp16 bf16 fp8 int8 int4 membw fillrate sysmem syslat asbuild triangle anyhit procedural mat_divergence incoherent divergence payload pathtracing rayscheduling all"
complete -c gpubench-gui -s g -l group -l groups -d "Benchmark group(s) to run" -r -a "compute memory raster graphics raytracing system all"
complete -c gpubench-gui -s d -l device -l devices -d "Target GPU device index(es)" -r
complete -c gpubench-gui -s k -l backend -d "Backend to select" -r -a "auto vulkan opencl rocm"
complete -c gpubench-gui -l dump -l dump-renders -d "Enable visual verification and render output dumping"
complete -c gpubench-gui -l auto-start -l run -d "Automatically start benchmark suite immediately on launch"
complete -c gpubench-gui -s r -l resolution -d "Resolution preset" -r -a "auto 1080p 1440p 4k"
