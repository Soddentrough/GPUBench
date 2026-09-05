# fish completion for gpubench

complete -c gpubench -s h -l help -d "Print help message and exit"
complete -c gpubench -l version -d "Display version information and exit"
complete -c gpubench -s b -l benchmark -l benchmarks -d "Benchmarks to run" -r -a "fp32 fp64 fp16 bf16 fp8 fp4 int8 int4 mem_bandwidth l1_cache pixel_fill ray_scheduling all"
complete -c gpubench -s g -l group -l groups -d "Benchmark group(s) to run" -r -a "compute memory raytracing graphics system all"
complete -c gpubench -l list-benchmarks -d "List available benchmarks"
complete -c gpubench -l list-groups -d "List available benchmark groups"
complete -c gpubench -s d -l device -d "Device(s) to use" -r
complete -c gpubench -s l -l list-devices -d "List available devices"
complete -c gpubench -l list-backends -d "List available backends"
complete -c gpubench -s k -l backend -d "Backend to use" -r -a "auto vulkan opencl rocm"
complete -c gpubench -l verbose -d "Enable verbose logging"
complete -c gpubench -l debug -d "Enable debug logging"
complete -c gpubench -l dump-geometry -d "Dump ray tracing geometry to OBJ files"
complete -c gpubench -l dump -l dump-renders -d "Dump and compare rendered frames"
complete -c gpubench -s s -l scene -d "Ray tracing benchmark scenario" -r -a "indoor outdoor all"
complete -c gpubench -s r -l resolution -d "Resolution preset" -r -a "auto 720p 1080p 1440p 4k 1024x1024"
complete -c gpubench -s c -l config -d "Run specific config index" -r
complete -c gpubench -l profile-snapshot -d "Profiling snapshot mode"
complete -c gpubench -l rra -d "Radeon Raytracing Analyzer capture"
complete -c gpubench -l output -d "Output format" -r -a "json csv"
complete -c gpubench -l output-file -d "Output file path" -r -F
