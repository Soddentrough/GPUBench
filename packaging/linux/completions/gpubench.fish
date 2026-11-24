# fish completion for gpubench

set -l gpubench_benchmarks "fp64 fp32 dualissue lds indirect fp16 bf16 fp8 fp6 fp4 int8 int4 sparsematrix coopvector membw interconnect l0b l1b l2b l3b cachebw l0_cache cache_latency_curve asyncoverlap pixelfill rayrawtraversal rayasbuild rayintersect rayanyhit rayomm rayprocedural raypayload rayscheduling raydivergence sysmembw sysmemlatency all"

complete -c gpubench -s h -l help -d "Print help message and exit"
complete -c gpubench -s v -l version -d "Display version information and exit"
complete -c gpubench -s b -l benchmark -l benchmarks -d "Benchmarks to run" -r -a "$gpubench_benchmarks"
complete -c gpubench -s g -l group -l groups -d "Benchmark group(s) to run" -r -a "compute memory scheduling graphics raster raytracing system all"
complete -c gpubench -l list -l list-benchmarks -d "List available benchmarks"
complete -c gpubench -l list-groups -d "List available benchmark groups"
complete -c gpubench -s d -l device -d "Device(s) to use" -r
complete -c gpubench -s l -l list-devices -d "List available devices"
complete -c gpubench -l list-backends -d "List available backends"
complete -c gpubench -s k -l backend -d "Backend to use" -r -a "auto vulkan opencl rocm"
complete -c gpubench -l verbose -d "Enable verbose logging"
complete -c gpubench -l debug -d "Enable debug logging"
complete -c gpubench -s q -l quiet -d "Suppress informational banners and progress output"
complete -c gpubench -l power -d "Display power and energy efficiency telemetry inline in benchmark tables"
complete -c gpubench -l telemetry -d "Display hardware performance counters and compiler register telemetry"
complete -c gpubench -l roofline -d "Compute and display the Empirical Roofline Model"
complete -c gpubench -l strict -d "Abort if active background GPU contention is detected"
complete -c gpubench -s f -l force -d "Force execution despite concurrent GPU activity"
complete -c gpubench -l verify-parity -d "Enforce strict Megakernel vs. DGC visual parity verification"
complete -c gpubench -l dump-geometry -d "Dump ray tracing geometry to OBJ files"
complete -c gpubench -l dump -l dump-renders -d "Dump and compare rendered frames"
complete -c gpubench -l no-dump -l no-dump-renders -d "Disable dumping rendered frames"
complete -c gpubench -s s -l scene -d "Ray tracing benchmark scenario" -r -a "all showroom indoor outdoor forest"
complete -c gpubench -s r -l res -l resolution -d "Resolution preset" -r -a "auto 720p 1080p 1440p 4k 8k 1024x1024"
complete -c gpubench -l bounces -d "Number of light bounces for path tracing (1..8)" -r -a "1 2 3 4 5 6 7 8"
complete -c gpubench -l spp -d "Number of samples per pixel for path tracing" -r -a "1 2 4 8 16 32 64 128 256"
complete -c gpubench -s c -l config -d "Run specific config index" -r
complete -c gpubench -l profile-snapshot -d "Profiling snapshot mode"
complete -c gpubench -l rra -d "Radeon Raytracing Analyzer capture (implies --profile-snapshot)"
complete -c gpubench -l dump-isa -d "Disassemble shader/kernel hardware ISA natively" -r -a "$gpubench_benchmarks"
complete -c gpubench -l asic -d "Target GPU architecture for ROCm disassembly" -r
complete -c gpubench -l full-isa -d "Print entire shader/kernel ISA disassembly"
complete -c gpubench -s o -l output -l output-json -d "Write benchmark results to JSON report file" -r -F
complete -c gpubench -l json -d "Write pure JSON report to stdout or file" -r -F
complete -c gpubench -s i -l import -l input -d "Load and display results from JSON file(s)" -r -F
complete -c gpubench -l compare -d "Compare benchmark JSON report files" -r -F
