# fish completion for gpubench-gui

complete -c gpubench-gui -s h -l help -d "Print help message and exit"
complete -c gpubench-gui -s v -l version -d "Display version information and exit"
complete -c gpubench-gui -s b -l benchmark -l benchmarks -s t -l test -d "Benchmarks to run" -r -a "fp64 fp32 dualissue lds indirect fp16 bf16 fp8 fp6 fp4 int8 int4 membw l0_cache cache_latency_curve pixelfill rayrawtraversal rayasbuild rayintersect rayanyhit rayprocedural rayscheduling raydivergence sysmembw sysmemlatency all"
complete -c gpubench-gui -s g -l group -l groups -d "Benchmark group(s) to run" -r -a "compute memory raster graphics raytracing system all"
complete -c gpubench-gui -s d -l device -d "Target GPU device index(es)" -r
complete -c gpubench-gui -s k -l backend -d "Backend to select" -r -a "auto vulkan opencl rocm"
complete -c gpubench-gui -s s -l scene -d "Pre-select Ray Tracing scene" -r -a "showroom indoor outdoor forest"
complete -c gpubench-gui -s r -l res -l resolution -d "Resolution preset" -r -a "720p 1080p 1440p 4k 8k"
complete -c gpubench-gui -l ui-scale -l scale -d "Override UI DPI scaling factor" -r -a "1.0 1.25 1.5 1.75 2.0 2.5"
complete -c gpubench-gui -l dump -l dump-renders -d "Enable visual verification and render output dumping"
complete -c gpubench-gui -l no-dump -l no-dump-renders -d "Disable render output dumping"
complete -c gpubench-gui -l auto-run -l auto-start -l run -d "Automatically start benchmark suite immediately on launch"
complete -c gpubench-gui -l exit-on-complete -d "Exit automatically when benchmark run finishes"
complete -c gpubench-gui -l frames -d "Run for specified number of frames and exit" -r
complete -c gpubench-gui -l verbose -d "Enable verbose logging"
complete -c gpubench-gui -l debug -d "Enable debug logging"
