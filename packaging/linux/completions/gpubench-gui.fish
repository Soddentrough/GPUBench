# fish completion for gpubench-gui

set -l gpubench_gui_benchmarks "fp64 fp32 dualissue lds indirect fp16 bf16 fp8 fp6 fp4 int8 int4 sparsematrix coopvector membw interconnect l0b l1b l2b l3b cachebw l0_cache cache_latency_curve asyncoverlap pixelfill rayrawtraversal rayasbuild rayintersect rayanyhit rayomm rayprocedural raypayload rayscheduling raydivergence sysmembw sysmemlatency all"

complete -c gpubench-gui -s h -l help -d "Print help message and exit"
complete -c gpubench-gui -s v -l version -d "Display program version information and exit"
complete -c gpubench-gui -s d -l device -d "Target GPU device index(es)" -r
complete -c gpubench-gui -s k -l backend -d "Pre-select compute backend" -r -a "auto vulkan opencl rocm"
complete -c gpubench-gui -s b -l benchmark -l benchmarks -s t -l test -d "Pre-select benchmark(s)" -r -a "$gpubench_gui_benchmarks"
complete -c gpubench-gui -s g -l group -l groups -d "Pre-select benchmark category" -r -a "compute memory scheduling graphics raster raytracing system all"
complete -c gpubench-gui -s s -l scene -d "Pre-select Ray Tracing scene" -r -a "all showroom indoor outdoor forest"
complete -c gpubench-gui -s r -l res -l resolution -d "Pre-select render resolution" -r -a "auto 720p 1080p 1440p 4k 8k 1024x1024"
complete -c gpubench-gui -l spp -d "Samples per pixel for path tracing (1..256)" -r -a "1 2 4 8 16 32 64 128 256"
complete -c gpubench-gui -s i -l import -l input -d "Import unified JSON session file on startup" -r -F
complete -c gpubench-gui -l compare -d "Load baseline JSON file for comparison" -r -F
complete -c gpubench-gui -s o -l output -l output-json -d "Export unified v2.0 JSON results upon completion" -r -F
complete -c gpubench-gui -l dump -l dump-renders -d "Enable saving rendered scene PNGs to disk"
complete -c gpubench-gui -l no-dump -l no-dump-renders -d "Disable saving rendered scene PNGs to disk"
complete -c gpubench-gui -l auto-run -l auto-start -l run -d "Automatically start benchmarks upon launch"
complete -c gpubench-gui -l roofline -d "Open the Empirical Roofline chart tab"
complete -c gpubench-gui -l exit-on-complete -d "Exit automatically when benchmark run finishes"
complete -c gpubench-gui -l frames -d "Run for specified number of frames and exit" -r
complete -c gpubench-gui -l screenshot -d "Save rendered GUI window PNG before exit" -r -F
complete -c gpubench-gui -l ui-scale -l scale -d "Override UI DPI scaling factor" -r -a "1.0 1.25 1.5 1.75 2.0 2.5"
complete -c gpubench-gui -l mode -d "Initial instrument mode" -r -a "f1 isa f2 charts perf roofline scorecard f3 rt viewport"
complete -c gpubench-gui -l palette -d "Bring-up phosphor palette" -r -a "amber green paper"
complete -c gpubench-gui -l verbose -d "Enable verbose logging output"
complete -c gpubench-gui -l debug -d "Enable Vulkan Khronos Validation Layers"
