pub mod context;
pub mod vulkan;
pub mod benchmarks;

use clap::Parser;

#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct ResultData {
    pub backendName: String,
    pub deviceName: String,
    pub deviceIndex: u32,
    pub component: String,
    pub subcategory: String,
    pub sortWeight: i32,
    pub benchmarkName: String,
    pub configIndex: u32,
    pub metric: String,
    pub operations: u64,
    pub time_ms: f64,
    pub isEmulated: bool,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "GPUBench")]
#[command(version = "1.1.0")]
#[command(about = "High-performance cross-platform GPU benchmarking tool", long_about = None)]
pub struct Cli {
    #[arg(short = 'b', long = "benchmarks", value_delimiter = ',')]
    pub benchmarks_to_run: Vec<String>,

    #[arg(long = "list-benchmarks", default_value_t = false)]
    pub list_benchmarks: bool,

    #[arg(short = 'd', long = "device", value_delimiter = ',')]
    pub device_indices: Vec<u32>,

    #[arg(short = 'l', long = "list-devices", default_value_t = false)]
    pub list_devices: bool,

    #[arg(long = "list-backends", default_value_t = false)]
    pub list_backends: bool,

    #[arg(short = 'k', long = "backend", value_delimiter = ',')]
    pub backend_strs: Vec<String>,

    #[arg(long = "verbose", default_value_t = false)]
    pub verbose: bool,

    #[arg(long = "debug", default_value_t = false)]
    pub debug: bool,

    #[arg(long = "dump-geometry", default_value_t = false)]
    pub dump_geometry: bool,
}

use std::sync::LazyLock;

static CACHED_HARDWARE: LazyLock<Vec<String>> = LazyLock::new(|| {
    let mut hw = Vec::new();
    
    // Pure Rust Vulkan Discovery
    if let Ok(entry) = unsafe { ash::Entry::load() } {
        let app_info = ash::vk::ApplicationInfo::default().api_version(ash::vk::make_api_version(0, 1, 2, 0));
        let create_info = ash::vk::InstanceCreateInfo::default().application_info(&app_info);
        if let Ok(instance) = unsafe { entry.create_instance(&create_info, None) } {
            if let Ok(devices) = unsafe { instance.enumerate_physical_devices() } {
                for (idx, device) in devices.iter().enumerate() {
                    let props = unsafe { instance.get_physical_device_properties(*device) };
                    if props.device_type == ash::vk::PhysicalDeviceType::CPU || props.device_type == ash::vk::PhysicalDeviceType::VIRTUAL_GPU { continue; }
                    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned() };
                    if name.to_lowercase().contains("llvmpipe") { continue; }
                    hw.push(format!("vulkan|{}|{}", idx, name));
                }
            }
            unsafe { instance.destroy_instance(None); }
        }
    }

    // Pure Rust OpenCL Discovery
    if let Ok(platforms) = opencl3::platform::get_platforms() {
        let mut cl_idx = 0;
        for platform in platforms {
            if let Ok(devices) = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU) {
                for device_id in devices {
                    let dev = opencl3::device::Device::new(device_id);
                    if let Ok(name) = dev.name() {
                        hw.push(format!("opencl|{}|{}", cl_idx, name));
                        cl_idx += 1;
                    }
                }
            }
        }
    }

    hw
});

pub fn get_available_hardware() -> Vec<String> {
    CACHED_HARDWARE.clone()
}

pub fn get_available_benchmarks() -> Vec<String> {
    // We will port these step by step, returning empty for now or dummy
    vec![
        "FP64".to_string(),
        "FP32".to_string(),
        "FP16".to_string(),
        "BF16".to_string(),
        "FP8".to_string(),
        "INT8".to_string(),
        "INT4".to_string(),
        "MemBandwidth".to_string(),
        "RayTracing".to_string(),
        "RayDivergence".to_string(),
        "RayAnyHit".to_string(),
        "RayIncoherent".to_string(),
        "RayPayload".to_string(),
        "RayASBuild".to_string(),
        "RayProcedural".to_string(),
        "SysMemBandwidth".to_string(),
        "SysMemLatency".to_string(),
    ]
}

use crate::context::ComputeContext;
use crate::benchmarks::Benchmark;
use crate::benchmarks::fp64::Fp64Bench;
use crate::benchmarks::fp32::Fp32Bench;
use crate::benchmarks::fp16::FP16Bench;
use crate::benchmarks::bf16::Bf16Bench;
use crate::benchmarks::fp8::Fp8Bench;
use crate::benchmarks::int8::Int8Bench;
use crate::benchmarks::int4::Int4Bench;
use crate::benchmarks::membw::MemBwBench;
use crate::benchmarks::sysmembw::SysMemBandwidthBench;
use crate::benchmarks::sysmemlat::SysMemLatencyBench;
use crate::vulkan::VulkanContext;

pub fn run_benchmarks(
    benchmarks: &Vec<String>,
    device_indices: &Vec<u32>,
    _backend_strs: &Vec<String>,
    _verbose: bool,
    _debug: bool,
    _dump_geometry: bool,
    callback: fn(&ResultData),
) -> Vec<ResultData> {
    let mut results = Vec::new();

    // Default to device 0 if none specified
    let device_idx = device_indices.first().copied().unwrap_or(0);
    
    // Create native context
    let mut context: Box<dyn ComputeContext> = match VulkanContext::new() {
        Ok(v) => Box::new(v),
        Err(e) => {
            eprintln!("Failed to initialize VulkanContext: {}", e);
            return results;
        }
    };

    if let Err(e) = context.pick_device(device_idx) {
        eprintln!("Failed to pick device {}: {}", device_idx, e);
        return results;
    }

    let dev_info = context.get_current_device_info().unwrap();

    if benchmarks.contains(&"FP64".to_string()) {
        let mut fp64_bench = Fp64Bench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";

        match fp64_bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                for config_idx in 0..fp64_bench.get_num_configs() {
                    if fp64_bench.run(context.as_mut(), config_idx).is_ok() {
                        let (ops, time_ms) = fp64_bench.get_result(config_idx);
                        
                        let res = ResultData {
                            backendName: "Vulkan".to_string(),
                            deviceName: dev_info.name.clone(),
                            deviceIndex: device_idx,
                            component: fp64_bench.get_component(config_idx).to_string(),
                            subcategory: fp64_bench.get_subcategory(config_idx).to_string(),
                            sortWeight: 0,
                            benchmarkName: fp64_bench.get_config_name(config_idx).to_string(),
                            configIndex: config_idx,
                            metric: fp64_bench.get_metric().to_string(),
                            operations: ops,
                            time_ms,
                            isEmulated: false,
                        };
                        callback(&res);
                        results.push(res);
                    }
                }
                fp64_bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("[DIAGNOSTIC] FP64 setup failed: {}", e),
        }
    }

    if benchmarks.contains(&"FP32".to_string()) {
        let mut fp32_bench = Fp32Bench::new();
        
        // We assume kernels are located in kernels dir
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";

        if fp32_bench.setup(context.as_mut(), kernel_dir).is_ok() {
            for config_idx in 0..fp32_bench.get_num_configs() {
                if fp32_bench.run(context.as_mut(), config_idx).is_ok() {
                    let (ops, time_ms) = fp32_bench.get_result(config_idx);
                    
                    let res = ResultData {
                        backendName: "Vulkan".to_string(),
                        deviceName: dev_info.name.clone(),
                        deviceIndex: device_idx,
                        component: fp32_bench.get_component(config_idx).to_string(),
                        subcategory: fp32_bench.get_subcategory(config_idx).to_string(),
                        sortWeight: 0,
                        benchmarkName: fp32_bench.get_config_name(config_idx).to_string(),
                        configIndex: config_idx,
                        metric: fp32_bench.get_metric().to_string(),
                        operations: ops,
                        time_ms,
                        isEmulated: false,
                    };
                    callback(&res);
                    results.push(res);
                }
            }
            fp32_bench.teardown(context.as_mut());
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
    }

    if benchmarks.contains(&"FP16".to_string()) {
        let mut bench = FP16Bench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                if bench.run(context.as_mut(), 0).is_ok() {
                    let (ops, time_ms) = bench.get_result(0);
                    let res = ResultData { backendName: "Vulkan".to_string(), deviceName: dev_info.name.clone(), deviceIndex: device_idx, component: bench.get_component(0).to_string(), subcategory: bench.get_subcategory(0).to_string(), sortWeight: 0, benchmarkName: bench.get_config_name(0).to_string(), configIndex: 0, metric: bench.get_metric().to_string(), operations: ops, time_ms, isEmulated: false };
                    callback(&res); results.push(res);
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("[DIAGNOSTIC] FP16 setup failed: {}", e),
        }
    }

    if benchmarks.contains(&"BF16".to_string()) {
        let mut bench = Bf16Bench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                if bench.run(context.as_mut(), 0).is_ok() {
                    let (ops, time_ms) = bench.get_result(0);
                    let res = ResultData { backendName: "Vulkan".to_string(), deviceName: dev_info.name.clone(), deviceIndex: device_idx, component: bench.get_component(0).to_string(), subcategory: bench.get_subcategory(0).to_string(), sortWeight: 0, benchmarkName: bench.get_config_name(0).to_string(), configIndex: 0, metric: bench.get_metric().to_string(), operations: ops, time_ms, isEmulated: false };
                    callback(&res); results.push(res);
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("[DIAGNOSTIC] BF16 setup failed: {}", e),
        }
    }

    if benchmarks.contains(&"FP8".to_string()) {
        let mut bench = Fp8Bench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                if bench.run(context.as_mut(), 0).is_ok() {
                    let (ops, time_ms) = bench.get_result(0);
                    let res = ResultData { backendName: "Vulkan".to_string(), deviceName: dev_info.name.clone(), deviceIndex: device_idx, component: bench.get_component(0).to_string(), subcategory: bench.get_subcategory(0).to_string(), sortWeight: 0, benchmarkName: bench.get_config_name(0).to_string(), configIndex: 0, metric: bench.get_metric().to_string(), operations: ops, time_ms, isEmulated: false };
                    callback(&res); results.push(res);
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("[DIAGNOSTIC] FP8 setup failed: {}", e),
        }
    }

    if benchmarks.contains(&"INT8".to_string()) {
        let mut bench = Int8Bench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                if bench.run(context.as_mut(), 0).is_ok() {
                    let (ops, time_ms) = bench.get_result(0);
                    let res = ResultData { backendName: "Vulkan".to_string(), deviceName: dev_info.name.clone(), deviceIndex: device_idx, component: bench.get_component(0).to_string(), subcategory: bench.get_subcategory(0).to_string(), sortWeight: 0, benchmarkName: bench.get_config_name(0).to_string(), configIndex: 0, metric: bench.get_metric().to_string(), operations: ops, time_ms, isEmulated: false };
                    callback(&res); results.push(res);
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("[DIAGNOSTIC] INT8 setup failed: {}", e),
        }
    }

    if benchmarks.contains(&"INT4".to_string()) {
        let mut bench = Int4Bench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                if bench.run(context.as_mut(), 0).is_ok() {
                    let (ops, time_ms) = bench.get_result(0);
                    let res = ResultData { backendName: "Vulkan".to_string(), deviceName: dev_info.name.clone(), deviceIndex: device_idx, component: bench.get_component(0).to_string(), subcategory: bench.get_subcategory(0).to_string(), sortWeight: 0, benchmarkName: bench.get_config_name(0).to_string(), configIndex: 0, metric: bench.get_metric().to_string(), operations: ops, time_ms, isEmulated: false };
                    callback(&res); results.push(res);
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("[DIAGNOSTIC] INT4 setup failed: {}", e),
        }
    }

    if benchmarks.contains(&"MemBandwidth".to_string()) {
        let mut membw_bench = MemBwBench::new();
        
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";

        match membw_bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                for config_idx in 0..membw_bench.get_num_configs() {
                    if membw_bench.run(context.as_mut(), config_idx).is_ok() {
                        let (ops, time_ms) = membw_bench.get_result(config_idx);
                        
                        let res = ResultData {
                            backendName: "Vulkan".to_string(),
                            deviceName: dev_info.name.clone(),
                            deviceIndex: device_idx,
                            component: membw_bench.get_component(config_idx).to_string(),
                            subcategory: membw_bench.get_subcategory(config_idx).to_string(),
                            sortWeight: 0,
                            benchmarkName: membw_bench.get_config_name(config_idx).to_string(),
                            configIndex: config_idx,
                            metric: membw_bench.get_metric().to_string(),
                            operations: ops,
                            time_ms,
                            isEmulated: false,
                        };
                        callback(&res);
                        results.push(res);
                    }
                }
                membw_bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => {
                eprintln!("[DIAGNOSTIC] MemBandwidth setup failed: {}", e);
            }
        }
    }

    if benchmarks.contains(&"SysMemBandwidth".to_string()) {
        let mut sys_bw = SysMemBandwidthBench::new();
        if sys_bw.setup(context.as_mut(), "").is_ok() {
            for config_idx in 0..sys_bw.get_num_configs() {
                if sys_bw.run(context.as_mut(), config_idx).is_ok() {
                    let (bytes, time_ms) = sys_bw.get_result(config_idx);
                    let res = ResultData {
                        backendName: "Native".to_string(),
                        deviceName: "Host CPU".to_string(),
                        deviceIndex: 0,
                        component: sys_bw.get_component(config_idx).to_string(),
                        subcategory: sys_bw.get_subcategory(config_idx).to_string(),
                        sortWeight: config_idx as i32,
                        benchmarkName: sys_bw.get_name().to_string(),
                        configIndex: config_idx,
                        metric: sys_bw.get_metric().to_string(),
                        operations: bytes,
                        time_ms,
                        isEmulated: false,
                    };
                    callback(&res);
                    results.push(res);
                }
            }
            sys_bw.teardown(context.as_mut());
        }
    }

    if benchmarks.contains(&"SysMemLatency".to_string()) {
        let mut sys_lat = SysMemLatencyBench::new();
        if sys_lat.setup(context.as_mut(), "").is_ok() {
            for config_idx in 0..sys_lat.get_num_configs() {
                if sys_lat.run(context.as_mut(), config_idx).is_ok() {
                    let (ops, time_ms) = sys_lat.get_result(config_idx);
                    let res = ResultData {
                        backendName: "Native".to_string(),
                        deviceName: "Host CPU".to_string(),
                        deviceIndex: 0,
                        component: sys_lat.get_component(config_idx).to_string(),
                        subcategory: sys_lat.get_subcategory(config_idx).to_string(),
                        sortWeight: config_idx as i32,
                        benchmarkName: sys_lat.get_name().to_string(),
                        configIndex: config_idx,
                        metric: sys_lat.get_metric().to_string(),
                        operations: ops,
                        time_ms,
                        isEmulated: false,
                    };
                    callback(&res);
                    results.push(res);
                }
            }
            sys_lat.teardown(context.as_mut());
        }
    }

    if benchmarks.contains(&"RayTracing".to_string()) {
        let mut bench = crate::benchmarks::rt_intersection::RayTracingBench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                for config_idx in 0..bench.get_num_configs() {
                    match bench.run(context.as_mut(), config_idx) {
                        Ok(_) => {
                            let (ops, time_ms) = bench.get_result(config_idx);
                            let res = ResultData {
                                backendName: "Vulkan".to_string(),
                                deviceName: dev_info.name.clone(),
                                deviceIndex: device_idx,
                                component: bench.get_component(config_idx).to_string(),
                                subcategory: bench.get_subcategory(config_idx).to_string(),
                                sortWeight: config_idx as i32,
                                benchmarkName: bench.get_config_name(config_idx).to_string(),
                                configIndex: config_idx,
                                metric: bench.get_metric().to_string(),
                                operations: ops,
                                time_ms,
                                isEmulated: false,
                            };
                            callback(&res);
                            results.push(res);
                        }
                        Err(e) => eprintln!("RayTracing run {} error: {}", config_idx, e),
                    }
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("RayTracing setup error: {}", e),
        }
    }

    if benchmarks.contains(&"RayDivergence".to_string()) {
        let mut bench = crate::benchmarks::rt_divergence::RayDivergenceBench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                for config_idx in 0..bench.get_num_configs() {
                    match bench.run(context.as_mut(), config_idx) {
                        Ok(_) => {
                            let (ops, time_ms) = bench.get_result(config_idx);
                            let res = ResultData {
                                backendName: "Vulkan".to_string(),
                                deviceName: dev_info.name.clone(),
                                deviceIndex: device_idx,
                                component: bench.get_component(config_idx).to_string(),
                                subcategory: bench.get_subcategory(config_idx).to_string(),
                                sortWeight: config_idx as i32,
                                benchmarkName: bench.get_config_name(config_idx).to_string(),
                                configIndex: config_idx,
                                metric: bench.get_metric().to_string(),
                                operations: ops,
                                time_ms,
                                isEmulated: false,
                            };
                            callback(&res);
                            results.push(res);
                        }
                        Err(e) => eprintln!("RayDivergence run {} error: {}", config_idx, e),
                    }
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("RayDivergence setup error: {}", e),
        }
    }

    if benchmarks.contains(&"RayAnyHit".to_string()) {
        let mut bench = crate::benchmarks::rt_anyhit::RayAnyHitBench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                for config_idx in 0..bench.get_num_configs() {
                    match bench.run(context.as_mut(), config_idx) {
                        Ok(_) => {
                            let (ops, time_ms) = bench.get_result(config_idx);
                            let res = ResultData {
                                backendName: "Vulkan".to_string(),
                                deviceName: dev_info.name.clone(),
                                deviceIndex: device_idx,
                                component: bench.get_component(config_idx).to_string(),
                                subcategory: bench.get_subcategory(config_idx).to_string(),
                                sortWeight: config_idx as i32,
                                benchmarkName: bench.get_config_name(config_idx).to_string(),
                                configIndex: config_idx,
                                metric: bench.get_metric().to_string(),
                                operations: ops,
                                time_ms,
                                isEmulated: false,
                            };
                            callback(&res);
                            results.push(res);
                        }
                        Err(e) => eprintln!("RayAnyHit run {} error: {}", config_idx, e),
                    }
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("RayAnyHit setup error: {}", e),
        }
    }

    if benchmarks.contains(&"RayIncoherent".to_string()) {
        let mut bench = crate::benchmarks::rt_incoherent::RayIncoherentBench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                for config_idx in 0..bench.get_num_configs() {
                    match bench.run(context.as_mut(), config_idx) {
                        Ok(_) => {
                            let (ops, time_ms) = bench.get_result(config_idx);
                            let res = ResultData {
                                backendName: "Vulkan".to_string(),
                                deviceName: dev_info.name.clone(),
                                deviceIndex: device_idx,
                                component: bench.get_component(config_idx).to_string(),
                                subcategory: bench.get_subcategory(config_idx).to_string(),
                                sortWeight: config_idx as i32,
                                benchmarkName: bench.get_config_name(config_idx).to_string(),
                                configIndex: config_idx,
                                metric: bench.get_metric().to_string(),
                                operations: ops,
                                time_ms,
                                isEmulated: false,
                            };
                            callback(&res);
                            results.push(res);
                        }
                        Err(e) => eprintln!("RayIncoherent run {} error: {}", config_idx, e),
                    }
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("RayIncoherent setup error: {}", e),
        }
    }

    if benchmarks.contains(&"RayPayload".to_string()) {
        let mut bench = crate::benchmarks::rt_payload::RayPayloadBench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                for config_idx in 0..bench.get_num_configs() {
                    match bench.run(context.as_mut(), config_idx) {
                        Ok(_) => {
                            let (ops, time_ms) = bench.get_result(config_idx);
                            let res = ResultData {
                                backendName: "Vulkan".to_string(),
                                deviceName: dev_info.name.clone(),
                                deviceIndex: device_idx,
                                component: bench.get_component(config_idx).to_string(),
                                subcategory: bench.get_subcategory(config_idx).to_string(),
                                sortWeight: config_idx as i32,
                                benchmarkName: bench.get_config_name(config_idx).to_string(),
                                configIndex: config_idx,
                                metric: bench.get_metric().to_string(),
                                operations: ops,
                                time_ms,
                                isEmulated: false,
                            };
                            callback(&res);
                            results.push(res);
                        }
                        Err(e) => eprintln!("RayPayload run {} error: {}", config_idx, e),
                    }
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("RayPayload setup error: {}", e),
        }
    }

    if benchmarks.contains(&"RayASBuild".to_string()) {
        let mut bench = crate::benchmarks::rt_asbuild::RayASBuildBench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        if bench.setup(context.as_mut(), kernel_dir).is_ok() {
            for config_idx in 0..bench.get_num_configs() {
                if bench.run(context.as_mut(), config_idx).is_ok() {
                    let (ops, time_ms) = bench.get_result(config_idx);
                    let res = ResultData {
                        backendName: "Vulkan".to_string(),
                        deviceName: dev_info.name.clone(),
                        deviceIndex: device_idx,
                        component: bench.get_component(config_idx).to_string(),
                        subcategory: bench.get_subcategory(config_idx).to_string(),
                        sortWeight: config_idx as i32,
                        benchmarkName: bench.get_config_name(config_idx).to_string(),
                        configIndex: config_idx,
                        metric: bench.get_metric().to_string(),
                        operations: ops,
                        time_ms,
                        isEmulated: false,
                    };
                    callback(&res);
                    results.push(res);
                }
            }
            bench.teardown(context.as_mut());
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
    }

    if benchmarks.contains(&"RayProcedural".to_string()) {
        let mut bench = crate::benchmarks::rt_procedural::RayProceduralBench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        match bench.setup(context.as_mut(), kernel_dir) {
            Ok(_) => {
                for config_idx in 0..bench.get_num_configs() {
                    match bench.run(context.as_mut(), config_idx) {
                        Ok(_) => {
                            let (ops, time_ms) = bench.get_result(config_idx);
                            let res = ResultData {
                                backendName: "Vulkan".to_string(),
                                deviceName: dev_info.name.clone(),
                                deviceIndex: device_idx,
                                component: bench.get_component(config_idx).to_string(),
                                subcategory: bench.get_subcategory(config_idx).to_string(),
                                sortWeight: config_idx as i32,
                                benchmarkName: bench.get_config_name(config_idx).to_string(),
                                configIndex: config_idx,
                                metric: bench.get_metric().to_string(),
                                operations: ops,
                                time_ms,
                                isEmulated: false,
                            };
                            callback(&res);
                            results.push(res);
                        }
                        Err(e) => eprintln!("RayProcedural run {} error: {}", config_idx, e),
                    }
                }
                bench.teardown(context.as_mut());
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            Err(e) => eprintln!("RayProcedural setup error: {}", e),
        }
    }

    if benchmarks.contains(&"RayMaterialDivergence".to_string()) {
        let mut bench = crate::benchmarks::rt_matdiv::RayMaterialDivergenceBench::new();
        let kernel_dir = "/home/naoki/Development/GPUBench/kernels";
        if bench.setup(context.as_mut(), kernel_dir).is_ok() {
            for config_idx in 0..bench.get_num_configs() {
                if bench.run(context.as_mut(), config_idx).is_ok() {
                    let (ops, time_ms) = bench.get_result(config_idx);
                    let res = ResultData {
                        backendName: "Vulkan".to_string(),
                        deviceName: dev_info.name.clone(),
                        deviceIndex: device_idx,
                        component: bench.get_component(config_idx).to_string(),
                        subcategory: bench.get_subcategory(config_idx).to_string(),
                        sortWeight: config_idx as i32,
                        benchmarkName: bench.get_config_name(config_idx).to_string(),
                        configIndex: config_idx,
                        metric: bench.get_metric().to_string(),
                        operations: ops,
                        time_ms,
                        isEmulated: false,
                    };
                    callback(&res);
                    results.push(res);
                }
            }
            bench.teardown(context.as_mut());
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
    }

    results
}

pub fn run_cli() {
    println!("GPUBench CLI has been ported to pure Rust. Implementation pending.");
}
