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

use std::sync::Mutex;

pub fn get_available_hardware() -> Vec<String> {
    gpubench_sys::ffi::gpubench_get_available_hardware()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

pub fn get_available_benchmarks() -> Vec<String> {
    gpubench_sys::ffi::gpubench_get_available_benchmarks()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

pub fn run_benchmarks(
    benchmarks: &Vec<String>,
    device_indices: &Vec<u32>,
    backend_strs: &Vec<String>,
    verbose: bool,
    debug: bool,
    dump_geometry: bool,
    callback: fn(&ResultData),
) -> Vec<ResultData> {
    static CALLBACK_MUTEX: Mutex<Option<fn(&ResultData)>> = Mutex::new(None);
    
    if let Ok(mut guard) = CALLBACK_MUTEX.lock() {
        *guard = Some(callback);
    }
    
    fn ffi_callback(ffi_res: &gpubench_sys::ffi::FfiResultData) {
        let res = ResultData {
            backendName: ffi_res.backendName.clone(),
            deviceName: ffi_res.deviceName.clone(),
            deviceIndex: ffi_res.deviceIndex,
            component: ffi_res.component.clone(),
            subcategory: ffi_res.subcategory.clone(),
            sortWeight: ffi_res.sortWeight,
            benchmarkName: ffi_res.benchmarkName.clone(),
            configIndex: ffi_res.configIndex,
            metric: ffi_res.metric.clone(),
            operations: ffi_res.operations,
            time_ms: ffi_res.time_ms,
            isEmulated: ffi_res.isEmulated,
        };
        if let Ok(guard) = CALLBACK_MUTEX.lock() {
            if let Some(cb) = *guard {
                cb(&res);
            }
        }
    }
    
    let ffi_results = gpubench_sys::ffi::gpubench_run_benchmarks(
        benchmarks,
        device_indices,
        backend_strs,
        verbose,
        debug,
        dump_geometry,
        ffi_callback,
    );
    
    ffi_results.into_iter().map(|ffi_res| ResultData {
        backendName: ffi_res.backendName,
        deviceName: ffi_res.deviceName,
        deviceIndex: ffi_res.deviceIndex,
        component: ffi_res.component,
        subcategory: ffi_res.subcategory,
        sortWeight: ffi_res.sortWeight,
        benchmarkName: ffi_res.benchmarkName,
        configIndex: ffi_res.configIndex,
        metric: ffi_res.metric,
        operations: ffi_res.operations,
        time_ms: ffi_res.time_ms,
        isEmulated: ffi_res.isEmulated,
    }).collect()
}

pub fn run_cli() {
    println!("GPUBench CLI has been ported to pure Rust. Implementation pending.");
}
