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
    pub isUnsupported: bool,
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
            isUnsupported: ffi_res.isUnsupported,
        };
        if let Ok(guard) = CALLBACK_MUTEX.lock() {
            if let Some(cb) = *guard {
                cb(&res);
            }
        }
    }
    
    let backend_strs_lower: Vec<String> = backend_strs
        .iter()
        .map(|s| s.to_lowercase())
        .collect();

    let ffi_results = gpubench_sys::ffi::gpubench_run_benchmarks(
        benchmarks,
        device_indices,
        &backend_strs_lower,
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
        isUnsupported: ffi_res.isUnsupported,
    }).collect()
}

/// Compute the display value and unit for a result, mirroring the C++
/// ResultFormatter logic.
fn format_result_value(res: &ResultData) -> (f64, String) {
    let ops = res.operations as f64;
    let time_s = res.time_ms / 1000.0;
    match res.metric.as_str() {
        "TFLOPS" | "TOPS" => (ops / time_s / 1e12, res.metric.clone()),
        "ns" => {
            let v = if res.operations > 0 { res.time_ms * 1e6 / ops } else { 0.0 };
            (v, "ns".to_string())
        }
        "GB/s" => (ops / time_s / 1e9, "GB/s".to_string()),
        other => (ops / time_s, other.to_string()),
    }
}

fn print_results_table(results: &[ResultData]) {
    if results.is_empty() {
        println!("No results.");
        return;
    }

    let rows: Vec<(String, String, String, String)> = results
        .iter()
        .map(|r| {
            let (value, unit) = format_result_value(r);
            (
                r.benchmarkName.clone(),
                r.component.clone(),
                format!("{:.2}", value),
                unit,
            )
        })
        .collect();

    let w_bench = rows.iter().map(|r| r.0.len()).max().unwrap_or(9).max(9);
    let w_comp = rows.iter().map(|r| r.1.len()).max().unwrap_or(9).max(9);
    let w_val = rows.iter().map(|r| r.2.len()).max().unwrap_or(5).max(5);

    println!(
        "{:<w_bench$} | {:<w_comp$} | {:>w_val$} | {}",
        "benchmark", "component", "value", "unit",
        w_bench = w_bench,
        w_comp = w_comp,
        w_val = w_val
    );
    println!(
        "{:-<w_bench$}-+-{:-<w_comp$}-+-{:-<w_val$}-+------",
        "",
        "",
        "",
        w_bench = w_bench,
        w_comp = w_comp,
        w_val = w_val
    );
    for (bench, comp, val, unit) in &rows {
        println!(
            "{:<w_bench$} | {:<w_comp$} | {:>w_val$} | {}",
            bench,
            comp,
            val,
            unit,
            w_bench = w_bench,
            w_comp = w_comp,
            w_val = w_val
        );
    }
}

pub fn run_cli() {
    // RADV prints a conformance warning to stderr unless this is set.
    if std::env::var_os("MESA_VK_IGNORE_CONFORMANCE_WARNING").is_none() {
        // SAFETY: called at process startup, before any threads are spawned.
        unsafe { std::env::set_var("MESA_VK_IGNORE_CONFORMANCE_WARNING", "1") };
    }

    let cli = Cli::parse();

    if cli.list_benchmarks {
        println!("Available benchmarks:");
        for name in get_available_benchmarks() {
            println!("  {}", name);
        }
        return;
    }

    if cli.list_devices {
        println!("Available devices (backend | index | name):");
        for entry in get_available_hardware() {
            println!("  {}", entry);
        }
        return;
    }

    if cli.list_backends {
        // Hardware entries are "backend|index|name"; collect unique backends,
        // excluding the pseudo "System" entry.
        let mut backends: Vec<String> = Vec::new();
        for entry in get_available_hardware() {
            if let Some(backend) = entry.split('|').next() {
                if backend != "System" && !backends.iter().any(|b| b == backend) {
                    backends.push(backend.to_string());
                }
            }
        }
        println!("Available backends:");
        for b in backends {
            println!("  {}", b);
        }
        return;
    }

    // Empty benchmark list = run all (C++ BenchmarkRunner convention).
    // Empty device list = device 0, empty backend list = auto (C++ RunnerAPI convention).
    let results = run_benchmarks(
        &cli.benchmarks_to_run,
        &cli.device_indices,
        &cli.backend_strs,
        cli.verbose,
        cli.debug,
        cli.dump_geometry,
        |_| {},
    );

    print_results_table(&results);

    // Exit non-zero when nothing ran (backend failure, unmatched benchmark
    // names, out-of-range device) so scripts can detect it.
    if results.is_empty() {
        eprintln!("Error: no benchmark results were produced.");
        std::process::exit(1);
    }
}
