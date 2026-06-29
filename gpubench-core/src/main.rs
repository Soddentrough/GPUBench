use gpubench_core::{run_benchmarks, ResultData};

fn print_result(res: &ResultData) {
    println!("{:?} - {}: {} => {:.2} {}", res.benchmarkName, res.component, res.subcategory, (res.operations as f64) / res.time_ms / 1e6, res.metric);
}

fn main() {
    let benchmarks = vec![
        "All".to_string(),
    ];
    let device_indices = vec![0];
    let backend_strs = vec!["Vulkan".to_string()];
    
    let results = run_benchmarks(&benchmarks, &device_indices, &backend_strs, true, true, false, print_result);
    println!("Total results: {}", results.len());
}
