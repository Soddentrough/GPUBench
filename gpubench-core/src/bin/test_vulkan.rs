fn main() {
    println!("Testing run_benchmarks...");
    let benchmarks = vec![
        "RayAnyHit".to_string(),
        "RayASBuild".to_string(),
        "RayIncoherent".to_string(),
        "RayTracing".to_string(),
        "RayDivergence".to_string(),
        "RayPayload".to_string(),
        "RayProcedural".to_string(),
    ];
    let device_indices = vec![0];
    let backend_strs = vec!["VULKAN".to_string()];

    let results = gpubench_core::run_benchmarks(
        &benchmarks,
        &device_indices,
        &backend_strs,
        true,
        true,
        false,
        |_res| {
            println!("Callback received result");
        }
    );
    println!("Done! Results: {}", results.len());
}
