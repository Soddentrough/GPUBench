#[cxx::bridge]
pub mod ffi {
    struct FfiResultData {
        backendName: String,
        deviceName: String,
        benchmarkName: String,
        component: String,
        subcategory: String,
        metric: String,
        operations: u64,
        time_ms: f64,
        isEmulated: bool,
        maxWorkGroupSize: u32,
        deviceIndex: u32,
        configIndex: u32,
        sortWeight: i32,
    }

    unsafe extern "C++" {
        include!("gpubench-sys/src/bridge.h");

        fn gpubench_init();
        fn gpubench_run_benchmarks(
            benchmarks_to_run: &Vec<String>,
            device_indices: &Vec<u32>,
            backend_strs: &Vec<String>,
            verbose: bool,
            debug: bool,
            dump_geometry: bool,
            callback: fn(&FfiResultData)
        ) -> Vec<FfiResultData>;

        fn gpubench_get_available_hardware() -> Vec<String>;
        fn gpubench_get_available_benchmarks() -> Vec<String>;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_init() {
        // This will print to standard output if FFI is correctly linked
        ffi::gpubench_init();
    }
}

