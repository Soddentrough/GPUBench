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
        isUnsupported: bool,
        supportNote: String,
        supportCategory: String,
        maxWorkGroupSize: u32,
        deviceIndex: u32,
        configIndex: u32,
        sortWeight: i32,
        width: u32,
        height: u32,
    }

    struct FfiDeviceProfile {
        backend: String,
        deviceIndex: u32,
        deviceName: String,
        vendorId: u32,
        deviceId: u32,
        driverName: String,
        driverInfo: String,
        driverVersion: String,
        apiVersion: String,
        vramTotalMb: u64,
        subgroupSize: u32,
        maxWorkGroupSize: u32,
        rayTracingSupported: bool,
        serSupported: bool,
        workGraphsSupported: bool,
        cooperativeMatrixSupported: bool,
        float16Supported: bool,
        int8Supported: bool,
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
            dump_renders: bool,
            render_width: u32,
            render_height: u32,
            callback: fn(&FfiResultData),
            scene: String,
            samples_per_pixel: u32,
        ) -> Vec<FfiResultData>;

        fn gpubench_get_available_hardware() -> Vec<String>;
        fn gpubench_get_available_benchmarks() -> Vec<String>;
        fn gpubench_get_device_profiles() -> Vec<FfiDeviceProfile>;
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

