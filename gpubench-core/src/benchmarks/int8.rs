use super::Benchmark;
use crate::context::{ComputeBuffer, ComputeContext, ComputeKernel};
use std::process::Command;
use std::time::Instant;

pub struct Int8Bench {
    kernel: ComputeKernel,
    buffer: ComputeBuffer,
    num_elements: u32,
    elapsed_ms: f64,
}

impl Int8Bench {
    pub fn new() -> Self {
        Self {
            kernel: 0,
            buffer: 0,
            num_elements: 0,
            elapsed_ms: 0.0,
        }
    }
}

impl Benchmark for Int8Bench {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String> {
        self.num_elements = 1024 * 64;
        let buffer_size = 8 * 1024 * 1024; // 8MB buffer to prevent out of bounds
        self.buffer = context.create_buffer(buffer_size, None)?;

        let glsl_path = format!("{}/vulkan/int8.comp", kernel_dir);
        let spv_path = format!("{}/vulkan/int8.comp.spv", kernel_dir);

        if !std::path::Path::new(&spv_path).exists() {
            let status = Command::new("glslc")
                .arg(&glsl_path)
                .arg("-o")
                .arg(&spv_path)
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                return Err(format!("glslc failed to compile {}", glsl_path));
            }
        }

        let spv_bytes = std::fs::read(&spv_path).map_err(|e| e.to_string())?;
        self.kernel = context.create_kernel(&spv_bytes, "main", 1)?;
        context.set_kernel_arg_buffer(self.kernel, 0, self.buffer)?;

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, _config_idx: u32) -> Result<(), String> {
        let start = Instant::now();
        let iters = 20;
        for _ in 0..iters {
            context.dispatch(self.kernel, 1024, 1, 1, 64, 1, 1)?;
            context.wait_idle()?;
        }
        self.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
        Ok(())
    }

    fn get_result(&self, _config_idx: u32) -> (u64, f64) {
        // Operations per iteration depend on the shader
        // Usually these loop 65536 times with 2 ops per loop per thread
        let num_ops = 1024u64 * 64u64 * 65536u64 * 256u64;
        (num_ops, self.elapsed_ms)
    }

    fn teardown(&mut self, context: &mut dyn ComputeContext) {
        if self.kernel != 0 {
            context.release_kernel(self.kernel);
            self.kernel = 0;
        }
        if self.buffer != 0 {
            context.release_buffer(self.buffer);
            self.buffer = 0;
        }
    }

    fn get_name(&self) -> &str {
        "INT8"
    }

    fn get_component(&self, _config_idx: u32) -> &str {
        "Compute"
    }

    fn get_metric(&self) -> &str {
        if "INT8".starts_with("INT") { "TOPS" } else { "TFLOPS" }
    }

    fn get_subcategory(&self, _config_idx: u32) -> &str {
        "INT8"
    }

    fn get_config_name(&self, _config_idx: u32) -> &str {
        "INT8 (Integer 8-Bit)"
    }

    fn get_num_configs(&self) -> u32 {
        1
    }
}
