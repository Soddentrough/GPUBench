use super::Benchmark;
use crate::context::{ComputeBuffer, ComputeContext, ComputeKernel};
use std::process::Command;
use std::time::Instant;

pub struct Fp32Bench {
    kernel: ComputeKernel,
    buffer: ComputeBuffer,
    num_elements: u32,
    elapsed_ms: f64,
}

impl Fp32Bench {
    pub fn new() -> Self {
        Self {
            kernel: 0,
            buffer: 0,
            num_elements: 0,
            elapsed_ms: 0.0,
        }
    }
}

impl Benchmark for Fp32Bench {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String> {
        self.num_elements = 1024 * 64; // 65536
        let buffer_size = 8 * 1024 * 1024; // 8MB buffer to prevent out of bounds
        self.buffer = context.create_buffer(buffer_size, None)?;

        let glsl_path = format!("{}/vulkan/fp32.comp", kernel_dir);
        let spv_path = format!("{}/vulkan/fp32.comp.spv", kernel_dir);

        // Compile SPV if not exists
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
        let multiplier: f32 = 1.0001;
        // In our pure Rust, we will implement push constants using set_kernel_arg_push_constant
        // For simplicity right now, we can pass it if we implemented it, or ignore if the shader doesn't strictly need it to run the benchmark loops.
        // Actually the cpp benchmark sets arg 1 to multiplier and arg 2 to numElements.
        // In Vulkan, args 1 and 2 are usually mapped to push constants if they are not buffers.
        
        // Push constants: [float multiplier, uint numElements] -> 8 bytes total
        let mut pc = [0u8; 8];
        let multiplier: f32 = 1.0;
        pc[0..4].copy_from_slice(&multiplier.to_le_bytes());
        pc[4..8].copy_from_slice(&self.num_elements.to_le_bytes());
        println!("PC bytes: {:?}", pc);
        context.set_kernel_arg_push_constant(self.kernel, &pc)?;

        // Run the benchmark
        let start = Instant::now();
        
        // The original cpp code dispatched 8192, 1, 1 but iterated outside.
        // Wait, the cpp code's `Run` is just 1 dispatch?
        // Ah, the CLI runner orchestrates the `Run` iteration loop.
        for iter in 0..5 {
            println!("[DIAGNOSTIC] FP32 iter {}", iter);
            let temp_buffer = context.create_buffer(8 * 1024 * 1024, None)?;
            context.set_kernel_arg_buffer(self.kernel, 0, temp_buffer)?;
            context.dispatch(self.kernel, 1024, 1, 1, 64, 1, 1)?;
            context.wait_idle()?;
        }
        
        self.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    fn get_result(&self, _config_idx: u32) -> (u64, f64) {
        // 32 vec4 FMAs per iteration = 32 * 4 * 2 = 256 FP32 operations per iteration
        // Vulkan kernel loops 16384 iters
        let iters: u64 = 16384; 
        let num_ops = iters * 256 * 8192 * 64;
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
        "FP32"
    }

    fn get_component(&self, _config_idx: u32) -> &str {
        "Compute"
    }

    fn get_metric(&self) -> &str {
        "TFLOPS"
    }

    fn get_subcategory(&self, _config_idx: u32) -> &str {
        "FP32"
    }

    fn get_config_name(&self, _config_idx: u32) -> &str {
        "FP32 (Vector)"
    }

    fn get_num_configs(&self) -> u32 {
        1
    }
}
