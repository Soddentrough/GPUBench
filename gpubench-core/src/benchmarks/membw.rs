use super::Benchmark;
use crate::context::{ComputeBuffer, ComputeContext, ComputeKernel};
use std::process::Command;
use std::time::Instant;

pub struct MemBwBench {
    kernel: ComputeKernel,
    buffer_in: ComputeBuffer,
    buffer_out: ComputeBuffer,
    num_elements: u32,
    elapsed_ms: f64,
}

impl MemBwBench {
    pub fn new() -> Self {
        Self {
            kernel: 0,
            buffer_in: 0,
            buffer_out: 0,
            num_elements: 0,
            elapsed_ms: 0.0,
        }
    }
}

impl Benchmark for MemBwBench {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String> {
        // Allocate 8 MB per buffer
        let buffer_size = 8 * 1024 * 1024;
        self.num_elements = (buffer_size / 16) as u32; // Number of vec4s
        
        self.buffer_in = context.create_buffer(buffer_size, None)?;

        let glsl_path = format!("{}/vulkan/membw.comp", kernel_dir);
        let spv_path = format!("{}/vulkan/membw.comp.spv", kernel_dir);

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
        context.set_kernel_arg_buffer(self.kernel, 0, self.buffer_in)?;
        
        let mut pc = [0u8; 8];
        let multiplier: f32 = 1.0;
        pc[0..4].copy_from_slice(&multiplier.to_le_bytes());
        pc[4..8].copy_from_slice(&self.num_elements.to_le_bytes());
        context.set_kernel_arg_push_constant(self.kernel, &pc)?;

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, _config_idx: u32) -> Result<(), String> {
        let start = Instant::now();
        
        let threads_needed = self.num_elements / 16;
        let workgroups = (threads_needed + 511) / 512;

        let mut actual_iters = 0;
        
        while start.elapsed().as_secs_f64() < 2.5 {
            context.dispatch(self.kernel, workgroups, 1, 1, 512, 1, 1)?;
            context.wait_idle()?;
            actual_iters += 1;
        }
        
        self.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / (actual_iters as f64);
        Ok(())
    }

    fn get_result(&self, _config_idx: u32) -> (u64, f64) {
        // Read 256MB and write 256MB = 512MB total traffic
        let bytes_transferred = (self.num_elements as u64) * 16 * 2;
        
        // Return bytes as operations. The GUI expects GB/s, so we must calculate bandwidth.
        // Wait, the core orchestrator sets 'metric' string. GUI just displays operations/elapsed_ms format.
        (bytes_transferred, self.elapsed_ms)
    }

    fn teardown(&mut self, context: &mut dyn ComputeContext) {
        if self.kernel != 0 {
            context.release_kernel(self.kernel);
            self.kernel = 0;
        }
        if self.buffer_in != 0 {
            context.release_buffer(self.buffer_in);
            self.buffer_in = 0;
        }
        if self.buffer_out != 0 {
            context.release_buffer(self.buffer_out);
            self.buffer_out = 0;
        }
    }

    fn get_name(&self) -> &str {
        "MemBandwidth"
    }

    fn get_component(&self, _config_idx: u32) -> &str {
        "Memory"
    }

    fn get_metric(&self) -> &str {
        "GB/s"
    }

    fn get_subcategory(&self, _config_idx: u32) -> &str {
        "Max Bandwidth"
    }

    fn get_config_name(&self, _config_idx: u32) -> &str {
        "MemBandwidth"
    }

    fn get_num_configs(&self) -> u32 {
        1
    }
}
