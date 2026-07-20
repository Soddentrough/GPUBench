use super::Benchmark;
use crate::context::{ComputeBuffer, ComputeContext, ComputeKernel};
use std::process::Command;
use std::time::Instant;

// Single source of truth for the grid size: used by BOTH the dispatch and the
// op-count math so they can never drift apart. Matches the C++ Fp32Bench.
const NUM_WORKGROUPS: u32 = 8192;
const WORKGROUP_SIZE: u32 = 64;
// Number of timed dispatch iterations inside run().
const NUM_ITERATIONS: u64 = 5;
// The Vulkan shader loops 16384 FMA iterations; each iteration is 32 vec4 FMAs
// = 32 * 4 * 2 = 256 FP32 ops.
const SHADER_LOOP_ITERS: u64 = 16384;
const OPS_PER_ITER: u64 = 256;

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
        self.num_elements = NUM_WORKGROUPS * WORKGROUP_SIZE; // 524288
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

        // All kernel argument setup happens here, before the timed region,
        // matching the C++ Setup/Run split.
        context.set_kernel_arg_buffer(self.kernel, 0, self.buffer)?;

        // Push constants: [float multiplier, uint numElements] -> 8 bytes total.
        // multiplier = 1.0001 (not 1.0) so the FMA chain cannot be constant-folded
        // by the shader compiler, matching the C++ path.
        let multiplier: f32 = 1.0001;
        let mut pc = [0u8; 8];
        pc[0..4].copy_from_slice(&multiplier.to_le_bytes());
        pc[4..8].copy_from_slice(&self.num_elements.to_le_bytes());
        context.set_kernel_arg_push_constant(self.kernel, &pc)?;

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, _config_idx: u32) -> Result<(), String> {
        // Timed region: dispatches only. Buffer creation and kernel argument
        // setup were already done in setup().
        let start = Instant::now();

        for _ in 0..NUM_ITERATIONS {
            context.dispatch(self.kernel, NUM_WORKGROUPS, 1, 1, WORKGROUP_SIZE, 1, 1)?;
            context.wait_idle()?;
        }

        self.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    fn get_result(&self, _config_idx: u32) -> (u64, f64) {
        // NUM_ITERATIONS dispatches, each running SHADER_LOOP_ITERS iterations of
        // OPS_PER_ITER FP32 ops on NUM_WORKGROUPS * WORKGROUP_SIZE threads.
        let num_ops = NUM_ITERATIONS
            * SHADER_LOOP_ITERS
            * OPS_PER_ITER
            * NUM_WORKGROUPS as u64
            * WORKGROUP_SIZE as u64;
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
