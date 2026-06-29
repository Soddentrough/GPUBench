use super::Benchmark;
use crate::context::ComputeContext;
use std::time::Instant;

pub struct SysMemLatencyBench {
    buffer: Vec<u32>,
    last_run_time_ms: f64,
    last_run_ops: u64,
}

impl SysMemLatencyBench {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            last_run_time_ms: 0.0,
            last_run_ops: 0,
        }
    }
}

impl Benchmark for SysMemLatencyBench {
    fn setup(&mut self, _context: &mut dyn ComputeContext, _kernel_dir: &str) -> Result<(), String> {
        let buffer_size = 512 * 1024 * 1024; // 512MB
        let num_elements = buffer_size / std::mem::size_of::<u32>();
        
        self.buffer = vec![0; num_elements];
        
        let mut indices: Vec<u32> = (0..num_elements as u32).collect();
        // Use a simple LCG to shuffle for deterministic pointer chasing
        let mut seed = 1337u32;
        for i in (1..indices.len()).rev() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let j = (seed as usize) % (i + 1);
            indices.swap(i, j);
        }
        
        for i in 0..num_elements - 1 {
            self.buffer[indices[i] as usize] = indices[i + 1];
        }
        self.buffer[indices[num_elements - 1] as usize] = indices[0];
        
        Ok(())
    }

    fn run(&mut self, _context: &mut dyn ComputeContext, _config_idx: u32) -> Result<(), String> {
        let p_buffer = &self.buffer;
        let mut index: usize = 0;
        
        // Warm up
        for _ in 0..1000 {
            index = p_buffer[index] as usize;
        }
        
        let iterations = 1_000_000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            index = p_buffer[index] as usize;
        }
        
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        
        // Prevent optimization out
        std::hint::black_box(index);
        
        self.last_run_time_ms = elapsed_ns / 1_000_000.0;
        self.last_run_ops = iterations as u64;
        
        Ok(())
    }

    fn get_result(&self, _config_idx: u32) -> (u64, f64) {
        (self.last_run_ops, self.last_run_time_ms)
    }

    fn teardown(&mut self, _context: &mut dyn ComputeContext) {
        self.buffer.clear();
    }

    fn get_name(&self) -> &str { "SysMemLatency" }
    fn get_component(&self, _config_idx: u32) -> &str { "System" }
    fn get_metric(&self) -> &str { "ns" }
    fn get_subcategory(&self, _config_idx: u32) -> &str { "Latency" }
    fn get_config_name(&self, _config_idx: u32) -> &str { "Default" }
    fn get_num_configs(&self) -> u32 { 1 }
}
