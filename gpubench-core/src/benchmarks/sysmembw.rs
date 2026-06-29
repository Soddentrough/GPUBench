use super::Benchmark;
use crate::context::ComputeContext;
use std::time::Instant;
use std::thread;

use std::ptr;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Read,
    Write,
    Copy,
}

struct Config {
    name: &'static str,
    mode: Mode,
    threads: usize,
}

pub struct SysMemBandwidthBench {
    buffer1: Vec<u8>,
    buffer2: Vec<u8>,
    configs: Vec<Config>,
    last_run_bytes: u64,
    last_run_time_ms: f64,
}

impl SysMemBandwidthBench {
    pub fn new() -> Self {
        Self {
            buffer1: Vec::new(),
            buffer2: Vec::new(),
            configs: vec![
                Config { name: "Read", mode: Mode::Read, threads: 0 },
                Config { name: "Write", mode: Mode::Write, threads: 0 },
                Config { name: "Copy", mode: Mode::Copy, threads: 0 },
                Config { name: "Read (1 Thread)", mode: Mode::Read, threads: 1 },
                Config { name: "Write (1 Thread)", mode: Mode::Write, threads: 1 },
                Config { name: "Copy (1 Thread)", mode: Mode::Copy, threads: 1 },
            ],
            last_run_bytes: 0,
            last_run_time_ms: 0.0,
        }
    }
}

impl Benchmark for SysMemBandwidthBench {
    fn setup(&mut self, _context: &mut dyn ComputeContext, _kernel_dir: &str) -> Result<(), String> {
        let buffer_size = 4 * 1024 * 1024 * 1024; // 4GB
        
        self.buffer1 = vec![1u8; buffer_size];
        self.buffer2 = vec![0u8; buffer_size];
        
        Ok(())
    }

    fn run(&mut self, _context: &mut dyn ComputeContext, config_idx: u32) -> Result<(), String> {
        let config = &self.configs[config_idx as usize];
        let mut threads_to_use = config.threads;
        if threads_to_use == 0 {
            threads_to_use = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        }
        
        let chunk_size = self.buffer1.len() / threads_to_use;
        let mode = config.mode;
        
        let ptr1 = self.buffer1.as_mut_ptr() as usize;
        let ptr2 = self.buffer2.as_mut_ptr() as usize;
        
        let mut handles = vec![];
        let start = Instant::now();
        
        for i in 0..threads_to_use {
            let offset = i * chunk_size;
            let current_chunk = chunk_size;
            
            handles.push(thread::spawn(move || {
                unsafe {
                    let p1 = (ptr1 + offset) as *mut u64;
                    let p2 = (ptr2 + offset) as *mut u64;
                    let count = current_chunk / 8;
                    
                    match mode {
                        Mode::Read => {
                            let mut accum: u64 = 0;
                            for j in 0..count {
                                accum ^= ptr::read_volatile(p1.add(j));
                            }
                            std::hint::black_box(accum);
                        }
                        Mode::Write => {
                            for j in 0..count {
                                ptr::write_volatile(p2.add(j), 0xAAAAAAAAAAAAAAAA);
                            }
                        }
                        Mode::Copy => {
                            ptr::copy_nonoverlapping(p1, p2, count);
                        }
                    }
                }
            }));
        }
        
        for h in handles {
            h.join().unwrap();
        }
        
        self.last_run_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        let mut total_bytes = (chunk_size * threads_to_use) as u64;
        if config.mode == Mode::Copy {
            total_bytes *= 2;
        }
        self.last_run_bytes = total_bytes;
        
        Ok(())
    }

    fn get_result(&self, _config_idx: u32) -> (u64, f64) {
        (self.last_run_bytes, self.last_run_time_ms)
    }

    fn teardown(&mut self, _context: &mut dyn ComputeContext) {
        self.buffer1.clear();
        self.buffer2.clear();
    }

    fn get_name(&self) -> &str { "SysMemBandwidth" }
    fn get_component(&self, _config_idx: u32) -> &str { "System" }
    fn get_metric(&self) -> &str { "GB/s" }
    fn get_subcategory(&self, config_idx: u32) -> &str {
        if config_idx >= 3 { "Bandwidth (Single-threaded)" } else { "Bandwidth (Multi-threaded)" }
    }
    fn get_config_name(&self, config_idx: u32) -> &str { self.configs[config_idx as usize].name }
    fn get_num_configs(&self) -> u32 { self.configs.len() as u32 }
}
