pub mod fp64;
pub mod fp32;
pub mod fp16;
pub mod bf16;
pub mod fp8;
pub mod int8;
pub mod int4;
pub mod membw;
pub mod sysmembw;
pub mod sysmemlat;
pub mod rt_intersection;
pub mod rt_divergence;
pub mod rt_anyhit;
pub mod rt_incoherent;
pub mod rt_payload;
pub mod rt_asbuild;
pub mod rt_procedural;
pub mod rt_matdiv;

use crate::context::ComputeContext;

pub trait Benchmark {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String>;
    fn run(&mut self, context: &mut dyn ComputeContext, config_idx: u32) -> Result<(), String>;
    fn get_result(&self, config_idx: u32) -> (u64, f64);
    fn teardown(&mut self, context: &mut dyn ComputeContext);

    fn get_name(&self) -> &str;
    fn get_component(&self, config_idx: u32) -> &str;
    fn get_metric(&self) -> &str;
    fn get_subcategory(&self, config_idx: u32) -> &str;
    fn get_config_name(&self, config_idx: u32) -> &str;
    fn get_num_configs(&self) -> u32;
}
