use crate::context::{ComputeContext, ComputeBuffer, AccelerationStructureHandle, BlasGeometry, GeometryInfo, TlasInstance};
use crate::benchmarks::Benchmark;
use std::time::Instant;

pub struct RayMaterialDivergenceBench {
    result_buffer: Option<ComputeBuffer>,
    kernel: Option<crate::context::ComputeKernel>,
    
    vertex_buffer: Option<ComputeBuffer>,
    
    triangle_blas: Option<AccelerationStructureHandle>,
    triangle_tlas: Option<AccelerationStructureHandle>,

    ray_count: u32,
    num_primitives: u32,
    num_instances: u32,
    time_ms: [f64; 2],
}

impl RayMaterialDivergenceBench {
    pub fn new() -> Self {
        Self {
            result_buffer: None,
            kernel: None,
            vertex_buffer: None,
            triangle_blas: None,
            triangle_tlas: None,
            ray_count: 4_000_000,
            num_primitives: 12,
            num_instances: 40_000,
            time_ms: [0.0; 2],
        }
    }
    
    fn build_as(&mut self, context: &mut dyn ComputeContext, config_idx: u32) -> Result<(), String> {
        if let Some(t) = self.triangle_tlas {
            context.release_acceleration_structure(t);
        }
        if let Some(b) = self.triangle_blas {
            context.release_acceleration_structure(b);
        }
        
        let mut rng_state: u32 = 1337;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state as f32) / (std::u32::MAX as f32)
        };

        if self.vertex_buffer.is_none() {
            let mut vertices: Vec<f32> = Vec::with_capacity((self.num_primitives * 9) as usize);
            for _ in 0..self.num_primitives {
                for _ in 0..9 {
                    vertices.push(rand() * 0.8 - 0.4);
                }
            }
            let vertices_bytes = unsafe { std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vertices.len() * 4) };
            self.vertex_buffer = Some(context.create_buffer(vertices_bytes.len(), Some(vertices_bytes))?);
        }

        let tri_geom = BlasGeometry::Triangles(GeometryInfo {
            vertex_buffer: self.vertex_buffer.unwrap(),
            vertex_count: self.num_primitives * 3,
            vertex_stride: 12,
            index_buffer: self.vertex_buffer.unwrap(),
            index_count: 0,
            is_opaque: true,
        });
        self.triangle_blas = Some(context.build_blas(&[tri_geom])?);

        let grid_width = (self.num_instances as f32).sqrt() as u32;
        let spacing = 200.0 / grid_width as f32;

        let mut rng_state2: u32 = 1337;
        let mut rand_int = || -> u32 {
            rng_state2 = rng_state2.wrapping_mul(1664525).wrapping_add(1013904223);
            rng_state2
        };

        let mut instances = Vec::with_capacity(self.num_instances as usize);
        for i in 0..self.num_instances {
            let ix = i % grid_width;
            let iy = i / grid_width;
            
            let mut transform = [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0];
            transform[3] = (ix as f32 * spacing) - 100.0;
            transform[7] = (iy as f32 * spacing) - 100.0;
            transform[11] = 0.0;
            
            let material_idx = if config_idx == 0 { 0 } else { rand_int() % 4 };

            instances.push(TlasInstance {
                blas: self.triangle_blas.unwrap(),
                transform,
                instance_id: i,
                mask: 0xFF,
                instance_offset: material_idx,
                flags: 1, // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
            });
        }
        
        self.triangle_tlas = Some(context.build_tlas(&instances)?);
        Ok(())
    }
}

impl Benchmark for RayMaterialDivergenceBench {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String> {
        self.ray_count = 4_000_000;
        self.result_buffer = Some(context.create_buffer(4, Some(&[0u8; 4]))?);
        
        let kdir = std::path::PathBuf::from(kernel_dir).join("vulkan");
        
        let rgen = kdir.join("raymatdiv.rgen.spv").to_string_lossy().into_owned();
        let rmiss = kdir.join("raymatdiv.rmiss.spv").to_string_lossy().into_owned();
        
        let rchit_paths = vec![
            kdir.join("raymatdiv_mat0.rchit.spv").to_string_lossy().into_owned(),
            kdir.join("raymatdiv_mat1.rchit.spv").to_string_lossy().into_owned(),
            kdir.join("raymatdiv_mat2.rchit.spv").to_string_lossy().into_owned(),
            kdir.join("raymatdiv_mat3.rchit.spv").to_string_lossy().into_owned(),
        ];
        let rchit: Vec<&str> = rchit_paths.iter().map(|s| s.as_str()).collect();

        self.kernel = Some(context.create_rt_pipeline(&rgen, &rmiss, &rchit, &[], &[], 2)?);

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, config_idx: u32) -> Result<(), String> {
        self.build_as(context, config_idx)?;
        
        let kernel = self.kernel.unwrap();
        let tlas = self.triangle_tlas.unwrap();
        
        context.set_kernel_arg_as(kernel, 0, tlas)?;
        context.set_kernel_arg_buffer(kernel, 1, self.result_buffer.unwrap())?;

        let mut pc = [0u8; 4];
        pc[0..4].copy_from_slice(&self.ray_count.to_ne_bytes());
        context.set_kernel_arg_push_constant(kernel, &pc)?;

        let start = Instant::now();
        context.dispatch(kernel, (self.ray_count + 31) / 32, 1, 1, 32, 1, 1)?;
        context.wait_idle()?;
        self.time_ms[config_idx as usize] = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(())
    }

    fn get_result(&self, config_idx: u32) -> (u64, f64) {
        ((self.ray_count as u64), self.time_ms[config_idx as usize])
    }

    fn teardown(&mut self, context: &mut dyn ComputeContext) {
        if let Some(k) = self.kernel { context.release_kernel(k); }
        if let Some(b) = self.result_buffer { context.release_buffer(b); }
        if let Some(b) = self.vertex_buffer { context.release_buffer(b); }
        if let Some(as_handle) = self.triangle_blas { context.release_acceleration_structure(as_handle); }
        if let Some(as_handle) = self.triangle_tlas { context.release_acceleration_structure(as_handle); }
    }

    fn get_name(&self) -> &str { "RayMaterialDivergence" }
    fn get_component(&self, _config_idx: u32) -> &str { "Ray Tracing" }
    fn get_metric(&self) -> &str { "GRays/s" }
    fn get_subcategory(&self, _config_idx: u32) -> &str { "Material Divergence" }
    
    fn get_config_name(&self, config_idx: u32) -> &str {
        if config_idx == 0 { "Coherent Material (1 Shader)" } else { "Divergent Material (4 Shaders)" }
    }
    
    fn get_num_configs(&self) -> u32 { 2 }
}
