use crate::context::{ComputeContext, ComputeBuffer, BlasGeometry, GeometryInfo, TlasInstance};
use crate::benchmarks::Benchmark;
use std::time::Instant;

pub struct RayASBuildBench {
    vertex_buffer: Option<ComputeBuffer>,
    num_primitives: u32,
    num_instances: u32,
    time_ms: [f64; 3],
}

impl RayASBuildBench {
    pub fn new() -> Self {
        Self {
            vertex_buffer: None,
            num_primitives: 1_000_000,
            num_instances: 10_000,
            time_ms: [0.0; 3],
        }
    }
}

impl Benchmark for RayASBuildBench {
    fn setup(&mut self, context: &mut dyn ComputeContext, _kernel_dir: &str) -> Result<(), String> {
        self.num_primitives = 1_000_000;
        self.num_instances = 10_000;

        let mut vertices: Vec<f32> = Vec::with_capacity((self.num_primitives * 9) as usize);

        let mut rng_state: u32 = 1337;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state as f32) / (std::u32::MAX as f32)
        };

        for _ in 0..(self.num_primitives * 3) {
            vertices.push(rand());
            vertices.push(rand());
            vertices.push(rand());
        }

        let vertices_bytes = unsafe { std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vertices.len() * 4) };
        let v_buf = context.create_buffer(vertices_bytes.len(), Some(vertices_bytes))?;
        self.vertex_buffer = Some(v_buf);

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, config_idx: u32) -> Result<(), String> {
        let iters = 10;
        
        let mut total_time = 0.0;

        for _ in 0..iters {
            if config_idx == 0 || config_idx == 2 {
                let tri_geom = BlasGeometry::Triangles(GeometryInfo {
                    vertex_buffer: self.vertex_buffer.unwrap(),
                    vertex_count: self.num_primitives * 3,
                    vertex_stride: 12,
                    index_buffer: self.vertex_buffer.unwrap(), // Unused
                    index_count: 0,
                    is_opaque: true,
                });
                let start = Instant::now();
                let blas = context.build_blas(&[tri_geom])?;
                context.wait_idle()?;
                total_time += start.elapsed().as_secs_f64() * 1000.0;
                context.release_acceleration_structure(blas);
            } else if config_idx == 1 {
                // To build TLAS, we need a BLAS first
                let tri_geom = BlasGeometry::Triangles(GeometryInfo {
                    vertex_buffer: self.vertex_buffer.unwrap(),
                    vertex_count: 3, // minimal for tlas test
                    vertex_stride: 12,
                    index_buffer: self.vertex_buffer.unwrap(),
                    index_count: 0,
                    is_opaque: true,
                });
                let blas = context.build_blas(&[tri_geom])?;
                
                let mut instances = Vec::with_capacity(self.num_instances as usize);
                for i in 0..self.num_instances {
                    instances.push(TlasInstance {
                        blas,
                        transform: [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0],
                        instance_id: i,
                        mask: 0xFF,
                        instance_offset: 0,
                        flags: 1, // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
                    });
                }
                
                let start = Instant::now();
                let tlas = context.build_tlas(&instances)?;
                context.wait_idle()?;
                total_time += start.elapsed().as_secs_f64() * 1000.0;
                
                context.release_acceleration_structure(tlas);
                context.release_acceleration_structure(blas);
            }
        }

        self.time_ms[config_idx as usize] = total_time / (iters as f64);
        
        Ok(())
    }

    fn get_result(&self, config_idx: u32) -> (u64, f64) {
        (1, self.time_ms[config_idx as usize])
    }

    fn teardown(&mut self, context: &mut dyn ComputeContext) {
        if let Some(b) = self.vertex_buffer { context.release_buffer(b); }
    }

    fn get_name(&self) -> &str { "RayASBuild" }
    fn get_component(&self, _config_idx: u32) -> &str { "Ray Tracing" }
    fn get_metric(&self) -> &str { "ms/op" }
    fn get_subcategory(&self, _config_idx: u32) -> &str { "AS Build Performance" }
    
    fn get_config_name(&self, config_idx: u32) -> &str {
        match config_idx {
            0 => "BLAS Build (1M Tris)",
            1 => "TLAS Build (10K Inst)",
            _ => "BLAS Update (1M Tris) [Emulated]",
        }
    }
    
    fn get_num_configs(&self) -> u32 { 3 }
}
