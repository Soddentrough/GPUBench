use crate::context::{ComputeContext, ComputeBuffer, AccelerationStructureHandle, BlasGeometry, GeometryInfo, TlasInstance};
use crate::benchmarks::Benchmark;
use std::time::Instant;

pub struct RayAnyHitBench {
    result_buffer: Option<ComputeBuffer>,
    kernel: Option<crate::context::ComputeKernel>,
    
    vertex_buffer: Option<ComputeBuffer>,
    
    triangle_blas: Option<AccelerationStructureHandle>,
    triangle_tlas: Option<AccelerationStructureHandle>,

    ray_count: u32,
    num_primitives: u32,
    time_ms: [f64; 5],
}

impl RayAnyHitBench {
    pub fn new() -> Self {
        Self {
            result_buffer: None,
            kernel: None,
            vertex_buffer: None,
            triangle_blas: None,
            triangle_tlas: None,
            ray_count: 4_000_000,
            num_primitives: 0,
            time_ms: [0.0; 5],
        }
    }
}

impl Benchmark for RayAnyHitBench {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String> {
        self.ray_count = 4_000_000;
        self.result_buffer = Some(context.create_buffer(4, Some(&[0u8; 4]))?);
        
        let grid_size = 256;
        let primitives_per_plane = grid_size * grid_size * 2;
        let num_planes = 16;
        self.num_primitives = primitives_per_plane * num_planes;

        let mut vertices: Vec<f32> = Vec::new();

        let mut add_plane = |z: f32| {
            let scale = 200.0 / grid_size as f32;
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let fx0 = x as f32 * scale - 100.0;
                    let fy0 = y as f32 * scale - 100.0;
                    let fx1 = (x + 1) as f32 * scale - 100.0;
                    let fy1 = (y + 1) as f32 * scale - 100.0;

                    vertices.push(fx0); vertices.push(fy0); vertices.push(z);
                    vertices.push(fx1); vertices.push(fy0); vertices.push(z);
                    vertices.push(fx0); vertices.push(fy1); vertices.push(z);

                    vertices.push(fx1); vertices.push(fy0); vertices.push(z);
                    vertices.push(fx1); vertices.push(fy1); vertices.push(z);
                    vertices.push(fx0); vertices.push(fy1); vertices.push(z);
                }
            }
        };

        for i in 0..num_planes {
            add_plane((i as f32) * 2.0);
        }
        
        let vertices_bytes = unsafe { std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vertices.len() * 4) };
        let v_buf = context.create_buffer(vertices_bytes.len(), Some(vertices_bytes))?;
        self.vertex_buffer = Some(v_buf);

        let tri_geom = BlasGeometry::Triangles(GeometryInfo {
            vertex_buffer: v_buf,
            vertex_count: self.num_primitives * 3,
            vertex_stride: 12,
            index_buffer: v_buf, // Unused
            index_count: 0,
            is_opaque: false, // Force any-hit
        });
        self.triangle_blas = Some(context.build_blas(&[tri_geom])?);

        let tri_inst = TlasInstance {
            blas: self.triangle_blas.unwrap(),
            transform: [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0],
            instance_id: 0,
            mask: 0xFF,
            instance_offset: 0,
            flags: 9, // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR (1) | VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR (8)
        };
        self.triangle_tlas = Some(context.build_tlas(&[tri_inst])?);

        let kdir = std::path::PathBuf::from(kernel_dir).join("vulkan");
        
        let rgen = kdir.join("rayanyhit_pipeline.rgen.spv").to_string_lossy().into_owned();
        let rmiss = kdir.join("rayanyhit_pipeline.rmiss.spv").to_string_lossy().into_owned();
        
        let rchit_paths = vec![kdir.join("rayanyhit_pipeline.rchit.spv").to_string_lossy().into_owned()];
        let rchit: Vec<&str> = rchit_paths.iter().map(|s| s.as_str()).collect();
        
        let rahit_paths = vec![kdir.join("rayanyhit_pipeline.rahit.spv").to_string_lossy().into_owned()];
        let rahit: Vec<&str> = rahit_paths.iter().map(|s| s.as_str()).collect();

        self.kernel = Some(context.create_rt_pipeline(&rgen, &rmiss, &rchit, &rahit, &[], 2)?);

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, config_idx: u32) -> Result<(), String> {
        let tlas = self.triangle_tlas.unwrap();
        context.set_kernel_arg_as(self.kernel.unwrap(), 0, tlas)?;
        context.set_kernel_arg_buffer(self.kernel.unwrap(), 1, self.result_buffer.unwrap())?;

        let alpha_density = 1.0 - (config_idx as f32 * 0.25);
        let seed = config_idx * 1337;

        let mut pc = [0u8; 12];
        pc[0..4].copy_from_slice(&self.ray_count.to_ne_bytes());
        pc[4..8].copy_from_slice(&alpha_density.to_ne_bytes());
        pc[8..12].copy_from_slice(&seed.to_ne_bytes());
        
        context.set_kernel_arg_push_constant(self.kernel.unwrap(), &pc)?;

        let start = Instant::now();
        context.dispatch(self.kernel.unwrap(), (self.ray_count + 31) / 32, 1, 1, 32, 1, 1)?;
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

    fn get_name(&self) -> &str { "RayAnyHit" }
    fn get_component(&self, _config_idx: u32) -> &str { "Ray Tracing" }
    fn get_metric(&self) -> &str { "GRays/s" }
    fn get_subcategory(&self, _config_idx: u32) -> &str { "Alpha-Tested Geometry" }
    
    fn get_config_name(&self, config_idx: u32) -> &str {
        let solid_percentage = 100 - (config_idx * 25);
        match solid_percentage {
            100 => "100% Solid (Min AnyHit Ignore)",
            75 => "75% Solid",
            50 => "50% Solid",
            25 => "25% Solid",
            0 => "0% Solid (Max AnyHit Ignore)",
            _ => "Unknown",
        }
    }
    
    fn get_num_configs(&self) -> u32 { 5 }
}
