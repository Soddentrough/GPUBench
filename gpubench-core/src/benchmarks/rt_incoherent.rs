use crate::context::{ComputeContext, ComputeBuffer, AccelerationStructureHandle, BlasGeometry, GeometryInfo, TlasInstance};
use crate::benchmarks::Benchmark;
use std::time::Instant;

pub struct RayIncoherentBench {
    result_buffer: Option<ComputeBuffer>,
    kernel: Option<crate::context::ComputeKernel>,
    
    vertex_buffer: Option<ComputeBuffer>,
    
    triangle_blas: Option<AccelerationStructureHandle>,
    triangle_tlas: Option<AccelerationStructureHandle>,

    ray_count: u32,
    num_primitives: u32,
    time_ms: [f64; 2],
}

impl RayIncoherentBench {
    pub fn new() -> Self {
        Self {
            result_buffer: None,
            kernel: None,
            vertex_buffer: None,
            triangle_blas: None,
            triangle_tlas: None,
            ray_count: 4_000_000,
            num_primitives: 200_000,
            time_ms: [0.0; 2],
        }
    }
}

impl Benchmark for RayIncoherentBench {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String> {
        self.ray_count = 4_000_000;
        self.result_buffer = Some(context.create_buffer(4, Some(&[0u8; 4]))?);
        
        self.num_primitives = 200_000;

        let mut vertices: Vec<f32> = Vec::with_capacity((self.num_primitives * 9) as usize);

        // We'll use a simple linear congruential generator for deterministic pseudo-random numbers
        let mut rng_state: u32 = 1337;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state as f32) / (std::u32::MAX as f32)
        };

        for _ in 0..self.num_primitives {
            let x = rand() * 200.0 - 100.0;
            let y = rand() * 200.0 - 100.0;
            let z = rand() * 200.0 - 100.0;
            let s = 2.0;

            vertices.push(x); vertices.push(y); vertices.push(z);
            vertices.push(x + s); vertices.push(y); vertices.push(z);
            vertices.push(x); vertices.push(y + s); vertices.push(z);
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
            is_opaque: true, 
        });
        self.triangle_blas = Some(context.build_blas(&[tri_geom])?);

        let tri_inst = TlasInstance {
            blas: self.triangle_blas.unwrap(),
            transform: [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0],
            instance_id: 0,
            mask: 0xFF,
            instance_offset: 0,
            flags: 1, // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR (1)
        };
        self.triangle_tlas = Some(context.build_tlas(&[tri_inst])?);

        let kdir = std::path::PathBuf::from(kernel_dir).join("vulkan");
        
        let rgen = kdir.join("rayincoherent.rgen.spv").to_string_lossy().into_owned();
        let rmiss = kdir.join("rayincoherent.rmiss.spv").to_string_lossy().into_owned();
        let rchit_paths = vec![kdir.join("rayincoherent.rchit.spv").to_string_lossy().into_owned()];
        let rchit: Vec<&str> = rchit_paths.iter().map(|s| s.as_str()).collect();

        self.kernel = Some(context.create_rt_pipeline(&rgen, &rmiss, &rchit, &[], &[], 2)?);

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, config_idx: u32) -> Result<(), String> {
        let tlas = self.triangle_tlas.unwrap();
        context.set_kernel_arg_as(self.kernel.unwrap(), 0, tlas)?;
        context.set_kernel_arg_buffer(self.kernel.unwrap(), 1, self.result_buffer.unwrap())?;

        let is_incoherent = config_idx;
        let seed = config_idx * 1337;

        let mut pc = [0u8; 12];
        pc[0..4].copy_from_slice(&self.ray_count.to_ne_bytes());
        pc[4..8].copy_from_slice(&is_incoherent.to_ne_bytes());
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

    fn get_name(&self) -> &str { "RayIncoherent" }
    fn get_component(&self, _config_idx: u32) -> &str { "Ray Tracing" }
    fn get_metric(&self) -> &str { "GRays/s" }
    fn get_subcategory(&self, _config_idx: u32) -> &str { "Incoherent Traversal" }
    
    fn get_config_name(&self, config_idx: u32) -> &str {
        if config_idx == 0 { "Coherent (Primary)" } else { "Incoherent (Diffuse Bounces)" }
    }
    
    fn get_num_configs(&self) -> u32 { 2 }
}
