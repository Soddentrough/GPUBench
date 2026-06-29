use crate::context::{ComputeContext, ComputeBuffer, AccelerationStructureHandle, BlasGeometry, AabbInfo, TlasInstance};
use crate::benchmarks::Benchmark;
use std::time::Instant;

pub struct RayProceduralBench {
    result_buffer: Option<ComputeBuffer>,
    kernel: Option<crate::context::ComputeKernel>,
    
    aabb_buffer: Option<ComputeBuffer>,
    sphere_buffer: Option<ComputeBuffer>,
    
    aabb_blas: Option<AccelerationStructureHandle>,
    aabb_tlas: Option<AccelerationStructureHandle>,

    ray_count: u32,
    num_primitives: u32,
    time_ms: f64,
}

impl RayProceduralBench {
    pub fn new() -> Self {
        Self {
            result_buffer: None,
            kernel: None,
            aabb_buffer: None,
            sphere_buffer: None,
            aabb_blas: None,
            aabb_tlas: None,
            ray_count: 4_000_000,
            num_primitives: 200_000,
            time_ms: 0.0,
        }
    }
}

impl Benchmark for RayProceduralBench {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String> {
        self.ray_count = 4_000_000;
        self.result_buffer = Some(context.create_buffer(4, Some(&[0u8; 4]))?);
        
        self.num_primitives = 200_000;

        let mut aabbs: Vec<f32> = Vec::with_capacity((self.num_primitives * 6) as usize);
        let mut spheres: Vec<f32> = Vec::with_capacity((self.num_primitives * 4) as usize);

        let mut rng_state: u32 = 1337;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state as f32) / (std::u32::MAX as f32)
        };

        for _ in 0..self.num_primitives {
            let x = rand() * 200.0 - 100.0;
            let y = rand() * 200.0 - 100.0;
            let z = rand() * 200.0 - 100.0;
            let r = 2.0;

            aabbs.push(x - r); aabbs.push(y - r); aabbs.push(z - r);
            aabbs.push(x + r); aabbs.push(y + r); aabbs.push(z + r);
            
            spheres.push(x); spheres.push(y); spheres.push(z); spheres.push(r);
        }

        let aabbs_bytes = unsafe { std::slice::from_raw_parts(aabbs.as_ptr() as *const u8, aabbs.len() * 4) };
        let a_buf = context.create_buffer(aabbs_bytes.len(), Some(aabbs_bytes))?;
        self.aabb_buffer = Some(a_buf);
        
        let spheres_bytes = unsafe { std::slice::from_raw_parts(spheres.as_ptr() as *const u8, spheres.len() * 4) };
        let s_buf = context.create_buffer(spheres_bytes.len(), Some(spheres_bytes))?;
        self.sphere_buffer = Some(s_buf);

        let aabb_geom = BlasGeometry::Aabbs(AabbInfo {
            aabb_buffer: a_buf,
            aabb_count: self.num_primitives,
            aabb_stride: 24, // 6 floats * 4 bytes
            is_opaque: true,
        });
        self.aabb_blas = Some(context.build_blas(&[aabb_geom])?);

        let inst = TlasInstance {
            blas: self.aabb_blas.unwrap(),
            transform: [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0],
            instance_id: 0,
            mask: 0xFF,
            instance_offset: 0,
            flags: 1, // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
        };
        self.aabb_tlas = Some(context.build_tlas(&[inst])?);

        let kdir = std::path::PathBuf::from(kernel_dir).join("vulkan");
        
        let rgen = kdir.join("rayprocedural.rgen").to_string_lossy().into_owned();
        let rmiss = kdir.join("rayprocedural.rmiss").to_string_lossy().into_owned();
        let rchit_paths = vec![kdir.join("rayprocedural.rchit").to_string_lossy().into_owned()];
        let rchit: Vec<&str> = rchit_paths.iter().map(|s| s.as_str()).collect();
        let rint_paths = vec![kdir.join("rayprocedural.rint").to_string_lossy().into_owned()];
        let rint: Vec<&str> = rint_paths.iter().map(|s| s.as_str()).collect();

        self.kernel = Some(context.create_rt_pipeline(&rgen, &rmiss, &rchit, &[], &rint, 3)?);

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, _config_idx: u32) -> Result<(), String> {
        let kernel = self.kernel.unwrap();
        let tlas = self.aabb_tlas.unwrap();
        
        context.set_kernel_arg_as(kernel, 0, tlas)?;
        context.set_kernel_arg_buffer(kernel, 1, self.result_buffer.unwrap())?;
        context.set_kernel_arg_buffer(kernel, 2, self.sphere_buffer.unwrap())?;

        let mut pc = [0u8; 128];
        pc[0..4].copy_from_slice(&self.ray_count.to_ne_bytes());
        context.set_kernel_arg_push_constant(kernel, &pc)?;

        let start = Instant::now();
        context.dispatch(kernel, (self.ray_count + 31) / 32, 1, 1, 32, 1, 1)?;
        context.wait_idle()?;
        self.time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(())
    }

    fn get_result(&self, _config_idx: u32) -> (u64, f64) {
        ((self.ray_count as u64 * 100), self.time_ms)
    }

    fn teardown(&mut self, context: &mut dyn ComputeContext) {
        if let Some(k) = self.kernel { context.release_kernel(k); }
        if let Some(b) = self.result_buffer { context.release_buffer(b); }
        if let Some(b) = self.aabb_buffer { context.release_buffer(b); }
        if let Some(b) = self.sphere_buffer { context.release_buffer(b); }
        if let Some(as_handle) = self.aabb_blas { context.release_acceleration_structure(as_handle); }
        if let Some(as_handle) = self.aabb_tlas { context.release_acceleration_structure(as_handle); }
    }

    fn get_name(&self) -> &str { "RayProcedural" }
    fn get_component(&self, _config_idx: u32) -> &str { "Ray Tracing" }
    fn get_metric(&self) -> &str { "GRays/s" }
    fn get_subcategory(&self, _config_idx: u32) -> &str { "Procedural Intersection" }
    
    fn get_config_name(&self, _config_idx: u32) -> &str {
        "Spheres (AABB)"
    }
    
    fn get_num_configs(&self) -> u32 { 1 }
}
