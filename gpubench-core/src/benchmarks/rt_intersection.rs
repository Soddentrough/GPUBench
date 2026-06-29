use crate::context::{ComputeContext, ComputeBuffer, AccelerationStructureHandle, BlasGeometry, GeometryInfo, AabbInfo, TlasInstance};
use crate::benchmarks::Benchmark;
use std::time::Instant;

pub struct RayTracingBench {
    result_buffer: Option<ComputeBuffer>,
    kernel: Option<crate::context::ComputeKernel>,
    
    vertex_buffer: Option<ComputeBuffer>,
    aabb_buffer: Option<ComputeBuffer>,
    
    triangle_blas: Option<AccelerationStructureHandle>,
    box_blas: Option<AccelerationStructureHandle>,
    triangle_tlas: Option<AccelerationStructureHandle>,
    box_tlas: Option<AccelerationStructureHandle>,

    ray_count: u32,
    num_primitives: u32,
    time_ms: [f64; 2],
}

impl RayTracingBench {
    pub fn new() -> Self {
        Self {
            result_buffer: None,
            kernel: None,
            vertex_buffer: None,
            aabb_buffer: None,
            triangle_blas: None,
            box_blas: None,
            triangle_tlas: None,
            box_tlas: None,
            ray_count: 1_048_576,
            num_primitives: 0,
            time_ms: [0.0, 0.0],
        }
    }
}

impl Benchmark for RayTracingBench {
    fn setup(&mut self, context: &mut dyn ComputeContext, kernel_dir: &str) -> Result<(), String> {
        self.ray_count = 1_048_576;
        self.result_buffer = Some(context.create_buffer(4, Some(&[0u8; 4]))?);
        
        let grid_size = 16;
        let layers = 64;
        self.num_primitives = grid_size * grid_size * layers;

        let mut vertices: Vec<f32> = Vec::new();
        for z in 0..layers {
            let jitter_x = (z % 8) as f32 * 0.05;
            let jitter_y = (z / 8) as f32 * 0.05;
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let fx = x as f32 - 8.0 + jitter_x;
                    let fy = y as f32 - 8.0 + jitter_y;
                    let fz = z as f32 * 0.1;
                    vertices.push(fx + 0.1); vertices.push(fy + 0.1); vertices.push(fz);
                    vertices.push(fx + 0.4); vertices.push(fy + 0.1); vertices.push(fz);
                    vertices.push(fx + 0.1); vertices.push(fy + 0.4); vertices.push(fz);
                }
            }
        }
        
        let vertices_bytes = unsafe { std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vertices.len() * 4) };
        let v_buf = context.create_buffer(vertices_bytes.len(), Some(vertices_bytes))?;
        self.vertex_buffer = Some(v_buf);

        let mut aabbs: Vec<f32> = Vec::new();
        for z in 0..layers {
            let jitter_x = (z % 8) as f32 * 0.05;
            let jitter_y = (z / 8) as f32 * 0.05;
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let fx = x as f32 - 8.0 + jitter_x;
                    let fy = y as f32 - 8.0 + jitter_y;
                    let fz = z as f32 * 0.1;
                    aabbs.push(fx + 0.1); aabbs.push(fy + 0.1); aabbs.push(fz - 0.01);
                    aabbs.push(fx + 0.4); aabbs.push(fy + 0.4); aabbs.push(fz + 0.01);
                }
            }
        }
        
        let aabbs_bytes = unsafe { std::slice::from_raw_parts(aabbs.as_ptr() as *const u8, aabbs.len() * 4) };
        let a_buf = context.create_buffer(aabbs_bytes.len(), Some(aabbs_bytes))?;
        self.aabb_buffer = Some(a_buf);

        let tri_geom = BlasGeometry::Triangles(GeometryInfo {
            vertex_buffer: v_buf,
            vertex_count: self.num_primitives * 3,
            vertex_stride: 12,
            index_buffer: v_buf, // Unused
            index_count: 0,
            is_opaque: true,
        });
        self.triangle_blas = Some(context.build_blas(&[tri_geom])?);

        let box_geom = BlasGeometry::Aabbs(AabbInfo {
            aabb_buffer: a_buf,
            aabb_count: self.num_primitives,
            aabb_stride: 24,
            is_opaque: false,
        });
        self.box_blas = Some(context.build_blas(&[box_geom])?);

        let tri_inst = TlasInstance {
            blas: self.triangle_blas.unwrap(),
            transform: [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0],
            instance_id: 0,
            mask: 0xFF,
            instance_offset: 0,
            flags: 1, // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR = 0x00000001
        };
        self.triangle_tlas = Some(context.build_tlas(&[tri_inst])?);

        let box_inst = TlasInstance {
            blas: self.box_blas.unwrap(),
            transform: [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0],
            instance_id: 0,
            mask: 0xFF,
            instance_offset: 0,
            flags: 1,
        };
        self.box_tlas = Some(context.build_tlas(&[box_inst])?);

        let spv_path = std::path::PathBuf::from(kernel_dir).join("vulkan").join("rt_benchmark.comp.spv");
        let spv_bytes = std::fs::read(&spv_path).map_err(|e| format!("Failed to read {}: {}", spv_path.display(), e))?;
        self.kernel = Some(context.create_kernel(&spv_bytes, "main", 2)?);

        Ok(())
    }

    fn run(&mut self, context: &mut dyn ComputeContext, config_idx: u32) -> Result<(), String> {
        let tlas = if config_idx == 0 { self.triangle_tlas.unwrap() } else { self.box_tlas.unwrap() };
        context.set_kernel_arg_as(self.kernel.unwrap(), 0, tlas)?;
        context.set_kernel_arg_buffer(self.kernel.unwrap(), 1, self.result_buffer.unwrap())?;

        let mut pc = [0u8; 8];
        pc[0..4].copy_from_slice(&self.ray_count.to_ne_bytes());
        pc[4..8].copy_from_slice(&config_idx.to_ne_bytes());
        context.set_kernel_arg_push_constant(self.kernel.unwrap(), &pc)?;

        let start = Instant::now();
        context.dispatch(self.kernel.unwrap(), (self.ray_count + 31) / 32, 1, 1, 32, 1, 1)?;
        context.wait_idle()?;
        self.time_ms[config_idx as usize] = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(())
    }

    fn get_result(&self, config_idx: u32) -> (u64, f64) {
        ((self.ray_count as u64 * 1024), self.time_ms[config_idx as usize])
    }

    fn teardown(&mut self, context: &mut dyn ComputeContext) {
        if let Some(k) = self.kernel { context.release_kernel(k); }
        if let Some(b) = self.result_buffer { context.release_buffer(b); }
        if let Some(b) = self.vertex_buffer { context.release_buffer(b); }
        if let Some(b) = self.aabb_buffer { context.release_buffer(b); }
        if let Some(as_handle) = self.triangle_blas { context.release_acceleration_structure(as_handle); }
        if let Some(as_handle) = self.box_blas { context.release_acceleration_structure(as_handle); }
        if let Some(as_handle) = self.triangle_tlas { context.release_acceleration_structure(as_handle); }
        if let Some(as_handle) = self.box_tlas { context.release_acceleration_structure(as_handle); }
    }

    fn get_name(&self) -> &str { "RayTracing" }
    fn get_component(&self, _config_idx: u32) -> &str { "Ray Tracing" }
    fn get_metric(&self) -> &str { "GIS/s" }
    fn get_subcategory(&self, _config_idx: u32) -> &str { "Intersection tests" }
    
    fn get_config_name(&self, config_idx: u32) -> &str {
        if config_idx == 0 { "Triangles" } else { "AABBs" }
    }
    
    fn get_num_configs(&self) -> u32 { 2 }
}
