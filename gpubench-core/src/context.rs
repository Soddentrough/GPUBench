#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub index: u32,
    pub vendor_id: u32,
    pub max_compute_units: u32,
    pub max_work_group_size: u32,
    pub ray_tracing_support: bool,
    pub fp64_support: bool,
    pub fp16_support: bool,
    pub int8_support: bool,
}

pub type ComputeBuffer = usize;
pub type ComputeKernel = usize;
pub type AccelerationStructureHandle = usize;

#[derive(Clone)]
pub struct GeometryInfo {
    pub vertex_buffer: ComputeBuffer,
    pub vertex_count: u32,
    pub vertex_stride: u32,
    pub index_buffer: ComputeBuffer,
    pub index_count: u32,
    pub is_opaque: bool,
}

#[derive(Clone)]
pub struct AabbInfo {
    pub aabb_buffer: ComputeBuffer,
    pub aabb_count: u32,
    pub aabb_stride: u32,
    pub is_opaque: bool,
}

pub enum BlasGeometry {
    Triangles(GeometryInfo),
    Aabbs(AabbInfo),
}

#[derive(Clone)]
pub struct TlasInstance {
    pub blas: AccelerationStructureHandle,
    pub transform: [f32; 12],
    pub instance_id: u32,
    pub mask: u8,
    pub instance_offset: u32,
    pub flags: u32,
}

pub trait ComputeContext {
    fn get_devices(&self) -> Vec<DeviceInfo>;
    fn pick_device(&mut self, index: u32) -> Result<(), String>;
    fn get_current_device_info(&self) -> Option<DeviceInfo>;

    fn create_buffer(&mut self, size: usize, host_ptr: Option<&[u8]>) -> Result<ComputeBuffer, String>;
    fn write_buffer(&mut self, buffer: ComputeBuffer, offset: usize, data: &[u8]) -> Result<(), String>;
    fn read_buffer(&self, buffer: ComputeBuffer, offset: usize, data: &mut [u8]) -> Result<(), String>;
    fn release_buffer(&mut self, buffer: ComputeBuffer);

    fn create_kernel(&mut self, spv_bytes: &[u8], entry_point: &str, num_buffer_args: u32) -> Result<ComputeKernel, String>;
    fn create_rt_pipeline(&mut self, rgen_path: &str, rmiss_path: &str, rchit_paths: &[&str], rahit_paths: &[&str], rint_paths: &[&str], num_buffer_args: u32) -> Result<ComputeKernel, String>;
    
    fn set_kernel_arg_buffer(&mut self, kernel: ComputeKernel, arg_index: u32, buffer: ComputeBuffer) -> Result<(), String>;
    fn set_kernel_arg_as(&mut self, kernel: ComputeKernel, arg_index: u32, as_handle: AccelerationStructureHandle) -> Result<(), String>;
    fn set_kernel_arg_push_constant(&mut self, kernel: ComputeKernel, data: &[u8]) -> Result<(), String>;
    fn dispatch(&mut self, kernel: ComputeKernel, grid_x: u32, grid_y: u32, grid_z: u32, block_x: u32, block_y: u32, block_z: u32) -> Result<(), String>;
    fn release_kernel(&mut self, kernel: ComputeKernel);

    fn build_blas(&mut self, geometry: &[BlasGeometry]) -> Result<AccelerationStructureHandle, String>;
    fn build_tlas(&mut self, instances: &[TlasInstance]) -> Result<AccelerationStructureHandle, String>;
    fn release_acceleration_structure(&mut self, as_handle: AccelerationStructureHandle);


    fn wait_idle(&mut self) -> Result<(), String>;
}
