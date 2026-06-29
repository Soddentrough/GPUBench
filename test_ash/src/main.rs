use ash::vk;
use ash::khr::ray_tracing_pipeline::Device as RayTracingPipeline;

fn test(rt_pipeline: &RayTracingPipeline, command_buffer: vk::CommandBuffer, rgen: &vk::StridedDeviceAddressRegionKHR, rmiss: &vk::StridedDeviceAddressRegionKHR, rhit: &vk::StridedDeviceAddressRegionKHR, rcall: &vk::StridedDeviceAddressRegionKHR) {
    unsafe {
        rt_pipeline.cmd_trace_rays(command_buffer, rgen, rmiss, rhit, rcall, 1, 1, 1);
    }
}

fn main() {}
