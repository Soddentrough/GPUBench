use std::sync::LazyLock;
use ash::vk;
use ash::{Device, Entry, Instance};
use std::collections::HashMap;
use std::ffi::CStr;
use crate::context::{ComputeBuffer, ComputeContext, ComputeKernel, DeviceInfo};
use ash::khr::acceleration_structure::Device as AccelerationStructure;
use ash::khr::ray_tracing_pipeline::Device as RayTracingPipeline;
use std::ffi::CString;

pub static VULKAN_CORE: LazyLock<Result<(Entry, Instance), String>> = LazyLock::new(|| {
    let entry = unsafe { Entry::load().map_err(|e| format!("Failed to load Vulkan: {}", e))? };
    let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&create_info, None).map_err(|e| e.to_string())? };
    Ok((entry, instance))
});

struct VulkanBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    _size: usize,
}

struct VulkanKernel {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    desc_set_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    _num_buffers: u32,
    bound_buffers: HashMap<u32, ComputeBuffer>,
    push_constants: Vec<u8>,
    
    is_rt: bool,
    rgen_region: vk::StridedDeviceAddressRegionKHR,
    rmiss_region: vk::StridedDeviceAddressRegionKHR,
    rhit_region: vk::StridedDeviceAddressRegionKHR,
    rcall_region: vk::StridedDeviceAddressRegionKHR,
    sbt_buffer: Option<ComputeBuffer>,
}

pub struct VulkanAS {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: ComputeBuffer,
    pub instance_buffer: Option<ComputeBuffer>,
}

pub struct VulkanContext {
    physical_devices: Vec<vk::PhysicalDevice>,
    
    device_infos: Vec<DeviceInfo>,
    selected_device_idx: u32,

    device: Option<Device>,
    queue_family_index: u32,
    queue: vk::Queue,
    
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    is_cmd_recorded: bool,
    fence: vk::Fence,

    buffers: HashMap<ComputeBuffer, VulkanBuffer>,
    next_buffer_id: ComputeBuffer,

    kernels: HashMap<ComputeKernel, VulkanKernel>,
    next_kernel_id: ComputeKernel,
    
    // RT Loaders
    pub rt_pipeline: Option<RayTracingPipeline>,
    pub rt_as: Option<AccelerationStructure>,
    
    pub acceleration_structures: HashMap<crate::context::AccelerationStructureHandle, VulkanAS>,
    pub next_as_id: crate::context::AccelerationStructureHandle,
}

impl VulkanContext {
    pub fn new() -> Result<Self, String> {
        let core = VULKAN_CORE.as_ref().map_err(|e| e.clone())?;
        let instance = &core.1;

        let physical_devices = unsafe { instance.enumerate_physical_devices().unwrap_or_default() };
        let mut filtered_devices = Vec::new();
        let mut device_infos = Vec::new();
        for (idx, pdevice) in physical_devices.iter().enumerate() {
            let props = unsafe { instance.get_physical_device_properties(*pdevice) };
            if props.device_type == vk::PhysicalDeviceType::CPU {
                continue;
            }
            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned() };
            if name.to_lowercase().contains("llvmpipe") {
                continue;
            }
            filtered_devices.push(*pdevice);
            let ext_props = unsafe { instance.enumerate_device_extension_properties(*pdevice).unwrap_or_default() };
            let mut rt_support = false;
            for ext in ext_props {
                let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()).to_string_lossy() };
                if ext_name == "VK_KHR_ray_tracing_pipeline" {
                    rt_support = true;
                }
            }

            device_infos.push(DeviceInfo {
                name,
                index: idx as u32,
                vendor_id: props.vendor_id,
                max_compute_units: 0, 
                max_work_group_size: props.limits.max_compute_work_group_invocations,
                ray_tracing_support: rt_support,
                fp64_support: true,
                fp16_support: true,
                int8_support: true,
            });
        }

        Ok(VulkanContext {
            physical_devices: filtered_devices,
            device_infos,
            selected_device_idx: 0,
            device: None,
            queue_family_index: 0,
            queue: vk::Queue::null(),
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            is_cmd_recorded: false,
            fence: vk::Fence::null(),
            buffers: HashMap::new(),
            next_buffer_id: 1,
            kernels: HashMap::new(),
            next_kernel_id: 1,
            rt_pipeline: None,
            rt_as: None,
            acceleration_structures: HashMap::new(),
            next_as_id: 1,
        })
    }

    pub fn get_buffer_device_address(&self, buffer: ComputeBuffer) -> Result<u64, String> {
        let device = self.device.as_ref().unwrap();
        let vbuf = self.buffers.get(&buffer).ok_or("Invalid buffer")?;
        let info = vk::BufferDeviceAddressInfo::default().buffer(vbuf.buffer);
        let addr = unsafe { device.get_buffer_device_address(&info) };
        Ok(addr)
    }

    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32, String> {
        let pdevice = self.physical_devices[self.selected_device_idx as usize];
        let instance = &VULKAN_CORE.as_ref().map_err(|e| e.clone())?.1;
        let mem_props = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0 && (mem_props.memory_types[i as usize].property_flags & properties) == properties {
                return Ok(i);
            }
        }
        Err("Failed to find suitable memory type".to_string())
    }
}

impl ComputeContext for VulkanContext {
    fn get_devices(&self) -> Vec<DeviceInfo> {
        self.device_infos.clone()
    }

    fn pick_device(&mut self, index: u32) -> Result<(), String> {
        if index as usize >= self.physical_devices.len() {
            return Err("Invalid device index".to_string());
        }
        self.selected_device_idx = index;
        let pdevice = self.physical_devices[index as usize];
        let instance = &VULKAN_CORE.as_ref().map_err(|e| e.clone())?.1;

        // Find compute queue
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        let mut compute_family = None;
        for (i, family) in queue_families.iter().enumerate() {
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                compute_family = Some(i as u32);
                break;
            }
        }
        let queue_family_index = compute_family.ok_or("No compute queue found")?;
        self.queue_family_index = queue_family_index;

        let priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let mut device_extensions = Vec::new();
        let bda_name = CString::new("VK_KHR_buffer_device_address").unwrap();
        let as_name = CString::new("VK_KHR_acceleration_structure").unwrap();
        let rt_name = CString::new("VK_KHR_ray_tracing_pipeline").unwrap();
        let dho_name = CString::new("VK_KHR_deferred_host_operations").unwrap();
        
        if self.device_infos[self.selected_device_idx as usize].ray_tracing_support {
            device_extensions.push(as_name.as_ptr());
            device_extensions.push(rt_name.as_ptr());
            device_extensions.push(dho_name.as_ptr());
            device_extensions.push(bda_name.as_ptr()); // Provided by Vulkan 1.2 normally, but we request the KHR extension explicitly
        }

        let mut as_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
        let mut rt_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
        let mut rq_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR::default().buffer_device_address(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        unsafe { instance.get_physical_device_features2(pdevice, &mut features2); }

        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(true)
            .shader_int8(true)
            .storage_buffer8_bit_access(true);
            
        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true);

        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions);
            
        let mut features2_create = vk::PhysicalDeviceFeatures2::default()
            .features(features2.features)
            .push_next(&mut features12)
            .push_next(&mut features11);

        // Explicitly enable 64-bit and 16-bit
        features2_create.features.shader_float64 = vk::TRUE;
        features2_create.features.shader_int64 = vk::TRUE;
        features2_create.features.shader_int16 = vk::TRUE;

        if self.device_infos[self.selected_device_idx as usize].ray_tracing_support {
            features2_create = features2_create
                .push_next(&mut bda_features)
                .push_next(&mut rt_features)
                .push_next(&mut as_features)
                .push_next(&mut rq_features);
        }
            
        device_info.p_next = &features2_create as *const _ as *const std::ffi::c_void;

        let device = unsafe { instance.create_device(pdevice, &device_info, None).map_err(|e| e.to_string())? };
        
        let rt_pipeline = if self.device_infos[self.selected_device_idx as usize].ray_tracing_support {
            Some(RayTracingPipeline::new(instance, &device))
        } else { None };
        
        let rt_as = if self.device_infos[self.selected_device_idx as usize].ray_tracing_support {
            Some(AccelerationStructure::new(instance, &device))
        } else { None };
        
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None).map_err(|e| e.to_string())? };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info).map_err(|e| e.to_string())? };
        let command_buffer = command_buffers[0];

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None).map_err(|e| e.to_string())? };

        self.device = Some(device);
        self.queue = queue;
        self.command_pool = command_pool;
        self.command_buffer = command_buffer;
        self.fence = fence;
        self.rt_pipeline = rt_pipeline;
        self.rt_as = rt_as;

        Ok(())
    }

    fn get_current_device_info(&self) -> Option<DeviceInfo> {
        self.device_infos.get(self.selected_device_idx as usize).cloned()
    }

    fn create_buffer(&mut self, size: usize, host_ptr: Option<&[u8]>) -> Result<ComputeBuffer, String> {
        let device = self.device.as_ref().unwrap();
        let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None).map_err(|e| e.to_string())? };
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
        
        let properties = if buffer_info.usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        } else {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        };
        
        let mem_type_idx = self.find_memory_type(mem_req.memory_type_bits, properties).unwrap_or_else(|_| {
            self.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap()
        });
        let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_idx)
            .push_next(&mut alloc_flags);

        let memory = unsafe { device.allocate_memory(&alloc_info, None).map_err(|e| e.to_string())? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0).map_err(|e| e.to_string())? };

        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        let bda_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        self.buffers.insert(id, VulkanBuffer { buffer, memory, _size: size });

        if let Some(data) = host_ptr {
            self.write_buffer(id, 0, data)?;
        }

        Ok(id)
    }

    fn write_buffer(&mut self, buffer: ComputeBuffer, offset: usize, data: &[u8]) -> Result<(), String> {
        let device = self.device.as_ref().unwrap();
        let vbuf = self.buffers.get(&buffer).ok_or("Invalid buffer")?;
        
        let staging_info = vk::BufferCreateInfo::default()
            .size(data.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
            
        let staging_buffer = unsafe { device.create_buffer(&staging_info, None).map_err(|e| e.to_string())? };
        let mem_req = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let mem_type_idx = self.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();
        
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_idx);
            
        let staging_memory = unsafe { device.allocate_memory(&alloc_info, None).map_err(|e| e.to_string())? };
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0).map_err(|e| e.to_string())? };
        
        let ptr = unsafe { device.map_memory(staging_memory, 0, data.len() as u64, vk::MemoryMapFlags::empty()).map_err(|e| e.to_string())? };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len()) };
        unsafe { device.unmap_memory(staging_memory) };
        
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = device.allocate_command_buffers(&alloc_info).map_err(|e| e.to_string())?;
            let cb = command_buffers[0];
            
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cb, &begin_info).map_err(|e| e.to_string())?;
            
            let copy_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(offset as u64)
                .size(data.len() as u64);
            device.cmd_copy_buffer(cb, staging_buffer, vbuf.buffer, std::slice::from_ref(&copy_region));
            
            device.end_command_buffer(cb).map_err(|e| e.to_string())?;
            
            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
            
            let fence_info = vk::FenceCreateInfo::default();
            let new_fence = device.create_fence(&fence_info, None).map_err(|e| e.to_string())?;
            
            device.queue_submit(self.queue, std::slice::from_ref(&submit_info), new_fence).map_err(|e| e.to_string())?;
            device.wait_for_fences(std::slice::from_ref(&new_fence), true, u64::MAX).map_err(|e| e.to_string())?;
            device.destroy_fence(new_fence, None);
            
            device.free_command_buffers(self.command_pool, &command_buffers);
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }
        Ok(())
    }

    fn read_buffer(&self, buffer: ComputeBuffer, offset: usize, data: &mut [u8]) -> Result<(), String> {
        let device = self.device.as_ref().unwrap();
        let vbuf = self.buffers.get(&buffer).ok_or("Invalid buffer")?;
        
        let staging_info = vk::BufferCreateInfo::default()
            .size(data.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
            
        let staging_buffer = unsafe { device.create_buffer(&staging_info, None).map_err(|e| e.to_string())? };
        let mem_req = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
        let mem_type_idx = self.find_memory_type(mem_req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();
        
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_idx);
            
        let staging_memory = unsafe { device.allocate_memory(&alloc_info, None).map_err(|e| e.to_string())? };
        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0).map_err(|e| e.to_string())? };
        
        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = device.allocate_command_buffers(&alloc_info).map_err(|e| e.to_string())?;
            let cb = command_buffers[0];
            
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cb, &begin_info).map_err(|e| e.to_string())?;
            
            let copy_region = vk::BufferCopy::default()
                .src_offset(offset as u64)
                .dst_offset(0)
                .size(data.len() as u64);
            device.cmd_copy_buffer(cb, vbuf.buffer, staging_buffer, std::slice::from_ref(&copy_region));
            
            device.end_command_buffer(cb).map_err(|e| e.to_string())?;
            
            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));
            
            let fence_info = vk::FenceCreateInfo::default();
            let new_fence = device.create_fence(&fence_info, None).map_err(|e| e.to_string())?;
            
            device.queue_submit(self.queue, std::slice::from_ref(&submit_info), new_fence).map_err(|e| e.to_string())?;
            device.wait_for_fences(std::slice::from_ref(&new_fence), true, u64::MAX).map_err(|e| e.to_string())?;
            device.destroy_fence(new_fence, None);
            
            device.free_command_buffers(self.command_pool, &command_buffers);
        }
        
        let ptr = unsafe { device.map_memory(staging_memory, 0, data.len() as u64, vk::MemoryMapFlags::empty()).map_err(|e| e.to_string())? };
        unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, data.as_mut_ptr(), data.len()) };
        unsafe { device.unmap_memory(staging_memory) };
        
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }
        Ok(())
    }

    fn release_buffer(&mut self, buffer: ComputeBuffer) {
        if let Some(vbuf) = self.buffers.remove(&buffer) {
            if let Some(device) = &self.device {
                unsafe {
                    device.destroy_buffer(vbuf.buffer, None);
                    device.free_memory(vbuf.memory, None);
                }
            }
        }
    }

    fn create_kernel(&mut self, spv_bytes: &[u8], entry_point: &str, num_buffer_args: u32) -> Result<ComputeKernel, String> {
        let device = self.device.as_ref().unwrap();

        // Convert slice to u32 slice for ash
        let mut code = Vec::new();
        for chunk in spv_bytes.chunks_exact(4) {
            let val = u32::from_le_bytes(chunk.try_into().unwrap());
            code.push(val);
        }

        let module_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe { device.create_shader_module(&module_info, None).map_err(|e| e.to_string())? };

        let entry_name = std::ffi::CString::new(entry_point).unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_name);

        let mut bindings = Vec::new();
        for i in 0..num_buffer_args {
            let desc_type = vk::DescriptorType::STORAGE_BUFFER;
            bindings.push(vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(desc_type)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::ANY_HIT_KHR | vk::ShaderStageFlags::MISS_KHR | vk::ShaderStageFlags::INTERSECTION_KHR));
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None).map_err(|e| e.to_string())? };

        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)]; // allocate 8 bytes of push constants

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&desc_set_layout))
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).map_err(|e| e.to_string())? };

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);
        
        let pipeline = unsafe { 
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| format!("{:?}", e))?[0] 
        };

        unsafe { device.destroy_shader_module(shader_module, None); }

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((num_buffer_args).max(1) * 2),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(2);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None).map_err(|e| e.to_string())? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(std::slice::from_ref(&desc_set_layout));
        let desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) };
        let desc_set = match desc_sets {
            Ok(sets) => sets[0],
            Err(e) => {
                println!("[DIAGNOSTIC] allocate_descriptor_sets failed: {:?}", e);
                vk::DescriptorSet::null()
            }
        };

        let id = self.next_kernel_id;
        self.next_kernel_id += 1;
        self.kernels.insert(id, VulkanKernel {
            pipeline_layout, pipeline, desc_set_layout, desc_pool, desc_set,
            _num_buffers: num_buffer_args,
            bound_buffers: HashMap::new(),
            push_constants: Vec::new(),
            is_rt: false,
            rgen_region: vk::StridedDeviceAddressRegionKHR::default(),
            rmiss_region: vk::StridedDeviceAddressRegionKHR::default(),
            rhit_region: vk::StridedDeviceAddressRegionKHR::default(),
            rcall_region: vk::StridedDeviceAddressRegionKHR::default(),
            sbt_buffer: None,
        });
        Ok(id)
    }

    fn create_rt_pipeline(&mut self, rgen_path: &str, rmiss_path: &str, rchit_paths: &[&str], rahit_paths: &[&str], rint_paths: &[&str], num_buffer_args: u32) -> Result<ComputeKernel, String> {
        let device = self.device.clone().unwrap();
        let rt_pipeline = self.rt_pipeline.clone().ok_or("RT not supported")?;
        
        let mut shader_modules = Vec::new();
        let mut stages = Vec::new();
        let mut groups = Vec::new();
        let mut c_names = Vec::new();

        let mut load_shader = |path: &str, stage: vk::ShaderStageFlags, modules: &mut Vec<vk::ShaderModule>, stgs: &mut Vec<vk::PipelineShaderStageCreateInfo>| -> Result<u32, String> {
            let spv_path = if path.ends_with(".spv") { path.to_string() } else { format!("{}.spv", path) };
            let spv_bytes = std::fs::read(&spv_path).map_err(|e| format!("Failed to read {}: {}", spv_path, e))?;
            let mut words = Vec::new();
            for chunk in spv_bytes.chunks(4) {
                let mut word = [0u8; 4];
                let len = chunk.len();
                word[..len].copy_from_slice(chunk);
                words.push(u32::from_ne_bytes(word));
            }
            let create_info = vk::ShaderModuleCreateInfo::default().code(&words);
            let module = unsafe { device.create_shader_module(&create_info, None).map_err(|e| e.to_string())? };
            modules.push(module);
            
            let stage_info = vk::PipelineShaderStageCreateInfo::default()
                .stage(stage)
                .module(module);
            stgs.push(stage_info);
            c_names.push(std::ffi::CString::new("main").unwrap());
            Ok((stgs.len() - 1) as u32)
        };

        let rgen_idx = load_shader(rgen_path, vk::ShaderStageFlags::RAYGEN_KHR, &mut shader_modules, &mut stages)?;
        groups.push(vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(rgen_idx)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
        );

        let rmiss_idx = load_shader(rmiss_path, vk::ShaderStageFlags::MISS_KHR, &mut shader_modules, &mut stages)?;
        groups.push(vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(rmiss_idx)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
        );

        let num_hit_groups = rchit_paths.len().max(rahit_paths.len()).max(rint_paths.len());
        
        let mut rchit_indices = Vec::new();
        for path in rchit_paths {
            rchit_indices.push(load_shader(path, vk::ShaderStageFlags::CLOSEST_HIT_KHR, &mut shader_modules, &mut stages)?);
        }
        let mut rahit_indices = Vec::new();
        for path in rahit_paths {
            rahit_indices.push(load_shader(path, vk::ShaderStageFlags::ANY_HIT_KHR, &mut shader_modules, &mut stages)?);
        }
        let mut rint_indices = Vec::new();
        for path in rint_paths {
            rint_indices.push(load_shader(path, vk::ShaderStageFlags::INTERSECTION_KHR, &mut shader_modules, &mut stages)?);
        }
        
        for i in 0..num_hit_groups {
            let ty = if i < rint_paths.len() { vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP } else { vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP };
            groups.push(vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ty)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(if i < rchit_paths.len() { rchit_indices[i] } else { vk::SHADER_UNUSED_KHR })
                .any_hit_shader(if i < rahit_paths.len() { rahit_indices[i] } else { vk::SHADER_UNUSED_KHR })
                .intersection_shader(if i < rint_paths.len() { rint_indices[i] } else { vk::SHADER_UNUSED_KHR })
            );
        }

        for i in 0..stages.len() {
            stages[i].p_name = c_names[i].as_ptr();
        }

        let mut bindings = Vec::new();
        for i in 0..num_buffer_args {
            let mut ty = vk::DescriptorType::STORAGE_BUFFER;
            if i == 0 { ty = vk::DescriptorType::ACCELERATION_STRUCTURE_KHR; }
            bindings.push(vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL));
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None).map_err(|e| e.to_string())? };

        let push_constant = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(128);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&desc_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None).map_err(|e| e.to_string())? };

        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1)
            .layout(pipeline_layout);

        let pipeline = unsafe { rt_pipeline.create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None).map_err(|e| e.1.to_string())?[0] };

        let pdevice = self.physical_devices[self.selected_device_idx as usize];
        let instance = &crate::vulkan::VULKAN_CORE.as_ref().unwrap().1;
        let mut rt_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_props);
        unsafe { instance.get_physical_device_properties2(pdevice, &mut props2); }

        let handle_size = rt_props.shader_group_handle_size;
        let alignment = rt_props.shader_group_handle_alignment;
        let base_alignment = rt_props.shader_group_base_alignment;
        
        let handle_size_aligned = (handle_size + alignment - 1) & !(alignment - 1);
        let base_aligned = (handle_size_aligned + base_alignment - 1) & !(base_alignment - 1);
        let group_count = groups.len() as u32;
        let sbt_size = base_aligned + base_aligned + (group_count - 2) * handle_size_aligned;

        let handles = unsafe { rt_pipeline.get_ray_tracing_shader_group_handles(pipeline, 0, group_count, (group_count * handle_size) as usize).map_err(|e| e.to_string())? };

        let mut sbt_data = vec![0u8; sbt_size as usize];
        
        let rgen_offset = 0;
        let rmiss_offset = base_aligned;
        let rhit_offset = base_aligned * 2;
        
        // Copy rgen
        sbt_data[rgen_offset as usize .. rgen_offset as usize + handle_size as usize].copy_from_slice(&handles[0..handle_size as usize]);
        // Copy rmiss
        sbt_data[rmiss_offset as usize .. rmiss_offset as usize + handle_size as usize].copy_from_slice(&handles[handle_size as usize .. 2 * handle_size as usize]);
        // Copy hit groups
        for i in 2..group_count {
            let src_start = (i * handle_size) as usize;
            let src_end = src_start + handle_size as usize;
            let dst_start = (rhit_offset + (i - 2) * handle_size_aligned) as usize;
            sbt_data[dst_start..dst_start + handle_size as usize].copy_from_slice(&handles[src_start..src_end]);
        }
        
        let sbt_buffer = self.create_buffer(sbt_size as usize, Some(&sbt_data))?;
        
        let sbt_addr = self.get_buffer_device_address(sbt_buffer)?;

        let rgen_region = vk::StridedDeviceAddressRegionKHR::default().device_address(sbt_addr + rgen_offset as u64).stride(base_aligned as u64).size(base_aligned as u64);
        let rmiss_region = vk::StridedDeviceAddressRegionKHR::default().device_address(sbt_addr + rmiss_offset as u64).stride(handle_size_aligned as u64).size(base_aligned as u64);
        let rhit_region = vk::StridedDeviceAddressRegionKHR::default().device_address(sbt_addr + rhit_offset as u64).stride(handle_size_aligned as u64).size(((group_count - 2) * handle_size_aligned) as u64);
        let rcall_region = vk::StridedDeviceAddressRegionKHR::default().device_address(0).stride(0).size(0);

        unsafe {
            for m in shader_modules {
                device.destroy_shader_module(m, None);
            }
        }

        let mut pool_sizes = vec![vk::DescriptorPoolSize::default().ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).descriptor_count(1)];
        if num_buffer_args > 1 {
            pool_sizes.push(vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(num_buffer_args - 1));
        }

        let pool_info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1);
        let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None).map_err(|e| e.to_string())? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(std::slice::from_ref(&desc_set_layout));
        let desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).map_err(|e| e.to_string())? };
        let desc_set = desc_sets[0];

        let id = self.next_kernel_id;
        self.next_kernel_id += 1;
        self.kernels.insert(id, VulkanKernel {
            pipeline_layout, pipeline, desc_set_layout, desc_pool, desc_set,
            _num_buffers: num_buffer_args,
            bound_buffers: HashMap::new(),
            push_constants: Vec::new(),
            is_rt: true,
            rgen_region,
            rmiss_region,
            rhit_region,
            rcall_region,
            sbt_buffer: Some(sbt_buffer),
        });
        Ok(id)
    }

    fn set_kernel_arg_buffer(&mut self, kernel: ComputeKernel, arg_index: u32, buffer: ComputeBuffer) -> Result<(), String> {
        let device = self.device.as_ref().unwrap();
        let k = self.kernels.get_mut(&kernel).ok_or("Invalid kernel")?;
        let vbuf = self.buffers.get(&buffer).ok_or("Invalid buffer")?;

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(vbuf.buffer)
            .offset(0)
            .range(vbuf._size as u64);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(k.desc_set)
            .dst_binding(arg_index)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe { device.update_descriptor_sets(&[write], &[]); }
        k.bound_buffers.insert(arg_index, buffer);
        Ok(())
    }

    fn set_kernel_arg_push_constant(&mut self, kernel: ComputeKernel, data: &[u8]) -> Result<(), String> {
        let k = self.kernels.get_mut(&kernel).ok_or("Invalid kernel")?;
        let mut padded = vec![0u8; 128];
        padded[..data.len().min(128)].copy_from_slice(&data[..data.len().min(128)]);
        k.push_constants = padded;
        Ok(())
    }

    fn set_kernel_arg_as(&mut self, kernel: ComputeKernel, arg_index: u32, as_handle: crate::context::AccelerationStructureHandle) -> Result<(), String> {
        let device = self.device.as_ref().unwrap();
        let k = self.kernels.get_mut(&kernel).ok_or("Invalid kernel")?;
        let v_as = self.acceleration_structures.get(&as_handle).ok_or("Invalid AS")?;
        
        let mut write_as = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&v_as.handle));
            
        let write = vk::WriteDescriptorSet::default()
            .dst_set(k.desc_set)
            .dst_binding(arg_index)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .push_next(&mut write_as);

        unsafe { device.update_descriptor_sets(&[write], &[]); }
        Ok(())
    }

    fn dispatch(&mut self, kernel_id: ComputeKernel, grid_x: u32, grid_y: u32, grid_z: u32, _block_x: u32, _block_y: u32, _block_z: u32) -> Result<(), String> {
        let kernel = self.kernels.get(&kernel_id).ok_or("Invalid kernel")?;
        
        unsafe {
            let device = self.device.as_ref().unwrap();
            
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = device.allocate_command_buffers(&alloc_info).map_err(|e| e.to_string())?;
            let cb = command_buffers[0];
            self.command_buffer = cb;
            
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            
            device.begin_command_buffer(cb, &begin_info).map_err(|e| e.to_string())?;
            
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            
            if kernel.desc_set != vk::DescriptorSet::null() {
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline_layout,
                    0,
                    &[kernel.desc_set],
                    &[]
                );
            }
            
            if !kernel.push_constants.is_empty() {
                device.cmd_push_constants(
                    cb,
                    kernel.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &kernel.push_constants
                );
            }
            
            device.cmd_dispatch(cb, grid_x, grid_y, grid_z);
            
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ)
                .dst_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ);
                
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[]
            );
            
            device.cmd_dispatch(cb, grid_x, grid_y, grid_z);
            
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&memory_barrier),
                &[],
                &[]
            );
            
            device.end_command_buffer(cb).map_err(|e| e.to_string())?;
            
            let fence_info = vk::FenceCreateInfo::default();
            let new_fence = device.create_fence(&fence_info, None).map_err(|e| e.to_string())?;

            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cb));
            device.queue_submit(self.queue, std::slice::from_ref(&submit_info), new_fence).map_err(|e| e.to_string())?;
            
            device.wait_for_fences(std::slice::from_ref(&new_fence), true, u64::MAX).map_err(|e| e.to_string())?;
            device.destroy_fence(new_fence, None);
            
            device.free_command_buffers(self.command_pool, &command_buffers);
        }
        Ok(())
    }

    fn release_kernel(&mut self, kernel: ComputeKernel) {
        if let Some(k) = self.kernels.remove(&kernel) {
            if let Some(sbt) = k.sbt_buffer {
                self.release_buffer(sbt);
            }
            if let Some(device) = &self.device {
                unsafe {
                    device.destroy_pipeline(k.pipeline, None);
                    device.destroy_pipeline_layout(k.pipeline_layout, None);
                    device.destroy_descriptor_pool(k.desc_pool, None);
                    device.destroy_descriptor_set_layout(k.desc_set_layout, None);
                }
            }
        }
    }

    fn wait_idle(&mut self) -> Result<(), String> {
        unsafe {
            let device = self.device.as_ref().unwrap();
            device.queue_wait_idle(self.queue).map_err(|e| e.to_string())?;
            device.reset_fences(&[self.fence]).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn build_blas(&mut self, geometry: &[crate::context::BlasGeometry]) -> Result<crate::context::AccelerationStructureHandle, String> {
        let rt_as = self.rt_as.as_ref().ok_or("RT not supported")?.clone();

        let mut as_geometries = Vec::new();
        let mut as_ranges = Vec::new();
        let mut max_primitive_counts = Vec::new();

        for geom in geometry {
            match geom {
                crate::context::BlasGeometry::Triangles(t) => {
                    let vertex_addr = self.get_buffer_device_address(t.vertex_buffer)?;
                    let index_addr = self.get_buffer_device_address(t.index_buffer)?;

                    let mut flags = vk::GeometryFlagsKHR::empty();
                    if t.is_opaque { flags |= vk::GeometryFlagsKHR::OPAQUE; }

                    let has_indices = t.index_count > 0;
                    let idx_type = if has_indices { vk::IndexType::UINT32 } else { vk::IndexType::NONE_KHR };
                    let prim_count = if has_indices { t.index_count / 3 } else { t.vertex_count / 3 };

                    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_addr })
                        .vertex_stride(t.vertex_stride as u64)
                        .max_vertex(t.vertex_count - 1)
                        .index_type(idx_type)
                        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_addr });

                    let geometry_data = vk::AccelerationStructureGeometryDataKHR { triangles };

                    as_geometries.push(vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                        .geometry(geometry_data)
                        .flags(flags));

                    as_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR::default()
                        .primitive_count(prim_count)
                        .primitive_offset(0)
                        .first_vertex(0)
                        .transform_offset(0));

                    max_primitive_counts.push(prim_count);
                }
                crate::context::BlasGeometry::Aabbs(a) => {
                    let aabb_addr = self.get_buffer_device_address(a.aabb_buffer)?;
                    
                    let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
                        .data(vk::DeviceOrHostAddressConstKHR { device_address: aabb_addr })
                        .stride(a.aabb_stride as u64);

                    let geometry_data = vk::AccelerationStructureGeometryDataKHR { aabbs };

                    let mut flags = vk::GeometryFlagsKHR::empty();
                    if a.is_opaque { flags |= vk::GeometryFlagsKHR::OPAQUE; }

                    as_geometries.push(vk::AccelerationStructureGeometryKHR::default()
                        .geometry_type(vk::GeometryTypeKHR::AABBS)
                        .geometry(geometry_data)
                        .flags(flags));

                    as_ranges.push(vk::AccelerationStructureBuildRangeInfoKHR::default()
                        .primitive_count(a.aabb_count)
                        .primitive_offset(0)
                        .first_vertex(0)
                        .transform_offset(0));

                    max_primitive_counts.push(a.aabb_count);
                }
            }
        }

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&as_geometries);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            rt_as.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &max_primitive_counts,
                &mut size_info,
            );
        }

        let as_buffer = self.create_buffer(size_info.acceleration_structure_size as usize, None)?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(self.buffers.get(&as_buffer).unwrap().buffer)
            .size(size_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        let handle = unsafe { rt_as.create_acceleration_structure(&create_info, None).map_err(|e| e.to_string())? };

        let scratch_buffer = self.create_buffer(size_info.build_scratch_size as usize, None)?;
        let scratch_addr = self.get_buffer_device_address(scratch_buffer)?;

        build_info.dst_acceleration_structure = handle;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch_addr };

        unsafe {
            let device = self.device.as_ref().unwrap();
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(self.command_buffer, &begin_info).map_err(|e| e.to_string())?;

            rt_as.cmd_build_acceleration_structures(
                self.command_buffer,
                std::slice::from_ref(&build_info),
                &[&as_ranges],
            );

            device.end_command_buffer(self.command_buffer).map_err(|e| e.to_string())?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
            device.reset_fences(std::slice::from_ref(&self.fence)).map_err(|e| e.to_string())?;
            device.queue_submit(self.queue, std::slice::from_ref(&submit_info), self.fence).map_err(|e| e.to_string())?;
            device.wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX).map_err(|e| e.to_string())?;
        }

        self.release_buffer(scratch_buffer);

        let id = self.next_as_id;
        self.next_as_id += 1;
        self.acceleration_structures.insert(id, VulkanAS { handle, buffer: as_buffer, instance_buffer: None });

        Ok(id)
    }

    fn build_tlas(&mut self, instances: &[crate::context::TlasInstance]) -> Result<crate::context::AccelerationStructureHandle, String> {
        let rt_as = self.rt_as.as_ref().ok_or("RT not supported")?.clone();

        let mut vk_instances = Vec::new();
        for inst in instances {
            let blas = self.acceleration_structures.get(&inst.blas).ok_or("Invalid BLAS")?;
            let info = vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas.handle);
            let bda = unsafe { rt_as.get_acceleration_structure_device_address(&info) };

            let transform = vk::TransformMatrixKHR {
                matrix: [
                    inst.transform[0], inst.transform[1], inst.transform[2], inst.transform[3],
                    inst.transform[4], inst.transform[5], inst.transform[6], inst.transform[7],
                    inst.transform[8], inst.transform[9], inst.transform[10], inst.transform[11],
                ]
            };

            vk_instances.push(vk::AccelerationStructureInstanceKHR {
                transform,
                instance_custom_index_and_mask: vk::Packed24_8::new(inst.instance_id, inst.mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(inst.instance_offset, inst.flags as u8),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: bda },
            });
        }

        let instance_buffer_size = (vk_instances.len() * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()) as usize;
        let instances_bytes = unsafe { std::slice::from_raw_parts(vk_instances.as_ptr() as *const u8, instance_buffer_size) };
        let instance_buffer = self.create_buffer(instance_buffer_size, Some(instances_bytes))?;
        let instance_addr = self.get_buffer_device_address(instance_buffer)?;

        let geometry_data = vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR { device_address: instance_addr })
        };

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(geometry_data)
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            rt_as.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[instances.len() as u32],
                &mut size_info,
            );
        }

        let as_buffer = self.create_buffer(size_info.acceleration_structure_size as usize, None)?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(self.buffers.get(&as_buffer).unwrap().buffer)
            .size(size_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let handle = unsafe { rt_as.create_acceleration_structure(&create_info, None).map_err(|e| e.to_string())? };

        let scratch_buffer = self.create_buffer(size_info.build_scratch_size as usize, None)?;
        let scratch_addr = self.get_buffer_device_address(scratch_buffer)?;

        build_info.dst_acceleration_structure = handle;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch_addr };

        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instances.len() as u32)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);

        unsafe {
            let device = self.device.as_ref().unwrap();
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(self.command_buffer, &begin_info).map_err(|e| e.to_string())?;

            rt_as.cmd_build_acceleration_structures(
                self.command_buffer,
                std::slice::from_ref(&build_info),
                &[&[range]],
            );

            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);

            self.device.as_ref().unwrap().cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );

            device.end_command_buffer(self.command_buffer).map_err(|e| e.to_string())?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
            device.reset_fences(std::slice::from_ref(&self.fence)).map_err(|e| e.to_string())?;
            device.queue_submit(self.queue, std::slice::from_ref(&submit_info), self.fence).map_err(|e| e.to_string())?;
            device.wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX).map_err(|e| e.to_string())?;
        }

        self.release_buffer(scratch_buffer);
        // Do not release instance_buffer! It might be needed by radv traversal.
        
        let id = self.next_as_id;
        self.next_as_id += 1;
        self.acceleration_structures.insert(id, VulkanAS { handle, buffer: as_buffer, instance_buffer: Some(instance_buffer) });

        Ok(id)
    }

    fn release_acceleration_structure(&mut self, as_handle: crate::context::AccelerationStructureHandle) {
        if let Some(v_as) = self.acceleration_structures.remove(&as_handle) {
            if let Some(rt_as) = &self.rt_as {
                unsafe { rt_as.destroy_acceleration_structure(v_as.handle, None); }
            }
            self.release_buffer(v_as.buffer);
            if let Some(ibuf) = v_as.instance_buffer {
                self.release_buffer(ibuf);
            }
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        if let Some(device) = &self.device {
            unsafe {
                let _ = device.device_wait_idle();
                for (_, k) in self.kernels.drain() {
                    device.destroy_pipeline(k.pipeline, None);
                    device.destroy_pipeline_layout(k.pipeline_layout, None);
                    device.destroy_descriptor_pool(k.desc_pool, None);
                    device.destroy_descriptor_set_layout(k.desc_set_layout, None);
                }
                for (_, vbuf) in self.buffers.drain() {
                    device.destroy_buffer(vbuf.buffer, None);
                    device.free_memory(vbuf.memory, None);
                }
                device.destroy_fence(self.fence, None);
                device.destroy_command_pool(self.command_pool, None);
                device.destroy_device(None);
            }
        }
    }
}
