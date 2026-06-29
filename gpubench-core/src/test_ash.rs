use ash::vk;
fn test() {
    let _data = vk::AccelerationStructureGeometryDataKHR {
        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
    };
    let _addr = vk::DeviceOrHostAddressConstKHR {
        device_address: 0
    };
}
