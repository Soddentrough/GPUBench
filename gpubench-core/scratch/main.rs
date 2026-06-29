use ash::vk;
fn main() {
    let packed = vk::Packed24_8::new(0xAA, 0xBB);
    println!("packed: {:x}", packed.as_u32());
}
