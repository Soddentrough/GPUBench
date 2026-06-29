fn main() {
    println!("Available hardware:");
    for hw in gpubench_core::get_available_hardware() {
        println!("{}", hw);
    }
}
