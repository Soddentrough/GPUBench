fn main() {
    println!("Testing FP32...");
    gpubench_core::run_benchmarks(&vec![], &vec![0], &vec![], true, true, false, false, 0, 0, "all", |res| {
        println!("Result: {:?}", res);
    });
    println!("Done!");
}
