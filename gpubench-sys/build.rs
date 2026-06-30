fn main() {
    let dst = cmake::Config::new("..")
        .build_target("gpubench_lib") // We will need to update CMakeLists.txt to build a lib
        .build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/build/Release", dst.display());
    println!("cargo:rustc-link-search=native={}/build/Debug", dst.display());
    println!("cargo:rustc-link-lib=static=gpubench_lib");
    println!("cargo:rustc-link-lib=vulkan");

    cxx_build::bridge("src/lib.rs")
        .file("src/bridge.cpp") // We'll create this to bridge complex types if needed
        .include("../cpp_src")
        .include("../external")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-maybe-uninitialized")
        .compile("gpubench-cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=../CMakeLists.txt");
}
