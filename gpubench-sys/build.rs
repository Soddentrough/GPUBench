fn main() {
    let dst = cmake::Config::new("..")
        .build_target("gpubench_lib")
        .build();

    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}/build/Release", dst.display());
    println!("cargo:rustc-link-search=native={}/build/Debug", dst.display());
    println!("cargo:rustc-link-lib=static=gpubench_lib");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        println!("cargo:rustc-link-lib=vulkan-1");
        if let Ok(sdk) = std::env::var("VULKAN_SDK") {
            println!("cargo:rustc-link-search=native={}/Lib", sdk);
        }
    } else {
        println!("cargo:rustc-link-lib=vulkan");
        if target_os == "macos" {
            if let Ok(sdk) = std::env::var("VULKAN_SDK") {
                println!("cargo:rustc-link-search=native={}/lib", sdk);
            }
        }
    }

    cxx_build::bridge("src/lib.rs")
        .file("src/bridge.cpp")
        .include("../cpp_src")
        .include("../external")
        .define("HAVE_VULKAN", None)
        .define("HAVE_OPENCL", None)
        .define("HAVE_ROCM", None)
        .define("__HIP_PLATFORM_AMD__", None)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-maybe-uninitialized")
        .compile("gpubench-cxx");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=../CMakeLists.txt");
    println!("cargo:rerun-if-changed=../cpp_src");
}
