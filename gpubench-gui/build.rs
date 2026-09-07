fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows" {
        let mut res = winres::WindowsResource::new();
        res.set_icon("../packaging/windows/icon.ico");
        res.set("ProductName", "GPUBench");
        res.set("FileDescription", "GPUBench Graphical User Interface");
        res.set("CompanyName", "Soddentrough");
        res.set("LegalCopyright", "Copyright (C) 2026 Soddentrough");
        res.set("OriginalFilename", "gpubench-gui.exe");
        res.set("ProductVersion", env!("CARGO_PKG_VERSION"));
        res.set("FileVersion", env!("CARGO_PKG_VERSION"));
        if let Err(e) = res.compile() {
            eprintln!("Failed to compile Windows resources: {}", e);
        }
    }
}
