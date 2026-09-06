#![windows_subsystem = "windows"]

use iced::widget::{button, column, container, progress_bar, row, scrollable, text, Space, tooltip};
use iced::{color, Background, Border, Command, Element, Length, Theme, executor, Application, Settings};
use gpubench_core::{get_available_benchmarks, run_benchmarks, ResultData, SystemInfo, DeviceProfile, get_device_profiles};
use std::sync::{mpsc, mpsc::Sender, Mutex, LazyLock};
use std::collections::{HashSet, HashMap};

struct SleekPrimaryButton;
impl iced::widget::button::StyleSheet for SleekPrimaryButton {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        use iced::gradient::Linear;
        let mut gradient = Linear::new(0.0);
        gradient = gradient.add_stop(0.0, color!(0x4F46E5, 0.95)); // Indigo 600
        gradient = gradient.add_stop(0.6, color!(0x6366F1, 0.95)); // Indigo 500
        gradient = gradient.add_stop(1.0, color!(0x818CF8, 0.95)); // Indigo 400
        iced::widget::button::Appearance {
            background: Some(Background::Gradient(gradient.into())),
            text_color: color!(0xFFFFFF),
            border: Border { radius: 10.0.into(), width: 1.0, color: color!(0xFFFFFF, 0.15) },
            shadow_offset: iced::Vector::new(0.0, 2.0),
            ..Default::default()
        }
    }
    fn hovered(&self, style: &Self::Style) -> iced::widget::button::Appearance {
        let mut app = self.active(style);
        use iced::gradient::Linear;
        let mut gradient = Linear::new(0.0);
        gradient = gradient.add_stop(0.0, color!(0x4338CA, 1.0)); // Indigo 700
        gradient = gradient.add_stop(0.6, color!(0x4F46E5, 1.0)); // Indigo 600
        gradient = gradient.add_stop(1.0, color!(0x6366F1, 1.0)); // Indigo 500
        app.background = Some(Background::Gradient(gradient.into()));
        app.border = Border { radius: 10.0.into(), width: 1.0, color: color!(0xFFFFFF, 0.3) };
        app
    }
}

struct SleekSecondaryButton;
impl iced::widget::button::StyleSheet for SleekSecondaryButton {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        iced::widget::button::Appearance {
            background: Some(Background::Color(color!(0x141722))),
            text_color: color!(0xCBD5E1),
            border: Border { radius: 10.0.into(), width: 1.0, color: color!(0x2D354B) },
            ..Default::default()
        }
    }
    fn hovered(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        iced::widget::button::Appearance {
            background: Some(Background::Color(color!(0x1D2232))),
            text_color: color!(0xF8FAFC),
            border: Border { radius: 10.0.into(), width: 1.0, color: color!(0x475569) },
            ..Default::default()
        }
    }
}

struct SleekPillToggle {
    is_active: bool,
    is_api_selector: bool,
}

impl iced::widget::button::StyleSheet for SleekPillToggle {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_active {
            if self.is_api_selector {
                iced::widget::button::Appearance {
                    background: Some(Background::Color(color!(0x0EA5E9, 0.16))),
                    text_color: color!(0x38BDF8),
                    border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x0EA5E9, 0.75) },
                    ..Default::default()
                }
            } else {
                iced::widget::button::Appearance {
                    background: Some(Background::Color(color!(0x6366F1, 0.16))),
                    text_color: color!(0xA5B4FC),
                    border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x6366F1, 0.75) },
                    ..Default::default()
                }
            }
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x12151F))),
                text_color: color!(0x64748B),
                border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1A1F2C) },
                ..Default::default()
            }
        }
    }
    fn hovered(&self, style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_active {
            self.active(style)
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x181C2A))),
                text_color: color!(0x94A3B8),
                border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x262D40) },
                ..Default::default()
            }
        }
    }
}

struct SleekDisabledPill;
impl iced::widget::button::StyleSheet for SleekDisabledPill {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        iced::widget::button::Appearance {
            background: Some(Background::Color(color!(0x0C0E14))),
            text_color: color!(0x334155),
            border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x161B26) },
            ..Default::default()
        }
    }
    fn hovered(&self, style: &Self::Style) -> iced::widget::button::Appearance {
        self.active(style)
    }
}

struct SleekDeviceCheckbox {
    is_checked: bool,
}

impl iced::widget::button::StyleSheet for SleekDeviceCheckbox {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_checked {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x141A2E))),
                text_color: color!(0xF1F5F9),
                border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x6366F1, 0.7) },
                ..Default::default()
            }
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x0E1017))),
                text_color: color!(0x64748B),
                border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1E2433) },
                ..Default::default()
            }
        }
    }
    fn hovered(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        iced::widget::button::Appearance {
            background: Some(Background::Color(color!(0x182038))),
            text_color: color!(0xFFFFFF),
            border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x818CF8, 0.85) },
            ..Default::default()
        }
    }
}

struct SleekGroupChip {
    is_highlighted: bool,
    is_disabled: bool,
}

impl iced::widget::button::StyleSheet for SleekGroupChip {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_disabled {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x0D0F16))),
                text_color: color!(0x334155),
                border: Border { radius: 16.0.into(), width: 1.0, color: color!(0x161B26) },
                ..Default::default()
            }
        } else if self.is_highlighted {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x4F46E5, 0.4))),
                text_color: color!(0xFFFFFF),
                border: Border { radius: 16.0.into(), width: 1.0, color: color!(0x818CF8, 0.9) },
                ..Default::default()
            }
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x131722))),
                text_color: color!(0x94A3B8),
                border: Border { radius: 16.0.into(), width: 1.0, color: color!(0x262E42) },
                ..Default::default()
            }
        }
    }
    fn hovered(&self, style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_disabled {
            self.active(style)
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x1B2132))),
                text_color: color!(0xF8FAFC),
                border: Border { radius: 16.0.into(), width: 1.0, color: color!(0x6366F1, 0.8) },
                ..Default::default()
            }
        }
    }
}

struct SleekDeviceTab {
    is_active: bool,
}

impl iced::widget::button::StyleSheet for SleekDeviceTab {
    type Style = Theme;
    fn active(&self, _style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_active {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x0EA5E9, 0.25))),
                text_color: color!(0x38BDF8),
                border: Border { radius: 5.0.into(), width: 1.0, color: color!(0x0EA5E9, 0.8) },
                ..Default::default()
            }
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x131722))),
                text_color: color!(0x64748B),
                border: Border { radius: 5.0.into(), width: 1.0, color: color!(0x1C2230) },
                ..Default::default()
            }
        }
    }
    fn hovered(&self, style: &Self::Style) -> iced::widget::button::Appearance {
        if self.is_active {
            self.active(style)
        } else {
            iced::widget::button::Appearance {
                background: Some(Background::Color(color!(0x1A2030))),
                text_color: color!(0x94A3B8),
                border: Border { radius: 5.0.into(), width: 1.0, color: color!(0x2D3748) },
                ..Default::default()
            }
        }
    }
}

// ============================================================================
// Hardware Telemetry & Dynamic API Detection
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct DeviceTelemetry {
    pub id: String,
    pub name: String,
    pub is_gpu: bool,
    pub hwmon_path: std::path::PathBuf,
    pub drm_dev_path: Option<std::path::PathBuf>,
    
    // Live values
    pub temp: f32,
    pub junction_temp: f32,
    pub mem_temp: f32,
    pub gpu_util: u32,
    pub fan_rpm: u32,
    pub power: f32,
    pub sclk: u32,
    pub mclk: u32,
    pub vram_used: u64,
    pub vram_total: u64,

    // Session Statistics
    pub temp_min: f32,
    pub temp_max: f32,
    pub temp_sum: f32,
    pub power_min: f32,
    pub power_max: f32,
    pub power_sum: f32,
    pub sample_count: u32,
}

impl DeviceTelemetry {
    pub fn reset_stats(&mut self) {
        self.temp_min = 0.0;
        self.temp_max = 0.0;
        self.temp_sum = 0.0;
        self.power_min = 0.0;
        self.power_max = 0.0;
        self.power_sum = 0.0;
        self.sample_count = 0;
    }

    pub fn avg_temp(&self) -> f32 {
        if self.sample_count > 0 { self.temp_sum / self.sample_count as f32 } else { self.temp }
    }

    pub fn avg_power(&self) -> f32 {
        if self.sample_count > 0 { self.power_sum / self.sample_count as f32 } else { self.power }
    }
}

fn read_sysfs_trimmed(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

fn discover_all_devices() -> Vec<DeviceTelemetry> {
    let mut devices = Vec::new();

    // 1. Discover GPUs via DRM
    let mut cards = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("card") && !name.contains('-') {
                    if path.join("device/hwmon").exists() {
                        cards.push(path);
                    }
                }
            }
        }
    }
    cards.sort();

    for (i, card) in cards.iter().enumerate() {
        let dev = card.join("device");
        let hwmon_base = dev.join("hwmon");
        let mut hwmon_path = std::path::PathBuf::new();
        if let Ok(entries) = std::fs::read_dir(&hwmon_base) {
            for entry in entries.flatten() {
                hwmon_path = entry.path();
                break;
            }
        }
        
        let prod_name = read_sysfs_trimmed(&dev.join("product_name").to_string_lossy())
            .or_else(|| read_sysfs_trimmed(&card.join("device/product_name").to_string_lossy()))
            .unwrap_or_else(|| format!("AMD Radeon GPU #{}", i));

        devices.push(DeviceTelemetry {
            id: format!("GPU {}", i),
            name: prod_name,
            is_gpu: true,
            hwmon_path,
            drm_dev_path: Some(dev),
            temp: 0.0,
            junction_temp: 0.0,
            mem_temp: 0.0,
            gpu_util: 0,
            fan_rpm: 0,
            power: 0.0,
            sclk: 0,
            mclk: 0,
            vram_used: 0,
            vram_total: 0,
            temp_min: 0.0,
            temp_max: 0.0,
            temp_sum: 0.0,
            power_min: 0.0,
            power_max: 0.0,
            power_sum: 0.0,
            sample_count: 0,
        });
    }

    // 2. Discover CPU via hwmon (k10temp / coretemp / zenpower)
    if let Ok(entries) = std::fs::read_dir("/sys/class/hwmon") {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(name) = read_sysfs_trimmed(&p.join("name").to_string_lossy()) {
                if name == "k10temp" || name == "coretemp" || name == "zenpower" || name.contains("cpu") {
                    devices.push(DeviceTelemetry {
                        id: "CPU".to_string(),
                        name: "Host CPU (Package)".to_string(),
                        is_gpu: false,
                        hwmon_path: p,
                        drm_dev_path: None,
                        temp: 0.0,
                        junction_temp: 0.0,
                        mem_temp: 0.0,
                        gpu_util: 0,
                        fan_rpm: 0,
                        power: 0.0,
                        sclk: 0,
                        mclk: 0,
                        vram_used: 0,
                        vram_total: 0,
                        temp_min: 0.0,
                        temp_max: 0.0,
                        temp_sum: 0.0,
                        power_min: 0.0,
                        power_max: 0.0,
                        power_sum: 0.0,
                        sample_count: 0,
                    });
                    break;
                }
            }
        }
    }

    devices
}

fn poll_all_devices(devices: &mut [DeviceTelemetry], is_benchmarking: bool) {
    for dev in devices.iter_mut() {
        if dev.is_gpu {
            if let Some(t) = read_sysfs_trimmed(&dev.hwmon_path.join("temp1_input").to_string_lossy()).and_then(|s| s.parse::<f32>().ok()) {
                dev.temp = t / 1000.0;
            }
            if let Some(t2) = read_sysfs_trimmed(&dev.hwmon_path.join("temp2_input").to_string_lossy()).and_then(|s| s.parse::<f32>().ok()) {
                dev.junction_temp = t2 / 1000.0;
            }
            if let Some(t3) = read_sysfs_trimmed(&dev.hwmon_path.join("temp3_input").to_string_lossy()).and_then(|s| s.parse::<f32>().ok()) {
                dev.mem_temp = t3 / 1000.0;
            }
            if let Some(fan) = read_sysfs_trimmed(&dev.hwmon_path.join("fan1_input").to_string_lossy()).and_then(|s| s.parse::<u32>().ok()) {
                dev.fan_rpm = fan;
            }
            if let Some(pw) = read_sysfs_trimmed(&dev.hwmon_path.join("power1_average").to_string_lossy())
                .or_else(|| read_sysfs_trimmed(&dev.hwmon_path.join("power1_input").to_string_lossy()))
                .and_then(|s| s.parse::<f32>().ok()) {
                dev.power = pw / 1_000_000.0;
            }
            if let Some(f1) = read_sysfs_trimmed(&dev.hwmon_path.join("freq1_input").to_string_lossy()).and_then(|s| s.parse::<u32>().ok()) {
                dev.sclk = f1 / 1_000_000;
            }
            if let Some(f2) = read_sysfs_trimmed(&dev.hwmon_path.join("freq2_input").to_string_lossy()).and_then(|s| s.parse::<u32>().ok()) {
                dev.mclk = f2 / 1_000_000;
            }
            if let Some(drm_dev) = &dev.drm_dev_path {
                if let Some(u) = read_sysfs_trimmed(&drm_dev.join("gpu_busy_percent").to_string_lossy()).and_then(|s| s.parse::<u32>().ok()) {
                    dev.gpu_util = u;
                }
                if dev.sclk == 0 {
                    if let Some(s) = read_sysfs_trimmed(&drm_dev.join("current_gfxclk").to_string_lossy()).and_then(|s| s.parse::<u32>().ok()) {
                        dev.sclk = s;
                    }
                }
                if dev.mclk == 0 {
                    if let Some(m) = read_sysfs_trimmed(&drm_dev.join("current_uclk").to_string_lossy()).and_then(|s| s.parse::<u32>().ok()) {
                        dev.mclk = m;
                    }
                }
                if let Some(vu) = read_sysfs_trimmed(&drm_dev.join("mem_info_vram_used").to_string_lossy()).and_then(|s| s.parse::<u64>().ok()) {
                    dev.vram_used = vu / (1024 * 1024);
                }
                if let Some(vt) = read_sysfs_trimmed(&drm_dev.join("mem_info_vram_total").to_string_lossy()).and_then(|s| s.parse::<u64>().ok()) {
                    dev.vram_total = vt / (1024 * 1024);
                }
            }
        } else {
            // CPU
            if let Some(t) = read_sysfs_trimmed(&dev.hwmon_path.join("temp1_input").to_string_lossy()).and_then(|s| s.parse::<f32>().ok()) {
                dev.temp = t / 1000.0;
            }
            if let Some(pw) = read_sysfs_trimmed(&dev.hwmon_path.join("power1_average").to_string_lossy())
                .or_else(|| read_sysfs_trimmed(&dev.hwmon_path.join("power1_input").to_string_lossy()))
                .and_then(|s| s.parse::<f32>().ok()) {
                dev.power = pw / 1_000_000.0;
            }
        }

        if is_benchmarking {
            if dev.temp > 0.0 {
                dev.temp_min = if dev.temp_min == 0.0 { dev.temp } else { dev.temp_min.min(dev.temp) };
                dev.temp_max = dev.temp_max.max(dev.temp);
                dev.temp_sum += dev.temp;
            }
            if dev.power > 0.0 {
                dev.power_min = if dev.power_min == 0.0 { dev.power } else { dev.power_min.min(dev.power) };
                dev.power_max = dev.power_max.max(dev.power);
                dev.power_sum += dev.power;
            }
            dev.sample_count += 1;
        }
    }
}

fn detect_dynamic_api_version(api: &str) -> String {
    match api.to_uppercase().as_str() {
        "VULKAN" => {
            #[cfg(target_os = "linux")]
            {
                type PfnEnumerate = unsafe extern "C" fn(*mut u32) -> i32;
                unsafe {
                    for lib_name in &[b"libvulkan.so.1\0".as_ptr(), b"libvulkan.so\0".as_ptr()] {
                        let handle = libc::dlopen(*lib_name as *const _, libc::RTLD_LAZY);
                        if !handle.is_null() {
                            let sym = libc::dlsym(handle, b"vkEnumerateInstanceVersion\0".as_ptr() as *const _);
                            if !sym.is_null() {
                                let func: PfnEnumerate = std::mem::transmute(sym);
                                let mut ver = 0u32;
                                if func(&mut ver) == 0 {
                                    libc::dlclose(handle);
                                    let major = (ver >> 22) & 0x7F;
                                    let minor = (ver >> 12) & 0x3FF;
                                    return format!("{}.{}", major, minor);
                                }
                            }
                            libc::dlclose(handle);
                        }
                    }
                }
            }
            "1.4".to_string()
        }
        "ROCM" => {
            #[cfg(target_os = "linux")]
            {
                type PfnHipVersion = unsafe extern "C" fn(*mut i32) -> i32;
                unsafe {
                    for lib_name in &[
                        b"libamdhip64.so.7\0".as_ptr(),
                        b"libamdhip64.so.6\0".as_ptr(),
                        b"libamdhip64.so\0".as_ptr(),
                    ] {
                        let handle = libc::dlopen(*lib_name as *const _, libc::RTLD_LAZY);
                        if !handle.is_null() {
                            let sym = libc::dlsym(handle, b"hipRuntimeGetVersion\0".as_ptr() as *const _);
                            if !sym.is_null() {
                                let func: PfnHipVersion = std::mem::transmute(sym);
                                let mut ver = 0i32;
                                if func(&mut ver) == 0 && ver > 0 {
                                    libc::dlclose(handle);
                                    let major = ver / 10000000;
                                    let minor = (ver / 100000) % 100;
                                    return format!("{}.{}", major, minor);
                                }
                            }
                            libc::dlclose(handle);
                        }
                    }
                }
            }
            "7.1".to_string()
        }
        "OPENCL" => "3.0".to_string(),
        _ => "".to_string(),
    }
}

fn get_benchmark_description(name: &str) -> &'static str {
    match name {
        "FP64" => "Double-precision (64-bit IEEE 754) floating point compute throughput.",
        "FP32" => "Single-precision (32-bit float) compute throughput, primary metric for standard 3D shaders and graphics compute.",
        "FP16" => "Half-precision (16-bit float) SIMD vector and dual-issue packed arithmetic throughput.",
        "BF16" => "16-bit Brain Floating Point vector throughput optimized for machine learning and neural network training.",
        "FP8" => "Quarter-precision 8-bit floating point arithmetic (E4M3 / E5M2) for modern low-bit AI inference.",
        "FP4" => "4-bit quantized floating point compute throughput for ultra-compact model execution.",
        "INT8" => "8-bit integer tensor and dot-product (DP4A) throughput for quantized neural network inference.",
        "INT4" => "4-bit integer quantized vector compute throughput for heavily compressed models.",
        "Device Memory Bandwidth" => "Peak streaming read/write bandwidth across the dedicated GPU VRAM bus.",
        "System Memory Bandwidth" => "Multi-threaded host RAM copy and streaming bandwidth from CPU to system DDR memory.",
        "System Memory Latency" => "Pointer-chasing memory access latency in nanoseconds (lower is better).",
        "Pixel Fill Rate" => "Rasterizer and ROP output fill throughput across 32-bit RGBA, 64-bit HDR, and alpha blending.",
        "RayTracing" | "RayIntersect" => "Peak BVH acceleration structure traversal and ray-triangle intersection throughput.",
        "RayDivergence" => "BVH traversal throughput under non-uniform ray branches and wavefront execution divergence.",
        "RayAnyHit" => "Traversal and shader invocation throughput against transparent, alpha-tested geometry.",
        "RayIncoherent" => "Cache hit rate and traversal speed under randomized non-coherent diffuse bounce distributions.",
        "RayPayload" => "Ray traversal performance under heavy recursive register payload pressure.",
        "RayASBuild" => "Acceleration structure construction (BLAS/TLAS) and dynamic mesh refit throughput.",
        "RayProcedural" => "Intersection evaluation against mathematically defined procedural primitives (spheres, curves).",
        "RayMaterialDivergence" => "Shading dispatch throughput when secondary rays scatter across dissimilar materials.",
        "RayPathTracing" => "Full multi-bounce stochastic Monte Carlo path tracing with global illumination and cosine sampling.",
        "RayScheduling" | "RayExecutionParadigm" => "Comparative ray scheduling architectures: Traditional Megakernel vs Work Lists / DGC vs GPU Work Graphs vs Hardware SER.",
        _ => "GPU workstation benchmark suite.",
    }
}

pub fn is_benchmark_requested(t: &str, requested_tokens: &[String]) -> bool {
    let t_lower = t.to_lowercase();
    let t_norm: String = t_lower.chars().filter(|c| c.is_alphanumeric()).collect();

    for req in requested_tokens {
        let req_lower = req.to_lowercase();
        let req_norm: String = req_lower.chars().filter(|c| c.is_alphanumeric()).collect();
        if req_norm.is_empty() {
            continue;
        }

        // 1. "all" matches everything
        if req_norm == "all" {
            return true;
        }

        // 2. Group matches
        // Compute: FP64, FP32, FP16, BF16, FP8, FP4, INT8, INT4
        if (req_norm == "compute" || req_norm == "comp")
            && (t.starts_with("FP") || t.starts_with("BF") || t.starts_with("INT"))
        {
            return true;
        }

        // Memory: Device Memory Bandwidth, Cache
        if (req_norm == "memory" || req_norm == "mem" || req_norm == "cache" || req_norm == "vram")
            && (t.contains("Memory") || t.contains("Mem") || t.contains("Cache"))
            && !t.contains("System")
        {
            return true;
        }

        // Graphics: combines Raster and Ray Tracing
        if (req_norm == "graphics" || req_norm == "gfx")
            && (t.contains("Fill Rate") || t.starts_with("Ray"))
        {
            return true;
        }

        // Raster: Pixel Fill Rate
        if (req_norm == "raster" || req_norm == "rop" || req_norm == "rasterization")
            && t.contains("Fill Rate")
        {
            return true;
        }

        // Ray Tracing: All Ray* benchmarks
        if (req_norm == "raytracing" || req_norm == "rt" || req_norm == "ray" || req_norm == "ray_tracing")
            && t.starts_with("Ray")
        {
            return true;
        }

        // System: System Memory Bandwidth, Latency
        if (req_norm == "system" || req_norm == "sys" || req_norm == "host")
            && t.contains("System Memory")
        {
            return true;
        }

        // 3. Direct or normalized benchmark name match
        if t_norm == req_norm || t_lower == req_lower {
            return true;
        }

        // 4. Individual benchmark aliases
        let matched_alias = match req_norm.as_str() {
            "fp64" => t == "FP64",
            "fp32" => t == "FP32",
            "fp16" => t == "FP16",
            "bf16" => t == "BF16",
            "fp8" => t == "FP8",
            "fp4" => t == "FP4",
            "int8" => t == "INT8",
            "int4" => t == "INT4",
            "membw" | "bandwidth" | "devicememorybandwidth" => t == "Device Memory Bandwidth",
            "fillrate" | "pixelfill" | "pixelfillrate" => t == "Pixel Fill Rate",
            "sysmem" | "sysbw" | "systemmemorybandwidth" => t == "System Memory Bandwidth",
            "syslat" | "latency" | "systemmemorylatency" => t == "System Memory Latency",
            "asbuild" | "blas" | "tlas" | "rayasbuild" => t == "RayASBuild",
            "triangle" | "intersect" | "rayintersect" | "raytriangle" => {
                t == "RayIntersect" || t == "RayTracing"
            }
            "anyhit" | "rayanyhit" => t == "RayAnyHit",
            "procedural" | "rayprocedural" => t == "RayProcedural",
            "matdivergence" | "materialdivergence" | "raymaterialdivergence" => {
                t == "RayMaterialDivergence" || t == "RayDivergence"
            }
            "incoherent" | "rayincoherent" => t == "RayIncoherent",
            "divergence" | "raydivergence" => {
                t == "RayDivergence" || t == "RayMaterialDivergence"
            }
            "payload" | "raypayload" => t == "RayPayload",
            "pathtracing" | "raypathtracing" => t == "RayPathTracing",
            "rayscheduling" | "scheduling" | "worklists" | "dgc" | "rayexecutionparadigm" => {
                t == "RayScheduling" || t == "RayExecutionParadigm" || t.starts_with("RayScheduling")
            }
            _ => false,
        };

        if matched_alias {
            return true;
        }
    }

    false
}

// ============================================================================
// Kernel Discovery & Entry Point
// ============================================================================

fn kernel_path_candidates() -> Vec<std::path::PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join("../share/gpubench/kernels"));
        }
    }
    candidates.push(std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../kernels")));
    candidates
}

fn resolve_kernel_path() {
    if std::env::var_os("GPUBENCH_KERNEL_PATH").is_some() {
        return;
    }
    if let Some(candidate) = kernel_path_candidates().into_iter().find(|p| p.exists()) {
        unsafe {
            std::env::set_var("GPUBENCH_KERNEL_PATH", candidate);
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuiCliArgs {
    pub backend: Option<String>,
    pub groups: Vec<String>,
    pub benchmarks: Vec<String>,
    pub devices: Vec<String>,
    pub resolution: Option<String>,
    pub scene: Option<String>,
    pub dump_renders: bool,
    pub auto_start: bool,
    pub should_exit: bool,
}

impl Default for GuiCliArgs {
    fn default() -> Self {
        Self {
            backend: None,
            groups: Vec::new(),
            benchmarks: Vec::new(),
            devices: Vec::new(),
            resolution: None,
            scene: None,
            dump_renders: true,
            auto_start: false,
            should_exit: false,
        }
    }
}

fn print_gui_help() {
    println!("GPUBench GUI {}", env!("CARGO_PKG_VERSION"));
    println!("Cross-platform GPU Hardware Profiling & Ray Tracing Benchmark Suite\n");
    println!("USAGE:");
    println!("    gpubench-gui [OPTIONS]\n");
    println!("OPTIONS:");
    println!("    -k, --backend <BACKEND>       Backend to select: vulkan, opencl, rocm (default: auto)");
    println!("    -g, --group, --groups <NAMES> Benchmark group(s) to enable: compute, memory, graphics, raster, raytracing, system, all");
    println!("                                  (graphics combines raster and raytracing)");
    println!("    -b, --benchmark <BENCHMARKS>  Specific benchmark(s) to enable (comma-separated or multiple flags)");
    println!("    -d, --device <DEVICES>        Target GPU device index(es) (e.g. 0, 1, or 0,1)");
    println!("    -r, --resolution <RES>        Resolution preset: auto, 1080p, 1440p, 4k");
    println!("    -s, --scene <SCENE>           Ray tracing scene preset: all, indoor, forest, outdoor, showroom (default: all)");
    println!("        --dump-renders, --dump    Enable visual verification and render output dumping (default: enabled)");
    println!("        --no-dump-renders, --no-dump Disable visual verification and render output dumping");
    println!("        --auto-start, --run       Automatically start the benchmark suite immediately on launch");
    println!("    -h, --help                    Print help information and exit");
    println!("    -v, --version                 Print version information and exit\n");
    println!("BENCHMARK GROUPS & INCLUDED TESTS:");
    println!("    compute     Compute arithmetic units (vector & matrix tensor operations):");
    println!("                FP64, FP32, FP16, BF16, FP8, INT8, INT4\n");
    println!("    memory      VRAM and GPU cache hierarchy:");
    println!("                Device Memory Bandwidth, L0/L1/L2/L3 Cache Bandwidth & Latency\n");
    println!("    graphics    All 3D graphics rendering pipelines (combines 'raster' and 'raytracing', alias: 'gfx'):");
    println!("                Runs both fixed-function rasterization (ROP) and hardware ray tracing\n");
    println!("    raster      Fixed-function rasterization & ROP pixel fill rates (subset of graphics):");
    println!("                Pixel Fill Rate (RGBA8, RGBA16F HDR, Alpha Blending)\n");
    println!("    raytracing  Hardware BVH traversal, intersection & scheduling (subset of graphics, alias: 'rt'):");
    println!("                RayTracing, RayAnyHit, RayProcedural, RayIncoherent, RayMaterialDivergence,");
    println!("                RayPayload, RayASBuild, RayPathTracing, RayScheduling (Work Lists / SER / Work Graphs),");
    println!("                Ray Pipeline Breakdown (Linear vs 2D Tiled vs Morton Z-Curve, Queue Compaction)\n");
    println!("    system      Host CPU & RAM system memory:");
    println!("                System Memory Bandwidth (Multi & Single-Threaded), System Memory Latency\n");
    println!("    all         Enable all benchmark groups across target devices\n");
    println!("EXAMPLES:");
    println!("    gpubench-gui -k vulkan -g raytracing -d 1");
    println!("    gpubench-gui -g compute,memory,raster -r 1080p");
    println!("    gpubench-gui -b rayscheduling -s all --auto-start");
}

pub fn parse_gui_cli_args() -> GuiCliArgs {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    parse_gui_cli_args_from(&raw_args)
}

pub fn parse_gui_cli_args_from(raw_args: &[String]) -> GuiCliArgs {
    let mut args = GuiCliArgs::default();
    let mut i = 0;
    while i < raw_args.len() {
        let arg = &raw_args[i];
        if arg == "-h" || arg == "--help" {
            print_gui_help();
            args.should_exit = true;
            return args;
        }
        if arg == "-v" || arg == "--version" {
            println!("gpubench-gui {}", env!("CARGO_PKG_VERSION"));
            args.should_exit = true;
            return args;
        }
        if arg == "--no-dump-renders" || arg == "--no-dump" {
            args.dump_renders = false;
            i += 1;
            continue;
        }
        if arg == "--dump-renders" || arg == "--dump" {
            args.dump_renders = true;
            i += 1;
            continue;
        }
        if arg == "--auto-start" || arg == "--run" {
            args.auto_start = true;
            i += 1;
            continue;
        }
        if let Some(val) = arg.strip_prefix("-k=").or_else(|| arg.strip_prefix("--backend=")) {
            args.backend = Some(val.to_string());
            i += 1;
            continue;
        }
        if arg == "-k" || arg == "--backend" {
            if i + 1 < raw_args.len() {
                args.backend = Some(raw_args[i + 1].clone());
                i += 2;
                continue;
            }
        }
        if arg.starts_with("-k") && arg.len() > 2 && !arg.starts_with("--") {
            args.backend = Some(arg[2..].to_string());
            i += 1;
            continue;
        }
        if let Some(val) = arg.strip_prefix("-g=").or_else(|| arg.strip_prefix("--group=")).or_else(|| arg.strip_prefix("--groups=")) {
            args.groups.push(val.to_string());
            i += 1;
            continue;
        }
        if arg == "-g" || arg == "--group" || arg == "--groups" {
            if i + 1 < raw_args.len() {
                args.groups.push(raw_args[i + 1].clone());
                i += 2;
                continue;
            }
        }
        if arg.starts_with("-g") && arg.len() > 2 && !arg.starts_with("--") {
            args.groups.push(arg[2..].to_string());
            i += 1;
            continue;
        }
        if let Some(val) = arg.strip_prefix("-b=").or_else(|| arg.strip_prefix("--benchmark=")).or_else(|| arg.strip_prefix("--benchmarks=")) {
            args.benchmarks.push(val.to_string());
            i += 1;
            continue;
        }
        if arg == "-b" || arg == "--benchmark" || arg == "--benchmarks" {
            if i + 1 < raw_args.len() {
                args.benchmarks.push(raw_args[i + 1].clone());
                i += 2;
                continue;
            }
        }
        if arg.starts_with("-b") && arg.len() > 2 && !arg.starts_with("--") {
            args.benchmarks.push(arg[2..].to_string());
            i += 1;
            continue;
        }
        if let Some(val) = arg.strip_prefix("-d=").or_else(|| arg.strip_prefix("--device=")).or_else(|| arg.strip_prefix("--devices=")) {
            args.devices.push(val.to_string());
            i += 1;
            continue;
        }
        if arg == "-d" || arg == "--device" || arg == "--devices" {
            if i + 1 < raw_args.len() {
                args.devices.push(raw_args[i + 1].clone());
                i += 2;
                continue;
            }
        }
        if arg.starts_with("-d") && arg.len() > 2 && !arg.starts_with("--") {
            args.devices.push(arg[2..].to_string());
            i += 1;
            continue;
        }
        if let Some(val) = arg.strip_prefix("-r=").or_else(|| arg.strip_prefix("--resolution=")) {
            args.resolution = Some(val.to_string());
            i += 1;
            continue;
        }
        if arg == "-r" || arg == "--resolution" {
            if i + 1 < raw_args.len() {
                args.resolution = Some(raw_args[i + 1].clone());
                i += 2;
                continue;
            }
        }
        if arg.starts_with("-r") && arg.len() > 2 && !arg.starts_with("--") {
            args.resolution = Some(arg[2..].to_string());
            i += 1;
            continue;
        }
        if let Some(val) = arg.strip_prefix("-s=").or_else(|| arg.strip_prefix("--scene=")) {
            args.scene = Some(val.to_string());
            i += 1;
            continue;
        }
        if arg == "-s" || arg == "--scene" {
            if i + 1 < raw_args.len() {
                args.scene = Some(raw_args[i + 1].clone());
                i += 2;
                continue;
            }
        }
        if arg.starts_with("-s") && arg.len() > 2 && !arg.starts_with("--") {
            args.scene = Some(arg[2..].to_string());
            i += 1;
            continue;
        }
        i += 1;
    }
    args
}

pub fn main() -> iced::Result {
    let cli_args = parse_gui_cli_args();
    if cli_args.should_exit {
        return Ok(());
    }

    if std::env::var_os("MESA_VK_IGNORE_CONFORMANCE_WARNING").is_none() {
        unsafe { std::env::set_var("MESA_VK_IGNORE_CONFORMANCE_WARNING", "1") };
    }
    resolve_kernel_path();
    let icon_data = include_bytes!("../../packaging/linux/icons/hicolor/256x256/apps/io.github.soddentrough.gpubench.png");
    let app_icon = image::load_from_memory(icon_data)
        .ok()
        .and_then(|img| {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            iced::window::icon::from_rgba(rgba.into_raw(), w, h).ok()
        });

    GPUBenchApp::run(Settings {
        flags: cli_args,
        antialiasing: true,
        window: iced::window::Settings {
            size: iced::Size::new(1280.0, 980.0),
            min_size: Some(iced::Size::new(960.0, 640.0)),
            icon: app_icon,
            #[cfg(target_os = "linux")]
            platform_specific: iced::window::settings::PlatformSpecific {
                application_id: String::from("io.github.soddentrough.gpubench"),
            },
            ..Default::default()
        },
        ..Settings::default()
    })
}

static PROGRESS_SENDER: LazyLock<Mutex<Option<Sender<ResultData>>>> = LazyLock::new(|| Mutex::new(None));

fn log_diagnostic(msg: &str) {
    use std::io::Write;
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let line = format!("[{}.{:03}] {}\n", secs, millis, msg);
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("gpubench.log") {
        let _ = f.write_all(line.as_bytes());
        let _ = f.flush();
    }
}

fn progress_callback(res: &ResultData) {
    if res.time_ms < 0.0 {
        log_diagnostic(&format!("[START] dev={} bench='{}' comp='{}'", res.deviceIndex, res.benchmarkName, res.component));
    } else {
        log_diagnostic(&format!("[FINISH] dev={} bench='{}' comp='{}' ops={} time_ms={:.3} metric='{}' unsupported={}",
            res.deviceIndex, res.benchmarkName, res.component, res.operations, res.time_ms, res.metric, res.isUnsupported));
    }
    if let Ok(guard) = PROGRESS_SENDER.lock() {
        if let Some(sender) = guard.as_ref() {
            let _ = sender.send(res.clone());
        }
    }
}

fn clean_device_name(raw: &str) -> String {
    let name = raw.to_string();
    let mut cleaned = String::new();
    let mut depth = 0i32;
    for c in name.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ if depth == 0 => cleaned.push(c),
            _ => {}
        }
    }
    cleaned.trim().to_string()
}

fn available_backends(hw: &[String]) -> Vec<String> {
    let mut backends: Vec<String> = Vec::new();
    for h in hw {
        let parts: Vec<&str> = h.split('|').collect();
        if parts.len() == 3 {
            let api = parts[0].to_uppercase();
            if api != "SYSTEM" && !backends.contains(&api) {
                backends.push(api);
            }
        }
    }
    if backends.is_empty() {
        backends.push("VULKAN".to_string());
    }
    backends
}

fn devices_for_backend(hw: &[String], selected_backend: &str) -> Vec<String> {
    let mut devices = Vec::new();
    devices.push("System: Host CPU & RAM".to_string());
    for h in hw {
        let parts: Vec<&str> = h.split('|').collect();
        if parts.len() == 3 && parts[0].eq_ignore_ascii_case(selected_backend) {
            let cleaned = clean_device_name(parts[2]);
            devices.push(format!("{}: {}", parts[1], cleaned));
        }
    }
    devices
}

fn match_device(dev_str: &str, tok: &str) -> bool {
    let tok_clean = tok.trim();
    if tok_clean.is_empty() {
        return false;
    }
    if tok_clean.eq_ignore_ascii_case("all") {
        return true;
    }
    if dev_str.eq_ignore_ascii_case(tok_clean) {
        return true;
    }

    // Parse device index prefix from "0: Name", "1: Name"
    let dev_idx_opt = if dev_str.starts_with("System") {
        Some(SYSTEM_DEVICE_ID)
    } else {
        dev_str.split(':').next().and_then(|s| s.trim().parse::<u32>().ok())
    };

    // 1. Direct numeric match: e.g. "1" or "0" or "999"
    if let Ok(num) = tok_clean.parse::<u32>() {
        if let Some(dev_idx) = dev_idx_opt {
            return dev_idx == num;
        }
    }

    let tok_lower = tok_clean.to_lowercase();

    // 2. "gpu 1", "gpu1", "gpu:1", "device 1", "device1", "d1"
    if let Some(rest) = tok_lower.strip_prefix("gpu")
        .or_else(|| tok_lower.strip_prefix("device"))
        .or_else(|| tok_lower.strip_prefix('d')) 
    {
        let num_str = rest.trim_start_matches(|c: char| c == ' ' || c == ':' || c == '-' || c == '_');
        if let Ok(num) = num_str.parse::<u32>() {
            if let Some(dev_idx) = dev_idx_opt {
                return dev_idx == num;
            }
        }
    }

    // 3. "system", "cpu", "host", "ram"
    if tok_lower == "system" || tok_lower == "cpu" || tok_lower == "host" || tok_lower == "ram" {
        return dev_str.starts_with("System");
    }

    // 4. Substring match on device name (e.g. "r9700", "radeon")
    if dev_str.to_lowercase().contains(&tok_lower) {
        return true;
    }

    false
}

pub const SYSTEM_DEVICE_ID: u32 = 999;

#[derive(Clone, Debug, Default)]
pub struct CellResult {
    pub value_str: String,
    pub numeric: f64,
    pub unit: String,
    pub is_running: bool,
    pub is_unsupported: bool,
    pub support_note: String,
    pub support_category: String,
    pub raw_operations: u64,
    pub raw_time_ms: f64,
}

#[derive(Clone, Debug)]
pub struct WorkloadDef {
    pub id: &'static str,
    pub category: &'static str,
    pub label: &'static str,
    pub approach: &'static str,
    pub default_unit: &'static str,
    pub desc: &'static str,
    pub api_extensions: &'static str,
    pub is_system: bool,
}

pub static WORKLOADS: &[WorkloadDef] = &[
    // COMPUTE PIPELINES
    WorkloadDef {
        id: "fp64",
        category: "COMPUTE PIPELINES",
        label: "FP64 (Double Precision)",
        approach: "64-bit IEEE vector FMA",
        default_unit: "TFLOPS",
        desc: "Double-precision 64-bit floating point compute throughput.",
        api_extensions: "Vulkan / OpenCL / ROCm Core Compute (Float64)",
        is_system: false,
    },
    WorkloadDef {
        id: "fp32",
        category: "COMPUTE PIPELINES",
        label: "FP32 (Single Precision)",
        approach: "32-bit vector FMA",
        default_unit: "TFLOPS",
        desc: "Single-precision 32-bit float compute throughput, standard for graphics and compute shaders.",
        api_extensions: "Vulkan / OpenCL / ROCm Core Compute (Float32)",
        is_system: false,
    },
    WorkloadDef {
        id: "fp16_vec",
        category: "COMPUTE PIPELINES",
        label: "FP16 (Vector)",
        approach: "16-bit packed SIMD",
        default_unit: "TFLOPS",
        desc: "Half-precision 16-bit float SIMD vector throughput.",
        api_extensions: "VK_KHR_shader_float16_int8 / cl_khr_fp16",
        is_system: false,
    },
    WorkloadDef {
        id: "fp16_mat",
        category: "COMPUTE PIPELINES",
        label: "FP16 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TFLOPS",
        desc: "Hardware matrix cores half-precision tensor compute throughput using VK_KHR_cooperative_matrix.",
        api_extensions: "VK_KHR_cooperative_matrix / ROCm WMMA",
        is_system: false,
    },
    WorkloadDef {
        id: "bf16_vec",
        category: "COMPUTE PIPELINES",
        label: "BF16 (Vector)",
        approach: "Bfloat16 SIMD",
        default_unit: "TFLOPS",
        desc: "16-bit Brain Floating Point vector throughput for machine learning models.",
        api_extensions: "VK_KHR_shader_float_controls2 / ROCm BF16",
        is_system: false,
    },
    WorkloadDef {
        id: "bf16_mat",
        category: "COMPUTE PIPELINES",
        label: "BF16 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TFLOPS",
        desc: "Hardware matrix cores Bfloat16 tensor compute throughput.",
        api_extensions: "VK_KHR_cooperative_matrix / ROCm WMMA",
        is_system: false,
    },
    WorkloadDef {
        id: "fp8_vec",
        category: "COMPUTE PIPELINES",
        label: "FP8 (Vector)",
        approach: "8-bit float vector",
        default_unit: "TFLOPS",
        desc: "Quarter-precision 8-bit float arithmetic (E4M3/E5M2) for modern low-bit AI inference.",
        api_extensions: "VK_EXT_shader_float8 / ROCm FP8",
        is_system: false,
    },
    WorkloadDef {
        id: "fp8_mat",
        category: "COMPUTE PIPELINES",
        label: "FP8 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TFLOPS",
        desc: "Hardware 8-bit float matrix multiplication throughput.",
        api_extensions: "VK_KHR_cooperative_matrix / ROCm WMMA",
        is_system: false,
    },
    WorkloadDef {
        id: "int8_vec",
        category: "COMPUTE PIPELINES",
        label: "INT8 (Vector)",
        approach: "8-bit DP4A integer",
        default_unit: "TOPS",
        desc: "8-bit quantized integer vector throughput.",
        api_extensions: "VK_KHR_shader_integer_dot_product (DP4A)",
        is_system: false,
    },
    WorkloadDef {
        id: "int8_mat",
        category: "COMPUTE PIPELINES",
        label: "INT8 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TOPS",
        desc: "Hardware matrix 8-bit integer tensor throughput.",
        api_extensions: "VK_KHR_cooperative_matrix / ROCm WMMA",
        is_system: false,
    },
    WorkloadDef {
        id: "int4_vec",
        category: "COMPUTE PIPELINES",
        label: "INT4 (Vector)",
        approach: "4-bit packed integer",
        default_unit: "TOPS",
        desc: "4-bit quantized vector arithmetic.",
        api_extensions: "Vulkan Subgroup Bit Packing (4-bit INT)",
        is_system: false,
    },
    WorkloadDef {
        id: "int4_mat",
        category: "COMPUTE PIPELINES",
        label: "INT4 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TOPS",
        desc: "Hardware matrix 4-bit integer throughput.",
        api_extensions: "VK_KHR_cooperative_matrix (Sub-byte INT4)",
        is_system: false,
    },

    // MEMORY & SYSTEM
    WorkloadDef {
        id: "gpu_vram_bw",
        category: "MEMORY & SYSTEM",
        label: "GPU VRAM Bandwidth",
        approach: "Coalesced device stream",
        default_unit: "GB/s",
        desc: "Peak memory read/write streaming bandwidth across dedicated GPU VRAM bus.",
        api_extensions: "Vulkan / OpenCL / ROCm Linear Buffer DMA",
        is_system: false,
    },
    WorkloadDef {
        id: "cache_l0",
        category: "MEMORY & SYSTEM",
        label: "L0 Cache Latency",
        approach: "L0 / LDS pointer chase",
        default_unit: "ns",
        desc: "Level 0 vector cache access latency in nanoseconds (lower is better).",
        api_extensions: "Compute Shared Memory (LDS) Pointer Chase",
        is_system: false,
    },
    WorkloadDef {
        id: "cache_l1",
        category: "MEMORY & SYSTEM",
        label: "L1 Cache Latency",
        approach: "GL1C pointer chase",
        default_unit: "ns",
        desc: "Level 1 cache access latency in nanoseconds (lower is better).",
        api_extensions: "Hardware Global L1 Cache Pointer Chase",
        is_system: false,
    },
    WorkloadDef {
        id: "cache_l2",
        category: "MEMORY & SYSTEM",
        label: "L2 Cache Latency",
        approach: "GL2C pointer chase",
        default_unit: "ns",
        desc: "Level 2 cache access latency in nanoseconds (lower is better).",
        api_extensions: "Hardware Global L2 Cache Pointer Chase",
        is_system: false,
    },
    WorkloadDef {
        id: "cache_l3",
        category: "MEMORY & SYSTEM",
        label: "L3 Cache Latency",
        approach: "MALL / Infinity Cache chase",
        default_unit: "ns",
        desc: "Level 3 / Infinity Cache access latency in nanoseconds (lower is better).",
        api_extensions: "Memory Attached Last Level (MALL) Chase",
        is_system: false,
    },
    WorkloadDef {
        id: "sys_mem_bw_multi",
        category: "MEMORY & SYSTEM",
        label: "System RAM (Multi-Thread)",
        approach: "AVX2 multi-thread stream",
        default_unit: "GB/s",
        desc: "Multi-threaded host RAM copy and streaming bandwidth from CPU to system DDR memory.",
        api_extensions: "x86_64 AVX2 / Non-Temporal Streaming Stores",
        is_system: true,
    },
    WorkloadDef {
        id: "sys_mem_bw_single",
        category: "MEMORY & SYSTEM",
        label: "System RAM (1 Thread)",
        approach: "AVX2 single-thread stream",
        default_unit: "GB/s",
        desc: "Single-threaded host system RAM streaming bandwidth.",
        api_extensions: "x86_64 AVX2 Single-Thread Streaming",
        is_system: true,
    },
    WorkloadDef {
        id: "sys_mem_lat",
        category: "MEMORY & SYSTEM",
        label: "System RAM Latency",
        approach: "CPU pointer chase",
        default_unit: "ns",
        desc: "Host memory access latency in nanoseconds (lower is better).",
        api_extensions: "Hardware DRAM Pointer Chase",
        is_system: true,
    },

    // RASTERIZATION & ROP
    WorkloadDef {
        id: "rop_rgba8",
        category: "RASTERIZATION & ROP",
        label: "RGBA8 Color Fill",
        approach: "Hardware 32-bit ROP",
        default_unit: "GPixels/s",
        desc: "Standard 32-bit RGBA ROP rasterization fill rate.",
        api_extensions: "Vulkan Graphics Pipeline (VK_FORMAT_R8G8B8A8_UNORM)",
        is_system: false,
    },
    WorkloadDef {
        id: "rop_rgba16f",
        category: "RASTERIZATION & ROP",
        label: "RGBA16F HDR Fill",
        approach: "Hardware 64-bit HDR ROP",
        default_unit: "GPixels/s",
        desc: "64-bit HDR framebuffer rasterization throughput.",
        api_extensions: "Vulkan Graphics Pipeline (VK_FORMAT_R16G16B16A16_SFLOAT)",
        is_system: false,
    },
    WorkloadDef {
        id: "rop_blend",
        category: "RASTERIZATION & ROP",
        label: "Alpha Blending Fill",
        approach: "Hardware ROP blending",
        default_unit: "GPixels/s",
        desc: "ROP hardware alpha blend rasterization rate.",
        api_extensions: "Vulkan Blend Operations (SrcAlpha, OneMinusSrcAlpha)",
        is_system: false,
    },

    // RAY TRACING ACCELERATION
    // Phase 1: Acceleration Structure Creation & Updates
    WorkloadDef {
        id: "rt_blas_build_1m",
        category: "RAY TRACING ACCELERATION",
        label: "BLAS Build (1M Tris)",
        approach: "Standard game mesh AS build",
        default_unit: "MTris/s",
        desc: "Bottom-level AS construction for standard game assets (1M triangles).",
        api_extensions: "VK_KHR_acceleration_structure",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_blas_update_1m",
        category: "RAY TRACING ACCELERATION",
        label: "BLAS Update (1M Tris)",
        approach: "Dynamic mesh BVH refit",
        default_unit: "MTris/s",
        desc: "Dynamic in-place vertex update refit rate (1M triangles).",
        api_extensions: "VK_KHR_acceleration_structure (VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_blas_build_5m",
        category: "RAY TRACING ACCELERATION",
        label: "BLAS Build (5M Tris)",
        approach: "Heavy mesh / L3 cache spill",
        default_unit: "MTris/s",
        desc: "AS construction exceeding L3 Infinity Cache, stressing memory controllers (5M triangles).",
        api_extensions: "VK_KHR_acceleration_structure",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_blas_update_5m",
        category: "RAY TRACING ACCELERATION",
        label: "BLAS Update (5M Tris)",
        approach: "Dynamic mesh refit (5M)",
        default_unit: "MTris/s",
        desc: "Dynamic in-place vertex update refit rate for heavy 5M triangle geometry.",
        api_extensions: "VK_KHR_acceleration_structure (VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_blas_build_10m",
        category: "RAY TRACING ACCELERATION",
        label: "BLAS Build (10M Tris)",
        approach: "Massive hero mesh AS build",
        default_unit: "MTris/s",
        desc: "Production-scale high-density mesh construction throughput (10M triangles).",
        api_extensions: "VK_KHR_acceleration_structure",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_tlas_indoor",
        category: "RAY TRACING ACCELERATION",
        label: "TLAS: Indoor Corridor (20k Inst)",
        approach: "Room & Hallway Hierarchy (5k Meshes)",
        default_unit: "MInst/s",
        desc: "Clustered room/hallway hierarchy with high mesh diversity (~1:4 instancing ratio).",
        api_extensions: "VK_KHR_acceleration_structure (Top-Level)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_tlas_jungle",
        category: "RAY TRACING ACCELERATION",
        label: "TLAS: Dense Jungle (50k Inst)",
        approach: "High-Overlap Foliage (500 Meshes)",
        default_unit: "MInst/s",
        desc: "Dense overlapping foliage canopy on undulating terrain (~1:100 instancing ratio).",
        api_extensions: "VK_KHR_acceleration_structure (Top-Level)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_tlas_openworld",
        category: "RAY TRACING ACCELERATION",
        label: "TLAS: Open World (200k Inst)",
        approach: "Multi-Scale Geographic (5k Meshes)",
        default_unit: "MInst/s",
        desc: "Vast multi-scale landscape hierarchy (terrain sectors, urban blocks, micro-props).",
        api_extensions: "VK_KHR_acceleration_structure (Top-Level)",
        is_system: false,
    },

    // Phase 2: Total Scene Render (Direct Visibility & Materials)
    WorkloadDef {
        id: "rt_sched_prim_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Total Scene Render (Megakernel)",
        approach: "Monolithic Megakernel",
        default_unit: "MRays/s",
        desc: "Complete end-to-end scene frame generation: camera ray generation, BVH traversal, multi-material BSDF shading, and framebuffer write.",
        api_extensions: "VK_KHR_ray_query (Vulkan Compute)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_prim_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Total Scene Render (Work Lists)",
        approach: "Subgroup Compaction + Specialized Micro-Kernels",
        default_unit: "MRays/s",
        desc: "Complete end-to-end scene frame generation using Work Lists / DGC: camera ray generation, wave compaction into material queues, and indirect dispatch of specialized BSDF shaders.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },

    // Phase 2b: Directional Shadows & Multi-Light Coherence
    WorkloadDef {
        id: "rt_sched_shadow_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Ray-Traced Shadows (Megakernel)",
        approach: "Monolithic In-Kernel Traversal",
        default_unit: "MRays/s",
        desc: "Directional shadow visibility testing executed directly within monolithic megakernel with early termination flags, exhibiting wave divergence stalls across mixed hit/miss lanes.",
        api_extensions: "VK_KHR_ray_query (gl_RayFlagsTerminateOnFirstHitEXT)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_shadow_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Ray-Traced Shadows (Work Lists)",
        approach: "Wavefront Compaction + Micro-Kernel",
        default_unit: "MRays/s",
        desc: "Directional shadow ray testing via Work Lists / DGC: surface hits ballot-compacted into dense queues and dispatched via specialized shadow micro-kernel at 100% active SIMD lane density.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_shadow_bin",
        category: "RAY TRACING ACCELERATION",
        label: "Ray-Traced Shadows (Directional Binning)",
        approach: "Multi-Light Coherent Binning",
        default_unit: "MRays/s",
        desc: "Directional shadow testing across multiple scene lights with spatial queue binning, clustering coherent shadow cones into dedicated dispatches.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_triangle",
        category: "RAY TRACING ACCELERATION",
        label: "Ray-Triangle Intersect",
        approach: "Hardware BVH Ray Query",
        default_unit: "GIS/s",
        desc: "Peak BVH acceleration structure traversal and ray-triangle intersection throughput.",
        api_extensions: "VK_KHR_ray_query (Ray-Triangle Intersection Engine)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_anyhit",
        category: "RAY TRACING ACCELERATION",
        label: "AnyHit (Alpha-Tested)",
        approach: "AnyHit opacity eval",
        default_unit: "GRays/s",
        desc: "Traversal and shader invocation throughput against transparent, alpha-tested geometry.",
        api_extensions: "VK_KHR_ray_query (Custom AnyHit Opacity Shader)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_procedural",
        category: "RAY TRACING ACCELERATION",
        label: "Procedural Geometry",
        approach: "Analytical AABB eval",
        default_unit: "GRays/s",
        desc: "Intersection evaluation against mathematically defined procedural primitives (spheres).",
        api_extensions: "VK_KHR_ray_query (AABB Intersection Traversal)",
        is_system: false,
    },

    // Phase 3: Material Shading & Divergence
    WorkloadDef {
        id: "rt_sched_mat_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Material Shading (Traditional)",
        approach: "Monolithic Megakernel",
        default_unit: "MHits/s",
        desc: "Traditional monolithic compute shader with dynamic loop branching across heterogeneous materials.",
        api_extensions: "VK_KHR_ray_query (Dynamic Loop Branching)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_mat_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Material Shading (Work Lists)",
        approach: "Using ExecuteIndirect (Work Lists)",
        default_unit: "MHits/s",
        desc: "Using ExecuteIndirect (Work Lists): 2-pass parallel compaction into uniform material queues to eliminate branch divergence.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },

    // Phase 4: Secondary Rays & Traversal Divergence
    WorkloadDef {
        id: "rt_sched_incoh_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Incoherent Rays (Traditional)",
        approach: "Monolithic Megakernel",
        default_unit: "MRays/s",
        desc: "Traditional unordered traversal of highly divergent secondary rays.",
        api_extensions: "VK_KHR_ray_query (Unordered BVH Traversal)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_incoh_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Incoherent Rays (Work Lists)",
        approach: "Directional Binning",
        default_unit: "MRays/s",
        desc: "Using ExecuteIndirect (Work Lists): Sorts scattered secondary rays into directional bins before BVH traversal.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands (Octant Binning)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_incoherent",
        category: "RAY TRACING ACCELERATION",
        label: "Incoherent Bounces",
        approach: "Cosine diffuse bounce",
        default_unit: "GRays/s",
        desc: "Cache hit rate and traversal speed under randomized non-coherent diffuse bounce distributions.",
        api_extensions: "VK_KHR_ray_query (Cosine Weighted Sampling)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_divergence",
        category: "RAY TRACING ACCELERATION",
        label: "Divergence Traversal",
        approach: "Coherence gradient rays",
        default_unit: "GRays/s",
        desc: "BVH traversal throughput under heavy branch and wave execution divergence.",
        api_extensions: "VK_KHR_ray_query (Wavefront Divergence)",
        is_system: false,
    },

    // Phase 5: Multi-Bounce Path Tracing
    WorkloadDef {
        id: "rt_sched_pt_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Path Tracing (Traditional)",
        approach: "Monolithic Megakernel",
        default_unit: "MRays/s",
        desc: "Traditional megakernel path tracing where terminated rays leave SIMD lanes idle across multiple bounces.",
        api_extensions: "VK_KHR_ray_query (Monte Carlo Megakernel)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_pt_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Path Tracing (Work Lists)",
        approach: "Active Ray Compaction",
        default_unit: "MRays/s",
        desc: "Using ExecuteIndirect (Work Lists): Compacts non-terminated bounce rays into packed queues, keeping wavefronts 100% full.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands (Ray Compaction)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_pathtracing",
        category: "RAY TRACING ACCELERATION",
        label: "Multi-Bounce Path Tracing",
        approach: "Stochastic 8-bounce GI",
        default_unit: "MRays/s",
        desc: "Full multi-bounce stochastic Monte Carlo path tracing with global illumination and cosine sampling.",
        api_extensions: "VK_KHR_ray_query (8-Bounce Global Illumination)",
        is_system: false,
    },

    // Phase 6: Architectural Stress & Advanced Features
    WorkloadDef {
        id: "rt_payload",
        category: "RAY TRACING ACCELERATION",
        label: "Payload Pressure",
        approach: "16B - 256B register payload",
        default_unit: "GRays/s",
        desc: "Ray traversal performance under heavy recursive register payload pressure.",
        api_extensions: "VK_KHR_ray_query (Spill-to-Scratch Register Pressure)",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_ser",
        category: "RAY TRACING ACCELERATION",
        label: "Hardware Reordering (SER)",
        approach: "Shader Execution Reordering (SER)",
        default_unit: "MRays/s",
        desc: "Hardware-level dynamic thread reordering during traversal (VK_EXT_ray_tracing_invocation_reorder).",
        api_extensions: "VK_EXT_ray_tracing_invocation_reorder",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_workgraph",
        category: "RAY TRACING ACCELERATION",
        label: "GPU Work Graphs",
        approach: "Autonomous Node Enqueue",
        default_unit: "MRays/s",
        desc: "Autonomous GPU-driven work creation via node record routing (VK_AMDX_shader_enqueue).",
        api_extensions: "VK_AMDX_shader_enqueue",
        is_system: false,
    },

    // Phase 7: Ray Pipeline Breakdown
    WorkloadDef {
        id: "rt_sched_stage_bvh_linear",
        category: "RAY PIPELINE BREAKDOWN",
        label: "BVH Traversal (Linear 1D)",
        approach: "Linear 1D Scanline Traversal",
        default_unit: "MRays/s",
        desc: "Evaluates raw BVH tree traversal without spatial coherency optimizations using linear 1D scanline wavefront dispatch.",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_stage_queue_compaction",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Queue Compaction Overhead",
        approach: "Wave Ballot Stream Sort",
        default_unit: "MRecords/s",
        desc: "Measures subgroup ballot prefix-sum stream compaction overhead for re-packing active rays into material work lists.",
        api_extensions: "VK_KHR_shader_subgroup_ballot",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_stage_bvh_tiled",
        category: "RAY PIPELINE BREAKDOWN",
        label: "BVH Traversal (2D Tiled 8x4)",
        approach: "2D Screen Tiled (8x4)",
        default_unit: "MRays/s",
        desc: "Evaluates BVH traversal cache locality and SIMD coherence using 2D screen-space (8x4) rectangular tile ray dispatch.",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_stage_bvh_morton8x4",
        category: "RAY PIPELINE BREAKDOWN",
        label: "BVH Traversal (Morton 8x4)",
        approach: "2D Morton Z-Curve (8x4)",
        default_unit: "MRays/s",
        desc: "Maximizes BVH node traversal cache reuse by dispatching wavefronts along space-filling Morton Z-order curves (8x4).",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_stage_bvh_morton4x8",
        category: "RAY PIPELINE BREAKDOWN",
        label: "BVH Traversal (Morton 4x8)",
        approach: "2D Morton Z-Curve (4x8)",
        default_unit: "MRays/s",
        desc: "BVH traversal cache locality evaluating tall Morton Z-curve (4x8) quad layouts across GPU wave frontlines.",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_full_showroom_trad",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Megakernel (Showroom)",
        approach: "Morton 8x4 + Megakernel (109k Tris)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render of Showroom Studio (109k Tris) combining 2D Morton spatial ray ordering with monolithic megakernel shading.",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_full_showroom_wl",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Work Lists (Showroom)",
        approach: "Morton 8x4 + Work Lists / DGC (109k Tris)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render of Showroom Studio (109k Tris) combining 2D Morton ray ordering with wavefront stream compaction and indirect dispatch.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_full_indoor_trad",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Megakernel (Indoor)",
        approach: "Morton 8x4 + Megakernel (262k Tris)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render of Indoor Atrium (262k Tris) combining 2D Morton spatial ray ordering with monolithic megakernel shading.",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_full_indoor_wl",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Work Lists (Indoor)",
        approach: "Morton 8x4 + Work Lists / DGC (262k Tris)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render of Indoor Atrium (262k Tris) combining 2D Morton ray ordering with wavefront stream compaction and indirect dispatch.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_full_outdoor_trad",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Megakernel (Outdoor)",
        approach: "Morton 8x4 + Megakernel (57k Tris)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render of Outdoor Landscape (57k Tris) combining 2D Morton spatial ray ordering with monolithic megakernel shading.",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_full_outdoor_wl",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Work Lists (Outdoor)",
        approach: "Morton 8x4 + Work Lists / DGC (57k Tris)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render of Outdoor Landscape (57k Tris) combining 2D Morton ray ordering with wavefront stream compaction and indirect dispatch.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_full_forest_trad",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Megakernel (Forest)",
        approach: "Morton 8x4 + Megakernel (1.0M Tris)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render of Open-World Forest (1.0M Tris) combining 2D Morton spatial ray ordering with monolithic megakernel shading.",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_full_forest_wl",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Work Lists (Forest)",
        approach: "Morton 8x4 + Work Lists / DGC (1.0M Tris)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render of Open-World Forest (1.0M Tris) combining 2D Morton ray ordering with wavefront stream compaction and indirect dispatch.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_stage_prim_morton_trad",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Megakernel",
        approach: "Morton 8x4 + Megakernel",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render combining 2D Morton spatial ray ordering with traditional monolithic megakernel shading.",
        api_extensions: "VK_KHR_ray_query",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_stage_prim_morton_wl",
        category: "RAY PIPELINE BREAKDOWN",
        label: "Full Scene Render: Work Lists",
        approach: "Morton 8x4 + Work Lists (DGC)",
        default_unit: "MRays/s",
        desc: "End-to-end full scene render combining 2D Morton ray ordering with wavefront stream compaction and indirect dispatch.",
        api_extensions: "VK_KHR_ray_query, VK_EXT_device_generated_commands",
        is_system: false,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionPreset {
    Fhd1080p,
    Qhd1440p,
    Uhd4k,
}

impl ResolutionPreset {
    pub const ALL: [ResolutionPreset; 3] = [
        ResolutionPreset::Fhd1080p,
        ResolutionPreset::Qhd1440p,
        ResolutionPreset::Uhd4k,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ResolutionPreset::Fhd1080p => "1080p (FHD)",
            ResolutionPreset::Qhd1440p => "1440p (QHD)",
            ResolutionPreset::Uhd4k => "4K (UHD)",
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            ResolutionPreset::Fhd1080p => (1920, 1080),
            ResolutionPreset::Qhd1440p => (2560, 1440),
            ResolutionPreset::Uhd4k => (3840, 2160),
        }
    }

    pub fn detect_from_devices(devices: &[DeviceTelemetry], initial_devices: &HashSet<String>) -> Self {
        let target_gpu = devices.iter().find(|d| {
            if !d.is_gpu { return false; }
            initial_devices.iter().any(|id| {
                if let Some(idx_str) = id.split(':').next() {
                    if let Ok(idx) = idx_str.trim().parse::<u32>() {
                        return d.id == format!("GPU {}", idx);
                    }
                }
                id.contains(&d.id) || id.contains(&d.name)
            })
        }).or_else(|| {
            devices.iter().find(|d| d.is_gpu && d.id.contains("1"))
                .or_else(|| devices.iter().find(|d| d.is_gpu))
        });

        if let Some(gpu) = target_gpu {
            if gpu.vram_total >= 15000 {
                ResolutionPreset::Uhd4k
            } else if gpu.vram_total >= 9000 {
                ResolutionPreset::Qhd1440p
            } else {
                ResolutionPreset::Fhd1080p
            }
        } else {
            ResolutionPreset::Uhd4k
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePreset {
    All,
    Indoor,
    Forest,
    Outdoor,
    Showroom,
}

impl ScenePreset {
    pub const ALL: [ScenePreset; 5] = [
        ScenePreset::All,
        ScenePreset::Indoor,
        ScenePreset::Forest,
        ScenePreset::Outdoor,
        ScenePreset::Showroom,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ScenePreset::All => "All (4 Scenes)",
            ScenePreset::Indoor => "Indoor Atrium (262k Tris)",
            ScenePreset::Forest => "Open-World Forest (1.0M Tris)",
            ScenePreset::Outdoor => "Outdoor Landscape (57k Tris)",
            ScenePreset::Showroom => "Showroom Studio (109k Tris)",
        }
    }

    pub fn id_str(&self) -> &'static str {
        match self {
            ScenePreset::All => "all",
            ScenePreset::Indoor => "indoor",
            ScenePreset::Forest => "forest",
            ScenePreset::Outdoor => "outdoor",
            ScenePreset::Showroom => "showroom",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "all" => ScenePreset::All,
            "forest" | "aaa_forest" => ScenePreset::Forest,
            "outdoor" => ScenePreset::Outdoor,
            "showroom" => ScenePreset::Showroom,
            _ => ScenePreset::Indoor,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize, Default)]
struct ParityTraditional {
    #[serde(default)]
    fps: f64,
    #[serde(default)]
    mrays: f64,
    #[serde(default)]
    frame_ms: f64,
    #[serde(default)]
    bvh_ms: f64,
    #[serde(default)]
    bvh_pct: f64,
    #[serde(default)]
    shading_ms: f64,
    #[serde(default)]
    shading_pct: f64,
    #[serde(default)]
    shading_mhits: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize, Default)]
struct ParityWorklist {
    #[serde(default)]
    fps: f64,
    #[serde(default)]
    mrays: f64,
    #[serde(default)]
    frame_ms: f64,
    #[serde(default)]
    bvh_ms: f64,
    #[serde(default)]
    bvh_pct: f64,
    #[serde(default)]
    compaction_ms: f64,
    #[serde(default)]
    compaction_pct: f64,
    #[serde(default)]
    compaction_mrecords: f64,
    #[serde(default)]
    shading_ms: f64,
    #[serde(default)]
    shading_pct: f64,
    #[serde(default)]
    shading_mhits: f64,
    #[serde(default)]
    shading_speedup: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize, Default)]
struct ParityMetrics {
    #[serde(default)]
    psnr: f64,
    #[serde(default)]
    mae: f64,
    #[serde(default)]
    rmse: f64,
    #[serde(default)]
    exact_pixels: u64,
    #[serde(default)]
    exact_pct: f64,
    #[serde(default)]
    near_exact_pixels: u64,
    #[serde(default)]
    near_exact_pct: f64,
    #[serde(default)]
    diff_pixels: u64,
    #[serde(default)]
    diff_pct: f64,
    #[serde(default)]
    status: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize, Default)]
struct ParityProfile {
    #[serde(default)]
    gpu: String,
    #[serde(default)]
    scene: String,
    #[serde(default)]
    resolution: String,
    #[serde(default)]
    traditional: ParityTraditional,
    #[serde(default)]
    worklist: ParityWorklist,
    #[serde(default)]
    parity: ParityMetrics,
}

fn find_renders_file(relative: &str) -> Option<String> {
    let mut candidates = vec![
        relative.to_string(),
        format!("../{}", relative),
        format!("../../{}", relative),
        format!("/home/naoki/Development/GPUBench/{}", relative),
        format!("/usr/share/gpubench/{}", relative),
    ];
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(relative).to_string_lossy().to_string());
        candidates.push(cwd.join("..").join(relative).to_string_lossy().to_string());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            candidates.push(exe_dir.join(relative).to_string_lossy().to_string());
            if let Some(parent) = exe_dir.parent() {
                candidates.push(parent.join(relative).to_string_lossy().to_string());
                candidates.push(parent.join("share/gpubench").join(relative).to_string_lossy().to_string());
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(local_appdata) = std::env::var("LOCALAPPDATA") {
            candidates.push(std::path::Path::new(&local_appdata).join("GPUBench").join(relative).to_string_lossy().to_string());
        }
    }
    for cand in candidates {
        let p = std::path::Path::new(&cand);
        if p.exists() {
            if let Ok(canonical) = std::fs::canonicalize(p) {
                return Some(canonical.to_string_lossy().to_string());
            }
            return Some(cand);
        }
    }
    None
}

fn load_parity_profile() -> Option<(ParityProfile, String)> {
    let mut candidates = Vec::new();
    for tag in &["showroom", "indoor", "outdoor", "forest", "pathtracing"] {
        let rel_path = format!("renders/render_{}_profile.json", tag);
        if let Some(actual_path) = find_renders_file(&rel_path) {
            let mtime = std::fs::metadata(&actual_path)
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
            candidates.push((actual_path, tag.to_string(), mtime));
        }
    }
    if let Some(actual_path) = find_renders_file("renders/render_profile.json") {
        let mtime = std::fs::metadata(&actual_path)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        candidates.push((actual_path, "indoor".to_string(), mtime));
    }
    candidates.sort_by(|a, b| b.2.cmp(&a.2));

    for (actual_path, tag, _) in candidates {
        match std::fs::read_to_string(&actual_path) {
            Ok(content) => match serde_json::from_str::<ParityProfile>(&content) {
                Ok(profile) => return Some((profile, tag)),
                Err(e) => eprintln!("[GPUBench GUI] Error parsing profile JSON {}: {}", actual_path, e),
            },
            Err(e) => eprintln!("[GPUBench GUI] Error reading profile JSON {}: {}", actual_path, e),
        }
    }
    None
}

fn open_file_in_desktop(path: &str) {
    let p = std::path::Path::new(path);
    if !p.exists() {
        if path.ends_with("renders") || path == "renders" {
            std::fs::create_dir_all(p).ok();
        }
        if !p.exists() {
            eprintln!("[GPUBench GUI] Cannot open non-existent desktop target: {}", path);
            return;
        }
    }
    let abs_path = if let Ok(canonical) = std::fs::canonicalize(p) {
        canonical.to_string_lossy().to_string()
    } else {
        path.to_string()
    };

    #[cfg(target_os = "windows")]
    {
        if p.is_dir() {
            let _ = std::process::Command::new("explorer")
                .arg(&abs_path)
                .spawn();
        } else {
            let _ = std::process::Command::new("cmd")
                .args(["/C", "start", "", &abs_path])
                .spawn();
        }
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open")
            .arg(&abs_path)
            .spawn();
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open")
            .arg(&abs_path)
            .spawn();
    }
}

fn open_render_target(rel: &str) {
    if let Some(p) = find_renders_file(rel) {
        open_file_in_desktop(&p);
    } else {
        eprintln!("[GPUBench GUI] Render file not found: {}", rel);
    }
}

fn parity_stat_box<'a>(label: &'static str, value: String, sub: String, val_color: iced::Color) -> Element<'a, Message> {
    container(
        column![
            text(label).size(10).style(color!(0x64748B)),
            Space::with_height(2),
            text(value).size(16).style(val_color),
            Space::with_height(1),
            text(sub).size(10).style(color!(0x94A3B8)),
        ]
    )
    .padding(10)
    .width(Length::FillPortion(1))
    .style(|_t: &Theme| container::Appearance {
        background: Some(Background::Color(color!(0x0C0E17))),
        border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1E2438) },
        ..Default::default()
    })
    .into()
}

enum AppState {
    Setup {
        available_backends: Vec<String>,
        selected_backend: String,
        available_devices: Vec<String>,
        available_tests: Vec<String>,
    },
    Running {
        progress_receiver: Option<mpsc::Receiver<ResultData>>,
        total_configs: usize,
        completed_count: usize,
    },
    Complete {
        total_configs: usize,
    },
    Error(String),
}

struct GPUBenchApp {
    state: AppState,
    current_benchmark: String,
    current_devices_label: String,
    selected_tests: HashSet<String>,
    selected_backend: String,
    available_devices: Vec<String>,
    selected_devices: HashSet<String>,
    available_tests: Vec<String>,
    dump_renders: bool,
    selected_resolution: ResolutionPreset,
    selected_scene: ScenePreset,
    parity_profile: Option<(ParityProfile, String)>,
    
    // Multi-Hardware Telemetry
    monitored_devices: Vec<DeviceTelemetry>,
    selected_telemetry_device: usize,

    // Multi-Device Results & Targets
    active_device_targets: Vec<(u32, String)>,
    results_map: HashMap<(u32, &'static str), CellResult>,
    completed_configs_count: usize,
    total_expected_configs: usize,
    validation_error: Option<String>,

    // Metrics
    gpu_bw: f32,
    
    sys_mem_bw: f32,
    sys_mem_bw_single: f32,
    sys_mem_lat: f32,
    
    gpu_fp64: f32,
    gpu_fp32: f32,
    gpu_fp16_vector: f32,
    gpu_fp16_matrix: f32,
    gpu_bf16_vector: f32,
    gpu_bf16_matrix: f32,
    gpu_fp8_vector: f32,
    gpu_fp8_matrix: f32,
    gpu_int8_vector: f32,
    gpu_int8_matrix: f32,
    gpu_int4_vector: f32,
    gpu_int4_matrix: f32,
    
    gpu_rt_anyhit: f32,
    gpu_rt_blas_build: f32,
    gpu_rt_blas_update: f32,
    gpu_rt_tlas_build: f32,
    gpu_rt_incoherent: f32,
    gpu_rt_intersect: f32,
    gpu_rt_divergence: f32,
    gpu_rt_payload: f32,
    gpu_rt_procedural: f32,
    gpu_rt_pathtracing: f32,
    gpu_rt_scheduling_workgraph: f32,
    gpu_rt_scheduling_worklist: f32,
    gpu_rt_scheduling_trad: f32,

    gpu_pixel_fill: f32,
    gpu_pixel_fill_hdr: f32,
    gpu_pixel_fill_blend: f32,

    // Dynamic Viewport Dimensions
    window_width: u32,
    window_height: u32,
}

#[derive(Debug, Clone)]
enum Message {
    BackendSelected(String),
    DeviceToggled(String),
    TestToggled(String, bool),
    TestGroupSelected(String),
    TelemetryDeviceSelected(usize),
    DumpRendersToggled(bool),
    StartBenchmarks,
    BenchmarksComplete,
    BenchmarksFailed(String),
    Tick,
    WindowResized(u32, u32),
    SaveResults,
    SaveResultsComplete(Option<(std::path::PathBuf, String)>),
    Retest,
    ResolutionSelected(ResolutionPreset),
    SceneSelected(ScenePreset),
    CopyDiagnostics,
    OpenTriptych(String),
    OpenHeatmap(String),
    OpenRendersFolder,
}

impl Application for GPUBenchApp {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = GuiCliArgs;

    fn new(flags: GuiCliArgs) -> (Self, Command<Message>) {
        let tests = get_available_benchmarks();
        let hw = gpubench_core::get_available_hardware();
        let backends = available_backends(&hw);

        let selected_backend = if let Some(ref req) = flags.backend {
            let req_upper = req.trim().to_uppercase();
            backends.iter().find(|b| {
                b.to_uppercase() == req_upper
                    || (req_upper == "VK" && *b == "VULKAN")
                    || (req_upper == "CL" && *b == "OPENCL")
                    || (req_upper == "HIP" && *b == "ROCM")
            })
            .cloned()
            .unwrap_or_else(|| backends.first().cloned().unwrap_or_else(|| "VULKAN".to_string()))
        } else {
            backends.first().cloned().unwrap_or_else(|| "VULKAN".to_string())
        };

        let devices = devices_for_backend(&hw, &selected_backend);

        let mut initial_devices = HashSet::new();
        if !flags.devices.is_empty() {
            let mut tokens = Vec::new();
            for item in &flags.devices {
                for t in item.split(',') {
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        tokens.push(trimmed.to_string());
                    }
                }
            }
            for tok in &tokens {
                for dev_str in &devices {
                    if match_device(dev_str, tok) {
                        initial_devices.insert(dev_str.clone());
                    }
                }
            }
        }
        if initial_devices.is_empty() {
            for d in &devices {
                initial_devices.insert(d.clone());
            }
        }

        let mut initial_tests = HashSet::new();
        let filter_by_cli = !flags.groups.is_empty() || !flags.benchmarks.is_empty();

        if filter_by_cli {
            let mut requested_tokens = Vec::new();
            for g in &flags.groups {
                for t in g.split(',') {
                    let s = t.trim().to_lowercase();
                    if !s.is_empty() {
                        requested_tokens.push(s);
                    }
                }
            }
            for b in &flags.benchmarks {
                for t in b.split(',') {
                    let s = t.trim().to_lowercase();
                    if !s.is_empty() {
                        requested_tokens.push(s);
                    }
                }
            }

            for t in &tests {
                if is_benchmark_requested(t, &requested_tokens) {
                    initial_tests.insert(t.clone());
                    if t == "RayIntersect" {
                        initial_tests.insert("RayTracing".to_string());
                    } else if t == "RayTracing" {
                        initial_tests.insert("RayIntersect".to_string());
                    }
                }
            }

            if flags.devices.is_empty() && initial_tests.iter().any(|t| t.contains("System Memory")) {
                if let Some(sys_dev) = devices.iter().find(|d| d.starts_with("System")) {
                    initial_devices.insert(sys_dev.clone());
                }
            }
        } else {
            for t in &tests {
                initial_tests.insert(t.clone());
            }
        }

        if selected_backend != "VULKAN" {
            initial_tests.retain(|t| !t.starts_with("Ray"));
        }

        let mut monitored = discover_all_devices();
        poll_all_devices(&mut monitored, false);

        let selected_resolution = if let Some(ref res) = flags.resolution {
            match res.trim().to_lowercase().as_str() {
                "1080p" | "fhd" => ResolutionPreset::Fhd1080p,
                "1440p" | "qhd" | "2k" => ResolutionPreset::Qhd1440p,
                "4k" | "uhd" | "2160p" => ResolutionPreset::Uhd4k,
                _ => ResolutionPreset::detect_from_devices(&monitored, &initial_devices),
            }
        } else {
            ResolutionPreset::detect_from_devices(&monitored, &initial_devices)
        };

        let selected_scene = if let Some(ref sc) = flags.scene {
            ScenePreset::from_str(sc)
        } else {
            ScenePreset::All
        };

        let dump_renders = flags.dump_renders;
        let auto_start = flags.auto_start && !initial_tests.is_empty();

        let initial_cmd = if auto_start {
            Command::perform(async {}, |_| Message::StartBenchmarks)
        } else {
            Command::none()
        };

        let selected_telemetry_device = {
            let target_gpu_idx = initial_devices.iter()
                .filter(|d| !d.starts_with("System"))
                .filter_map(|d| d.split(':').next().and_then(|s| s.trim().parse::<u32>().ok()))
                .next();

            if let Some(gpu_idx) = target_gpu_idx {
                monitored.iter().position(|m| m.is_gpu && m.id == format!("GPU {}", gpu_idx))
                    .unwrap_or(0)
            } else {
                0
            }
        };

        (
            Self {
                state: AppState::Setup {
                    available_backends: backends.clone(),
                    selected_backend: selected_backend.clone(),
                    available_devices: devices.clone(),
                    available_tests: tests.clone(),
                },
                selected_backend,
                available_devices: devices,
                selected_devices: initial_devices,
                available_tests: tests,
                selected_tests: initial_tests,
                dump_renders,
                selected_resolution,
                selected_scene,
                parity_profile: None,
                current_benchmark: String::from("Waiting to start..."),
                current_devices_label: String::from(""),
                monitored_devices: monitored,
                selected_telemetry_device,
                active_device_targets: Vec::new(),
                results_map: HashMap::new(),
                completed_configs_count: 0,
                total_expected_configs: 1,
                validation_error: None,
                gpu_bw: 0.0,
                sys_mem_bw: 0.0,
                sys_mem_bw_single: 0.0,
                sys_mem_lat: 0.0,
                gpu_fp64: 0.0,
                gpu_fp32: 0.0,
                gpu_fp16_vector: 0.0,
                gpu_fp16_matrix: 0.0,
                gpu_bf16_vector: 0.0,
                gpu_bf16_matrix: 0.0,
                gpu_fp8_vector: 0.0,
                gpu_fp8_matrix: 0.0,
                gpu_int8_vector: 0.0,
                gpu_int8_matrix: 0.0,
                gpu_int4_vector: 0.0,
                gpu_int4_matrix: 0.0,
                gpu_rt_anyhit: 0.0,
                gpu_rt_blas_build: 0.0,
                gpu_rt_blas_update: 0.0,
                gpu_rt_tlas_build: 0.0,
                gpu_rt_incoherent: 0.0,
                gpu_rt_intersect: 0.0,
                gpu_rt_divergence: 0.0,
                gpu_rt_payload: 0.0,
                gpu_rt_procedural: 0.0,
                gpu_rt_pathtracing: 0.0,
                gpu_rt_scheduling_workgraph: 0.0,
                gpu_rt_scheduling_worklist: 0.0,
                gpu_rt_scheduling_trad: 0.0,
                gpu_pixel_fill: 0.0,
                gpu_pixel_fill_hdr: 0.0,
                gpu_pixel_fill_blend: 0.0,
                window_width: 1280,
                window_height: 980,
            },
            initial_cmd
        )
    }

    fn title(&self) -> String {
        String::from("GPUBench — Workstation GPU Profiler")
    }

    fn subscription(&self) -> iced::Subscription<Message> {
        let tick_sub = iced::time::every(std::time::Duration::from_millis(500)).map(|_| Message::Tick);
        let resize_sub = iced::event::listen_with(|event, _| {
            if let iced::Event::Window(_, iced::window::Event::Resized { width, height }) = event {
                Some(Message::WindowResized(width, height))
            } else {
                None
            }
        });
        iced::Subscription::batch([tick_sub, resize_sub])
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::WindowResized(width, height) => {
                self.window_width = width;
                self.window_height = height;
                Command::none()
            }
            Message::BackendSelected(backend) => {
                if let AppState::Setup { selected_backend, available_devices, .. } = &mut self.state {
                    *selected_backend = backend.clone();
                    self.selected_backend = backend.clone();
                    
                    let hw = gpubench_core::get_available_hardware();
                    let new_devices = devices_for_backend(&hw, selected_backend);
                    *available_devices = new_devices.clone();
                    self.available_devices = new_devices.clone();
                    
                    // Preserve selected device indices if possible across backend change
                    let prev_indices: Vec<u32> = self.selected_devices.iter()
                        .filter_map(|d| d.split(':').next().and_then(|s| s.trim().parse::<u32>().ok()))
                        .collect();
                    let had_system = self.selected_devices.iter().any(|d| d.starts_with("System"));

                    self.selected_devices.clear();
                    for d in &self.available_devices {
                        if d.starts_with("System") {
                            if had_system {
                                self.selected_devices.insert(d.clone());
                            }
                        } else if let Some(idx) = d.split(':').next().and_then(|s| s.trim().parse::<u32>().ok()) {
                            if prev_indices.contains(&idx) {
                                self.selected_devices.insert(d.clone());
                            }
                        }
                    }
                    if self.selected_devices.is_empty() {
                        for d in &self.available_devices {
                            self.selected_devices.insert(d.clone());
                        }
                    }

                    if backend != "VULKAN" {
                        self.selected_tests.retain(|t| !t.starts_with("Ray"));
                    }
                }
                self.validation_error = None;
                Command::none()
            }
            Message::DeviceToggled(device) => {
                if self.selected_devices.contains(&device) {
                    if self.selected_devices.len() > 1 {
                        self.selected_devices.remove(&device);
                        if device.starts_with("System") {
                            self.selected_tests.remove("System Memory Bandwidth");
                            self.selected_tests.remove("System Memory Latency");
                        }
                    }
                } else {
                    self.selected_devices.insert(device.clone());
                    if device.starts_with("System") {
                        self.selected_tests.insert("System Memory Bandwidth".to_string());
                        self.selected_tests.insert("System Memory Latency".to_string());
                    }
                }
                self.validation_error = None;
                Command::none()
            }
            Message::DumpRendersToggled(val) => {
                self.dump_renders = val;
                Command::none()
            }
            Message::ResolutionSelected(preset) => {
                self.selected_resolution = preset;
                Command::none()
            }
            Message::SceneSelected(preset) => {
                self.selected_scene = preset;
                Command::none()
            }
            Message::TelemetryDeviceSelected(idx) => {
                if idx < self.monitored_devices.len() {
                    self.selected_telemetry_device = idx;
                }
                Command::none()
            }
            Message::TestToggled(name, is_checked) => {
                if self.selected_backend != "VULKAN" && name.starts_with("Ray") {
                    return Command::none();
                }
                if is_checked {
                    self.selected_tests.insert(name.clone());
                    if name == "RayIntersect" {
                        self.selected_tests.insert("RayTracing".to_string());
                    } else if name == "RayTracing" {
                        self.selected_tests.insert("RayIntersect".to_string());
                    } else if name == "RayScheduling" {
                        self.selected_tests.insert("RayExecutionParadigm".to_string());
                    }
                    if name.contains("System Memory") {
                        self.selected_devices.insert("System: Host CPU & RAM".to_string());
                    }
                } else {
                    self.selected_tests.remove(&name);
                    if name == "RayIntersect" {
                        self.selected_tests.remove("RayTracing");
                    } else if name == "RayTracing" {
                        self.selected_tests.remove("RayIntersect");
                    } else if name == "RayScheduling" {
                        self.selected_tests.remove("RayExecutionParadigm");
                        self.selected_tests.retain(|t| !t.starts_with("RayScheduling"));
                    }
                    if name.contains("System Memory") {
                        let other_sys = if name == "System Memory Bandwidth" {
                            self.selected_tests.contains("System Memory Latency")
                        } else {
                            self.selected_tests.contains("System Memory Bandwidth")
                        };
                        if !other_sys && self.selected_devices.len() > 1 {
                            self.selected_devices.remove("System: Host CPU & RAM");
                        }
                    }
                }
                self.validation_error = None;
                Command::none()
            }
            Message::TestGroupSelected(group) => {
                if let AppState::Setup { available_tests, .. } = &mut self.state {
                    match group.as_str() {
                        "ALL" => {
                            for t in available_tests.iter() {
                                if self.selected_backend == "VULKAN" || (!t.starts_with("Ray") && t.as_str() != "Pixel Fill Rate") {
                                    self.selected_tests.insert(t.clone());
                                }
                            }
                            self.selected_devices.insert("System: Host CPU & RAM".to_string());
                        }
                        "NONE" => {
                            self.selected_tests.clear();
                            if self.selected_devices.len() > 1 {
                                self.selected_devices.remove("System: Host CPU & RAM");
                            }
                        }
                        "COMPUTE" => {
                            let compute: Vec<String> = available_tests.iter()
                                .filter(|t| !t.starts_with("Ray") && !t.contains("Memory") && !t.contains("SysMem") && !t.contains("Pixel"))
                                .cloned().collect();
                            let all_selected = compute.iter().all(|t| self.selected_tests.contains(t));
                            if all_selected {
                                for t in &compute { self.selected_tests.remove(t); }
                            } else {
                                for t in compute { self.selected_tests.insert(t); }
                            }
                        }
                        "MEMORY" => {
                            let mem: Vec<String> = available_tests.iter()
                                .filter(|t| t.as_str() == "Device Memory Bandwidth")
                                .cloned().collect();
                            let all_selected = mem.iter().all(|t| self.selected_tests.contains(t));
                            if all_selected {
                                for t in &mem { self.selected_tests.remove(t); }
                            } else {
                                for t in mem { self.selected_tests.insert(t); }
                            }
                        }
                        "RASTER" => {
                            if self.selected_backend == "VULKAN" {
                                let raster: Vec<String> = available_tests.iter()
                                    .filter(|t| t.as_str() == "Pixel Fill Rate")
                                    .cloned().collect();
                                let all_selected = raster.iter().all(|t| self.selected_tests.contains(t));
                                if all_selected {
                                    for t in &raster { self.selected_tests.remove(t); }
                                } else {
                                    for t in raster { self.selected_tests.insert(t); }
                                }
                            }
                        }
                        "GRAPHICS" => {
                            if self.selected_backend == "VULKAN" {
                                let gfx: Vec<String> = available_tests.iter()
                                    .filter(|t| t.as_str() == "Pixel Fill Rate" || t.starts_with("Ray"))
                                    .cloned().collect();
                                let all_selected = gfx.iter().all(|t| self.selected_tests.contains(t));
                                if all_selected {
                                    for t in &gfx { self.selected_tests.remove(t); }
                                } else {
                                    for t in gfx { self.selected_tests.insert(t); }
                                }
                            }
                        }
                        "RAY TRACING" => {
                            if self.selected_backend == "VULKAN" {
                                let rt: Vec<String> = available_tests.iter()
                                    .filter(|t| t.starts_with("Ray"))
                                    .cloned().collect();
                                let all_selected = rt.iter().all(|t| self.selected_tests.contains(t));
                                if all_selected {
                                    for t in &rt { self.selected_tests.remove(t); }
                                } else {
                                    for t in rt { self.selected_tests.insert(t); }
                                }
                            }
                        }
                        "SYSTEM" => {
                            let sys: Vec<String> = available_tests.iter()
                                .filter(|t| t.contains("System Memory"))
                                .cloned().collect();
                            let all_selected = sys.iter().all(|t| self.selected_tests.contains(t));
                            if all_selected {
                                for t in &sys { self.selected_tests.remove(t); }
                                if self.selected_devices.len() > 1 {
                                    self.selected_devices.remove("System: Host CPU & RAM");
                                }
                            } else {
                                for t in sys { self.selected_tests.insert(t); }
                                self.selected_devices.insert("System: Host CPU & RAM".to_string());
                            }
                        }
                        _ => {}
                    }
                }
                self.validation_error = None;
                Command::none()
            }
            Message::StartBenchmarks => {
                if let AppState::Setup { selected_backend, .. } = &self.state {
                    let b_str = selected_backend.clone();
                    
                    let mut gpu_indices: Vec<u32> = Vec::new();
                    let mut dev_names = Vec::new();
                    let mut target_devices: Vec<(u32, String)> = Vec::new();

                    let has_system_device = self.selected_devices.iter().any(|d| d.starts_with("System"));
                    let has_system_tests = self.selected_tests.iter().any(|t| t.contains("System Memory"));
                    let has_system = has_system_device && has_system_tests;
                    
                    for dev in &self.selected_devices {
                        if !dev.starts_with("System") {
                            if let Some(idx_str) = dev.split(':').next() {
                                if let Ok(idx) = idx_str.parse::<u32>() {
                                    if !gpu_indices.contains(&idx) {
                                        gpu_indices.push(idx);
                                    }
                                }
                            }
                            dev_names.push(dev.clone());
                        }
                    }
                    gpu_indices.sort();

                    if gpu_indices.is_empty() && !has_system {
                        self.validation_error = Some("Please select at least one target device to start.".to_string());
                        log_diagnostic("StartBenchmarks blocked: no devices selected. Displayed visual validation reminder.");
                        return Command::none();
                    }

                    let mut tests_to_run: Vec<String> = Vec::new();
                    for t in &self.available_tests {
                        if self.selected_tests.contains(t)
                            || (t == "RayScheduling" && (self.selected_tests.contains("RayExecutionParadigm") || self.selected_tests.iter().any(|s| s.starts_with("RayScheduling"))))
                        {
                            tests_to_run.push(t.clone());
                        }
                    }
                    if b_str == "VULKAN"
                        && (self.selected_tests.contains("RayScheduling") || self.selected_tests.contains("RayExecutionParadigm") || self.selected_tests.iter().any(|s| s.starts_with("RayScheduling")))
                        && !tests_to_run.iter().any(|t| t == "RayScheduling")
                    {
                        tests_to_run.push("RayScheduling".to_string());
                    }
                    for t in &mut tests_to_run {
                        if t.starts_with("RayScheduling") || t.starts_with("RayExecutionParadigm") {
                            *t = "RayScheduling".to_string();
                        }
                    }
                    tests_to_run.dedup();
                    if !has_system {
                        tests_to_run.retain(|t| !t.contains("System Memory") && !t.contains("SysMem"));
                    }
                    if gpu_indices.is_empty() {
                        tests_to_run.retain(|t| t.contains("System Memory"));
                    }
                    if b_str != "VULKAN" {
                        tests_to_run.retain(|t| !t.starts_with("Ray"));
                    }

                    if tests_to_run.is_empty() {
                        self.validation_error = Some("Please select at least one benchmark or group to start.".to_string());
                        log_diagnostic("StartBenchmarks blocked: no benchmarks selected. Displayed visual validation reminder.");
                        return Command::none();
                    }

                    self.validation_error = None;

                    for &idx in &gpu_indices {
                        let dname = dev_names.iter()
                            .find(|d| d.starts_with(&format!("{}:", idx)))
                            .cloned()
                            .unwrap_or_else(|| format!("GPU {}", idx));
                        target_devices.push((idx, dname));
                    }
                    
                    if has_system {
                        dev_names.insert(0, "System".to_string());
                        target_devices.push((SYSTEM_DEVICE_ID, "System (Host CPU)".to_string()));
                    }
                    self.current_devices_label = dev_names.join(", ");
                    self.active_device_targets = target_devices;
                    self.results_map.clear();
                    self.completed_configs_count = 0;
                    self.parity_profile = None;

                    // Estimate total expected configs across all targets
                    let mut gpu_configs = 0;
                    for t in &tests_to_run {
                        if t.contains("System Memory") { continue; }
                        gpu_configs += match t.as_str() {
                            "Device Memory Bandwidth" => 9,
                            "Pixel Fill Rate" => 3,
                            "FP16" | "BF16" | "FP8" | "INT8" | "INT4" => 2,
                            "RayASBuild" => 8,
                            "RayPathTracing" => 3,
                            "RayScheduling" | "RayExecutionParadigm" => {
                                if self.selected_scene == ScenePreset::All { 28 * 4 } else { 28 }
                            }
                            _ => 1,
                        };
                    }
                    let mut sys_configs = 0;
                    if has_system {
                        for t in &tests_to_run {
                            if t == "System Memory Bandwidth" { sys_configs += 2; }
                            else if t == "System Memory Latency" { sys_configs += 1; }
                        }
                    }
                    let total_configs = (gpu_indices.len() * gpu_configs) + sys_configs;
                    self.total_expected_configs = total_configs.max(1);

                    let (tx, rx) = mpsc::channel();
                    if let Ok(mut guard) = PROGRESS_SENDER.lock() {
                        *guard = Some(tx);
                    }

                    for dev in &mut self.monitored_devices {
                        dev.reset_stats();
                    }

                    let dump_renders_val = self.dump_renders;
                    let (res_w, res_h) = self.selected_resolution.dimensions();
                    let scene_str = self.selected_scene.id_str().to_string();

                    log_diagnostic(&format!(
                        "=== Benchmark run started: backend='{}', resolution='{}' ({}x{}), scene='{}', devices={:?}, tests={:?} ===",
                        b_str, self.selected_resolution.label(), res_w, res_h, self.selected_scene.label(), dev_names, tests_to_run
                    ));

                    self.state = AppState::Running {
                        progress_receiver: Some(rx),
                        total_configs: self.total_expected_configs,
                        completed_count: 0,
                    };

                    return Command::perform(
                        async move {
                            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                            tokio::task::spawn_blocking(move || {
                                run_benchmarks(
                                    &tests_to_run,
                                    &gpu_indices,
                                    &vec![b_str],
                                    false,
                                    false,
                                    false,
                                    dump_renders_val,
                                    res_w,
                                    res_h,
                                    &scene_str,
                                    progress_callback
                                )
                            }).await
                        },
                        |res| match res {
                            Ok(_) => Message::BenchmarksComplete,
                            Err(e) => Message::BenchmarksFailed(
                                format!("Benchmark worker task failed: {}", e)
                            ),
                        }
                    );
                }
                Command::none()
            }
            Message::Tick => {
                let is_running = matches!(self.state, AppState::Running { .. });
                if is_running {
                    poll_all_devices(&mut self.monitored_devices, true);
                }

                let mut results = Vec::new();
                if let AppState::Running { progress_receiver, .. } = &self.state {
                    if let Some(rx) = progress_receiver.as_ref() {
                        while let Ok(res) = rx.try_recv() {
                            results.push(res);
                        }
                    }
                }
                
                for res in results {
                    self.process_result(&res);
                }

                if let AppState::Running { completed_count, .. } = &mut self.state {
                    *completed_count = self.completed_configs_count;
                }

                Command::none()
            }
            Message::BenchmarksComplete => {
                let mut results_to_process = Vec::new();
                if let AppState::Running { ref mut progress_receiver, .. } = self.state {
                    if let Some(rx) = progress_receiver.take() {
                        while let Ok(res) = rx.try_recv() {
                            results_to_process.push(res);
                        }
                    }
                }
                for res in results_to_process {
                    self.process_result(&res);
                }
                let final_count = self.completed_configs_count.max(self.total_expected_configs);
                self.completed_configs_count = final_count;
                self.total_expected_configs = final_count;
                let ran_ray_scheduling = self.selected_tests.iter().any(|t| {
                    let tl = t.to_lowercase();
                    tl.contains("ray") || tl == "graphics" || tl == "all" || tl == "raytracing"
                });
                self.parity_profile = if ran_ray_scheduling && self.dump_renders {
                    load_parity_profile()
                } else {
                    None
                };
                self.state = AppState::Complete { total_configs: final_count };
                self.current_benchmark = String::from("Complete");
                log_diagnostic("=== Benchmark suite completed successfully ===");
                return Command::none();
            }
            Message::BenchmarksFailed(err) => {
                log_diagnostic(&format!("=== Benchmark suite failed: {} ===", err));
                self.state = AppState::Error(err);
                self.current_benchmark = String::from("");
                return Command::none();
            }
            Message::SaveResults => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                #[derive(serde::Serialize)]
                struct BenchmarkReport {
                    app_version: String,
                    timestamp: u64,
                    backend: String,
                    resolution: String,
                    system_info: SystemInfo,
                    device_profiles: Vec<DeviceProfile>,
                    devices: Vec<String>,
                    results: Vec<DeviceReport>,
                    telemetry: Vec<TelemetryReport>,
                }

                #[derive(serde::Serialize)]
                struct DeviceReport {
                    device_id: u32,
                    device_name: String,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    profile: Option<DeviceProfile>,
                    benchmarks: Vec<WorkloadResultReport>,
                }

                #[derive(serde::Serialize)]
                struct WorkloadResultReport {
                    id: String,
                    label: String,
                    category: String,
                    approach: String,
                    value: String,
                    numeric: f64,
                    unit: String,
                    status: String,
                    #[serde(skip_serializing_if = "String::is_empty")]
                    support_note: String,
                    #[serde(skip_serializing_if = "String::is_empty")]
                    support_category: String,
                    #[serde(skip_serializing_if = "String::is_empty")]
                    unsupported_reason: String,
                    raw_operations: u64,
                    raw_time_ms: f64,
                }

                #[derive(serde::Serialize)]
                struct TelemetryReport {
                    id: String,
                    name: String,
                    peak_temp_c: f32,
                    avg_temp_c: f32,
                    peak_power_w: f32,
                    avg_power_w: f32,
                }

                let all_profiles = get_device_profiles();
                let mut device_reports = Vec::new();
                for (dev_id, dev_name) in &self.active_device_targets {
                    let dev_profile = all_profiles.iter().find(|p| p.device_index == *dev_id).cloned();
                    let mut b_list = Vec::new();
                    for w in WORKLOADS {
                        if (w.is_system && *dev_id != SYSTEM_DEVICE_ID) || (!w.is_system && *dev_id == SYSTEM_DEVICE_ID) {
                            continue;
                        }
                        if let Some(cell) = self.results_map.get(&(*dev_id, w.id)) {
                            let status = if cell.is_unsupported {
                                "unsupported".to_string()
                            } else if cell.is_running {
                                "running".to_string()
                            } else if !cell.value_str.is_empty() {
                                "completed".to_string()
                            } else {
                                "pending".to_string()
                            };
                            b_list.push(WorkloadResultReport {
                                id: w.id.to_string(),
                                label: w.label.to_string(),
                                category: w.category.to_string(),
                                approach: w.approach.to_string(),
                                value: cell.value_str.clone(),
                                numeric: cell.numeric,
                                unit: if cell.unit.is_empty() { w.default_unit.to_string() } else { cell.unit.clone() },
                                status,
                                support_note: cell.support_note.clone(),
                                support_category: cell.support_category.clone(),
                                unsupported_reason: cell.support_note.clone(),
                                raw_operations: cell.raw_operations,
                                raw_time_ms: cell.raw_time_ms,
                            });
                        }
                    }
                    device_reports.push(DeviceReport {
                        device_id: *dev_id,
                        device_name: dev_name.clone(),
                        profile: dev_profile,
                        benchmarks: b_list,
                    });
                }

                let mut telem_reports = Vec::new();
                for dev in &self.monitored_devices {
                    telem_reports.push(TelemetryReport {
                        id: dev.id.clone(),
                        name: dev.name.clone(),
                        peak_temp_c: dev.temp_max,
                        avg_temp_c: dev.avg_temp(),
                        peak_power_w: dev.power_max,
                        avg_power_w: dev.avg_power(),
                    });
                }

                let (res_w, res_h) = self.selected_resolution.dimensions();
                let report = BenchmarkReport {
                    app_version: env!("CARGO_PKG_VERSION").to_string(),
                    timestamp: now,
                    backend: self.selected_backend.clone(),
                    resolution: format!("{} ({}x{})", self.selected_resolution.label(), res_w, res_h),
                    system_info: SystemInfo::collect(),
                    device_profiles: all_profiles,
                    devices: self.active_device_targets.iter().map(|(_, n)| n.clone()).collect(),
                    results: device_reports,
                    telemetry: telem_reports,
                };

                let filename = format!("gpubench_results_{}.json", now);
                if let Ok(json_str) = serde_json::to_string_pretty(&report) {
                    let _ = std::fs::write(&filename, &json_str);
                    self.current_benchmark = format!("Results auto-saved: {}. Choose destination in dialog...", filename);
                    return Command::perform(
                        async move {
                            let dialog = rfd::AsyncFileDialog::new()
                                .set_file_name(&filename)
                                .add_filter("JSON Report", &["json"]);
                            if let Some(handle) = dialog.save_file().await {
                                Some((handle.path().to_path_buf(), json_str))
                            } else {
                                None
                            }
                        },
                        Message::SaveResultsComplete,
                    );
                }
                return Command::none();
            }
            Message::SaveResultsComplete(maybe_save) => {
                if let Some((path, json_str)) = maybe_save {
                    let _ = std::fs::write(&path, &json_str);
                    self.current_benchmark = format!("Results exported: {}", path.display());
                }
                Command::none()
            }
            Message::CopyDiagnostics => {
                let summary = self.generate_diagnostic_summary();
                let _ = std::fs::write("gpubench_diagnostics.txt", &summary);
                log_diagnostic("Copied diagnostic report to clipboard and saved to gpubench_diagnostics.txt");
                self.current_benchmark = "Diagnostics copied to clipboard & saved to gpubench_diagnostics.txt".to_string();
                return iced::clipboard::write(summary);
            }
            Message::Retest => {
                let hw = gpubench_core::get_available_hardware();
                let backends = available_backends(&hw);
                let devices = devices_for_backend(&hw, &self.selected_backend);
                
                self.state = AppState::Setup {
                    available_backends: backends.clone(),
                    selected_backend: self.selected_backend.clone(),
                    available_devices: devices.clone(),
                    available_tests: self.available_tests.clone(),
                };
                self.current_benchmark = String::from("Waiting to start...");
                self.current_devices_label = String::from("");
                self.results_map.clear();
                self.completed_configs_count = 0;
                self.active_device_targets.clear();
                self.gpu_bw = 0.0;
                self.sys_mem_bw = 0.0;
                self.sys_mem_bw_single = 0.0;
                self.sys_mem_lat = 0.0;
                self.gpu_fp64 = 0.0;
                self.gpu_fp32 = 0.0;
                self.gpu_fp16_vector = 0.0;
                self.gpu_fp16_matrix = 0.0;
                self.gpu_bf16_vector = 0.0;
                self.gpu_bf16_matrix = 0.0;
                self.gpu_fp8_vector = 0.0;
                self.gpu_fp8_matrix = 0.0;
                self.gpu_int8_vector = 0.0;
                self.gpu_int8_matrix = 0.0;
                self.gpu_int4_vector = 0.0;
                self.gpu_int4_matrix = 0.0;
                self.gpu_rt_anyhit = 0.0;
                self.gpu_rt_blas_build = 0.0;
                self.gpu_rt_blas_update = 0.0;
                self.gpu_rt_tlas_build = 0.0;
                self.gpu_rt_incoherent = 0.0;
                self.gpu_rt_intersect = 0.0;
                self.gpu_rt_divergence = 0.0;
                self.gpu_rt_payload = 0.0;
                self.gpu_rt_procedural = 0.0;
                self.gpu_rt_pathtracing = 0.0;
                self.gpu_rt_scheduling_workgraph = 0.0;
                self.gpu_rt_scheduling_worklist = 0.0;
                self.gpu_rt_scheduling_trad = 0.0;
                self.gpu_pixel_fill = 0.0;
                self.gpu_pixel_fill_hdr = 0.0;
                self.gpu_pixel_fill_blend = 0.0;
                self.parity_profile = None;
                return Command::none();
            }
            Message::OpenTriptych(tag) => {
                let grid = "renders/render_comparison_grid.png";
                let primary_diptych = format!("renders/render_{}_comparison.png", tag);
                let primary = format!("renders/render_{}_comparison_triptych.png", tag);
                let fallback = "renders/render_comparison_triptych.png";
                if (self.selected_scene == ScenePreset::All || tag == "all" || tag == "grid") && find_renders_file(grid).is_some() {
                    open_render_target(grid);
                } else if find_renders_file(&primary_diptych).is_some() {
                    open_render_target(&primary_diptych);
                } else if find_renders_file(grid).is_some() {
                    open_render_target(grid);
                } else if find_renders_file(&primary).is_some() {
                    open_render_target(&primary);
                } else {
                    open_render_target(fallback);
                }
                return Command::none();
            }
            Message::OpenHeatmap(tag) => {
                let primary = format!("renders/render_{}_difference_heatmap.png", tag);
                let fallback = "renders/render_difference_heatmap.png";
                if find_renders_file(&primary).is_some() {
                    open_render_target(&primary);
                } else {
                    open_render_target(fallback);
                }
                return Command::none();
            }
            Message::OpenRendersFolder => {
                if let Some(p) = find_renders_file("renders") {
                    open_file_in_desktop(&p);
                } else {
                    open_file_in_desktop("renders");
                }
                return Command::none();
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let (status_text, status_badge_color, status_text_color) = match &self.state {
            AppState::Setup { .. } => ("IDLE", color!(0x1E2333), color!(0x94A3B8)),
            AppState::Running { .. } => ("RUNNING", color!(0x10B981, 0.2), color!(0x34D399)),
            AppState::Complete { .. } => ("READY", color!(0x6366F1, 0.2), color!(0xA5B4FC)),
            AppState::Error(_) => ("FAILED", color!(0xEF4444, 0.2), color!(0xF87171)),
        };

        // Header Title / Brand block
        let brand_block = column![
            row![
                text("GPUBench").size(24).style(color!(0xF8FAFC)),
                Space::with_width(8),
                container(text(concat!("v", env!("CARGO_PKG_VERSION"))).size(10).style(color!(0x818CF8)))
                    .padding([2, 6])
                    .style(|_t: &Theme| container::Appearance {
                        background: Some(Background::Color(color!(0x6366F1, 0.15))),
                        border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x6366F1, 0.4) },
                        ..Default::default()
                    })
            ].align_items(iced::Alignment::Center),
            text("Workstation Profiler").size(11).style(color!(0x64748B))
        ].spacing(2);

        // Multi-Hardware Telemetry Panel
        let telemetry_panel = {
            let active_dev = self.monitored_devices.get(self.selected_telemetry_device)
                .or_else(|| self.monitored_devices.first());

            let mut tabs_row = row![].spacing(4);
            for (idx, dev) in self.monitored_devices.iter().enumerate() {
                let is_sel = idx == self.selected_telemetry_device;
                let btn = button(text(&dev.id).size(9).horizontal_alignment(iced::alignment::Horizontal::Center))
                    .padding([3, 6])
                    .on_press(Message::TelemetryDeviceSelected(idx))
                    .style(iced::theme::Button::Custom(Box::new(SleekDeviceTab { is_active: is_sel })));
                tabs_row = tabs_row.push(btn);
            }

            let mut hud_content = column![
                row![
                    text("HARDWARE MONITOR").size(10).style(color!(0x64748B)),
                    Space::with_width(Length::Fill),
                    container(text(status_text).size(9).style(status_text_color))
                        .padding([2, 6])
                        .style(move |_t: &Theme| container::Appearance {
                            background: Some(Background::Color(status_badge_color)),
                            border: Border { radius: 4.0.into(), width: 1.0, color: color!(0x2D3748) },
                            ..Default::default()
                        })
                ].align_items(iced::Alignment::Center),
                tabs_row,
            ].spacing(6);

            if let Some(dev) = active_dev {
                let temp_color = if dev.temp > 80.0 { color!(0xEF4444) } else if dev.temp > 0.0 { color!(0x10B981) } else { color!(0x64748B) };
                let power_pct = (dev.power / 350.0).clamp(0.0, 1.0);
                let power_str = if dev.power > 0.0 { format!("{:.1} W", dev.power) } else { "-- W".to_string() };

                if dev.is_gpu {
                    let util_pct = (dev.gpu_util as f32 / 100.0).clamp(0.0, 1.0);
                    let util_str = format!("{}%", dev.gpu_util);
                    let util_color = if dev.gpu_util > 80 { color!(0x10B981) } else if dev.gpu_util > 20 { color!(0x38BDF8) } else { color!(0x64748B) };

                    let edge_str = if dev.temp > 0.0 { format!("{:.0}°C", dev.temp) } else { "--".to_string() };
                    let junc_str = if dev.junction_temp > 0.0 { format!("{:.0}°C", dev.junction_temp) } else { "--".to_string() };
                    let mem_t_str = if dev.mem_temp > 0.0 { format!("{:.0}°C", dev.mem_temp) } else { "--".to_string() };

                    let sclk_str = if dev.sclk > 0 { format!("{} MHz", dev.sclk) } else { "-- MHz".to_string() };
                    let mclk_str = if dev.mclk > 0 { format!("{} MHz", dev.mclk) } else { "-- MHz".to_string() };
                    let vram_pct = if dev.vram_total > 0 { (dev.vram_used as f32 / dev.vram_total as f32).clamp(0.0, 1.0) } else { 0.0 };
                    let vram_str = if dev.vram_total > 0 { format!("{}/{} MB", dev.vram_used, dev.vram_total) } else { "-- / -- MB".to_string() };

                    hud_content = hud_content.push(
                        column![
                            // GPU Utilization and Power side-by-side
                            row![
                                column![
                                    row![text("GPU UTIL").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(util_str).size(9).style(util_color)],
                                    progress_bar(0.0..=1.0, util_pct).height(3.0)
                                ].width(Length::FillPortion(1)).spacing(1),
                                Space::with_width(10),
                                column![
                                    row![text("POWER").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(power_str).size(9).style(color!(0x38BDF8))],
                                    progress_bar(0.0..=1.0, power_pct).height(3.0)
                                ].width(Length::FillPortion(1)).spacing(1),
                            ].align_items(iced::Alignment::Center),
                            // Temperatures
                            row![
                                column![text("EDGE").size(7).style(color!(0x64748B)), text(edge_str).size(9).style(temp_color)].spacing(1),
                                Space::with_width(Length::Fill),
                                column![text("HOTSPOT").size(7).style(color!(0x64748B)), text(junc_str).size(9).style(color!(0xF59E0B))].spacing(1),
                                Space::with_width(Length::Fill),
                                column![text("VRAM").size(7).style(color!(0x64748B)), text(mem_t_str).size(9).style(color!(0xA5B4FC))].spacing(1),
                            ],
                            // Clocks
                            row![
                                column![text("CORE CLK").size(7).style(color!(0x64748B)), text(sclk_str).size(9).style(color!(0xF1F5F9))].spacing(1),
                                Space::with_width(Length::Fill),
                                column![text("MEM CLK").size(7).style(color!(0x64748B)), text(mclk_str).size(9).style(color!(0xF1F5F9))].spacing(1),
                            ],
                            // VRAM gauge
                            column![
                                row![text("VRAM").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(vram_str).size(8).style(color!(0xA5B4FC))],
                                progress_bar(0.0..=1.0, vram_pct).height(3.0)
                            ].spacing(1),
                        ].spacing(4)
                    );
                } else {
                    let temp_str = if dev.temp > 0.0 { format!("{:.1} °C", dev.temp) } else { "-- °C".to_string() };
                    let temp_pct = (dev.temp / 100.0).clamp(0.0, 1.0);
                    hud_content = hud_content.push(
                        row![
                            column![
                                row![text("CPU TEMP").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(temp_str).size(9).style(temp_color)],
                                progress_bar(0.0..=1.0, temp_pct).height(3.0)
                            ].width(Length::FillPortion(1)).spacing(1),
                            Space::with_width(10),
                            column![
                                row![text("CPU POWER").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(power_str).size(9).style(color!(0x38BDF8))],
                                progress_bar(0.0..=1.0, power_pct).height(3.0)
                            ].width(Length::FillPortion(1)).spacing(1),
                        ].align_items(iced::Alignment::Center)
                    );
                }
            }

            container(hud_content)
                .padding([10, 12])
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x11141E))),
                    border: Border { radius: 10.0.into(), width: 1.0, color: color!(0x1F2536) },
                    ..Default::default()
                })
        };

        match &self.state {
            AppState::Setup { available_backends, selected_backend, available_devices, available_tests } => {
                let mut device_col = column![].spacing(5);
                for dev in available_devices {
                    let is_checked = self.selected_devices.contains(dev);
                    let check_box = text(if is_checked { "[X] " } else { "[   ] " })
                        .size(11)
                        .style(if is_checked { color!(0x818CF8) } else { color!(0x475569) });

                    let dev_row = button(
                        row![
                            check_box,
                            text(dev).size(11).style(if is_checked { color!(0xF8FAFC) } else { color!(0x64748B) })
                        ].align_items(iced::Alignment::Center)
                    )
                    .padding([6, 8])
                    .width(Length::Fill)
                    .on_press(Message::DeviceToggled(dev.clone()))
                    .style(iced::theme::Button::Custom(Box::new(SleekDeviceCheckbox { is_checked })));
                    
                    device_col = device_col.push(dev_row);
                }

                let mut api_row = row![].spacing(6);
                for api in available_backends {
                    let is_sel = api == selected_backend;
                    let ver = detect_dynamic_api_version(api);
                    let label = if ver.is_empty() { api.clone() } else { format!("{} {}", api, ver) };
                    
                    let api_btn = button(
                        text(label).size(10).horizontal_alignment(iced::alignment::Horizontal::Center)
                    )
                    .padding([7, 0])
                    .width(Length::Fill)
                    .on_press(Message::BackendSelected(api.clone()))
                    .style(iced::theme::Button::Custom(Box::new(SleekPillToggle { is_active: is_sel, is_api_selector: true })));
                    
                    api_row = api_row.push(api_btn);
                }

                let is_selection_empty = self.selected_tests.is_empty();
                let sel_count = self.selected_tests.len();
                let start_btn = if is_selection_empty {
                    button(
                        container(
                            column![
                                row![
                                    text("▶").size(12).style(color!(0x94A3B8)),
                                    Space::with_width(6),
                                    text("BEGIN TESTING").size(13).style(color!(0x94A3B8)),
                                ].align_items(iced::Alignment::Center),
                                Space::with_height(2),
                                text("Select at least 1 benchmark").size(10).style(color!(0xF59E0B)),
                            ].align_items(iced::Alignment::Center)
                        )
                        .width(Length::Fill)
                        .center_x()
                    )
                    .width(Length::Fill)
                    .padding([10, 0])
                    .on_press(Message::StartBenchmarks)
                    .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton)))
                } else {
                    button(
                        container(
                            column![
                                row![
                                    text("▶").size(13).style(color!(0xFFFFFF)),
                                    Space::with_width(6),
                                    text("BEGIN TESTING").size(14).style(color!(0xFFFFFF)),
                                ].align_items(iced::Alignment::Center),
                                Space::with_height(2),
                                text(format!("{} benchmark{} selected", sel_count, if sel_count == 1 { "" } else { "s" }))
                                    .size(10).style(color!(0xC7D2FE)),
                            ].align_items(iced::Alignment::Center)
                        )
                        .width(Length::Fill)
                        .center_x()
                    )
                    .width(Length::Fill)
                    .padding([10, 0])
                    .on_press(Message::StartBenchmarks)
                    .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton)))
                };

                let resolution_section = {
                    let mut res_row = row![].spacing(6);
                    for preset in ResolutionPreset::ALL {
                        let is_sel = self.selected_resolution == preset;
                        let btn = button(
                            text(preset.label()).size(10).horizontal_alignment(iced::alignment::Horizontal::Center)
                        )
                        .padding([6, 2])
                        .width(Length::Fill)
                        .on_press(Message::ResolutionSelected(preset))
                        .style(iced::theme::Button::Custom(Box::new(SleekPillToggle { is_active: is_sel, is_api_selector: false })));

                        let tip = match preset {
                            ResolutionPreset::Fhd1080p => "Fixed 1920x1080 (2.07M rays / 66 MB) — Fits on-chip cache on high-end GPUs",
                            ResolutionPreset::Qhd1440p => "Fixed 2560x1440 (3.69M rays / 118 MB) — Mid-range cache-stress benchmark",
                            ResolutionPreset::Uhd4k => "Fixed 3840x2160 (8.29M rays / 265 MB) — Spills L3 Infinity Cache on all GPUs",
                        };

                        let row_with_tip = tooltip(
                            btn,
                            container(text(tip).size(10).style(color!(0xE2E8F0)))
                                .width(Length::Fixed(240.0))
                                .padding(8)
                                .style(|_t: &Theme| container::Appearance {
                                    background: Some(Background::Color(color!(0x141824))),
                                    border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x2A3248) },
                                    ..Default::default()
                                }),
                            tooltip::Position::Top
                        )
                        .gap(4)
                        .style(iced::theme::Container::Transparent);

                        res_row = res_row.push(row_with_tip);
                    }
                    res_row
                };

                let make_scene_btn = |preset: ScenePreset, label: &str, is_sel: bool| {
                    let btn = button(
                        row![
                            text(if is_sel { "(•)" } else { "( )" }).size(9).style(if is_sel { color!(0x10B981) } else { color!(0x475569) }),
                            Space::with_width(4),
                            text(label).size(10).style(if is_sel { color!(0xF8FAFC) } else { color!(0x94A3B8) })
                        ].align_items(iced::Alignment::Center)
                    )
                    .padding([5, 6])
                    .width(Length::Fill)
                    .on_press(Message::SceneSelected(preset))
                    .style(iced::theme::Button::Custom(Box::new(SleekDeviceCheckbox { is_checked: is_sel })));

                    let tip = match preset {
                        ScenePreset::All => "Executes all 4 scenarios sequentially (Showroom -> Indoor -> Outdoor -> Forest) and produces comparative grid",
                        ScenePreset::Indoor => "Crytek Sponza Atrium (262k Tris, 25 materials, Cook-Torrance GGX & normal maps)",
                        ScenePreset::Forest => "High-density Open-World Forest (1.0M Tris, 850 trees, terrain, water bathymetry, 8 PBR nature shaders)",
                        ScenePreset::Outdoor => "Mountain Valley Landscape (57k Tris, alpine lake, 100 conifer trees, Rayleigh-Mie atmosphere)",
                        ScenePreset::Showroom => "Khronos ToyCar Studio (109k Tris, metallic flake clearcoat, decals, turntable pedestal)",
                    };

                    tooltip(
                        btn,
                        container(text(tip).size(10).style(color!(0xE2E8F0)))
                            .width(Length::Fixed(250.0))
                            .padding(8)
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x141824))),
                                border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x2A3248) },
                                ..Default::default()
                            }),
                        tooltip::Position::Top
                    )
                    .gap(4)
                    .style(iced::theme::Container::Transparent)
                };

                let scene_section = column![
                    make_scene_btn(ScenePreset::All, "All (4 Scenes)", self.selected_scene == ScenePreset::All),
                    row![
                        make_scene_btn(ScenePreset::Indoor, "Indoor (262k)", self.selected_scene == ScenePreset::Indoor),
                        make_scene_btn(ScenePreset::Forest, "Forest (1.0M)", self.selected_scene == ScenePreset::Forest),
                    ].spacing(6),
                    row![
                        make_scene_btn(ScenePreset::Outdoor, "Outdoor (57k)", self.selected_scene == ScenePreset::Outdoor),
                        make_scene_btn(ScenePreset::Showroom, "Showroom (109k)", self.selected_scene == ScenePreset::Showroom),
                    ].spacing(6),
                ].spacing(4);

                let dump_renders_btn = {
                    let is_checked = self.dump_renders;
                    let check_box = text(if is_checked { "[X] " } else { "[   ] " })
                        .size(11)
                        .style(if is_checked { color!(0x10B981) } else { color!(0x475569) });

                    let btn = button(
                        row![
                            check_box,
                            text("Dump Scene Renders").size(11).style(if is_checked { color!(0xF8FAFC) } else { color!(0x64748B) })
                        ].align_items(iced::Alignment::Center)
                    )
                    .padding([6, 8])
                    .width(Length::Fill)
                    .on_press(Message::DumpRendersToggled(!is_checked))
                    .style(iced::theme::Button::Custom(Box::new(SleekDeviceCheckbox { is_checked })));

                    tooltip(
                        btn,
                        container(text("Exports tonemapped PNG and PPM render buffers for traditional megakernel vs work lists to the 'renders/' directory.").size(10).style(color!(0xE2E8F0)))
                            .width(Length::Fixed(240.0))
                            .padding(8)
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x141824))),
                                border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x2A3248) },
                                ..Default::default()
                            }),
                        tooltip::Position::Top
                    )
                    .gap(4)
                    .style(iced::theme::Container::Transparent)
                };

                let validation_banner = if let Some(ref err) = self.validation_error {
                    Some(
                        container(
                            row![
                                text("⚠️").size(12),
                                Space::with_width(6),
                                text(err.clone()).size(10).style(color!(0xFBBF24))
                            ].align_items(iced::Alignment::Center)
                        )
                        .padding([8, 10])
                        .width(Length::Fill)
                        .style(|_t: &Theme| container::Appearance {
                            background: Some(Background::Color(color!(0x2D1B00))),
                            border: Border { radius: 6.0.into(), width: 1.0, color: color!(0xB45309) },
                            ..Default::default()
                        })
                    )
                } else {
                    None
                };

                let sidebar_scroll_content = column![
                    brand_block,
                    Space::with_height(10),
                    telemetry_panel,
                    Space::with_height(10),
                    text("COMPUTE API").size(10).style(color!(0x64748B)),
                    Space::with_height(4),
                    api_row,
                    Space::with_height(10),
                    text("TARGET DEVICES").size(10).style(color!(0x64748B)),
                    Space::with_height(4),
                    device_col,
                    Space::with_height(10),
                    text("TARGET RESOLUTION").size(10).style(color!(0x64748B)),
                    Space::with_height(4),
                    resolution_section,
                    Space::with_height(10),
                    text("RAY TRACING SCENE").size(10).style(color!(0x64748B)),
                    Space::with_height(4),
                    scene_section,
                    Space::with_height(10),
                    text("OPTIONS").size(10).style(color!(0x64748B)),
                    Space::with_height(4),
                    dump_renders_btn,
                    Space::with_height(6),
                ];

                let mut footer_col = column![].spacing(6);
                if let Some(banner) = validation_banner {
                    footer_col = footer_col.push(banner);
                }
                footer_col = footer_col.push(start_btn);

                let sidebar_col = column![
                    scrollable(
                        container(sidebar_scroll_content)
                            .padding([0, 4, 0, 0])
                            .width(Length::Fill)
                    ).height(Length::Fill),
                    container(footer_col)
                        .padding([8, 0])
                        .width(Length::Fill),
                ];

                let sidebar = container(sidebar_col)
                .width(Length::Fixed(self.dynamic_sidebar_width()))
                .height(Length::Fill)
                .padding(16)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0A0B10))),
                    border: Border { color: color!(0x1A1E2B), width: 1.0, ..Default::default() },
                    ..Default::default()
                });

                let create_pill_grid_with_tooltips = |title: &str, accent_color: iced::Color, is_rt: bool, items: Vec<(&str, &str)>| -> iced::widget::Column<'_, Message> {
                    let is_rt_disabled = is_rt && selected_backend != "VULKAN";
                    
                    let header_row = if is_rt_disabled {
                        row![
                            container(Space::with_width(3)).height(14).style(move |_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x475569))),
                                border: Border { radius: 2.0.into(), ..Default::default() },
                                ..Default::default()
                            }),
                            Space::with_width(8),
                            text(title).size(13).style(color!(0x94A3B8)),
                            Space::with_width(Length::Fill),
                            container(text("VULKAN ONLY").size(9).style(color!(0xF59E0B)))
                                .padding([2, 5])
                                .style(|_t: &Theme| container::Appearance {
                                    background: Some(Background::Color(color!(0xF59E0B, 0.15))),
                                    border: Border { radius: 4.0.into(), width: 1.0, color: color!(0xF59E0B, 0.4) },
                                    ..Default::default()
                                })
                        ].align_items(iced::Alignment::Center)
                    } else {
                        row![
                            container(Space::with_width(3)).height(14).style(move |_t: &Theme| container::Appearance {
                                background: Some(Background::Color(accent_color)),
                                border: Border { radius: 2.0.into(), ..Default::default() },
                                ..Default::default()
                            }),
                            Space::with_width(8),
                            text(title).size(13).style(color!(0xF1F5F9))
                        ].align_items(iced::Alignment::Center)
                    };

                    let mut col = column![
                        header_row,
                        Space::with_height(6)
                    ].spacing(8);

                    let mut current_row = row![].spacing(8);
                    let mut count = 0;
                    for (key, display_label) in items {
                        let is_available = available_tests.contains(&key.to_string())
                            || (key == "RayTracing" && available_tests.contains(&"RayIntersect".to_string()))
                            || (key == "RayIntersect" && available_tests.contains(&"RayTracing".to_string()));
                        if is_available {
                            let is_checked = self.selected_tests.contains(key)
                                || (key == "RayTracing" && self.selected_tests.contains("RayIntersect"))
                                || (key == "RayIntersect" && self.selected_tests.contains("RayTracing"));
                            let name = if available_tests.contains(&key.to_string()) {
                                key.to_string()
                            } else if key == "RayTracing" && available_tests.contains(&"RayIntersect".to_string()) {
                                "RayIntersect".to_string()
                            } else {
                                key.to_string()
                            };
                            let tip_text = if is_rt_disabled {
                                "Hardware Ray Tracing requires the Vulkan backend."
                            } else {
                                get_benchmark_description(key)
                            };

                            let pill_btn = if is_rt_disabled {
                                button(text(display_label).size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                                    .padding([10, 0])
                                    .width(Length::Fill)
                                    .style(iced::theme::Button::Custom(Box::new(SleekDisabledPill)))
                            } else {
                                button(text(display_label).size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                                    .padding([10, 0])
                                    .width(Length::Fill)
                                    .on_press(Message::TestToggled(name.clone(), !is_checked))
                                    .style(iced::theme::Button::Custom(Box::new(SleekPillToggle { is_active: is_checked, is_api_selector: false })))
                            };

                            let pill_with_tip = tooltip(
                                pill_btn,
                                container(text(tip_text).size(11).style(color!(0xE2E8F0)))
                                    .width(Length::Fixed(260.0))
                                    .padding(8)
                                    .style(|_t: &Theme| container::Appearance {
                                        background: Some(Background::Color(color!(0x141824))),
                                        border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x2A3248) },
                                        ..Default::default()
                                    }),
                                tooltip::Position::Top
                            )
                            .gap(4)
                            .style(iced::theme::Container::Transparent);

                            current_row = current_row.push(pill_with_tip);
                            count += 1;
                            if count % 2 == 0 {
                                col = col.push(current_row);
                                current_row = row![].spacing(8);
                            }
                        }
                    }
                    if count % 2 != 0 {
                        col = col.push(current_row.push(Space::with_width(Length::Fill)));
                    }
                    col
                };

                let comp_col = container(
                    create_pill_grid_with_tooltips("Compute Pipelines", color!(0x8B5CF6), false, vec![
                        ("FP64", "FP64 Double"),
                        ("FP32", "FP32 Single"),
                        ("FP16", "FP16 Vector"),
                        ("BF16", "BF16 Vector"),
                        ("FP8", "FP8 Vector"),
                        ("FP4", "FP4 Vector"),
                        ("INT8", "INT8 Vector"),
                        ("INT4", "INT4 Vector"),
                    ])
                )
                .padding(16)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0F121A))),
                    border: Border { radius: 12.0.into(), width: 1.0, color: color!(0x1C2230) },
                    ..Default::default()
                });

                let sys_col = container(
                    column![
                        create_pill_grid_with_tooltips("Graphics: Rasterization (ROP)", color!(0xF59E0B), false, vec![
                            ("Pixel Fill Rate", "Pixel Fill Rate"),
                        ]),
                        Space::with_height(12),
                        create_pill_grid_with_tooltips("Memory: VRAM & Host RAM", color!(0x0EA5E9), false, vec![
                            ("Device Memory Bandwidth", "GPU VRAM Bandwidth"),
                            ("System Memory Bandwidth", "System RAM Bandwidth"),
                            ("System Memory Latency", "System RAM Latency"),
                        ]),
                        Space::with_height(12),
                        container(
                            column![
                                text("STREAMING & ROP PROFILER").size(10).style(color!(0x64748B)),
                                Space::with_height(4),
                                text("Measures multi-threaded host RAM copy, PCIe host-to-device transfers, cache latency, and 32/64-bit frame buffer rasterization throughput.")
                                    .size(11).style(color!(0x94A3B8)),
                            ]
                        )
                        .padding(10)
                        .style(|_t: &Theme| container::Appearance {
                            background: Some(Background::Color(color!(0x0B0E16))),
                            border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x161C2A) },
                            ..Default::default()
                        })
                    ]
                )
                .padding(16)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0F121A))),
                    border: Border { radius: 12.0.into(), width: 1.0, color: color!(0x1C2230) },
                    ..Default::default()
                });

                let rt_col = {
                    let is_rt_disabled = selected_backend != "VULKAN";
                    let rt_top = create_pill_grid_with_tooltips("Graphics: Ray Tracing", color!(0x10B981), true, vec![
                        ("RayASBuild", "BLAS & TLAS Build"),
                        ("RayIntersect", "Ray-Triangle Intersect"),
                        ("RayAnyHit", "AnyHit Alpha-Tested"),
                        ("RayProcedural", "Procedural Geometry"),
                        ("RayMaterialDivergence", "Material Divergence"),
                        ("RayIncoherent", "Incoherent Bounces"),
                        ("RayDivergence", "Divergence Traversal"),
                        ("RayPayload", "Payload Pressure"),
                    ]);

                    let is_pt_checked = self.selected_tests.contains("RayPathTracing");
                    let pt_tip_text = if is_rt_disabled {
                        "Hardware Ray Tracing requires the Vulkan backend."
                    } else {
                        get_benchmark_description("RayPathTracing")
                    };

                    let pt_btn = if is_rt_disabled {
                        button(text("Path Tracing (Full Stochastic GI)").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                            .padding([10, 0])
                            .width(Length::Fill)
                            .style(iced::theme::Button::Custom(Box::new(SleekDisabledPill)))
                    } else {
                        button(text("Path Tracing (Full Stochastic GI)").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                            .padding([10, 0])
                            .width(Length::Fill)
                            .on_press(Message::TestToggled("RayPathTracing".to_string(), !is_pt_checked))
                            .style(iced::theme::Button::Custom(Box::new(SleekPillToggle { is_active: is_pt_checked, is_api_selector: false })))
                    };

                    let pt_with_tip = tooltip(
                        pt_btn,
                        container(text(pt_tip_text).size(11).style(color!(0xE2E8F0)))
                            .width(Length::Fixed(260.0))
                            .padding(8)
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x141824))),
                                border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x2A3248) },
                                ..Default::default()
                            }),
                        tooltip::Position::Top
                    )
                    .gap(4)
                    .style(iced::theme::Container::Transparent);

                    let is_scheduling_checked = self.selected_tests.contains("RayScheduling")
                        || self.selected_tests.contains("RayExecutionParadigm")
                        || self.selected_tests.iter().any(|t| t.starts_with("RayScheduling"));
                    let scheduling_tip_text = if is_rt_disabled {
                        "Hardware Ray Tracing requires the Vulkan backend."
                    } else {
                        get_benchmark_description("RayScheduling")
                    };

                    let scheduling_btn = if is_rt_disabled {
                        button(text("Ray Scheduling (Megakernel vs Work Lists vs Work Graphs vs SER)").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                            .padding([10, 0])
                            .width(Length::Fill)
                            .style(iced::theme::Button::Custom(Box::new(SleekDisabledPill)))
                    } else {
                        button(text("Ray Scheduling (Megakernel vs Work Lists vs Work Graphs vs SER)").size(12).horizontal_alignment(iced::alignment::Horizontal::Center))
                            .padding([10, 0])
                            .width(Length::Fill)
                            .on_press(Message::TestToggled("RayScheduling".to_string(), !is_scheduling_checked))
                            .style(iced::theme::Button::Custom(Box::new(SleekPillToggle { is_active: is_scheduling_checked, is_api_selector: false })))
                    };

                    let scheduling_with_tip = tooltip(
                        scheduling_btn,
                        container(text(scheduling_tip_text).size(11).style(color!(0xE2E8F0)))
                            .width(Length::Fixed(260.0))
                            .padding(8)
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x141824))),
                                border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x2A3248) },
                                ..Default::default()
                            }),
                        tooltip::Position::Top
                    )
                    .gap(4)
                    .style(iced::theme::Container::Transparent);

                    container(
                        column![
                            rt_top,
                            pt_with_tip,
                            scheduling_with_tip
                        ].spacing(8)
                    )
                    .padding(16)
                    .style(|_t: &Theme| container::Appearance {
                        background: Some(Background::Color(color!(0x0F121A))),
                        border: Border { radius: 12.0.into(), width: 1.0, color: color!(0x1C2230) },
                        ..Default::default()
                    })
                };

                let compute_tests: Vec<&String> = available_tests.iter()
                    .filter(|t| !t.starts_with("Ray") && !t.contains("Memory") && !t.contains("SysMem") && !t.contains("Pixel")).collect();
                let mem_tests: Vec<&String> = available_tests.iter()
                    .filter(|t| *t == "Device Memory Bandwidth").collect();
                let gfx_tests: Vec<&String> = available_tests.iter()
                    .filter(|t| *t == "Pixel Fill Rate" || t.starts_with("Ray")).collect();
                let raster_tests: Vec<&String> = available_tests.iter()
                    .filter(|t| *t == "Pixel Fill Rate").collect();
                let rt_tests: Vec<&String> = available_tests.iter()
                    .filter(|t| t.starts_with("Ray")).collect();
                let sys_tests: Vec<&String> = available_tests.iter()
                    .filter(|t| t.contains("System Memory")).collect();
                
                let all_selected = available_tests.iter().all(|t| self.selected_tests.contains(t));
                let none_selected = self.selected_tests.is_empty();
                let compute_all = !compute_tests.is_empty() && compute_tests.iter().all(|t| self.selected_tests.contains(*t));
                let mem_all = !mem_tests.is_empty() && mem_tests.iter().all(|t| self.selected_tests.contains(*t));
                let gfx_all = !gfx_tests.is_empty() && gfx_tests.iter().all(|t| self.selected_tests.contains(*t));
                let raster_all = !raster_tests.is_empty() && raster_tests.iter().all(|t| self.selected_tests.contains(*t));
                let rt_all = !rt_tests.is_empty() && rt_tests.iter().all(|t| self.selected_tests.contains(*t));
                let sys_all = !sys_tests.is_empty() && sys_tests.iter().all(|t| self.selected_tests.contains(*t));
                let is_vulkan = selected_backend == "VULKAN";

                let group_toggles = row![
                    button(text("All").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 12])
                        .on_press(Message::TestGroupSelected("ALL".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: all_selected, is_disabled: false }))),
                    button(text("None").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 12])
                        .on_press(Message::TestGroupSelected("NONE".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: none_selected, is_disabled: false }))),
                    button(text("Compute").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 12])
                        .on_press(Message::TestGroupSelected("COMPUTE".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: compute_all, is_disabled: false }))),
                    button(text("Memory").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 12])
                        .on_press(Message::TestGroupSelected("MEMORY".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: mem_all, is_disabled: false }))),
                    button(text("Graphics (All)").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 12])
                        .on_press(Message::TestGroupSelected("GRAPHICS".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: is_vulkan && gfx_all, is_disabled: !is_vulkan }))),
                    button(text("Raster").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 12])
                        .on_press(Message::TestGroupSelected("RASTER".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: is_vulkan && raster_all, is_disabled: !is_vulkan }))),
                    button(text("Ray Tracing").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 12])
                        .on_press(Message::TestGroupSelected("RAY TRACING".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: is_vulkan && rt_all, is_disabled: !is_vulkan }))),
                    button(text("System").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 12])
                        .on_press(Message::TestGroupSelected("SYSTEM".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: sys_all, is_disabled: false }))),
                ].spacing(5);

                let header_start_btn = if is_selection_empty {
                    button(
                        row![
                            text("▶").size(11).style(color!(0x94A3B8)),
                            Space::with_width(6),
                            text("BEGIN TESTING").size(11).style(color!(0x94A3B8)),
                        ].align_items(iced::Alignment::Center)
                    )
                    .padding([6, 14])
                    .on_press(Message::StartBenchmarks)
                    .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton)))
                } else {
                    button(
                        row![
                            text("▶").size(11).style(color!(0xFFFFFF)),
                            Space::with_width(6),
                            text(format!("BEGIN TESTING ({})", sel_count)).size(11).style(color!(0xFFFFFF)),
                        ].align_items(iced::Alignment::Center)
                    )
                    .padding([6, 16])
                    .on_press(Message::StartBenchmarks)
                    .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton)))
                };

                let test_categories_grid: Element<'_, Message> = if self.dynamic_main_width() >= 880.0 {
                    row![
                        comp_col.width(Length::FillPortion(1)),
                        sys_col.width(Length::FillPortion(1)),
                        rt_col.width(Length::FillPortion(1)),
                    ].spacing(16).into()
                } else if self.dynamic_main_width() >= 560.0 {
                    row![
                        column![comp_col, Space::with_height(16), sys_col].spacing(0).width(Length::FillPortion(1)),
                        rt_col.width(Length::FillPortion(1)),
                    ].spacing(16).into()
                } else {
                    column![
                        comp_col.width(Length::Fill),
                        Space::with_height(16),
                        sys_col.width(Length::Fill),
                        Space::with_height(16),
                        rt_col.width(Length::Fill),
                    ].spacing(0).into()
                };

                let main_area = container(
                    scrollable(
                        column![
                            row![
                                column![
                                    text("Benchmarks").size(20).style(color!(0xF8FAFC)),
                                    text("Select GPU compute, memory, and ray tracing benchmarks to profile").size(12).style(color!(0x64748B))
                                ].spacing(2),
                                Space::with_width(Length::Fill),
                                header_start_btn,
                            ].align_items(iced::Alignment::Center),
                            Space::with_height(10),
                            row![
                                group_toggles,
                                Space::with_width(Length::Fill),
                            ].align_items(iced::Alignment::Center),
                            Space::with_height(16),
                            test_categories_grid,
                        ]
                    ).height(Length::Fill)
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(24)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x07080D))),
                    ..Default::default()
                });

                row![sidebar, main_area].width(Length::Fill).height(Length::Fill).into()
            }
            AppState::Error(err) => {
                let copy_diag_btn = button(
                    container(text("COPY DIAGNOSTICS").size(13).style(color!(0xFFFFFF)))
                        .width(Length::Fill)
                        .center_x()
                )
                .width(Length::Fill)
                .padding([12, 0])
                .on_press(Message::CopyDiagnostics)
                .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton)));

                let retry_btn = button(
                    container(text("RUN NEW TEST").size(13).style(color!(0xCBD5E1)))
                        .width(Length::Fill)
                        .center_x()
                )
                .width(Length::Fill)
                .padding([12, 0])
                .on_press(Message::Retest)
                .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton)));

                let sidebar = container(
                    column![
                        brand_block,
                        Space::with_height(20),
                        telemetry_panel,
                        Space::with_height(Length::Fill),
                        copy_diag_btn,
                        Space::with_height(8),
                        retry_btn
                    ]
                )
                .width(Length::Fixed(self.dynamic_sidebar_width()))
                .height(Length::Fill)
                .padding(16)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0A0B10))),
                    border: Border { color: color!(0x1A1E2B), width: 1.0, ..Default::default() },
                    ..Default::default()
                });

                let error_panel = container(
                    column![
                        text("Benchmark Run Failed").size(20).style(color!(0xEF4444)),
                        Space::with_height(12),
                        text(err).size(13).style(color!(0xCBD5E1)),
                        Space::with_height(20),
                        text("Click COPY DIAGNOSTICS to copy system and benchmark debug logs to clipboard, or RUN NEW TEST to reconfigure.").size(12).style(color!(0x64748B)),
                    ]
                    .width(Length::Fill)
                )
                .width(Length::Fill)
                .padding(24)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x141016))),
                    border: Border { radius: 12.0.into(), width: 1.0, color: color!(0xEF4444, 0.4) },
                    ..Default::default()
                });

                let main_area = container(
                    column![error_panel].width(Length::Fill)
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .padding(24)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x07080D))),
                    ..Default::default()
                });

                row![sidebar, main_area].width(Length::Fill).height(Length::Fill).into()
            }
            state @ AppState::Running { .. } | state @ AppState::Complete { .. } => {
                let (total, completed) = match state {
                    AppState::Running { total_configs, completed_count, .. } => (*total_configs as f32, *completed_count as f32),
                    AppState::Complete { total_configs } => (*total_configs as f32, *total_configs as f32),
                    _ => (1.0, 0.0),
                };

                let global_progress = row![
                    column![
                        row![
                            text("Suite Execution Progress").size(13).style(color!(0xF8FAFC)),
                            Space::with_width(Length::Fill),
                            text(format!("{:.0} of {:.0} configs completed", completed, total)).size(12).style(color!(0x818CF8))
                        ],
                        progress_bar(0.0..=total.max(completed.max(1.0)), completed).height(6.0),
                    ].spacing(6).width(Length::Fill),
                ].align_items(iced::Alignment::Center);

                let targets = if !self.active_device_targets.is_empty() {
                    self.active_device_targets.clone()
                } else {
                    let mut t = Vec::new();
                    let mut gpu_indices = Vec::new();
                    let has_system = self.selected_devices.iter().any(|d| d.starts_with("System"));
                    if has_system {
                        t.push((SYSTEM_DEVICE_ID, "System (Host CPU)".to_string()));
                    }
                    for dev in &self.selected_devices {
                        if !dev.starts_with("System") {
                            if let Some(idx_str) = dev.split(':').next() {
                                if let Ok(idx) = idx_str.parse::<u32>() {
                                    if !gpu_indices.contains(&idx) {
                                        gpu_indices.push(idx);
                                    }
                                }
                            }
                        }
                    }
                    gpu_indices.sort();
                    for &idx in &gpu_indices {
                        let dname = self.selected_devices.iter()
                            .find(|d| d.starts_with(&format!("{}:", idx)))
                            .cloned()
                            .unwrap_or_else(|| format!("GPU {}", idx));
                        t.push((idx, dname));
                    }
                    if t.is_empty() {
                        vec![(0, "GPU 0".to_string())]
                    } else {
                        t
                    }
                };

                // Table Header Row
                let mut header_row = row![
                    container(
                        text("WORKLOAD & ARCHITECTURE").size(11).style(color!(0x64748B))
                    )
                    .width(Length::Fixed(320.0))
                    .padding([8, 12])
                ].spacing(8);

                for (dev_id, dev_name) in &targets {
                    let (dev_badge, badge_color) = if *dev_id == SYSTEM_DEVICE_ID {
                        ("HOST CPU", color!(0x38BDF8))
                    } else if *dev_id == 0 {
                        ("GPU 0", color!(0x818CF8))
                    } else if *dev_id == 1 {
                        ("GPU 1", color!(0x34D399))
                    } else {
                        ("GPU", color!(0xF59E0B))
                    };

                    let clean_name = dev_name.split(':').nth(1).unwrap_or(dev_name.as_str()).trim();
                    let dev_header_card = container(
                        column![
                            text(dev_badge).size(12).style(badge_color),
                            text(clean_name).size(10).style(color!(0x94A3B8)),
                        ].spacing(1)
                    )
                    .width(Length::FillPortion(1))
                    .padding([6, 10])
                    .style(|_t: &Theme| container::Appearance {
                        background: Some(Background::Color(color!(0x11141E))),
                        border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x1E2538) },
                        ..Default::default()
                    });

                    header_row = header_row.push(dev_header_card);
                }

                let categories = [
                    ("COMPUTE PIPELINES", "COMPUTE PIPELINES", color!(0x8B5CF6)),
                    ("MEMORY & SYSTEM", "MEMORY & SYSTEM", color!(0x0EA5E9)),
                    ("RASTERIZATION & ROP", "GRAPHICS: RASTERIZATION (ROP)", color!(0xF59E0B)),
                    ("RAY TRACING ACCELERATION", "GRAPHICS: RAY TRACING ACCELERATION", color!(0x10B981)),
                    ("RAY PIPELINE BREAKDOWN", "GRAPHICS: RAY PIPELINE BREAKDOWN", color!(0x06B6D4)),
                ];

                let mut table_column = column![header_row].spacing(12);
                let mut has_any_category = false;

                for (cat_id, cat_display, cat_color) in categories {
                    let cat_workloads: Vec<&WorkloadDef> = WORKLOADS.iter()
                        .filter(|w| w.category == cat_id)
                        .filter(|w| !w.is_system || targets.iter().any(|(d, _)| *d == SYSTEM_DEVICE_ID))
                        .filter(|w| !w.category.starts_with("RAY") || self.selected_backend == "VULKAN")
                        .filter(|w| w.category != "RASTERIZATION & ROP" || self.selected_backend == "VULKAN")
                        .filter(|w| {
                            self.is_workload_selected(w)
                                || targets.iter().any(|(dev_id, _)| {
                                    self.results_map.get(&(*dev_id, w.id))
                                        .map_or(false, |c| !c.value_str.is_empty() || c.is_unsupported || c.is_running)
                                })
                        })
                        .collect();

                    if cat_workloads.is_empty() {
                        continue;
                    }
                    has_any_category = true;

                    let cat_header = row![
                        container(Space::with_width(3)).height(12).style(move |_t: &Theme| container::Appearance {
                            background: Some(Background::Color(cat_color)),
                            border: Border { radius: 2.0.into(), ..Default::default() },
                            ..Default::default()
                        }),
                        Space::with_width(6),
                        text(cat_display).size(12).style(color!(0xE2E8F0))
                    ].align_items(iced::Alignment::Center);

                    let mut cat_rows = column![].spacing(4);

                    let is_single_target = targets.len() == 1;
                    let mut card_list: Vec<Element<'_, Message>> = Vec::new();

                    for w in cat_workloads {
                        let is_selected = self.is_workload_selected(w);
                        let any_running = targets.iter().any(|(dev_id, _)| {
                            self.results_map.get(&(*dev_id, w.id)).map_or(false, |c| c.is_running)
                        });
                        let any_completed = targets.iter().any(|(dev_id, _)| {
                            self.results_map.get(&(*dev_id, w.id)).map_or(false, |c| !c.value_str.is_empty() || c.is_unsupported)
                        });

                        let (name_color, approach_color) = if any_running {
                            (color!(0x38BDF8), color!(0x0284C7)) // Active Cyan
                        } else if any_completed {
                            (color!(0xE2E8F0), color!(0x94A3B8)) // Legible completed
                        } else if is_selected {
                            (color!(0xF1F5F9), color!(0x818CF8)) // Selected pending
                        } else {
                            (color!(0x64748B), color!(0x475569)) // Dimmed unselected
                        };

                        let name_block = column![
                            text(w.label).size(12).style(name_color),
                            text(w.approach).size(10).style(approach_color),
                        ].spacing(1);

                        let info_tip = tooltip(
                            text("ⓘ").size(11).style(color!(0x475569)),
                            container(
                                column![
                                    text(w.label).size(12).style(color!(0xF8FAFC)),
                                    Space::with_height(2),
                                    text(format!("Method: {}", w.approach)).size(10).style(color!(0x38BDF8)),
                                    Space::with_height(4),
                                    text(w.desc).size(10).style(color!(0xCBD5E1)),
                                ].spacing(2)
                            )
                            .width(Length::Fixed(290.0))
                            .padding(10)
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x141824))),
                                border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x2A3248) },
                                ..Default::default()
                            }),
                            tooltip::Position::Right
                        )
                        .gap(4)
                        .style(iced::theme::Container::Transparent);

                        let left_col = container(
                            row![name_block, Space::with_width(Length::Fill), info_tip]
                                .align_items(iced::Alignment::Center)
                        )
                        .width(if is_single_target { Length::Fill } else { Length::Fixed(320.0) })
                        .padding([3, 6]);

                        let mut row_cells = row![left_col].spacing(8).align_items(iced::Alignment::Center);

                        for (dev_id, _) in &targets {
                            let cell_element: Element<'_, Message> = if (w.is_system && *dev_id != SYSTEM_DEVICE_ID)
                                || (!w.is_system && *dev_id == SYSTEM_DEVICE_ID)
                            {
                                container(text("—").size(11).style(color!(0x334155)))
                                    .width(if is_single_target { Length::Fixed(210.0) } else { Length::FillPortion(1) })
                                    .padding([5, 8])
                                    .center_x()
                                    .into()
                            } else {
                                let cell = self.results_map.get(&(*dev_id, w.id));
                                let (val_str, text_color, bg_color, border_color, note_opt) = if let Some(c) = cell {
                                    if c.is_running {
                                        ("RUNNING...".to_string(), color!(0x22D3EE), color!(0x06B6D4, 0.18), color!(0x06B6D4, 0.5), None)
                                    } else if c.is_unsupported {
                                        let tag = match c.support_category.as_str() {
                                            "toolchain" => "UNSUPPORTED (Toolchain)",
                                            "api" => "UNSUPPORTED (API)",
                                            "hardware" => "UNSUPPORTED (No HW)",
                                            _ => "UNSUPPORTED",
                                        };
                                        (tag.to_string(), color!(0xF87171), color!(0xEF4444, 0.12), color!(0xEF4444, 0.35), if c.support_note.is_empty() { None } else { Some(c.support_note.clone()) })
                                    } else if !c.value_str.is_empty() {
                                        (c.value_str.clone(), color!(0x34D399), color!(0x10B981, 0.14), color!(0x10B981, 0.4), None)
                                    } else if matches!(self.state, AppState::Complete { .. }) {
                                        ("UNSUPPORTED".to_string(), color!(0xF87171), color!(0xEF4444, 0.12), color!(0xEF4444, 0.35), None)
                                    } else {
                                        ("PENDING".to_string(), color!(0x64748B), color!(0x181C28), color!(0x222838), None)
                                    }
                                } else if !is_selected {
                                    ("—".to_string(), color!(0x475569), color!(0x10131B), color!(0x161B26), None)
                                } else if matches!(self.state, AppState::Complete { .. }) {
                                    ("UNSUPPORTED".to_string(), color!(0xF87171), color!(0xEF4444, 0.12), color!(0xEF4444, 0.35), None)
                                } else {
                                    ("PENDING".to_string(), color!(0x64748B), color!(0x181C28), color!(0x222838), None)
                                };

                                let cell_box = container(
                                    text(val_str).size(10).style(text_color)
                                )
                                .width(if is_single_target { Length::Fixed(210.0) } else { Length::FillPortion(1) })
                                .padding([5, 8])
                                .center_x()
                                .style(move |_t: &Theme| container::Appearance {
                                    background: Some(Background::Color(bg_color)),
                                    border: Border { radius: 6.0.into(), width: 1.0, color: border_color },
                                    ..Default::default()
                                });

                                if let Some(note) = note_opt {
                                    tooltip(
                                        cell_box,
                                        container(
                                            column![
                                                text("Unsupported Reason").size(11).style(color!(0xF87171)),
                                                Space::with_height(2),
                                                text(note).size(10).style(color!(0xE2E8F0)),
                                            ]
                                        )
                                        .width(Length::Fixed(280.0))
                                        .padding(8)
                                        .style(|_t: &Theme| container::Appearance {
                                            background: Some(Background::Color(color!(0x141824))),
                                            border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x2A3248) },
                                            ..Default::default()
                                        }),
                                        tooltip::Position::Top
                                    ).into()
                                } else {
                                    cell_box.into()
                                }
                            };

                            row_cells = row_cells.push(cell_element);
                        }

                        let row_card = container(row_cells)
                            .width(if is_single_target { Length::FillPortion(1) } else { Length::Fill })
                            .padding([4, 8])
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x0C0E16))),
                                border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x171C2B) },
                                ..Default::default()
                            });

                        if is_single_target {
                            card_list.push(row_card.into());
                        } else {
                            cat_rows = cat_rows.push(row_card);
                        }
                    }

                    if is_single_target {
                        let num_cols = self.dynamic_test_columns();
                        while !card_list.is_empty() {
                            let mut row_items = row![].spacing(8).width(Length::Fill);
                            let take_count = card_list.len().min(num_cols);
                            for _ in 0..take_count {
                                let card = card_list.remove(0);
                                row_items = row_items.push(card);
                            }
                            for _ in take_count..num_cols {
                                let dummy = container(Space::with_width(Length::Fill)).width(Length::FillPortion(1));
                                row_items = row_items.push(dummy);
                            }
                            cat_rows = cat_rows.push(row_items);
                        }
                    }

                    let cat_panel = column![
                        cat_header,
                        Space::with_height(2),
                        create_panel(cat_rows.into())
                    ].spacing(2);

                    table_column = table_column.push(cat_panel);
                }

                if !has_any_category {
                    table_column = table_column.push(
                        container(
                            text("No test results to display. Click 'RUN NEW TEST' to configure and run benchmarks.")
                                .size(13).style(color!(0x94A3B8))
                        )
                        .padding(24)
                        .center_x()
                    );
                }

                // Hardware Thermal & Power Profile Section (on Complete)
                let thermal_profile_section = if matches!(self.state, AppState::Complete { .. }) {
                    let mut cards_row = row![].spacing(12);
                    for dev in &self.monitored_devices {
                        let dev_card = container(
                            column![
                                row![
                                    text(&dev.id).size(12).style(color!(0x818CF8)),
                                    Space::with_width(6),
                                    text(&dev.name).size(11).style(color!(0x94A3B8)),
                                ].align_items(iced::Alignment::Center),
                                Space::with_height(4),
                                row![
                                    column![
                                        text("TEMPERATURE").size(9).style(color!(0x64748B)),
                                        text(format!("Peak: {:.1} °C", dev.temp_max)).size(11).style(color!(0x10B981)),
                                        text(format!("Avg:  {:.1} °C", dev.avg_temp())).size(10).style(color!(0x94A3B8)),
                                    ].spacing(1),
                                    Space::with_width(Length::Fill),
                                    column![
                                        text("BOARD POWER").size(9).style(color!(0x64748B)),
                                        text(format!("Peak: {:.1} W", dev.power_max)).size(11).style(color!(0x38BDF8)),
                                        text(format!("Avg:  {:.1} W", dev.avg_power())).size(10).style(color!(0x94A3B8)),
                                    ].spacing(1),
                                ]
                            ].spacing(4)
                        )
                        .padding(12)
                        .width(Length::FillPortion(1))
                        .style(|_t: &Theme| container::Appearance {
                            background: Some(Background::Color(color!(0x0C0E16))),
                            border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1F2538) },
                            ..Default::default()
                        });
                        cards_row = cards_row.push(dev_card);
                    }

                    column![
                        row![
                            container(Space::with_width(3)).height(12).style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0xEC4899))),
                                border: Border { radius: 2.0.into(), ..Default::default() },
                                ..Default::default()
                            }),
                            Space::with_width(6),
                            text("HARDWARE THERMAL & POWER PROFILE").size(12).style(color!(0xE2E8F0))
                        ].align_items(iced::Alignment::Center),
                        Space::with_height(4),
                        cards_row
                    ].spacing(2).width(Length::Fill)
                } else {
                    column![]
                };
                let visual_parity_section = if matches!(self.state, AppState::Complete { .. }) {
                    if let Some((ref prof, ref tag)) = self.parity_profile {
                        let passed = prof.parity.diff_pixels == 0;
                        let badge_color = if passed { color!(0x10B981) } else { color!(0xF59E0B) };
                        let badge_bg = if passed { color!(0x10B981, 0.15) } else { color!(0xF59E0B, 0.15) };
                        let badge_text = if passed { "BIT-EXACT MATCH (PASS)" } else { "NEAR-EXACT MATCH" };

                        let card_psnr = parity_stat_box(
                            "PSNR (FIDELITY)",
                            format!("{:.2} dB", prof.parity.psnr),
                            format!("Diff Pixels: {}", prof.parity.diff_pixels),
                            color!(0x38BDF8),
                        );

                        let card_exact = parity_stat_box(
                            "BIT-EXACT RATIO",
                            format!("{:.2}%", prof.parity.exact_pct),
                            format!("{} / {}", prof.parity.exact_pixels, prof.parity.exact_pixels + prof.parity.diff_pixels),
                            color!(0x10B981),
                        );

                        let card_shading = parity_stat_box(
                            "MATERIAL DIVERGENCE SPEEDUP",
                            format!("{:.2}x", prof.worklist.shading_speedup),
                            format!("{:.0} vs {:.0} MHits/s", prof.worklist.shading_mhits, prof.traditional.shading_mhits),
                            color!(0xA78BFA),
                        );

                        let card_compaction = parity_stat_box(
                            "COMPACTION TIME & OVERHEAD",
                            format!("{:.2} ms", prof.worklist.compaction_ms),
                            format!("{:.1}% total frame time", prof.worklist.compaction_pct),
                            color!(0xF472B6),
                        );

                        let stats_grid: Element<'_, Message> = if self.window_width < 1100 {
                            column![
                                row![card_psnr, card_exact].spacing(8).width(Length::Fill),
                                row![card_shading, card_compaction].spacing(8).width(Length::Fill),
                            ].spacing(8).into()
                        } else {
                            row![card_psnr, card_exact, card_shading, card_compaction].spacing(10).width(Length::Fill).into()
                        };

                        let inspector_buttons = row![
                            button(
                                container(text("VIEW PERFORMANCE COMPARISON").size(11).style(color!(0xFFFFFF)))
                                    .padding([6, 12])
                                    .center_x()
                            )
                            .on_press(Message::OpenTriptych(tag.clone()))
                            .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton))),

                            button(
                                container(text("VIEW HEATMAP (10x)").size(11).style(color!(0xCBD5E1)))
                                    .padding([6, 12])
                                    .center_x()
                            )
                            .on_press(Message::OpenHeatmap(tag.clone()))
                            .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton))),

                            button(
                                container(text("OPEN RENDERS FOLDER").size(11).style(color!(0x94A3B8)))
                                    .padding([6, 12])
                                    .center_x()
                            )
                            .on_press(Message::OpenRendersFolder)
                            .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton))),
                        ].spacing(8);

                        column![
                            container(
                                column![
                                    row![
                                        container(Space::with_width(3)).height(14).style(|_t: &Theme| container::Appearance {
                                            background: Some(Background::Color(color!(0x10B981))),
                                            border: Border { radius: 2.0.into(), ..Default::default() },
                                            ..Default::default()
                                        }),
                                        Space::with_width(8),
                                        text("RAY SCHEDULING VISUAL PARITY & ANALYTICAL SCORECARD").size(12).style(color!(0xF1F5F9)),
                                        Space::with_width(12),
                                        container(
                                            text(badge_text).size(10).style(badge_color)
                                        )
                                        .padding([2, 8])
                                        .style(move |_t: &Theme| container::Appearance {
                                            background: Some(Background::Color(badge_bg)),
                                            border: Border { radius: 4.0.into(), width: 1.0, color: badge_color },
                                            ..Default::default()
                                        }),
                                        Space::with_width(Length::Fill),
                                        text(format!("{} | {}", prof.scene, prof.resolution)).size(11).style(color!(0x64748B)),
                                    ].align_items(iced::Alignment::Center),

                                    Space::with_height(8),

                                    stats_grid,

                                    Space::with_height(10),

                                    inspector_buttons,
                                ].spacing(2)
                            )
                            .padding(14)
                            .width(Length::Fill)
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x0A0D15))),
                                border: Border { radius: 10.0.into(), width: 1.0, color: color!(0x1E253A) },
                                ..Default::default()
                            })
                        ].spacing(2).width(Length::Fill)
                    } else if self.dump_renders && self.selected_tests.iter().any(|t| {
                        let tl = t.to_lowercase();
                        tl.contains("ray") || tl == "graphics" || tl == "all" || tl == "raytracing"
                    }) {
                        let inspector_buttons = row![
                            button(
                                container(text("VIEW PERFORMANCE COMPARISON").size(11).style(color!(0xFFFFFF)))
                                    .padding([6, 12])
                                    .center_x()
                            )
                            .on_press(Message::OpenTriptych("all".to_string()))
                            .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton))),

                            button(
                                container(text("OPEN RENDERS FOLDER").size(11).style(color!(0xCBD5E1)))
                                    .padding([6, 12])
                                    .center_x()
                            )
                            .on_press(Message::OpenRendersFolder)
                            .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton))),
                        ].spacing(8);

                        column![
                            container(
                                column![
                                    row![
                                        container(Space::with_width(3)).height(14).style(|_t: &Theme| container::Appearance {
                                            background: Some(Background::Color(color!(0x38BDF8))),
                                            border: Border { radius: 2.0.into(), ..Default::default() },
                                            ..Default::default()
                                        }),
                                        Space::with_width(8),
                                        text("RAY SCHEDULING VISUAL VERIFICATION & RENDERS").size(12).style(color!(0xF1F5F9)),
                                        Space::with_width(Length::Fill),
                                        text("Render dumps active").size(11).style(color!(0x64748B)),
                                    ].align_items(iced::Alignment::Center),
                                    Space::with_height(6),
                                    text("Comparison images and frame renders were dumped during execution.").size(10).style(color!(0x94A3B8)),
                                    Space::with_height(8),
                                    inspector_buttons,
                                ].spacing(2)
                            )
                            .padding(14)
                            .width(Length::Fill)
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x0A0D15))),
                                border: Border { radius: 10.0.into(), width: 1.0, color: color!(0x1E253A) },
                                ..Default::default()
                            })
                        ].spacing(2).width(Length::Fill)
                    } else {
                        column![]
                    }
                } else {
                    column![]
                };

                let action_buttons: Element<'_, Message> = if matches!(self.state, AppState::Complete { .. }) {
                    let ran_ray = self.selected_tests.iter().any(|t| {
                        let tl = t.to_lowercase();
                        tl.contains("ray") || tl == "graphics" || tl == "all" || tl == "raytracing"
                    });
                    let has_renders = self.parity_profile.is_some() || (ran_ray && self.dump_renders);
                    let tag = self.parity_profile.as_ref().map(|(_, t)| t.clone()).unwrap_or_else(|| "indoor".to_string());

                    let export_btn = button(
                        container(text("EXPORT JSON").size(11).style(color!(0xFFFFFF)))
                            .width(Length::Fill)
                            .center_x()
                    )
                    .width(Length::FillPortion(1))
                    .padding([8, 2])
                    .on_press(Message::SaveResults)
                    .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton)));

                    let copy_btn = button(
                        container(text("COPY DIAGNOSTICS").size(11).style(color!(0xCBD5E1)))
                            .width(Length::Fill)
                            .center_x()
                    )
                    .width(Length::FillPortion(1))
                    .padding([8, 2])
                    .on_press(Message::CopyDiagnostics)
                    .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton)));

                    let row_top = row![export_btn, copy_btn].spacing(6).width(Length::Fill);

                    let row_bottom = if has_renders {
                        let view_btn = button(
                            container(text("VIEW RENDERS").size(11).style(color!(0x34D399)))
                                .width(Length::Fill)
                                .center_x()
                        )
                        .width(Length::FillPortion(1))
                        .padding([8, 2])
                        .on_press(Message::OpenTriptych(tag))
                        .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton)));

                        let retest_btn = button(
                            container(text("RUN NEW TEST").size(11).style(color!(0xCBD5E1)))
                                .width(Length::Fill)
                                .center_x()
                        )
                        .width(Length::FillPortion(1))
                        .padding([8, 2])
                        .on_press(Message::Retest)
                        .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton)));

                        row![view_btn, retest_btn].spacing(6).width(Length::Fill)
                    } else {
                        let retest_btn = button(
                            container(text("RUN NEW TEST").size(11).style(color!(0xCBD5E1)))
                                .width(Length::Fill)
                                .center_x()
                        )
                        .width(Length::Fill)
                        .padding([8, 4])
                        .on_press(Message::Retest)
                        .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton)));

                        row![retest_btn].width(Length::Fill)
                    };

                    column![row_top, Space::with_height(6), row_bottom].width(Length::Fill).into()
                } else {
                    Space::with_height(0).into()
                };

                let current_workload_card: Element<'_, Message> = {
                    let is_complete = matches!(self.state, AppState::Complete { .. });
                    let workload_opt = if is_complete {
                        None
                    } else {
                        find_workload_for_benchmark(&self.current_benchmark)
                    };

                    if is_complete {
                        container(
                            column![
                                row![
                                    container(Space::with_width(3)).height(10).style(|_t: &Theme| container::Appearance {
                                        background: Some(Background::Color(color!(0x10B981))),
                                        border: Border { radius: 2.0.into(), ..Default::default() },
                                        ..Default::default()
                                    }),
                                    Space::with_width(6),
                                    text("SUITE FINISHED").size(9).style(color!(0x10B981)),
                                ].align_items(iced::Alignment::Center),
                                Space::with_height(4),
                                text("All tests completed successfully.").size(10).style(color!(0xF8FAFC)),
                                text(format!("{}/{} configs evaluated.", self.completed_configs_count, self.total_expected_configs)).size(9).style(color!(0x94A3B8)),
                            ].spacing(2)
                        )
                        .padding(10)
                        .width(Length::Fill)
                        .style(|_t: &Theme| container::Appearance {
                            background: Some(Background::Color(color!(0x0C121E))),
                            border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1E293B) },
                            ..Default::default()
                        })
                        .into()
                    } else if let Some(w) = workload_opt {
                        container(
                            column![
                                row![
                                    container(Space::with_width(3)).height(10).style(|_t: &Theme| container::Appearance {
                                        background: Some(Background::Color(color!(0x38BDF8))),
                                        border: Border { radius: 2.0.into(), ..Default::default() },
                                        ..Default::default()
                                    }),
                                    Space::with_width(6),
                                    text("ACTIVE WORKLOAD").size(9).style(color!(0x38BDF8)),
                                    Space::with_width(Length::Fill),
                                    text("RUNNING").size(9).style(color!(0x22D3EE))
                                ].align_items(iced::Alignment::Center),
                                Space::with_height(4),
                                text(w.label).size(11).style(color!(0xF8FAFC)),
                                text(w.category).size(8).style(color!(0x818CF8)),
                                Space::with_height(4),
                                column![
                                    text("APPROACH / METHOD").size(8).style(color!(0x64748B)),
                                    text(w.approach).size(9).style(color!(0x38BDF8)),
                                ].spacing(1),
                                Space::with_height(3),
                                column![
                                    text("WHAT IT MEASURES").size(8).style(color!(0x64748B)),
                                    text(w.desc).size(9).style(color!(0x94A3B8)),
                                ].spacing(1),
                                Space::with_height(3),
                                column![
                                    text("API & EXTENSIONS").size(8).style(color!(0x64748B)),
                                    text(w.api_extensions).size(9).style(color!(0x10B981)),
                                ].spacing(1),
                            ].spacing(2)
                        )
                        .padding(10)
                        .width(Length::Fill)
                        .style(|_t: &Theme| container::Appearance {
                            background: Some(Background::Color(color!(0x0E1322))),
                            border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1E2842) },
                            ..Default::default()
                        })
                        .into()
                    } else {
                        container(
                            column![
                                text("CURRENT WORKLOAD").size(9).style(color!(0x64748B)),
                                Space::with_height(3),
                                text(if self.current_benchmark.is_empty() { "Preparing..." } else { &self.current_benchmark }).size(11).style(color!(0xA5B4FC)),
                            ].spacing(2)
                        )
                        .padding(10)
                        .width(Length::Fill)
                        .style(|_t: &Theme| container::Appearance {
                            background: Some(Background::Color(color!(0x0E1322))),
                            border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1E2842) },
                            ..Default::default()
                        })
                        .into()
                    }
                };

                let live_suite_stats_card = container(
                    column![
                        row![
                            container(Space::with_width(3)).height(10).style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x818CF8))),
                                border: Border { radius: 2.0.into(), ..Default::default() },
                                ..Default::default()
                            }),
                            Space::with_width(6),
                            text("EXECUTION METRICS").size(9).style(color!(0x818CF8)),
                        ].align_items(iced::Alignment::Center),
                        Space::with_height(6),
                        row![
                            text("Completed:").size(10).style(color!(0x64748B)),
                            Space::with_width(Length::Fill),
                            text(format!("{}/{}", self.completed_configs_count, self.total_expected_configs)).size(10).style(color!(0xE2E8F0)),
                        ],
                        row![
                            text("Evaluated:").size(10).style(color!(0x64748B)),
                            Space::with_width(Length::Fill),
                            text(if self.completed_configs_count > 0 {
                                let passed = self.results_map.values().filter(|c| !c.value_str.is_empty()).count();
                                format!("{} completed", passed)
                            } else {
                                "—".to_string()
                            }).size(10).style(color!(0x34D399)),
                        ],
                        row![
                            text("Diagnostics:").size(10).style(color!(0x64748B)),
                            Space::with_width(Length::Fill),
                            text(if self.completed_configs_count > 0 {
                                let unsupp = self.results_map.values().filter(|c| c.is_unsupported).count();
                                format!("{} notes", unsupp)
                            } else {
                                "—".to_string()
                            }).size(10).style(color!(0x94A3B8)),
                        ],
                    ].spacing(3)
                )
                .padding(10)
                .width(Length::Fill)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x11141E))),
                    border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1A202C) },
                    ..Default::default()
                });

                let sidebar_scroll_content = column![
                    brand_block,
                    Space::with_height(10),
                    telemetry_panel,
                    Space::with_height(10),
                    text("ACTIVE CONFIGURATION").size(10).style(color!(0x64748B)),
                    Space::with_height(4),
                    container(
                        column![
                            row![text("API:").size(10).style(color!(0x64748B)), Space::with_width(Length::Fill), text(format!("{} {}", &self.selected_backend, detect_dynamic_api_version(&self.selected_backend))).size(11).style(color!(0x38BDF8))],
                            row![text("DEVICES:").size(10).style(color!(0x64748B)), Space::with_width(Length::Fill), text(if self.current_devices_label.is_empty() { "—" } else { &self.current_devices_label }).size(10).style(color!(0xE2E8F0))],
                            row![text("RES:").size(10).style(color!(0x64748B)), Space::with_width(Length::Fill), text(self.selected_resolution.label()).size(10).style(color!(0xA5B4FC))],
                            row![text("SCENE:").size(10).style(color!(0x64748B)), Space::with_width(Length::Fill), text(self.selected_scene.label()).size(10).style(color!(0x34D399))],
                        ].spacing(3)
                    )
                    .padding(10)
                    .style(|_t: &Theme| container::Appearance {
                        background: Some(Background::Color(color!(0x11141E))),
                        border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1A202C) },
                        ..Default::default()
                    }),
                    Space::with_height(10),
                    current_workload_card,
                    Space::with_height(10),
                    live_suite_stats_card,
                ];

                let sidebar_col = column![
                    scrollable(
                        container(sidebar_scroll_content)
                            .padding([0, 4, 0, 0])
                            .width(Length::Fill)
                    ).height(Length::Fill),
                    container(action_buttons)
                        .padding([8, 0])
                        .width(Length::Fill),
                ];

                let sidebar = container(sidebar_col)
                .width(Length::Fixed(self.dynamic_sidebar_width()))
                .height(Length::Fill)
                .padding(16)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0A0B10))),
                    border: Border { color: color!(0x1A1E2B), width: 1.0, ..Default::default() },
                    ..Default::default()
                });

                let main_content = if matches!(self.state, AppState::Complete { .. }) {
                    column![
                        visual_parity_section,
                        thermal_profile_section,
                        table_column
                    ]
                    .spacing(16)
                    .width(Length::Fill)
                } else {
                    column![
                        table_column
                    ]
                    .spacing(16)
                    .width(Length::Fill)
                };

                let main_area = container(
                    column![
                        global_progress,
                        Space::with_height(14),
                        scrollable(main_content)
                            .width(Length::Fill)
                            .height(Length::Fill)
                    ]
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding(24)
                )
                .width(Length::Fill)
                .height(Length::Fill)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x07080D))),
                    ..Default::default()
                });

                row![sidebar, main_area].width(Length::Fill).height(Length::Fill).into()
            }
        }
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }
}

impl GPUBenchApp {
    fn dynamic_sidebar_width(&self) -> f32 {
        if self.window_width >= 1600 {
            360.0
        } else if self.window_width >= 1200 {
            330.0
        } else if self.window_width >= 960 {
            290.0
        } else {
            260.0
        }
    }

    fn dynamic_main_width(&self) -> f32 {
        (self.window_width as f32 - self.dynamic_sidebar_width() - 48.0).max(300.0)
    }

    fn dynamic_test_columns(&self) -> usize {
        let mw = self.dynamic_main_width();
        if mw >= 1350.0 {
            4
        } else if mw >= 960.0 {
            3
        } else if mw >= 560.0 {
            2
        } else {
            1
        }
    }

    fn is_workload_selected(&self, w: &WorkloadDef) -> bool {
        match w.id {
            "fp64" => self.selected_tests.contains("FP64"),
            "fp32" => self.selected_tests.contains("FP32"),
            "fp16_vec" | "fp16_mat" => self.selected_tests.contains("FP16"),
            "bf16_vec" | "bf16_mat" => self.selected_tests.contains("BF16"),
            "fp8_vec" | "fp8_mat" => self.selected_tests.contains("FP8"),
            "int8_vec" | "int8_mat" => self.selected_tests.contains("INT8"),
            "int4_vec" | "int4_mat" => self.selected_tests.contains("INT4"),
            "gpu_vram_bw" => self.selected_tests.contains("Device Memory Bandwidth"),
            "cache_l0" | "cache_l1" | "cache_l2" | "cache_l3" => {
                self.selected_tests.iter().any(|t| t.contains("Cache"))
            }
            "sys_mem_bw_multi" | "sys_mem_bw_single" => self.selected_tests.contains("System Memory Bandwidth"),
            "sys_mem_lat" => self.selected_tests.contains("System Memory Latency"),
            "rop_rgba8" | "rop_rgba16f" | "rop_blend" => self.selected_tests.contains("Pixel Fill Rate"),
            "rt_triangle" => self.selected_tests.contains("RayTracing") || self.selected_tests.contains("RayIntersect"),
            "rt_divergence" => self.selected_tests.contains("RayDivergence") || self.selected_tests.contains("RayMaterialDivergence"),
            "rt_anyhit" => self.selected_tests.contains("RayAnyHit"),
            "rt_incoherent" => self.selected_tests.contains("RayIncoherent"),
            "rt_payload" => self.selected_tests.contains("RayPayload"),
            "rt_blas_build_1m" | "rt_blas_update_1m" | "rt_blas_build_5m" | "rt_blas_update_5m"
            | "rt_blas_build_10m" | "rt_tlas_indoor" | "rt_tlas_jungle" | "rt_tlas_openworld"
            | "rt_blas_build" | "rt_blas_update" | "rt_tlas_build" | "rt_tlas_build_10k" | "rt_tlas_build_100k" => self.selected_tests.contains("RayASBuild"),
            "rt_procedural" => self.selected_tests.contains("RayProcedural"),
            "rt_pathtracing" => self.selected_tests.contains("RayPathTracing"),
            "rt_sched_full_showroom_trad" | "rt_sched_full_showroom_wl" => {
                (self.selected_scene == ScenePreset::All || self.selected_scene == ScenePreset::Showroom)
                    && self.selected_tests.iter().any(|t| t.contains("RayScheduling") || t.contains("RayExecutionParadigm") || t.contains("RayShadows"))
            }
            "rt_sched_full_indoor_trad" | "rt_sched_full_indoor_wl" => {
                (self.selected_scene == ScenePreset::All || self.selected_scene == ScenePreset::Indoor)
                    && self.selected_tests.iter().any(|t| t.contains("RayScheduling") || t.contains("RayExecutionParadigm") || t.contains("RayShadows"))
            }
            "rt_sched_full_outdoor_trad" | "rt_sched_full_outdoor_wl" => {
                (self.selected_scene == ScenePreset::All || self.selected_scene == ScenePreset::Outdoor)
                    && self.selected_tests.iter().any(|t| t.contains("RayScheduling") || t.contains("RayExecutionParadigm") || t.contains("RayShadows"))
            }
            "rt_sched_full_forest_trad" | "rt_sched_full_forest_wl" => {
                (self.selected_scene == ScenePreset::All || self.selected_scene == ScenePreset::Forest)
                    && self.selected_tests.iter().any(|t| t.contains("RayScheduling") || t.contains("RayExecutionParadigm") || t.contains("RayShadows"))
            }
            "rt_sched_stage_prim_morton_trad" | "rt_sched_stage_prim_morton_wl" => {
                false
            }
            id if id.starts_with("rt_sched_") => {
                self.selected_tests.iter().any(|t| t.contains("RayScheduling") || t.contains("RayExecutionParadigm") || t.contains("RayShadows"))
            }
            _ => false,
        }
    }

    fn process_result(&mut self, res: &ResultData) {
        let is_system = res.backendName == "Native" || res.backendName == "System" || res.component == "System";
        let dev_key = if is_system { SYSTEM_DEVICE_ID } else { res.deviceIndex };

        // Handle start notification (time_ms < 0.0)
        if res.time_ms < 0.0 {
            self.current_benchmark = res.benchmarkName.clone();
            // Auto-switch telemetry monitor tab to active device
            if is_system {
                if let Some(pos) = self.monitored_devices.iter().position(|d| !d.is_gpu || d.id == "CPU") {
                    self.selected_telemetry_device = pos;
                }
            } else {
                let target_id = format!("GPU {}", res.deviceIndex);
                if let Some(pos) = self.monitored_devices.iter().position(|d| d.id == target_id) {
                    self.selected_telemetry_device = pos;
                } else if (res.deviceIndex as usize) < self.monitored_devices.len() {
                    self.selected_telemetry_device = res.deviceIndex as usize;
                }
            }

            if let Some(wid) = map_result_to_workload_id(res) {
                let entry = self.results_map.entry((dev_key, wid)).or_default();
                entry.is_running = true;
            }
            return;
        }

        // Completion notification (time_ms >= 0.0)
        let raw_rate = if res.time_ms > 0.0 {
            (res.operations as f64) / (res.time_ms / 1000.0)
        } else {
            0.0
        };
        let mut value = raw_rate;
        if res.metric == "ms/op" {
            value = if res.operations > 0 { res.time_ms / (res.operations as f64) } else { 0.0 };
        } else if res.metric == "ns" {
            value = if res.operations > 0 { (res.time_ms * 1_000_000.0) / (res.operations as f64) } else { 0.0 };
        } else if res.metric == "GIS/s" || res.metric == "GRays/s" || res.metric == "GB/s" || res.metric == "GPixels/s" {
            value /= 1e9;
        } else if res.metric == "TFLOPS" || res.metric == "TOPS" {
            value /= 1e12;
        } else if res.metric == "MTris/s" || res.metric == "MInst/s" || res.metric == "MRays/s" || res.metric == "MHits/s" || res.metric == "MRecords/s" {
            value /= 1e6;
        }

        let is_unsupported = res.isUnsupported || res.time_ms == 0.0 || (res.operations == 0 && res.time_ms > 0.0);

        let value_str = if is_unsupported {
            "UNSUPPORTED".to_string()
        } else if res.metric == "ns" {
            format!("{} ns", format_num_with_commas(value, 2))
        } else if res.metric == "MRays/s" || res.metric == "MHits/s" {
            let (res_w, res_h) = self.selected_resolution.dimensions();
            let eff_w = if res.width > 0 { res.width } else if res_w > 0 { res_w } else { 1920 };
            let eff_h = if res.height > 0 { res.height } else if res_h > 0 { res_h } else { 1080 };
            let rays_per_frame = (eff_w * eff_h).max(1) as f64;
            let fps = (value * 1e6) / rays_per_frame;
            format!("{} {} ({} FPS)", format_num_with_commas(value, 1), res.metric, format_num_with_commas(fps, 0))
        } else if value < 10.0 {
            format!("{} {}", format_num_with_commas(value, 2), res.metric)
        } else {
            format!("{} {}", format_num_with_commas(value, 1), res.metric)
        };

        if let Some(wid) = map_result_to_workload_id(res) {
            let entry = self.results_map.entry((dev_key, wid)).or_default();
            if entry.numeric > 0.0 && !is_unsupported {
                if res.metric == "ns" {
                    if value < entry.numeric {
                        entry.numeric = value;
                        entry.value_str = value_str.clone();
                        entry.support_note = res.supportNote.clone();
                        entry.support_category = res.supportCategory.clone();
                        entry.raw_operations = res.operations;
                        entry.raw_time_ms = res.time_ms;
                    }
                } else if value > entry.numeric {
                    entry.numeric = value;
                    entry.value_str = value_str.clone();
                    entry.support_note = res.supportNote.clone();
                    entry.support_category = res.supportCategory.clone();
                    entry.raw_operations = res.operations;
                    entry.raw_time_ms = res.time_ms;
                }
            } else {
                entry.numeric = value;
                entry.value_str = value_str.clone();
                entry.unit = res.metric.clone();
                entry.is_unsupported = is_unsupported;
                entry.support_note = res.supportNote.clone();
                entry.support_category = res.supportCategory.clone();
                entry.raw_operations = res.operations;
                entry.raw_time_ms = res.time_ms;
            }
            entry.is_running = false;

            if is_unsupported {
                if wid == "fp8_vec" && !self.results_map.contains_key(&(dev_key, "fp8_mat")) {
                    let mat_entry = self.results_map.entry((dev_key, "fp8_mat")).or_default();
                    mat_entry.value_str = "UNSUPPORTED".to_string();
                    mat_entry.is_unsupported = true;
                    mat_entry.support_note = res.supportNote.clone();
                    mat_entry.support_category = res.supportCategory.clone();
                    mat_entry.is_running = false;
                } else if wid == "int4_vec" && !self.results_map.contains_key(&(dev_key, "int4_mat")) {
                    let mat_entry = self.results_map.entry((dev_key, "int4_mat")).or_default();
                    mat_entry.value_str = "UNSUPPORTED".to_string();
                    mat_entry.is_unsupported = true;
                    mat_entry.support_note = res.supportNote.clone();
                    mat_entry.support_category = res.supportCategory.clone();
                    mat_entry.is_running = false;
                }
            }
        }

        self.completed_configs_count += 1;

        // Legacy fields for backward compatibility
        let val_f32 = value as f32;
        if is_system {
            if res.subcategory == "Bandwidth (Multi-threaded)" {
                self.sys_mem_bw = self.sys_mem_bw.max(val_f32);
            } else if res.subcategory == "Bandwidth (Single-threaded)" {
                self.sys_mem_bw_single = self.sys_mem_bw_single.max(val_f32);
            } else if res.subcategory == "Latency" {
                if self.sys_mem_lat == 0.0 { self.sys_mem_lat = val_f32; }
                else { self.sys_mem_lat = self.sys_mem_lat.min(val_f32); }
            }
        } else {
            match res.component.as_str() {
                "Memory" => { self.gpu_bw = self.gpu_bw.max(val_f32); }
                "Graphics" | "Raster" => {
                    if res.configIndex == 0 { self.gpu_pixel_fill = self.gpu_pixel_fill.max(val_f32); }
                    else if res.configIndex == 1 { self.gpu_pixel_fill_hdr = self.gpu_pixel_fill_hdr.max(val_f32); }
                    else { self.gpu_pixel_fill_blend = self.gpu_pixel_fill_blend.max(val_f32); }
                }
                "Compute" => {
                    if res.subcategory == "FP64" { self.gpu_fp64 = self.gpu_fp64.max(val_f32); }
                    if res.subcategory == "FP32" { self.gpu_fp32 = self.gpu_fp32.max(val_f32); }
                    if res.subcategory == "FP16" {
                        if res.configIndex == 0 { self.gpu_fp16_vector = self.gpu_fp16_vector.max(val_f32); }
                        else { self.gpu_fp16_matrix = self.gpu_fp16_matrix.max(val_f32); }
                    }
                    if res.subcategory == "BF16" {
                        if res.configIndex == 0 { self.gpu_bf16_vector = self.gpu_bf16_vector.max(val_f32); }
                        else { self.gpu_bf16_matrix = self.gpu_bf16_matrix.max(val_f32); }
                    }
                    if res.subcategory == "FP8" {
                        if res.configIndex == 0 { self.gpu_fp8_vector = self.gpu_fp8_vector.max(val_f32); }
                        else { self.gpu_fp8_matrix = self.gpu_fp8_matrix.max(val_f32); }
                    }
                    if res.subcategory == "INT8" {
                        if res.configIndex == 0 { self.gpu_int8_vector = self.gpu_int8_vector.max(val_f32); }
                        else { self.gpu_int8_matrix = self.gpu_int8_matrix.max(val_f32); }
                    }
                    if res.subcategory == "INT4" {
                        if res.configIndex == 0 { self.gpu_int4_vector = self.gpu_int4_vector.max(val_f32); }
                        else { self.gpu_int4_matrix = self.gpu_int4_matrix.max(val_f32); }
                    }
                }
                "Ray Tracing" => {
                    if res.subcategory == "Alpha-Tested Geometry" { self.gpu_rt_anyhit = self.gpu_rt_anyhit.max(val_f32); }
                    if res.subcategory.starts_with("BLAS Build") { self.gpu_rt_blas_build = self.gpu_rt_blas_build.max(val_f32); }
                    if res.subcategory.starts_with("BLAS Update") { self.gpu_rt_blas_update = self.gpu_rt_blas_update.max(val_f32); }
                    if res.subcategory.starts_with("TLAS Build") { self.gpu_rt_tlas_build = self.gpu_rt_tlas_build.max(val_f32); }
                    if res.subcategory == "Incoherent Traversal" { self.gpu_rt_incoherent = self.gpu_rt_incoherent.max(val_f32); }
                    if res.subcategory == "Intersection tests" { self.gpu_rt_intersect = self.gpu_rt_intersect.max(val_f32); }
                    if res.subcategory == "Material Divergence" || res.subcategory == "Execution Divergence" { self.gpu_rt_divergence = self.gpu_rt_divergence.max(val_f32); }
                    if res.subcategory == "Payload Register Pressure" { self.gpu_rt_payload = self.gpu_rt_payload.max(val_f32); }
                    if res.subcategory == "Procedural Intersection" { self.gpu_rt_procedural = self.gpu_rt_procedural.max(val_f32); }
                    if res.subcategory == "Path Tracing" || res.benchmarkName.contains("PathTracing") { self.gpu_rt_pathtracing = self.gpu_rt_pathtracing.max(val_f32); }
                    if (res.benchmarkName.contains("RayScheduling") || res.benchmarkName.contains("RayExecutionParadigm"))
                        && !res.benchmarkName.contains("Stage Breakdown")
                        && !res.benchmarkName.contains("Pipeline Breakdown")
                        && !res.subcategory.contains("Pipeline Breakdown")
                        && res.configIndex < 16 {
                        if res.benchmarkName.contains("Work Graphs") {
                            self.gpu_rt_scheduling_workgraph = self.gpu_rt_scheduling_workgraph.max(val_f32);
                        } else if res.benchmarkName.contains("Work Lists") {
                            self.gpu_rt_scheduling_worklist = self.gpu_rt_scheduling_worklist.max(val_f32);
                        } else if res.benchmarkName.contains("Traditional") && !res.benchmarkName.contains("+ SER") {
                            self.gpu_rt_scheduling_trad = self.gpu_rt_scheduling_trad.max(val_f32);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn generate_diagnostic_summary(&self) -> String {
        let sys = SystemInfo::collect();
        let profiles = get_device_profiles();
        let (res_w, res_h) = self.selected_resolution.dimensions();

        let mut s = String::new();
        s.push_str("# GPUBench Diagnostic Report\n\n");
        s.push_str(&format!("- **GPUBench Version**: v{}\n", env!("CARGO_PKG_VERSION")));
        s.push_str(&format!("- **OS**: {}\n", sys.os_name));
        s.push_str(&format!("- **Kernel**: {}\n", sys.kernel_version));
        s.push_str(&format!("- **Architecture**: {}\n", sys.arch));
        s.push_str(&format!("- **CPU**: {} ({} logical threads)\n", sys.cpu_model, sys.cpu_logical_cores));
        s.push_str(&format!("- **Total System RAM**: {:.1} GB\n", sys.total_ram_gb));
        s.push_str(&format!("- **Selected Backend**: {}\n", self.selected_backend));
        s.push_str(&format!("- **Selected Resolution**: {} ({}x{})\n", self.selected_resolution.label(), res_w, res_h));
        s.push_str(&format!("- **Selected RT Scene**: {}\n\n", self.selected_scene.label()));

        s.push_str("## Detected GPU Profiles\n\n");
        if profiles.is_empty() {
            s.push_str("*No GPU profiles returned by backend.*\n\n");
        } else {
            for (idx, p) in profiles.iter().enumerate() {
                s.push_str(&format!("### GPU #{}: {}\n", idx, p.device_name));
                s.push_str(&format!("- **Vendor ID**: {}, **Device ID**: {}\n", p.vendor_id, p.device_id_hex));
                s.push_str(&format!("- **Driver**: {} | Info: {} | Version: {}\n", p.driver_name, p.driver_info, p.driver_version));
                s.push_str(&format!("- **API Version**: {}\n", p.api_version));
                s.push_str(&format!("- **Total VRAM**: {} MB\n", p.vram_total_mb));
                s.push_str(&format!("- **Subgroup Size**: {}, **Max Workgroup Size**: {}\n", p.subgroup_size, p.max_workgroup_size));
                s.push_str(&format!("- **Ray Tracing Pipeline**: {}\n", if p.ray_tracing_supported { "Supported" } else { "Unsupported" }));
                s.push_str(&format!("- **Hardware SER**: {}\n", if p.ser_supported { "Supported" } else { "Unsupported" }));
                s.push_str(&format!("- **Work Graphs**: {}\n", if p.work_graphs_supported { "Supported" } else { "Unsupported" }));
                s.push_str(&format!("- **Cooperative Matrix**: {}\n", if p.cooperative_matrix_supported { "Supported" } else { "Unsupported" }));
                s.push_str(&format!("- **Float16 Compute**: {}, **Int8 Compute**: {}\n\n",
                    if p.float16_supported { "Supported" } else { "Unsupported" },
                    if p.int8_supported { "Supported" } else { "Unsupported" }
                ));
            }
        }

        s.push_str("## Active Device Results\n\n");
        for (dev_id, dev_name) in &self.active_device_targets {
            s.push_str(&format!("### Device {}: {}\n\n", dev_id, dev_name));
            s.push_str("| Workload | Result | Unit | Ops | Time (ms) | Status |\n");
            s.push_str("| :--- | :--- | :--- | :--- | :--- | :--- |\n");
            for w in WORKLOADS {
                if (w.is_system && *dev_id != SYSTEM_DEVICE_ID) || (!w.is_system && *dev_id == SYSTEM_DEVICE_ID) {
                    continue;
                }
                if let Some(cell) = self.results_map.get(&(*dev_id, w.id)) {
                    let status = if cell.is_unsupported {
                        if !cell.support_note.is_empty() {
                            format!("UNSUPPORTED ({})", cell.support_note)
                        } else {
                            "UNSUPPORTED".to_string()
                        }
                    } else if cell.is_running {
                        "RUNNING".to_string()
                    } else if !cell.value_str.is_empty() {
                        "COMPLETED".to_string()
                    } else {
                        "PENDING".to_string()
                    };
                    s.push_str(&format!("| {} | {} | {} | {} | {:.2} | {} |\n",
                        w.label,
                        if cell.value_str.is_empty() { "-" } else { &cell.value_str },
                        if cell.unit.is_empty() { w.default_unit } else { &cell.unit },
                        cell.raw_operations,
                        cell.raw_time_ms,
                        status
                    ));
                }
            }
            s.push_str("\n");
        }

        s.push_str("## Recent Execution Log\n\n```text\n");
        if let Ok(log_content) = std::fs::read_to_string("gpubench.log") {
            let lines: Vec<&str> = log_content.lines().collect();
            let start = if lines.len() > 60 { lines.len() - 60 } else { 0 };
            for line in &lines[start..] {
                s.push_str(line);
                s.push('\n');
            }
        } else {
            s.push_str("(No log recorded yet)\n");
        }
        s.push_str("```\n");

        s
    }
}


fn format_num_with_commas(val: f64, decimals: usize) -> String {
    let s = format!("{:.*}", decimals, val);
    let parts: Vec<&str> = s.split('.').collect();
    let int_part = parts[0];
    let is_negative = int_part.starts_with('-');
    let raw_digits = if is_negative { &int_part[1..] } else { int_part };
    
    let mut formatted = String::new();
    let chars: Vec<char> = raw_digits.chars().collect();
    let len = chars.len();
    for (i, &ch) in chars.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(ch);
    }
    let res = if is_negative { format!("-{}", formatted) } else { formatted };
    if parts.len() > 1 && decimals > 0 {
        format!("{}.{}", res, parts[1])
    } else {
        res
    }
}

fn find_workload_for_benchmark(bench_name: &str) -> Option<&'static WorkloadDef> {
    if bench_name.is_empty() {
        return None;
    }
    if bench_name.contains("RayScheduling") || bench_name.contains("RayExecutionParadigm") {
        if bench_name.contains("Material Shading - Traditional") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_mat_trad");
        } else if bench_name.contains("Material Shading - Work Lists") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_mat_wl");
        } else if bench_name.contains("Path Tracing - Traditional") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_pt_trad");
        } else if bench_name.contains("Path Tracing - Work Lists") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_pt_wl");
        } else if bench_name.contains("Incoherent Ray Tracing - Traditional") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_incoh_trad");
        } else if bench_name.contains("Incoherent Ray Tracing - Work Lists") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_incoh_wl");
        } else if bench_name.contains("Total Scene Render - Traditional") || bench_name.contains("Primary Ray Tracing - Traditional") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_prim_trad");
        } else if bench_name.contains("Total Scene Render - Work Lists") || bench_name.contains("Primary Ray Tracing - Work Lists") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_prim_wl");
        } else if bench_name.contains("SER") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_ser");
        } else if bench_name.contains("Work Graph") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_workgraph");
        } else if bench_name.contains("Linear 32x1") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_stage_bvh_linear");
        } else if bench_name.contains("Queue Compaction") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_stage_queue_compaction");
        } else if bench_name.contains("2D Tiled 8x4") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_stage_bvh_tiled");
        } else if bench_name.contains("Morton 8x4") && bench_name.contains("BVH Traversal") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_stage_bvh_morton8x4");
        } else if bench_name.contains("Morton 4x8") && bench_name.contains("BVH Traversal") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_stage_bvh_morton4x8");
        } else if (bench_name.contains("Full Scene Render") || bench_name.contains("Full Render") || bench_name.contains("Morton")) && (bench_name.contains("Traditional") || bench_name.contains("Megakernel")) {
            if bench_name.contains("Forest") {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_full_forest_trad");
            } else if bench_name.contains("Outdoor") {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_full_outdoor_trad");
            } else if bench_name.contains("Showroom") {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_full_showroom_trad");
            } else if bench_name.contains("Indoor") {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_full_indoor_trad");
            } else {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_stage_prim_morton_trad");
            }
        } else if (bench_name.contains("Full Scene Render") || bench_name.contains("Full Render") || bench_name.contains("Morton")) && (bench_name.contains("Work Lists") || bench_name.contains("DGC")) {
            if bench_name.contains("Forest") {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_full_forest_wl");
            } else if bench_name.contains("Outdoor") {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_full_outdoor_wl");
            } else if bench_name.contains("Showroom") {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_full_showroom_wl");
            } else if bench_name.contains("Indoor") {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_full_indoor_wl");
            } else {
                return WORKLOADS.iter().find(|w| w.id == "rt_sched_stage_prim_morton_wl");
            }
        } else if bench_name.contains("Ray-Traced Shadows - Traditional") || bench_name.contains("Shadows - Traditional") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_shadow_trad");
        } else if bench_name.contains("Ray-Traced Shadows - Work Lists (Directional Binning") || bench_name.contains("Shadows - Work Lists (Directional Binning") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_shadow_bin");
        } else if bench_name.contains("Ray-Traced Shadows - Work Lists") || bench_name.contains("Shadows - Work Lists") {
            return WORKLOADS.iter().find(|w| w.id == "rt_sched_shadow_wl");
        }
    } else if bench_name.contains("RayTracing") || bench_name.contains("RayIntersect") || bench_name.contains("Triangle") {
        return WORKLOADS.iter().find(|w| w.id == "rt_triangle");
    } else if bench_name.contains("AnyHit") {
        return WORKLOADS.iter().find(|w| w.id == "rt_anyhit");
    } else if bench_name.contains("Procedural") {
        return WORKLOADS.iter().find(|w| w.id == "rt_procedural");
    } else if bench_name.contains("Incoherent") {
        return WORKLOADS.iter().find(|w| w.id == "rt_incoherent");
    } else if bench_name.contains("Divergence") {
        return WORKLOADS.iter().find(|w| w.id == "rt_divergence");
    } else if bench_name.contains("PathTracing") {
        return WORKLOADS.iter().find(|w| w.id == "rt_pathtracing");
    } else if bench_name.contains("Payload") {
        return WORKLOADS.iter().find(|w| w.id == "rt_payload");
    } else if bench_name.contains("RayASBuild") {
        if bench_name.contains("1M") {
            if bench_name.contains("Update") {
                return WORKLOADS.iter().find(|w| w.id == "rt_blas_update_1m");
            } else {
                return WORKLOADS.iter().find(|w| w.id == "rt_blas_build_1m");
            }
        } else if bench_name.contains("5M") {
            if bench_name.contains("Update") {
                return WORKLOADS.iter().find(|w| w.id == "rt_blas_update_5m");
            } else {
                return WORKLOADS.iter().find(|w| w.id == "rt_blas_build_5m");
            }
        } else if bench_name.contains("10M") {
            return WORKLOADS.iter().find(|w| w.id == "rt_blas_build_10m");
        } else if bench_name.contains("Indoor") {
            return WORKLOADS.iter().find(|w| w.id == "rt_tlas_indoor");
        } else if bench_name.contains("Jungle") {
            return WORKLOADS.iter().find(|w| w.id == "rt_tlas_jungle");
        } else if bench_name.contains("Open World") {
            return WORKLOADS.iter().find(|w| w.id == "rt_tlas_openworld");
        }
    }

    for w in WORKLOADS {
        if bench_name.contains(w.label) || bench_name == w.id {
            return Some(w);
        }
    }
    None
}

fn map_result_to_workload_id(res: &ResultData) -> Option<&'static str> {
    if res.backendName == "Native" || res.backendName == "System" || res.component == "System" {
        if res.subcategory.contains("Multi-threaded") || (res.subcategory == "Bandwidth" && res.configIndex < 3) {
            return Some("sys_mem_bw_multi");
        } else if res.subcategory.contains("Single-threaded") || res.subcategory.contains("1 Thread") || (res.subcategory == "Bandwidth" && res.configIndex >= 3) {
            return Some("sys_mem_bw_single");
        } else if res.subcategory == "Latency" || res.benchmarkName.contains("Latency") {
            return Some("sys_mem_lat");
        }
    }

    let comp = if res.component.is_empty() {
        if res.benchmarkName.starts_with("Ray") {
            "Ray Tracing"
        } else if res.benchmarkName.starts_with("FP") || res.benchmarkName.starts_with("BF") || res.benchmarkName.starts_with("INT") {
            "Compute"
        } else if res.benchmarkName.contains("Fill Rate") {
            "Graphics"
        } else {
            ""
        }
    } else {
        res.component.as_str()
    };

    match comp {
        "Compute" => {
            match res.subcategory.as_str() {
                "FP64" => Some("fp64"),
                "FP32" => Some("fp32"),
                "FP16" => if res.configIndex == 0 || res.benchmarkName.contains("Vector") { Some("fp16_vec") } else { Some("fp16_mat") },
                "BF16" => if res.configIndex == 0 || res.benchmarkName.contains("Vector") { Some("bf16_vec") } else { Some("bf16_mat") },
                "FP8"  => if res.configIndex == 0 || res.benchmarkName.contains("Vector") { Some("fp8_vec") } else { Some("fp8_mat") },
                "INT8" => if res.configIndex == 0 || res.benchmarkName.contains("Vector") { Some("int8_vec") } else { Some("int8_mat") },
                "INT4" => if res.configIndex == 0 || res.benchmarkName.contains("Vector") { Some("int4_vec") } else { Some("int4_mat") },
                _ => None,
            }
        }
        "Memory" => {
            if res.benchmarkName.contains("L0 Cache") {
                Some("cache_l0")
            } else if res.benchmarkName.contains("L1 Cache") {
                Some("cache_l1")
            } else if res.benchmarkName.contains("L2 Cache") {
                Some("cache_l2")
            } else if res.benchmarkName.contains("L3 Cache") {
                Some("cache_l3")
            } else {
                Some("gpu_vram_bw")
            }
        }
        "Graphics" | "Raster" => {
            if res.configIndex == 0 {
                Some("rop_rgba8")
            } else if res.configIndex == 1 {
                Some("rop_rgba16f")
            } else {
                Some("rop_blend")
            }
        }
        "Ray Tracing" => {
            if res.benchmarkName.contains("RayScheduling") || res.benchmarkName.contains("RayExecutionParadigm") {
                if res.configIndex == 0 || (res.subcategory.contains("Traditional") && res.subcategory.contains("Material")) {
                    Some("rt_sched_mat_trad")
                } else if res.configIndex == 2 || (res.subcategory.contains("Work Lists") && res.subcategory.contains("Material")) {
                    Some("rt_sched_mat_wl")
                } else if res.configIndex == 4 || (res.subcategory.contains("Traditional") && res.subcategory.contains("Path Tracing")) {
                    Some("rt_sched_pt_trad")
                } else if res.configIndex == 6 || (res.subcategory.contains("Work Lists") && res.subcategory.contains("Path Tracing")) {
                    Some("rt_sched_pt_wl")
                } else if res.configIndex == 8 || (res.subcategory.contains("Traditional") && res.subcategory.contains("Incoherent")) {
                    Some("rt_sched_incoh_trad")
                } else if res.configIndex == 10 || (res.subcategory.contains("Work Lists") && res.subcategory.contains("Incoherent")) {
                    Some("rt_sched_incoh_wl")
                } else if res.configIndex == 12 || (res.subcategory.contains("Traditional") && (res.subcategory.contains("Primary") || res.subcategory.contains("Total Scene Render") || res.subcategory.contains("Scene Render"))) {
                    Some("rt_sched_prim_trad")
                } else if res.configIndex == 14 || (res.subcategory.contains("Work Lists") && (res.subcategory.contains("Primary") || res.subcategory.contains("Total Scene Render") || res.subcategory.contains("Scene Render"))) {
                    Some("rt_sched_prim_wl")
                } else if res.configIndex == 1 || res.configIndex == 5 || res.configIndex == 9 || res.configIndex == 13 || res.subcategory.contains("SER") {
                    Some("rt_sched_ser")
                } else if res.configIndex == 3 || res.configIndex == 7 || res.configIndex == 11 || res.configIndex == 15 || res.subcategory.contains("Work Graph") {
                    Some("rt_sched_workgraph")
                } else if res.configIndex == 16 || res.benchmarkName.contains("Linear 32x1") || res.benchmarkName.contains("Linear 1D") {
                    Some("rt_sched_stage_bvh_linear")
                } else if res.configIndex == 17 || res.benchmarkName.contains("Queue Compaction") {
                    Some("rt_sched_stage_queue_compaction")
                } else if res.configIndex == 18 || res.benchmarkName.contains("2D Tiled 8x4") || res.benchmarkName.contains("Screen Tiled") {
                    Some("rt_sched_stage_bvh_tiled")
                } else if res.configIndex == 19 || (res.benchmarkName.contains("Morton 8x4") && res.benchmarkName.contains("BVH Traversal")) {
                    Some("rt_sched_stage_bvh_morton8x4")
                } else if res.configIndex == 20 || (res.benchmarkName.contains("Morton 4x8") && res.benchmarkName.contains("BVH Traversal")) {
                    Some("rt_sched_stage_bvh_morton4x8")
                } else if res.configIndex == 21 || ((res.benchmarkName.contains("Full Scene Render") || res.benchmarkName.contains("Full Render") || res.benchmarkName.contains("Morton")) && (res.benchmarkName.contains("Megakernel") || res.benchmarkName.contains("Traditional"))) {
                    if res.benchmarkName.contains("Forest") {
                        Some("rt_sched_full_forest_trad")
                    } else if res.benchmarkName.contains("Outdoor") {
                        Some("rt_sched_full_outdoor_trad")
                    } else if res.benchmarkName.contains("Showroom") {
                        Some("rt_sched_full_showroom_trad")
                    } else if res.benchmarkName.contains("Indoor") {
                        Some("rt_sched_full_indoor_trad")
                    } else {
                        Some("rt_sched_stage_prim_morton_trad")
                    }
                } else if res.configIndex == 22 || ((res.benchmarkName.contains("Full Scene Render") || res.benchmarkName.contains("Full Render") || res.benchmarkName.contains("Morton")) && (res.benchmarkName.contains("Work Lists") || res.benchmarkName.contains("DGC"))) {
                    if res.benchmarkName.contains("Forest") {
                        Some("rt_sched_full_forest_wl")
                    } else if res.benchmarkName.contains("Outdoor") {
                        Some("rt_sched_full_outdoor_wl")
                    } else if res.benchmarkName.contains("Showroom") {
                        Some("rt_sched_full_showroom_wl")
                    } else if res.benchmarkName.contains("Indoor") {
                        Some("rt_sched_full_indoor_wl")
                    } else {
                        Some("rt_sched_stage_prim_morton_wl")
                    }
                } else if res.configIndex == 23 || (res.subcategory.contains("Traditional") && res.subcategory.contains("Shadow")) {
                    Some("rt_sched_shadow_trad")
                } else if res.configIndex == 24 {
                    Some("rt_sched_ser")
                } else if res.configIndex == 25 || (res.subcategory.contains("Work Lists") && res.subcategory.contains("Shadow") && !res.subcategory.contains("Binning")) {
                    Some("rt_sched_shadow_wl")
                } else if res.configIndex == 26 {
                    Some("rt_sched_workgraph")
                } else if res.configIndex == 27 || (res.subcategory.contains("Binning") && res.subcategory.contains("Shadow")) {
                    Some("rt_sched_shadow_bin")
                } else {
                    None
                }
            } else if res.benchmarkName.contains("PathTracing") || res.subcategory.contains("Path Tracing") {
                Some("rt_pathtracing")
            } else if res.subcategory == "Alpha-Tested Geometry" || res.benchmarkName.contains("AnyHit") {
                Some("rt_anyhit")
            } else if res.subcategory.contains("BLAS Build (1M)") || (res.subcategory == "BLAS Build" && res.configIndex == 0) {
                Some("rt_blas_build_1m")
            } else if res.subcategory.contains("BLAS Update (1M)") || (res.subcategory == "BLAS Update" && res.configIndex == 1) {
                Some("rt_blas_update_1m")
            } else if res.subcategory.contains("BLAS Build (5M)") || (res.subcategory == "BLAS Build" && res.configIndex == 2) {
                Some("rt_blas_build_5m")
            } else if res.subcategory.contains("BLAS Update (5M)") || (res.subcategory == "BLAS Update" && res.configIndex == 3) {
                Some("rt_blas_update_5m")
            } else if res.subcategory.contains("BLAS Build (10M)") || (res.subcategory == "BLAS Build" && res.configIndex == 4) {
                Some("rt_blas_build_10m")
            } else if res.subcategory.contains("Indoor") || (res.benchmarkName == "RayASBuild" && res.configIndex == 5) {
                Some("rt_tlas_indoor")
            } else if res.subcategory.contains("Jungle") || (res.benchmarkName == "RayASBuild" && res.configIndex == 6) {
                Some("rt_tlas_jungle")
            } else if res.subcategory.contains("Open World") || (res.benchmarkName == "RayASBuild" && res.configIndex == 7) {
                Some("rt_tlas_openworld")
            } else if res.subcategory == "Incoherent Traversal" || res.benchmarkName.contains("Incoherent") {
                Some("rt_incoherent")
            } else if res.subcategory == "Intersection tests" || res.benchmarkName == "RayTracing" || res.benchmarkName == "RayIntersect" {
                Some("rt_triangle")
            } else if res.subcategory == "Material Divergence" || res.subcategory == "Execution Divergence" || res.benchmarkName.contains("Divergence") {
                Some("rt_divergence")
            } else if res.subcategory == "Payload Register Pressure" || res.benchmarkName.contains("Payload") {
                Some("rt_payload")
            } else if res.subcategory == "Procedural Intersection" || res.benchmarkName.contains("Procedural") {
                Some("rt_procedural")
            } else {
                None
            }
        }
        _ => {
            if res.benchmarkName.contains("FP8") {
                if res.configIndex == 0 || res.benchmarkName.contains("Vector") { Some("fp8_vec") } else { Some("fp8_mat") }
            } else if res.benchmarkName.contains("INT4") {
                if res.configIndex == 0 || res.benchmarkName.contains("Vector") { Some("int4_vec") } else { Some("int4_mat") }
            } else {
                None
            }
        }
    }
}

fn create_panel<'a>(children: Element<'a, Message>) -> iced::widget::Container<'a, Message> {
    container(children)
        .width(Length::Fill)
        .padding(12)
        .style(|_t: &Theme| container::Appearance {
            background: Some(Background::Color(color!(0x0C0E15))),
            border: Border { radius: 10.0.into(), width: 1.0, color: color!(0x1A1F2C) },
            ..Default::default()
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gui_cli_args() {
        let args = vec![
            "-k".to_string(), "vulkan".to_string(),
            "-g".to_string(), "compute,raytracing".to_string(),
            "-b".to_string(), "rayscheduling".to_string(),
            "-d".to_string(), "1".to_string(),
            "-r".to_string(), "1080p".to_string(),
            "--dump-renders".to_string(),
            "--auto-start".to_string(),
        ];
        let parsed = parse_gui_cli_args_from(&args);
        assert_eq!(parsed.backend.as_deref(), Some("vulkan"));
        assert_eq!(parsed.groups, vec!["compute,raytracing"]);
        assert_eq!(parsed.benchmarks, vec!["rayscheduling"]);
        assert_eq!(parsed.devices, vec!["1"]);
        assert_eq!(parsed.resolution.as_deref(), Some("1080p"));
        assert!(parsed.dump_renders);
        assert!(parsed.auto_start);
        assert!(!parsed.should_exit);
    }

    #[test]
    fn test_parse_gui_cli_equals_syntax() {
        let args = vec![
            "--backend=opencl".to_string(),
            "--groups=all".to_string(),
            "--benchmarks=fp32,fp64".to_string(),
            "--devices=0,1".to_string(),
            "--resolution=4k".to_string(),
            "--run".to_string(),
            "--dump".to_string(),
        ];
        let parsed = parse_gui_cli_args_from(&args);
        assert_eq!(parsed.backend.as_deref(), Some("opencl"));
        assert_eq!(parsed.groups, vec!["all"]);
        assert_eq!(parsed.benchmarks, vec!["fp32,fp64"]);
        assert_eq!(parsed.devices, vec!["0,1"]);
        assert_eq!(parsed.resolution.as_deref(), Some("4k"));
        assert!(parsed.dump_renders);
        assert!(parsed.auto_start);
        assert!(!parsed.should_exit);
    }

    #[test]
    fn test_parse_gui_cli_raster_group() {
        let args = vec![
            "-g".to_string(), "raster".to_string(),
        ];
        let parsed = parse_gui_cli_args_from(&args);
        assert_eq!(parsed.groups, vec!["raster"]);
        assert!(!parsed.should_exit);
    }

    #[test]
    fn test_parse_gui_cli_graphics_group() {
        let args = vec![
            "-g".to_string(), "raster".to_string(),
        ];
        let parsed = parse_gui_cli_args_from(&args);
        assert_eq!(parsed.groups, vec!["raster"]);
        assert!(!parsed.should_exit);
    }

    #[test]
    fn test_parse_gui_cli_graphics_and_raytracing_benchmarks_flag() {
        let args = vec![
            "-k".to_string(), "vulkan".to_string(),
            "-d".to_string(), "1".to_string(),
            "-b".to_string(), "graphics,raytracing".to_string(),
        ];
        let parsed = parse_gui_cli_args_from(&args);
        assert_eq!(parsed.backend.as_deref(), Some("vulkan"));
        assert_eq!(parsed.devices, vec!["1"]);
        assert_eq!(parsed.benchmarks, vec!["graphics,raytracing"]);

        let mut requested_tokens = Vec::new();
        for b in &parsed.benchmarks {
            for t in b.split(',') {
                let s = t.trim().to_lowercase();
                if !s.is_empty() {
                    requested_tokens.push(s);
                }
            }
        }

        // Graphics & Ray Tracing benchmarks should all match
        assert!(is_benchmark_requested("Pixel Fill Rate", &requested_tokens));
        assert!(is_benchmark_requested("RayASBuild", &requested_tokens));
        assert!(is_benchmark_requested("RayIntersect", &requested_tokens));
        assert!(is_benchmark_requested("RayTracing", &requested_tokens));
        assert!(is_benchmark_requested("RayScheduling", &requested_tokens));
        assert!(is_benchmark_requested("RayPathTracing", &requested_tokens));
        assert!(is_benchmark_requested("RayAnyHit", &requested_tokens));
        assert!(is_benchmark_requested("RayProcedural", &requested_tokens));
        assert!(is_benchmark_requested("RayMaterialDivergence", &requested_tokens));
        assert!(is_benchmark_requested("RayIncoherent", &requested_tokens));
        assert!(is_benchmark_requested("RayDivergence", &requested_tokens));
        assert!(is_benchmark_requested("RayPayload", &requested_tokens));

        // Compute and memory benchmarks should NOT match
        assert!(!is_benchmark_requested("FP32", &requested_tokens));
        assert!(!is_benchmark_requested("FP64", &requested_tokens));
        assert!(!is_benchmark_requested("Device Memory Bandwidth", &requested_tokens));
        assert!(!is_benchmark_requested("System Memory Bandwidth", &requested_tokens));
        assert!(!is_benchmark_requested("System Memory Latency", &requested_tokens));
    }

    #[test]
    fn test_empty_selection_shows_validation_error_and_does_not_autostart() {
        let args = vec![
            "-k".to_string(), "vulkan".to_string(),
            "-d".to_string(), "1".to_string(),
        ];
        let parsed = parse_gui_cli_args_from(&args);
        let (mut app, _cmd) = GPUBenchApp::new(parsed);

        // Deselect all tests (like clicking the "None" group chip)
        let _ = app.update(Message::TestGroupSelected("NONE".to_string()));
        assert!(app.selected_tests.is_empty());
        assert_eq!(app.validation_error, None);

        // Attempting to start benchmark with empty selection must NOT auto-select all tests
        let _ = app.update(Message::StartBenchmarks);
        assert!(app.selected_tests.is_empty(), "Empty benchmark selection must not be silently replaced with all benchmarks");
        assert_eq!(
            app.validation_error.as_deref(),
            Some("Please select at least one benchmark or group to start.")
        );

        // Selecting a group or toggling a test should clear the validation error
        let _ = app.update(Message::TestGroupSelected("COMPUTE".to_string()));
        assert!(!app.selected_tests.is_empty());
        assert_eq!(app.validation_error, None);
    }

    #[test]
    fn test_resolution_presets_and_detection() {
        assert_eq!(ResolutionPreset::ALL.len(), 3);
        assert_eq!(ResolutionPreset::ALL[0], ResolutionPreset::Fhd1080p);
        assert_eq!(ResolutionPreset::ALL[1], ResolutionPreset::Qhd1440p);
        assert_eq!(ResolutionPreset::ALL[2], ResolutionPreset::Uhd4k);

        // Test VRAM detection tiers
        let mut dev32gb = DeviceTelemetry::default();
        dev32gb.is_gpu = true;
        dev32gb.id = "GPU 1".to_string();
        dev32gb.vram_total = 32768; // 32 GB R9700

        let mut dev12gb = DeviceTelemetry::default();
        dev12gb.is_gpu = true;
        dev12gb.id = "GPU 0".to_string();
        dev12gb.vram_total = 12288; // 12 GB

        let mut dev8gb = DeviceTelemetry::default();
        dev8gb.is_gpu = true;
        dev8gb.id = "GPU 2".to_string();
        dev8gb.vram_total = 8192; // 8 GB

        let mut initial = HashSet::new();
        initial.insert("GPU 1".to_string());
        assert_eq!(ResolutionPreset::detect_from_devices(&[dev32gb.clone()], &initial), ResolutionPreset::Uhd4k);

        initial.clear();
        initial.insert("GPU 0".to_string());
        assert_eq!(ResolutionPreset::detect_from_devices(&[dev12gb.clone()], &initial), ResolutionPreset::Qhd1440p);

        initial.clear();
        initial.insert("GPU 2".to_string());
        assert_eq!(ResolutionPreset::detect_from_devices(&[dev8gb.clone()], &initial), ResolutionPreset::Fhd1080p);

        // Verify scene naming does not contain "AAA"
        assert_eq!(ScenePreset::Forest.label(), "Open-World Forest (1.0M Tris)");
        assert!(!ScenePreset::Forest.label().contains("AAA"));

        // Verify live GPUBenchApp startup selects 4K UHD automatically on this 32GB R9700 host
        let (app, _) = GPUBenchApp::new(GuiCliArgs::default());
        assert_eq!(app.selected_resolution, ResolutionPreset::Uhd4k);
    }

    #[test]
    fn test_device_cli_flags_respected() {
        // 1. Test "-d 1" specifically selects GPU 1 and excludes other GPUs and System Memory
        let args_d1 = vec!["-k".to_string(), "vulkan".to_string(), "-d".to_string(), "1".to_string()];
        let parsed_d1 = parse_gui_cli_args_from(&args_d1);
        assert_eq!(parsed_d1.devices, vec!["1"]);
        let (app_d1, _) = GPUBenchApp::new(parsed_d1);
        assert_eq!(app_d1.selected_devices.len(), 1, "Passing -d 1 must select exactly 1 device");
        let selected_dev = app_d1.selected_devices.iter().next().unwrap();
        assert!(selected_dev.starts_with("1:"), "Selected device must start with '1:', got: {}", selected_dev);
        assert!(!selected_dev.contains("System"), "Passing -d 1 must not select System Memory");

        // Telemetry HUD must focus on GPU 1
        assert_eq!(app_d1.selected_telemetry_device, 1);
        if let Some(target_monitored) = app_d1.monitored_devices.get(app_d1.selected_telemetry_device) {
            assert_eq!(target_monitored.id, "GPU 1", "Telemetry HUD must target GPU 1");
        }

        // 2. Test "-d1" (compact syntax)
        let args_d1_compact = vec!["-d1".to_string()];
        let parsed_compact = parse_gui_cli_args_from(&args_d1_compact);
        assert_eq!(parsed_compact.devices, vec!["1"]);
        let (app_compact, _) = GPUBenchApp::new(parsed_compact);
        assert_eq!(app_compact.selected_devices.len(), 1);
        assert!(app_compact.selected_devices.iter().next().unwrap().starts_with("1:"));

        // 3. Test "--device=0" specifically selects GPU 0
        let args_d0 = vec!["--device=0".to_string()];
        let parsed_d0 = parse_gui_cli_args_from(&args_d0);
        assert_eq!(parsed_d0.devices, vec!["0"]);
        let (app_d0, _) = GPUBenchApp::new(parsed_d0);
        assert_eq!(app_d0.selected_devices.len(), 1);
        let selected_dev_0 = app_d0.selected_devices.iter().next().unwrap();
        assert!(selected_dev_0.starts_with("0:"), "Selected device must start with '0:', got: {}", selected_dev_0);

        // 4. Test "--devices=0,1" selects both GPU 0 and GPU 1
        let args_d01 = vec!["--devices=0,1".to_string()];
        let parsed_d01 = parse_gui_cli_args_from(&args_d01);
        let (app_d01, _) = GPUBenchApp::new(parsed_d01);
        assert_eq!(app_d01.selected_devices.len(), 2);
        assert!(app_d01.selected_devices.iter().any(|d| d.starts_with("0:")));
        assert!(app_d01.selected_devices.iter().any(|d| d.starts_with("1:")));
        assert!(!app_d01.selected_devices.iter().any(|d| d.starts_with("System")));

        // 5. Test "-d 'GPU 1'" or "-d gpu1"
        let args_gpu1 = vec!["-d".to_string(), "gpu 1".to_string()];
        let parsed_gpu1 = parse_gui_cli_args_from(&args_gpu1);
        let (app_gpu1, _) = GPUBenchApp::new(parsed_gpu1);
        assert_eq!(app_gpu1.selected_devices.len(), 1);
        assert!(app_gpu1.selected_devices.iter().next().unwrap().starts_with("1:"));
    }

    #[test]
    fn test_dynamic_layout_scaling() {
        let (mut app, _) = GPUBenchApp::new(GuiCliArgs::default());

        // Initial default size should be 1280x980
        assert_eq!(app.window_width, 1280);
        assert_eq!(app.window_height, 980);
        assert_eq!(app.dynamic_sidebar_width(), 330.0);
        assert_eq!(app.dynamic_main_width(), 1280.0 - 330.0 - 48.0);
        assert_eq!(app.dynamic_test_columns(), 2);

        // Test window resizing via WindowResized message
        let _ = app.update(Message::WindowResized(800, 600));
        assert_eq!(app.window_width, 800);
        assert_eq!(app.window_height, 600);
        assert_eq!(app.dynamic_sidebar_width(), 260.0);
        assert_eq!(app.dynamic_main_width(), 492.0);
        assert_eq!(app.dynamic_test_columns(), 1);

        // Intermediate size: 1000px width
        let _ = app.update(Message::WindowResized(1000, 700));
        assert_eq!(app.dynamic_sidebar_width(), 290.0);
        assert_eq!(app.dynamic_main_width(), 662.0);
        assert_eq!(app.dynamic_test_columns(), 2);

        // QHD width: 1440px
        let _ = app.update(Message::WindowResized(1440, 900));
        assert_eq!(app.dynamic_sidebar_width(), 330.0);
        assert_eq!(app.dynamic_main_width(), 1062.0);
        assert_eq!(app.dynamic_test_columns(), 3);

        // 1080p full width: 1920px -> 4 columns
        let _ = app.update(Message::WindowResized(1920, 1080));
        assert_eq!(app.dynamic_sidebar_width(), 360.0);
        assert_eq!(app.dynamic_main_width(), 1512.0);
        assert_eq!(app.dynamic_test_columns(), 4);

        // Ultrawide: 3440px -> 4 columns
        let _ = app.update(Message::WindowResized(3440, 1440));
        assert_eq!(app.dynamic_sidebar_width(), 360.0);
        assert_eq!(app.dynamic_test_columns(), 4);
    }

    #[test]
    fn test_rayscheduling_selection_and_scene_mapping() {
        let (mut app, _) = GPUBenchApp::new(GuiCliArgs::default());

        // Verify available_tests contains canonical "RayScheduling"
        assert!(app.available_tests.contains(&"RayScheduling".to_string()));

        // Toggle RayScheduling
        let _ = app.update(Message::TestToggled("RayScheduling".to_string(), true));
        assert!(app.selected_tests.contains("RayScheduling"));

        // With ScenePreset::All, all 8 scene render workloads must be selected
        app.selected_scene = ScenePreset::All;
        let get_workload = |id: &str| WORKLOADS.iter().find(|w| w.id == id).unwrap();
        assert!(app.is_workload_selected(get_workload("rt_sched_full_showroom_trad")));
        assert!(app.is_workload_selected(get_workload("rt_sched_full_showroom_wl")));
        assert!(app.is_workload_selected(get_workload("rt_sched_full_indoor_trad")));
        assert!(app.is_workload_selected(get_workload("rt_sched_full_indoor_wl")));
        assert!(app.is_workload_selected(get_workload("rt_sched_full_outdoor_trad")));
        assert!(app.is_workload_selected(get_workload("rt_sched_full_outdoor_wl")));
        assert!(app.is_workload_selected(get_workload("rt_sched_full_forest_trad")));
        assert!(app.is_workload_selected(get_workload("rt_sched_full_forest_wl")));

        // Verify result mapping for each scene produces correct workload ID
        let mut res = ResultData::default();
        res.configIndex = 21;
        res.benchmarkName = "RayScheduling (Showroom Studio) (Full Scene Render - Megakernel)".to_string();
        assert_eq!(map_result_to_workload_id(&res), Some("rt_sched_full_showroom_trad"));

        res.configIndex = 22;
        res.benchmarkName = "RayScheduling (Showroom Studio) (Full Scene Render - Work Lists)".to_string();
        assert_eq!(map_result_to_workload_id(&res), Some("rt_sched_full_showroom_wl"));

        res.configIndex = 21;
        res.benchmarkName = "RayScheduling (Indoor Atrium) (Full Scene Render - Megakernel)".to_string();
        assert_eq!(map_result_to_workload_id(&res), Some("rt_sched_full_indoor_trad"));

        res.configIndex = 22;
        res.benchmarkName = "RayScheduling (Indoor Atrium) (Full Scene Render - Work Lists)".to_string();
        assert_eq!(map_result_to_workload_id(&res), Some("rt_sched_full_indoor_wl"));

        res.configIndex = 21;
        res.benchmarkName = "RayScheduling (Outdoor Landscape) (Full Scene Render - Megakernel)".to_string();
        assert_eq!(map_result_to_workload_id(&res), Some("rt_sched_full_outdoor_trad"));

        res.configIndex = 22;
        res.benchmarkName = "RayScheduling (Outdoor Landscape) (Full Scene Render - Work Lists)".to_string();
        assert_eq!(map_result_to_workload_id(&res), Some("rt_sched_full_outdoor_wl"));

        res.configIndex = 21;
        res.benchmarkName = "RayScheduling (Open-World Forest) (Full Scene Render - Megakernel)".to_string();
        assert_eq!(map_result_to_workload_id(&res), Some("rt_sched_full_forest_trad"));

        res.configIndex = 22;
        res.benchmarkName = "RayScheduling (Open-World Forest) (Full Scene Render - Work Lists)".to_string();
        assert_eq!(map_result_to_workload_id(&res), Some("rt_sched_full_forest_wl"));
    }

    #[test]
    fn test_results_layout_and_render_path_resolution() {
        let (mut app, _) = GPUBenchApp::new(GuiCliArgs::default());

        // Test path resolution for renders
        let renders_dir = find_renders_file("renders");
        assert!(renders_dir.is_some(), "find_renders_file('renders') must locate the repository renders directory");

        // Switch app state to complete
        app.state = AppState::Complete { total_configs: 64 };
        app.completed_configs_count = 64;
        app.total_expected_configs = 64;

        // Test rendering view at standard scale
        let _ = app.view();

        // Test rendering view at Windows UI scaling resolution (1024x675)
        let _ = app.update(Message::WindowResized(1024, 675));
        let _ = app.view();

        // Test rendering view at minimum window size (960x640)
        let _ = app.update(Message::WindowResized(960, 640));
        let _ = app.view();
    }
}

