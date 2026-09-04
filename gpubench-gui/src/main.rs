#![windows_subsystem = "windows"]

use iced::widget::{button, column, container, progress_bar, row, scrollable, text, Space, tooltip};
use iced::{color, Background, Border, Command, Element, Length, Theme, executor, Application, Settings};
use gpubench_core::{get_available_benchmarks, run_benchmarks, ResultData};
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

#[derive(Debug, Clone)]
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
        "RayTracing" => "Peak BVH acceleration structure traversal and ray-triangle intersection throughput.",
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

pub fn main() -> iced::Result {
    if std::env::var_os("MESA_VK_IGNORE_CONFORMANCE_WARNING").is_none() {
        unsafe { std::env::set_var("MESA_VK_IGNORE_CONFORMANCE_WARNING", "1") };
    }
    resolve_kernel_path();
    GPUBenchApp::run(Settings {
        antialiasing: true,
        window: iced::window::Settings {
            size: iced::Size::new(1280.0, 840.0),
            min_size: Some(iced::Size::new(1080.0, 720.0)),
            ..Default::default()
        },
        ..Settings::default()
    })
}

static PROGRESS_SENDER: LazyLock<Mutex<Option<Sender<ResultData>>>> = LazyLock::new(|| Mutex::new(None));

fn progress_callback(res: &ResultData) {
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

pub const SYSTEM_DEVICE_ID: u32 = 999;

#[derive(Clone, Debug, Default)]
pub struct CellResult {
    pub value_str: String,
    pub numeric: f64,
    pub unit: String,
    pub is_running: bool,
    pub is_unsupported: bool,
}

#[derive(Clone, Debug)]
pub struct WorkloadDef {
    pub id: &'static str,
    pub category: &'static str,
    pub label: &'static str,
    pub approach: &'static str,
    pub default_unit: &'static str,
    pub desc: &'static str,
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
        is_system: false,
    },
    WorkloadDef {
        id: "fp32",
        category: "COMPUTE PIPELINES",
        label: "FP32 (Single Precision)",
        approach: "32-bit vector FMA",
        default_unit: "TFLOPS",
        desc: "Single-precision 32-bit float compute throughput, standard for graphics and compute shaders.",
        is_system: false,
    },
    WorkloadDef {
        id: "fp16_vec",
        category: "COMPUTE PIPELINES",
        label: "FP16 (Vector)",
        approach: "16-bit packed SIMD",
        default_unit: "TFLOPS",
        desc: "Half-precision 16-bit float SIMD vector throughput.",
        is_system: false,
    },
    WorkloadDef {
        id: "fp16_mat",
        category: "COMPUTE PIPELINES",
        label: "FP16 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TFLOPS",
        desc: "Hardware matrix cores half-precision tensor compute throughput using VK_KHR_cooperative_matrix.",
        is_system: false,
    },
    WorkloadDef {
        id: "bf16_vec",
        category: "COMPUTE PIPELINES",
        label: "BF16 (Vector)",
        approach: "Bfloat16 SIMD",
        default_unit: "TFLOPS",
        desc: "16-bit Brain Floating Point vector throughput for machine learning models.",
        is_system: false,
    },
    WorkloadDef {
        id: "bf16_mat",
        category: "COMPUTE PIPELINES",
        label: "BF16 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TFLOPS",
        desc: "Hardware matrix cores Bfloat16 tensor compute throughput.",
        is_system: false,
    },
    WorkloadDef {
        id: "fp8_vec",
        category: "COMPUTE PIPELINES",
        label: "FP8 (Vector)",
        approach: "8-bit float vector",
        default_unit: "TFLOPS",
        desc: "Quarter-precision 8-bit float arithmetic (E4M3/E5M2) for modern low-bit AI inference.",
        is_system: false,
    },
    WorkloadDef {
        id: "fp8_mat",
        category: "COMPUTE PIPELINES",
        label: "FP8 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TFLOPS",
        desc: "Hardware 8-bit float matrix multiplication throughput.",
        is_system: false,
    },
    WorkloadDef {
        id: "int8_vec",
        category: "COMPUTE PIPELINES",
        label: "INT8 (Vector)",
        approach: "8-bit DP4A integer",
        default_unit: "TOPS",
        desc: "8-bit quantized integer vector throughput.",
        is_system: false,
    },
    WorkloadDef {
        id: "int8_mat",
        category: "COMPUTE PIPELINES",
        label: "INT8 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TOPS",
        desc: "Hardware matrix 8-bit integer tensor throughput.",
        is_system: false,
    },
    WorkloadDef {
        id: "int4_vec",
        category: "COMPUTE PIPELINES",
        label: "INT4 (Vector)",
        approach: "4-bit packed integer",
        default_unit: "TOPS",
        desc: "4-bit quantized vector arithmetic.",
        is_system: false,
    },
    WorkloadDef {
        id: "int4_mat",
        category: "COMPUTE PIPELINES",
        label: "INT4 (Matrix / Tensor)",
        approach: "Cooperative Matrix",
        default_unit: "TOPS",
        desc: "Hardware matrix 4-bit integer throughput.",
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
        is_system: false,
    },
    WorkloadDef {
        id: "cache_l0",
        category: "MEMORY & SYSTEM",
        label: "L0 Cache Latency",
        approach: "L0 / LDS pointer chase",
        default_unit: "ns",
        desc: "Level 0 vector cache access latency in nanoseconds (lower is better).",
        is_system: false,
    },
    WorkloadDef {
        id: "cache_l1",
        category: "MEMORY & SYSTEM",
        label: "L1 Cache Latency",
        approach: "GL1C pointer chase",
        default_unit: "ns",
        desc: "Level 1 cache access latency in nanoseconds (lower is better).",
        is_system: false,
    },
    WorkloadDef {
        id: "cache_l2",
        category: "MEMORY & SYSTEM",
        label: "L2 Cache Latency",
        approach: "GL2C pointer chase",
        default_unit: "ns",
        desc: "Level 2 cache access latency in nanoseconds (lower is better).",
        is_system: false,
    },
    WorkloadDef {
        id: "cache_l3",
        category: "MEMORY & SYSTEM",
        label: "L3 Cache Latency",
        approach: "MALL / Infinity Cache chase",
        default_unit: "ns",
        desc: "Level 3 / Infinity Cache access latency in nanoseconds (lower is better).",
        is_system: false,
    },
    WorkloadDef {
        id: "sys_mem_bw_multi",
        category: "MEMORY & SYSTEM",
        label: "System RAM (Multi-Thread)",
        approach: "AVX2 multi-thread stream",
        default_unit: "GB/s",
        desc: "Multi-threaded host RAM copy and streaming bandwidth from CPU to system DDR memory.",
        is_system: true,
    },
    WorkloadDef {
        id: "sys_mem_bw_single",
        category: "MEMORY & SYSTEM",
        label: "System RAM (1 Thread)",
        approach: "AVX2 single-thread stream",
        default_unit: "GB/s",
        desc: "Single-threaded host system RAM streaming bandwidth.",
        is_system: true,
    },
    WorkloadDef {
        id: "sys_mem_lat",
        category: "MEMORY & SYSTEM",
        label: "System RAM Latency",
        approach: "CPU pointer chase",
        default_unit: "ns",
        desc: "Host memory access latency in nanoseconds (lower is better).",
        is_system: true,
    },

    // GRAPHICS & ROP
    WorkloadDef {
        id: "rop_rgba8",
        category: "GRAPHICS & ROP",
        label: "RGBA8 Color Fill",
        approach: "Hardware 32-bit ROP",
        default_unit: "GPixels/s",
        desc: "Standard 32-bit RGBA ROP rasterization fill rate.",
        is_system: false,
    },
    WorkloadDef {
        id: "rop_rgba16f",
        category: "GRAPHICS & ROP",
        label: "RGBA16F HDR Fill",
        approach: "Hardware 64-bit HDR ROP",
        default_unit: "GPixels/s",
        desc: "64-bit HDR framebuffer rasterization throughput.",
        is_system: false,
    },
    WorkloadDef {
        id: "rop_blend",
        category: "GRAPHICS & ROP",
        label: "Alpha Blending Fill",
        approach: "Hardware ROP blending",
        default_unit: "GPixels/s",
        desc: "ROP hardware alpha blend rasterization rate.",
        is_system: false,
    },

    // RAY TRACING ACCELERATION
    WorkloadDef {
        id: "rt_triangle",
        category: "RAY TRACING ACCELERATION",
        label: "Ray-Triangle Intersect",
        approach: "Hardware BVH Ray Query",
        default_unit: "GIS/s",
        desc: "Peak BVH acceleration structure traversal and ray-triangle intersection throughput.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_divergence",
        category: "RAY TRACING ACCELERATION",
        label: "Divergence Traversal",
        approach: "Coherence gradient rays",
        default_unit: "GRays/s",
        desc: "BVH traversal throughput under heavy branch and wave execution divergence.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_anyhit",
        category: "RAY TRACING ACCELERATION",
        label: "AnyHit (Alpha-Tested)",
        approach: "AnyHit opacity eval",
        default_unit: "GRays/s",
        desc: "Traversal and shader invocation throughput against transparent, alpha-tested geometry.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_incoh_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Incoherent Rays (Traditional)",
        approach: "Monolithic Megakernel",
        default_unit: "MRays/s",
        desc: "Traditional unordered traversal of highly divergent secondary rays.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_incoh_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Incoherent Rays (Work Lists)",
        approach: "Directional Binning",
        default_unit: "MRays/s",
        desc: "Using ExecuteIndirect (Work Lists): Sorts scattered secondary rays into directional bins before BVH traversal.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_incoherent",
        category: "RAY TRACING ACCELERATION",
        label: "Incoherent Bounces",
        approach: "Cosine diffuse bounce",
        default_unit: "GRays/s",
        desc: "Cache hit rate and traversal speed under randomized non-coherent diffuse bounce distributions.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_prim_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Primary Rays (Traditional)",
        approach: "Monolithic Megakernel",
        default_unit: "MRays/s",
        desc: "Traditional monolithic primary camera ray dispatch.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_prim_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Primary Rays (Work Lists)",
        approach: "Material Sorting",
        default_unit: "MRays/s",
        desc: "Using ExecuteIndirect (Work Lists): Separates camera ray generation and material shading into dedicated work queues.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_mat_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Material Shading (Traditional)",
        approach: "Monolithic Megakernel",
        default_unit: "MHits/s",
        desc: "Traditional monolithic compute shader with dynamic loop branching across heterogeneous materials.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_mat_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Material Shading (Work Lists)",
        approach: "Using ExecuteIndirect (Work Lists)",
        default_unit: "MHits/s",
        desc: "Using ExecuteIndirect (Work Lists): 2-pass parallel compaction into uniform material queues to eliminate branch divergence.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_pt_trad",
        category: "RAY TRACING ACCELERATION",
        label: "Path Tracing (Traditional)",
        approach: "Monolithic Megakernel",
        default_unit: "MRays/s",
        desc: "Traditional megakernel path tracing where terminated rays leave SIMD lanes idle across multiple bounces.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_pt_wl",
        category: "RAY TRACING ACCELERATION",
        label: "Path Tracing (Work Lists)",
        approach: "Active Ray Compaction",
        default_unit: "MRays/s",
        desc: "Using ExecuteIndirect (Work Lists): Compacts non-terminated bounce rays into packed queues, keeping wavefronts 100% full.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_pathtracing",
        category: "RAY TRACING ACCELERATION",
        label: "Multi-Bounce Path Tracing",
        approach: "Stochastic 8-bounce GI",
        default_unit: "MRays/s",
        desc: "Full multi-bounce stochastic Monte Carlo path tracing with global illumination and cosine sampling.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_payload",
        category: "RAY TRACING ACCELERATION",
        label: "Payload Pressure",
        approach: "16B - 256B register payload",
        default_unit: "GRays/s",
        desc: "Ray traversal performance under heavy recursive register payload pressure.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_blas_build",
        category: "RAY TRACING ACCELERATION",
        label: "BLAS Build (1M Tris)",
        approach: "Bottom-level AS build",
        default_unit: "MTris/s",
        desc: "Hardware bottom-level acceleration structure construction rate.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_blas_update",
        category: "RAY TRACING ACCELERATION",
        label: "BLAS Update (1M Tris)",
        approach: "Dynamic in-place BVH refit",
        default_unit: "MTris/s",
        desc: "Bottom-level acceleration structure dynamic mesh refit speed.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_tlas_build",
        category: "RAY TRACING ACCELERATION",
        label: "TLAS Build (10k Inst)",
        approach: "Top-level hierarchy build",
        default_unit: "MInst/s",
        desc: "Top-level instance hierarchy construction throughput.",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_procedural",
        category: "RAY TRACING ACCELERATION",
        label: "Procedural Geometry",
        approach: "Analytical AABB eval",
        default_unit: "GRays/s",
        desc: "Intersection evaluation against mathematically defined procedural primitives (spheres).",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_ser",
        category: "RAY TRACING ACCELERATION",
        label: "Hardware Reordering (SER)",
        approach: "Shader Execution Reordering (SER)",
        default_unit: "MRays/s",
        desc: "Hardware-level dynamic thread reordering during traversal (VK_NV_ray_tracing_invocation_reorder).",
        is_system: false,
    },
    WorkloadDef {
        id: "rt_sched_workgraph",
        category: "RAY TRACING ACCELERATION",
        label: "GPU Work Graphs",
        approach: "Autonomous Node Enqueue",
        default_unit: "MRays/s",
        desc: "Autonomous GPU-driven work creation via node record routing (VK_AMDX_shader_enqueue).",
        is_system: false,
    },
];

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
    
    // Multi-Hardware Telemetry
    monitored_devices: Vec<DeviceTelemetry>,
    selected_telemetry_device: usize,

    // Multi-Device Results & Targets
    active_device_targets: Vec<(u32, String)>,
    results_map: HashMap<(u32, &'static str), CellResult>,
    completed_configs_count: usize,
    total_expected_configs: usize,

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
    SaveResults,
    Retest,
}

impl Application for GPUBenchApp {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = ();

    fn new(_flags: ()) -> (Self, Command<Message>) {
        let tests = get_available_benchmarks();
        
        let hw = gpubench_core::get_available_hardware();
        let backends = available_backends(&hw);
        let selected_backend = backends.first().cloned().unwrap_or_else(|| "VULKAN".to_string());
        let devices = devices_for_backend(&hw, &selected_backend);
        
        let mut initial_devices = HashSet::new();
        for d in &devices {
            initial_devices.insert(d.clone());
        }

        let mut initial_tests = HashSet::new();
        for t in &tests {
            initial_tests.insert(t.clone());
        }

        let mut monitored = discover_all_devices();
        poll_all_devices(&mut monitored, false);

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
                dump_renders: false,
                current_benchmark: String::from("Waiting to start..."),
                current_devices_label: String::from(""),
                monitored_devices: monitored,
                selected_telemetry_device: 0,
                active_device_targets: Vec::new(),
                results_map: HashMap::new(),
                completed_configs_count: 0,
                total_expected_configs: 1,
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
            },
            Command::none()
        )
    }

    fn title(&self) -> String {
        String::from("GPUBench — Workstation GPU Profiler")
    }

    fn subscription(&self) -> iced::Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(500)).map(|_| Message::Tick)
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::BackendSelected(backend) => {
                if let AppState::Setup { selected_backend, available_devices, .. } = &mut self.state {
                    *selected_backend = backend.clone();
                    self.selected_backend = backend.clone();
                    
                    let hw = gpubench_core::get_available_hardware();
                    let new_devices = devices_for_backend(&hw, selected_backend);
                    *available_devices = new_devices.clone();
                    self.available_devices = new_devices.clone();
                    
                    self.selected_devices.clear();
                    for d in &self.available_devices {
                        self.selected_devices.insert(d.clone());
                    }

                    if backend != "VULKAN" {
                        self.selected_tests.retain(|t| !t.starts_with("Ray"));
                    }
                }
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
                Command::none()
            }
            Message::DumpRendersToggled(val) => {
                self.dump_renders = val;
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
                    if name.contains("System Memory") {
                        self.selected_devices.insert("System: Host CPU & RAM".to_string());
                    }
                } else {
                    self.selected_tests.remove(&name);
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
                Command::none()
            }
            Message::TestGroupSelected(group) => {
                if let AppState::Setup { available_tests, .. } = &mut self.state {
                    match group.as_str() {
                        "ALL" => {
                            for t in available_tests.iter() {
                                if self.selected_backend == "VULKAN" || !t.starts_with("Ray") {
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
                                .filter(|t| t.as_str() == "Device Memory Bandwidth" || t.as_str() == "Pixel Fill Rate")
                                .cloned().collect();
                            let all_selected = mem.iter().all(|t| self.selected_tests.contains(t));
                            if all_selected {
                                for t in &mem { self.selected_tests.remove(t); }
                            } else {
                                for t in mem { self.selected_tests.insert(t); }
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

                    let mut tests_to_run: Vec<String> = Vec::new();
                    for t in &self.available_tests {
                        if self.selected_tests.contains(t) {
                            tests_to_run.push(t.clone());
                        }
                    }
                    if !has_system {
                        tests_to_run.retain(|t| !t.contains("System Memory") && !t.contains("SysMem"));
                    }
                    if gpu_indices.is_empty() {
                        tests_to_run.retain(|t| t.contains("System Memory"));
                    }
                    if b_str != "VULKAN" {
                        tests_to_run.retain(|t| !t.starts_with("Ray"));
                    }

                    if tests_to_run.is_empty() { return Command::none(); }

                    // Estimate total expected configs across all targets
                    let mut gpu_configs = 0;
                    for t in &tests_to_run {
                        if t.contains("System Memory") { continue; }
                        gpu_configs += match t.as_str() {
                            "Device Memory Bandwidth" => 9,
                            "Pixel Fill Rate" => 3,
                            "FP16" | "BF16" | "FP8" | "INT8" | "INT4" => 2,
                            "RayASBuild" => 3,
                            "RayPathTracing" => 3,
                            "RayScheduling" | "RayExecutionParadigm" => 8,
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
                poll_all_devices(&mut self.monitored_devices, is_running);

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
                self.state = AppState::Complete { total_configs: final_count };
                self.current_benchmark = String::from("Complete");
                return Command::none();
            }
            Message::BenchmarksFailed(err) => {
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
                    timestamp: u64,
                    backend: String,
                    devices: Vec<String>,
                    results: Vec<DeviceReport>,
                    telemetry: Vec<TelemetryReport>,
                }

                #[derive(serde::Serialize)]
                struct DeviceReport {
                    device_id: u32,
                    device_name: String,
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

                let mut device_reports = Vec::new();
                for (dev_id, dev_name) in &self.active_device_targets {
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
                            });
                        }
                    }
                    device_reports.push(DeviceReport {
                        device_id: *dev_id,
                        device_name: dev_name.clone(),
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

                let report = BenchmarkReport {
                    timestamp: now,
                    backend: self.selected_backend.clone(),
                    devices: self.active_device_targets.iter().map(|(_, n)| n.clone()).collect(),
                    results: device_reports,
                    telemetry: telem_reports,
                };

                let filename = format!("gpubench_results_{}.json", now);
                if let Ok(json_str) = serde_json::to_string_pretty(&report) {
                    let _ = std::fs::write(&filename, &json_str);
                    if let Some(path) = rfd::FileDialog::new()
                        .set_file_name(&filename)
                        .add_filter("JSON Report", &["json"])
                        .save_file()
                    {
                        let _ = std::fs::write(path, &json_str);
                    }
                    self.current_benchmark = format!("Results exported: {}", filename);
                }
                return Command::none();
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
                            // GPU Utilization
                            column![
                                row![text("GPU UTIL").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(util_str).size(9).style(util_color)],
                                progress_bar(0.0..=1.0, util_pct).height(3.0)
                            ].spacing(1),
                            // Temperatures
                            row![
                                column![text("EDGE").size(7).style(color!(0x64748B)), text(edge_str).size(9).style(temp_color)].spacing(1),
                                Space::with_width(Length::Fill),
                                column![text("HOTSPOT").size(7).style(color!(0x64748B)), text(junc_str).size(9).style(color!(0xF59E0B))].spacing(1),
                                Space::with_width(Length::Fill),
                                column![text("VRAM").size(7).style(color!(0x64748B)), text(mem_t_str).size(9).style(color!(0xA5B4FC))].spacing(1),
                            ],
                            // Power gauge
                            column![
                                row![text("POWER").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(power_str).size(9).style(color!(0x38BDF8))],
                                progress_bar(0.0..=1.0, power_pct).height(3.0)
                            ].spacing(1),
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
                        column![
                            column![
                                row![text("CPU TEMP").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(temp_str).size(9).style(temp_color)],
                                progress_bar(0.0..=1.0, temp_pct).height(3.0)
                            ].spacing(1),
                            column![
                                row![text("CPU POWER").size(8).style(color!(0x94A3B8)), Space::with_width(Length::Fill), text(power_str).size(9).style(color!(0x38BDF8))],
                                progress_bar(0.0..=1.0, power_pct).height(3.0)
                            ].spacing(1),
                        ].spacing(4)
                    );
                }
            }

            container(hud_content)
                .padding(12)
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

                let start_btn = button(
                    container(text("START BENCHMARK").size(13).style(color!(0xFFFFFF)))
                        .width(Length::Fill)
                        .center_x()
                )
                .width(Length::Fill)
                .padding([12, 0])
                .on_press(Message::StartBenchmarks)
                .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton)));

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

                let sidebar = container(
                    column![
                        brand_block,
                        Space::with_height(14),
                        telemetry_panel,
                        Space::with_height(14),
                        text("COMPUTE API").size(10).style(color!(0x64748B)),
                        Space::with_height(4),
                        api_row,
                        Space::with_height(14),
                        text("TARGET DEVICES").size(10).style(color!(0x64748B)),
                        Space::with_height(4),
                        device_col,
                        Space::with_height(14),
                        text("OPTIONS").size(10).style(color!(0x64748B)),
                        Space::with_height(4),
                        dump_renders_btn,
                        Space::with_height(Length::Fill),
                        start_btn
                    ]
                )
                .width(Length::Fixed(270.0))
                .height(Length::Fill)
                .padding(18)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0A0B10))),
                    border: Border { color: color!(0x1A1E2B), width: 1.0, ..Default::default() },
                    ..Default::default()
                });

                let create_pill_grid_with_tooltips = |title: &str, accent_color: iced::Color, is_rt: bool, items: Vec<(&str, &str)>| {
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
                        if available_tests.contains(&key.to_string()) {
                            let is_checked = self.selected_tests.contains(key);
                            let name = key.to_string();
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
                        create_pill_grid_with_tooltips("GPU Memory & Raster", color!(0x0EA5E9), false, vec![
                            ("Device Memory Bandwidth", "GPU VRAM Bandwidth"),
                            ("Pixel Fill Rate", "Pixel Fill Rate"),
                        ]),
                        Space::with_height(12),
                        create_pill_grid_with_tooltips("Host System Memory", color!(0x38BDF8), false, vec![
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
                    let rt_top = create_pill_grid_with_tooltips("Ray Tracing Acceleration", color!(0x10B981), true, vec![
                        ("RayTracing", "Ray-Triangle Intersect"),
                        ("RayDivergence", "Divergence Traversal"),
                        ("RayAnyHit", "AnyHit Alpha-Tested"),
                        ("RayIncoherent", "Incoherent Bounces"),
                        ("RayPayload", "Payload Pressure"),
                        ("RayASBuild", "BLAS & TLAS Build"),
                        ("RayProcedural", "Procedural Geometry"),
                        ("RayMaterialDivergence", "Material Divergence"),
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

                    let is_scheduling_checked = self.selected_tests.contains("RayScheduling") || self.selected_tests.contains("RayExecutionParadigm");
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
                    .filter(|t| *t == "Device Memory Bandwidth" || *t == "Pixel Fill Rate").collect();
                let sys_tests: Vec<&String> = available_tests.iter()
                    .filter(|t| t.contains("System Memory")).collect();
                let rt_tests: Vec<&String> = available_tests.iter()
                    .filter(|t| t.starts_with("Ray")).collect();
                
                let all_selected = available_tests.iter().all(|t| self.selected_tests.contains(t));
                let none_selected = self.selected_tests.is_empty();
                let compute_all = !compute_tests.is_empty() && compute_tests.iter().all(|t| self.selected_tests.contains(*t));
                let mem_all = !mem_tests.is_empty() && mem_tests.iter().all(|t| self.selected_tests.contains(*t));
                let sys_all = !sys_tests.is_empty() && sys_tests.iter().all(|t| self.selected_tests.contains(*t));
                let rt_all = !rt_tests.is_empty() && rt_tests.iter().all(|t| self.selected_tests.contains(*t));
                let is_rt_avail = selected_backend == "VULKAN";

                let group_toggles = row![
                    button(text("All").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 14])
                        .on_press(Message::TestGroupSelected("ALL".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: all_selected, is_disabled: false }))),
                    button(text("None").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 14])
                        .on_press(Message::TestGroupSelected("NONE".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: none_selected, is_disabled: false }))),
                    button(text("Compute").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 14])
                        .on_press(Message::TestGroupSelected("COMPUTE".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: compute_all, is_disabled: false }))),
                    button(text("Memory").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 14])
                        .on_press(Message::TestGroupSelected("MEMORY".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: mem_all, is_disabled: false }))),
                    button(text("Ray Tracing").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 14])
                        .on_press(Message::TestGroupSelected("RAY TRACING".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: is_rt_avail && rt_all, is_disabled: !is_rt_avail }))),
                    button(text("System").size(11).horizontal_alignment(iced::alignment::Horizontal::Center))
                        .padding([5, 14])
                        .on_press(Message::TestGroupSelected("SYSTEM".to_string()))
                        .style(iced::theme::Button::Custom(Box::new(SleekGroupChip { is_highlighted: sys_all, is_disabled: false }))),
                ].spacing(6);

                let main_area = container(
                    scrollable(
                        column![
                            row![
                                column![
                                    text("Benchmarks").size(20).style(color!(0xF8FAFC)),
                                    text("Select GPU compute, memory, and ray tracing benchmarks to profile").size(12).style(color!(0x64748B))
                                ].spacing(2),
                                Space::with_width(Length::Fill),
                                group_toggles
                            ].align_items(iced::Alignment::Center),
                            Space::with_height(20),
                            row![
                                comp_col.width(Length::FillPortion(1)),
                                sys_col.width(Length::FillPortion(1)),
                                rt_col.width(Length::FillPortion(1)),
                            ].spacing(16)
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
                let retry_btn = button(
                    container(text("RUN NEW TEST").size(13).style(color!(0xFFFFFF)))
                        .width(Length::Fill)
                        .center_x()
                )
                .width(Length::Fill)
                .padding([12, 0])
                .on_press(Message::Retest)
                .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton)));

                let sidebar = container(
                    column![
                        brand_block,
                        Space::with_height(20),
                        telemetry_panel,
                        Space::with_height(Length::Fill),
                        retry_btn
                    ]
                )
                .width(Length::Fixed(270.0))
                .height(Length::Fill)
                .padding(20)
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
                        text("Click RUN NEW TEST to reconfigure the benchmark suite.").size(12).style(color!(0x64748B)),
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
                    Space::with_width(16),
                    button(
                        container(text("SAVE RESULTS").size(12).style(color!(0xFFFFFF)))
                            .padding([6, 14])
                            .center_x()
                    )
                    .on_press(Message::SaveResults)
                    .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton))),
                ].align_items(iced::Alignment::Center);

                let targets = if self.active_device_targets.is_empty() {
                    vec![(0, "GPU 0".to_string())]
                } else {
                    self.active_device_targets.clone()
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
                    ("COMPUTE PIPELINES", color!(0x8B5CF6)),
                    ("MEMORY & SYSTEM", color!(0x0EA5E9)),
                    ("GRAPHICS & ROP", color!(0xF59E0B)),
                    ("RAY TRACING ACCELERATION", color!(0x10B981)),
                ];

                let mut table_column = column![header_row].spacing(12);
                let mut has_any_category = false;

                for (cat_name, cat_color) in categories {
                    let cat_workloads: Vec<&WorkloadDef> = WORKLOADS.iter()
                        .filter(|w| w.category == cat_name)
                        .filter(|w| !w.is_system || targets.iter().any(|(d, _)| *d == SYSTEM_DEVICE_ID))
                        .filter(|w| !w.category.starts_with("RAY") || self.selected_backend == "VULKAN")
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
                        text(cat_name).size(12).style(color!(0xE2E8F0))
                    ].align_items(iced::Alignment::Center);

                    let mut cat_rows = column![].spacing(4);

                    for w in cat_workloads {
                        let is_selected = self.is_workload_selected(w);

                        let name_block = column![
                            text(w.label).size(12).style(if is_selected { color!(0xF1F5F9) } else { color!(0x64748B) }),
                            text(w.approach).size(10).style(if is_selected { color!(0x818CF8) } else { color!(0x475569) }),
                        ].spacing(2);

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
                        .width(Length::Fixed(320.0))
                        .padding([4, 10]);

                        let mut row_cells = row![left_col].spacing(8).align_items(iced::Alignment::Center);

                        for (dev_id, _) in &targets {
                            let cell_element: Element<'_, Message> = if (w.is_system && *dev_id != SYSTEM_DEVICE_ID)
                                || (!w.is_system && *dev_id == SYSTEM_DEVICE_ID)
                            {
                                container(text("—").size(11).style(color!(0x334155)))
                                    .width(Length::FillPortion(1))
                                    .padding([5, 8])
                                    .center_x()
                                    .into()
                            } else {
                                let cell = self.results_map.get(&(*dev_id, w.id));
                                let (val_str, text_color, bg_color, border_color) = if let Some(c) = cell {
                                    if c.is_running {
                                        ("RUNNING...", color!(0x22D3EE), color!(0x06B6D4, 0.18), color!(0x06B6D4, 0.5))
                                    } else if c.is_unsupported {
                                        ("UNSUPPORTED", color!(0xF87171), color!(0xEF4444, 0.12), color!(0xEF4444, 0.35))
                                    } else if !c.value_str.is_empty() {
                                        (c.value_str.as_str(), color!(0x34D399), color!(0x10B981, 0.14), color!(0x10B981, 0.4))
                                    } else if matches!(self.state, AppState::Complete { .. }) {
                                        ("UNSUPPORTED", color!(0xF87171), color!(0xEF4444, 0.12), color!(0xEF4444, 0.35))
                                    } else {
                                        ("PENDING", color!(0x64748B), color!(0x181C28), color!(0x222838))
                                    }
                                } else if !is_selected {
                                    ("—", color!(0x475569), color!(0x10131B), color!(0x161B26))
                                } else if matches!(self.state, AppState::Complete { .. }) {
                                    ("UNSUPPORTED", color!(0xF87171), color!(0xEF4444, 0.12), color!(0xEF4444, 0.35))
                                } else {
                                    ("PENDING", color!(0x64748B), color!(0x181C28), color!(0x222838))
                                };

                                container(
                                    text(val_str).size(11).style(text_color)
                                )
                                .width(Length::FillPortion(1))
                                .padding([5, 8])
                                .center_x()
                                .style(move |_t: &Theme| container::Appearance {
                                    background: Some(Background::Color(bg_color)),
                                    border: Border { radius: 6.0.into(), width: 1.0, color: border_color },
                                    ..Default::default()
                                })
                                .into()
                            };

                            row_cells = row_cells.push(cell_element);
                        }

                        let row_card = container(row_cells)
                            .width(Length::Fill)
                            .padding([4, 6])
                            .style(|_t: &Theme| container::Appearance {
                                background: Some(Background::Color(color!(0x0C0E16))),
                                border: Border { radius: 6.0.into(), width: 1.0, color: color!(0x171C2B) },
                                ..Default::default()
                            });

                        cat_rows = cat_rows.push(row_card);
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

                let action_buttons: Element<'_, Message> = if matches!(self.state, AppState::Complete { .. }) {
                    column![
                        button(
                            container(text("EXPORT JSON").size(13).style(color!(0xFFFFFF)))
                                .width(Length::Fill)
                                .center_x()
                        )
                        .width(Length::Fill)
                        .padding([11, 0])
                        .on_press(Message::SaveResults)
                        .style(iced::theme::Button::Custom(Box::new(SleekPrimaryButton))),
                        
                        Space::with_height(8),
                        
                        button(
                            container(text("RUN NEW TEST").size(13).style(color!(0xCBD5E1)))
                                .width(Length::Fill)
                                .center_x()
                        )
                        .width(Length::Fill)
                        .padding([11, 0])
                        .on_press(Message::Retest)
                        .style(iced::theme::Button::Custom(Box::new(SleekSecondaryButton)))
                    ]
                    .width(Length::Fill)
                    .into()
                } else {
                    Space::with_height(0).into()
                };

                let sidebar = container(
                    column![
                        brand_block,
                        Space::with_height(14),
                        telemetry_panel,
                        Space::with_height(14),
                        text("ACTIVE CONFIGURATION").size(10).style(color!(0x64748B)),
                        Space::with_height(4),
                        container(
                            column![
                                row![text("API:").size(10).style(color!(0x64748B)), Space::with_width(Length::Fill), text(format!("{} {}", &self.selected_backend, detect_dynamic_api_version(&self.selected_backend))).size(11).style(color!(0x38BDF8))],
                                row![text("DEVICES:").size(10).style(color!(0x64748B)), Space::with_width(Length::Fill), text(if self.current_devices_label.is_empty() { "—" } else { &self.current_devices_label }).size(10).style(color!(0xE2E8F0))],
                            ].spacing(3)
                        )
                        .padding(10)
                        .style(|_t: &Theme| container::Appearance {
                            background: Some(Background::Color(color!(0x11141E))),
                            border: Border { radius: 8.0.into(), width: 1.0, color: color!(0x1A202C) },
                            ..Default::default()
                        }),
                        Space::with_height(14),
                        text("CURRENT WORKLOAD").size(10).style(color!(0x64748B)),
                        Space::with_height(4),
                        text(if self.current_benchmark.is_empty() { "Complete" } else { &self.current_benchmark }).size(12).style(color!(0xA5B4FC)),
                        Space::with_height(Length::Fill),
                        action_buttons
                    ]
                )
                .width(Length::Fixed(270.0))
                .height(Length::Fill)
                .padding(18)
                .style(|_t: &Theme| container::Appearance {
                    background: Some(Background::Color(color!(0x0A0B10))),
                    border: Border { color: color!(0x1A1E2B), width: 1.0, ..Default::default() },
                    ..Default::default()
                });

                let main_area = container(
                    scrollable(
                        column![
                            global_progress,
                            table_column,
                            thermal_profile_section
                        ]
                        .spacing(20)
                        .width(Length::Fill)
                    )
                    .width(Length::Fill)
                    .height(Length::Fill)
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
        }
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }
}

impl GPUBenchApp {
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
            "rt_triangle" => self.selected_tests.contains("RayTracing"),
            "rt_divergence" => self.selected_tests.contains("RayDivergence") || self.selected_tests.contains("RayMaterialDivergence"),
            "rt_anyhit" => self.selected_tests.contains("RayAnyHit"),
            "rt_incoherent" => self.selected_tests.contains("RayIncoherent"),
            "rt_payload" => self.selected_tests.contains("RayPayload"),
            "rt_blas_build" | "rt_blas_update" | "rt_tlas_build" => self.selected_tests.contains("RayASBuild"),
            "rt_procedural" => self.selected_tests.contains("RayProcedural"),
            "rt_pathtracing" => self.selected_tests.contains("RayPathTracing"),
            id if id.starts_with("rt_sched_") => {
                self.selected_tests.iter().any(|t| t.contains("RayScheduling") || t.contains("RayExecutionParadigm"))
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
            format!("{:.2} ns", value)
        } else if res.metric == "MRays/s" || res.metric == "MHits/s" {
            let fps = (value * 1e6) / 2073600.0;
            format!("{:.1} {} ({:.0} FPS)", value, res.metric, fps)
        } else if value < 10.0 {
            format!("{:.2} {}", value, res.metric)
        } else {
            format!("{:.1} {}", value, res.metric)
        };

        if let Some(wid) = map_result_to_workload_id(res) {
            let entry = self.results_map.entry((dev_key, wid)).or_default();
            if entry.numeric > 0.0 && !is_unsupported {
                if res.metric == "ns" {
                    if value < entry.numeric {
                        entry.numeric = value;
                        entry.value_str = value_str.clone();
                    }
                } else if value > entry.numeric {
                    entry.numeric = value;
                    entry.value_str = value_str.clone();
                }
            } else {
                entry.numeric = value;
                entry.value_str = value_str.clone();
                entry.unit = res.metric.clone();
                entry.is_unsupported = is_unsupported;
            }
            entry.is_running = false;

            if is_unsupported {
                if wid == "fp8_vec" && !self.results_map.contains_key(&(dev_key, "fp8_mat")) {
                    let mat_entry = self.results_map.entry((dev_key, "fp8_mat")).or_default();
                    mat_entry.value_str = "UNSUPPORTED".to_string();
                    mat_entry.is_unsupported = true;
                    mat_entry.is_running = false;
                } else if wid == "int4_vec" && !self.results_map.contains_key(&(dev_key, "int4_mat")) {
                    let mat_entry = self.results_map.entry((dev_key, "int4_mat")).or_default();
                    mat_entry.value_str = "UNSUPPORTED".to_string();
                    mat_entry.is_unsupported = true;
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
                "Graphics" => {
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
                    if res.subcategory == "BLAS Build" { self.gpu_rt_blas_build = self.gpu_rt_blas_build.max(val_f32); }
                    if res.subcategory == "BLAS Update" { self.gpu_rt_blas_update = self.gpu_rt_blas_update.max(val_f32); }
                    if res.subcategory == "TLAS Build" { self.gpu_rt_tlas_build = self.gpu_rt_tlas_build.max(val_f32); }
                    if res.subcategory == "Incoherent Traversal" { self.gpu_rt_incoherent = self.gpu_rt_incoherent.max(val_f32); }
                    if res.subcategory == "Intersection tests" { self.gpu_rt_intersect = self.gpu_rt_intersect.max(val_f32); }
                    if res.subcategory == "Material Divergence" || res.subcategory == "Execution Divergence" { self.gpu_rt_divergence = self.gpu_rt_divergence.max(val_f32); }
                    if res.subcategory == "Payload Register Pressure" { self.gpu_rt_payload = self.gpu_rt_payload.max(val_f32); }
                    if res.subcategory == "Procedural Intersection" { self.gpu_rt_procedural = self.gpu_rt_procedural.max(val_f32); }
                    if res.subcategory == "Path Tracing" || res.benchmarkName.contains("PathTracing") { self.gpu_rt_pathtracing = self.gpu_rt_pathtracing.max(val_f32); }
                    if (res.benchmarkName.contains("RayScheduling") || res.benchmarkName.contains("RayExecutionParadigm"))
                        && !res.benchmarkName.contains("Stage Breakdown") {
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

    match res.component.as_str() {
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
        "Graphics" => {
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
                } else if res.configIndex == 12 || (res.subcategory.contains("Traditional") && res.subcategory.contains("Primary")) {
                    Some("rt_sched_prim_trad")
                } else if res.configIndex == 14 || (res.subcategory.contains("Work Lists") && res.subcategory.contains("Primary")) {
                    Some("rt_sched_prim_wl")
                } else if res.configIndex == 1 || res.configIndex == 5 || res.configIndex == 9 || res.configIndex == 13 || res.subcategory.contains("SER") {
                    Some("rt_sched_ser")
                } else if res.configIndex == 3 || res.configIndex == 7 || res.configIndex == 11 || res.configIndex == 15 || res.subcategory.contains("Work Graph") {
                    Some("rt_sched_workgraph")
                } else {
                    None
                }
            } else if res.benchmarkName.contains("PathTracing") || res.subcategory.contains("Path Tracing") {
                Some("rt_pathtracing")
            } else if res.subcategory == "Alpha-Tested Geometry" || res.benchmarkName.contains("AnyHit") {
                Some("rt_anyhit")
            } else if res.subcategory == "BLAS Build" {
                Some("rt_blas_build")
            } else if res.subcategory == "BLAS Update" {
                Some("rt_blas_update")
            } else if res.subcategory == "TLAS Build" {
                Some("rt_tlas_build")
            } else if res.subcategory == "Incoherent Traversal" || res.benchmarkName.contains("Incoherent") {
                Some("rt_incoherent")
            } else if res.subcategory == "Intersection tests" || res.benchmarkName == "RayTracing" {
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
