#[cfg(target_os = "windows")]
use std::process::Command;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemInfo {
    pub os_name: String,
    pub kernel_version: String,
    pub arch: String,
    pub cpu_model: String,
    pub cpu_logical_cores: usize,
    pub total_ram_gb: f64,
}

impl SystemInfo {
    pub fn collect() -> Self {
        let (os_name, kernel_version) = detect_os_and_kernel();
        let arch = std::env::consts::ARCH.to_string();
        let cpu_model = detect_cpu_model();
        let cpu_logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or_else(|_| {
                std::env::var("NUMBER_OF_PROCESSORS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
            });
        let total_ram_gb = detect_total_ram_gb();

        Self {
            os_name,
            kernel_version,
            arch,
            cpu_model,
            cpu_logical_cores,
            total_ram_gb,
        }
    }
}

#[cfg(target_os = "linux")]
fn detect_os_and_kernel() -> (String, String) {
    let mut os_name = "Linux".to_string();
    if let Ok(content) = std::fs::read_to_string("/etc/os-release") {
        for line in content.lines() {
            if let Some(val) = line.strip_prefix("PRETTY_NAME=") {
                os_name = val.trim_matches('"').to_string();
                break;
            } else if let Some(val) = line.strip_prefix("NAME=") {
                if os_name == "Linux" {
                    os_name = val.trim_matches('"').to_string();
                }
            }
        }
    }
    let kernel_version = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "Unknown Kernel".to_string());
    (os_name, kernel_version)
}

#[cfg(target_os = "linux")]
fn detect_cpu_model() -> String {
    if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in content.lines() {
            if line.to_lowercase().starts_with("model name") {
                if let Some((_, model)) = line.split_once(':') {
                    return model.trim().to_string();
                }
            }
        }
    }
    "Generic x86_64 CPU".to_string()
}

#[cfg(target_os = "linux")]
fn detect_total_ram_gb() -> f64 {
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<f64>() {
                        let gb = kb / 1024.0 / 1024.0;
                        return (gb * 10.0).round() / 10.0;
                    }
                }
            }
        }
    }
    0.0
}

#[cfg(target_os = "windows")]
fn detect_os_and_kernel() -> (String, String) {
    let mut os_name = "Windows".to_string();
    let mut kernel_version = "Unknown".to_string();

    // Query 'cmd /c ver' -> e.g. "Microsoft Windows [Version 10.0.26100.1742]"
    if let Ok(output) = Command::new("cmd").args(["/c", "ver"]).output() {
        let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !s.is_empty() {
            os_name = s.clone();
            if let Some(start) = s.find("[Version ") {
                if let Some(end) = s[start..].find(']') {
                    kernel_version = s[start + 9..start + end].trim().to_string();
                }
            }
        }
    }
    (os_name, kernel_version)
}

#[cfg(target_os = "windows")]
fn detect_cpu_model() -> String {
    // 1. Try registry query
    if let Ok(output) = Command::new("reg")
        .args(["query", r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "/v", "ProcessorNameString"])
        .output()
    {
        let s = String::from_utf8_lossy(&output.stdout);
        for line in s.lines() {
            if line.contains("ProcessorNameString") {
                if let Some(pos) = line.find("REG_SZ") {
                    let name = line[pos + 6..].trim();
                    if !name.is_empty() {
                        return name.to_string();
                    }
                }
            }
        }
    }
    // 2. Fallback to PROCESSOR_IDENTIFIER
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "Generic Windows CPU".to_string())
}

#[cfg(target_os = "windows")]
fn detect_total_ram_gb() -> f64 {
    // Query powershell CIM instance: (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    if let Ok(output) = Command::new("powershell")
        .args(["-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        .output()
    {
        let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if let Ok(bytes) = s.parse::<f64>() {
            let gb = bytes / 1024.0 / 1024.0 / 1024.0;
            return (gb * 10.0).round() / 10.0;
        }
    }
    0.0
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn detect_os_and_kernel() -> (String, String) {
    (std::env::consts::OS.to_string(), "Unknown".to_string())
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn detect_cpu_model() -> String {
    "Generic CPU".to_string()
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn detect_total_ram_gb() -> f64 {
    0.0
}
