
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
mod win32 {
    #[repr(C)]
    pub struct MEMORYSTATUSEX {
        pub dwLength: u32,
        pub dwMemoryLoad: u32,
        pub ullTotalPhys: u64,
        pub ullAvailPhys: u64,
        pub ullTotalPageFile: u64,
        pub ullAvailPageFile: u64,
        pub ullTotalVirtual: u64,
        pub ullAvailVirtual: u64,
        pub ullAvailExtendedVirtual: u64,
    }

    #[repr(C)]
    pub struct OSVERSIONINFOEXW {
        pub dwOSVersionInfoSize: u32,
        pub dwMajorVersion: u32,
        pub dwMinorVersion: u32,
        pub dwBuildNumber: u32,
        pub dwPlatformId: u32,
        pub szCSDVersion: [u16; 128],
        pub wServicePackMajor: u16,
        pub wServicePackMinor: u16,
        pub wSuiteMask: u16,
        pub wProductType: u8,
        pub wReserved: u8,
    }

    pub const HKEY_LOCAL_MACHINE: isize = 0x80000002_u32 as i32 as isize;
    pub const KEY_READ: u32 = 0x20019;

    #[link(name = "kernel32")]
    #[link(name = "advapi32")]
    #[link(name = "ntdll")]
    unsafe extern "system" {
        pub fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
        pub fn RtlGetVersion(lpVersionInformation: *mut OSVERSIONINFOEXW) -> i32;
        pub fn RegOpenKeyExA(
            hKey: isize,
            lpSubKey: *const u8,
            ulOptions: u32,
            samDesired: u32,
            phkResult: *mut isize,
        ) -> i32;
        pub fn RegQueryValueExA(
            hKey: isize,
            lpValueName: *const u8,
            lpReserved: *mut u32,
            lpType: *mut u32,
            lpData: *mut u8,
            lpcbData: *mut u32,
        ) -> i32;
        pub fn RegCloseKey(hKey: isize) -> i32;
    }
}

#[cfg(target_os = "windows")]
fn detect_os_and_kernel() -> (String, String) {
    let mut os_name = "Windows".to_string();
    let mut kernel_version = "Unknown".to_string();

    // 1. Read exact OS build version via RtlGetVersion
    let mut osinfo = win32::OSVERSIONINFOEXW {
        dwOSVersionInfoSize: std::mem::size_of::<win32::OSVERSIONINFOEXW>() as u32,
        dwMajorVersion: 0,
        dwMinorVersion: 0,
        dwBuildNumber: 0,
        dwPlatformId: 0,
        szCSDVersion: [0; 128],
        wServicePackMajor: 0,
        wServicePackMinor: 0,
        wSuiteMask: 0,
        wProductType: 0,
        wReserved: 0,
    };
    if unsafe { win32::RtlGetVersion(&mut osinfo) } == 0 {
        kernel_version = format!("{}.{}.{}", osinfo.dwMajorVersion, osinfo.dwMinorVersion, osinfo.dwBuildNumber);
        if osinfo.dwBuildNumber >= 22000 {
            os_name = format!("Microsoft Windows 11 (Build {})", osinfo.dwBuildNumber);
        } else if osinfo.dwMajorVersion == 10 {
            os_name = format!("Microsoft Windows 10 (Build {})", osinfo.dwBuildNumber);
        } else {
            os_name = format!("Microsoft Windows {}.{} (Build {})", osinfo.dwMajorVersion, osinfo.dwMinorVersion, osinfo.dwBuildNumber);
        }
    }

    // 2. Try reading ProductName from registry to get e.g. "Windows 11 Pro"
    unsafe {
        let subkey = b"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\0";
        let val_name = b"ProductName\0";
        let mut hkey: isize = 0;
        if win32::RegOpenKeyExA(win32::HKEY_LOCAL_MACHINE, subkey.as_ptr(), 0, win32::KEY_READ, &mut hkey) == 0 {
            let mut buf = [0u8; 256];
            let mut size = buf.len() as u32;
            let mut val_type = 0u32;
            let ret = win32::RegQueryValueExA(
                hkey,
                val_name.as_ptr(),
                std::ptr::null_mut(),
                &mut val_type,
                buf.as_mut_ptr(),
                &mut size,
            );
            win32::RegCloseKey(hkey);
            if ret == 0 && size > 1 {
                let bytes = &buf[..((size - 1) as usize)];
                if let Ok(name) = std::str::from_utf8(bytes) {
                    let cleaned = name.trim();
                    if !cleaned.is_empty() {
                        if osinfo.dwBuildNumber >= 22000 && cleaned.contains("Windows 10") {
                            os_name = cleaned.replace("Windows 10", "Windows 11");
                        } else {
                            os_name = cleaned.to_string();
                        }
                    }
                }
            }
        }
    }

    (os_name, kernel_version)
}

#[cfg(target_os = "windows")]
fn detect_cpu_model() -> String {
    unsafe {
        let subkey = b"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0\0";
        let val_name = b"ProcessorNameString\0";
        let mut hkey: isize = 0;
        if win32::RegOpenKeyExA(win32::HKEY_LOCAL_MACHINE, subkey.as_ptr(), 0, win32::KEY_READ, &mut hkey) == 0 {
            let mut buf = [0u8; 256];
            let mut size = buf.len() as u32;
            let mut val_type = 0u32;
            let ret = win32::RegQueryValueExA(
                hkey,
                val_name.as_ptr(),
                std::ptr::null_mut(),
                &mut val_type,
                buf.as_mut_ptr(),
                &mut size,
            );
            win32::RegCloseKey(hkey);
            if ret == 0 && size > 1 {
                let bytes = &buf[..((size - 1) as usize)];
                if let Ok(name) = std::str::from_utf8(bytes) {
                    let cleaned = name.trim();
                    if !cleaned.is_empty() {
                        return cleaned.to_string();
                    }
                }
            }
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "Generic Windows CPU".to_string())
}

#[cfg(target_os = "windows")]
fn detect_total_ram_gb() -> f64 {
    let mut mem = win32::MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<win32::MEMORYSTATUSEX>() as u32,
        dwMemoryLoad: 0,
        ullTotalPhys: 0,
        ullAvailPhys: 0,
        ullTotalPageFile: 0,
        ullAvailPageFile: 0,
        ullTotalVirtual: 0,
        ullAvailVirtual: 0,
        ullAvailExtendedVirtual: 0,
    };
    let success = unsafe { win32::GlobalMemoryStatusEx(&mut mem) };
    if success != 0 && mem.ullTotalPhys > 0 {
        let gb = mem.ullTotalPhys as f64 / 1024.0 / 1024.0 / 1024.0;
        return (gb * 10.0).round() / 10.0;
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
