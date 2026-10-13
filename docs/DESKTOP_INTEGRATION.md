# Desktop Integration

## 1. Linux (FreeDesktop, XDG, Wayland, and Package Ecosystems)
Linux has a modular desktop architecture (GNOME, KDE Plasma, COSMIC, wlroots/Sway/Hyprland). Proper desktop integration on Linux relies on FreeDesktop / XDG specifications.

### 1.1. Application Discovery & Launching
XDG Desktop Entry (`packaging/linux/io.github.soddentrough.gpubench.desktop`):
- Installed to `/usr/share/applications/` (RPM, DEB, and Flatpak packages).
- Specifies categories: `Categories=System;Settings;HardwareSettings;Monitor;`.
- Executable target: `Exec=gpubench-gui`.
- Secondary desktop actions provide quick terminal/CLI launches (`RunCLI`, `RunAllCLI`, `ListDevices`).
```ini
[Desktop Entry]
Type=Application
Version=1.5
Name=GPUBench
GenericName=GPU Compute & Ray Tracing Benchmark
Comment=Cross-backend GPU compute, memory, and ray tracing benchmark suite (Vulkan, ROCm, OpenCL)
Exec=gpubench-gui
TryExec=gpubench-gui
Icon=io.github.soddentrough.gpubench
Terminal=false
Categories=System;Settings;HardwareSettings;Monitor;
Keywords=GPU;Benchmark;Vulkan;ROCm;HIP;OpenCL;Compute;RayTracing;TFLOPS;Bandwidth;AMD;NVIDIA;Intel;
StartupNotify=true
StartupWMClass=gpubench-gui
Actions=RunCLI;RunAllCLI;ListDevices;

[Desktop Action RunCLI]
Name=Run Interactive CLI Benchmark
Exec=gpubench
Terminal=true

[Desktop Action RunAllCLI]
Name=Run Full Benchmark Suite (CLI)
Exec=gpubench --benchmark all
Terminal=true

[Desktop Action ListDevices]
Name=List Detected GPUs
Exec=gpubench --list-devices
Terminal=true
```

Wayland `app_id` and X11 `WM_CLASS` Alignment:
Wayland compositors (Mutter/GNOME, KWin/KDE) match running windows to `.desktop` files using the window's `app_id`.
In `gpubench-gui` (`cpp_src/gui/VulkanContext.cpp`), the window properties explicitly set the application ID so the taskbar, dash, and Alt-Tab switcher display the GPUBench icon:

```cpp
// SDL3 window creation property
SDL_SetStringProperty(props, SDL_PROP_WINDOW_CREATE_WAYLAND_APP_ID_STRING, "io.github.soddentrough.gpubench");
```

### 1.2. AppStream Metainfo (`metainfo.xml`)
For inclusion in GNOME Software, KDE Discover, Fedora Software, and Flathub packaging:
- Installed from `packaging/linux/io.github.soddentrough.gpubench.metainfo.xml` to `/usr/share/metainfo/io.github.soddentrough.gpubench.metainfo.xml`.
- Provides screenshots, release notes, hardware developer tags, and URLs to the repository and issue tracker.

### 1.3. XDG Icon Theme Layout
Multi-resolution PNGs and vector SVG assets (`packaging/linux/icons/`) are installed to:
- `/usr/share/icons/hicolor/16x16/apps/io.github.soddentrough.gpubench.png`
- `/usr/share/icons/hicolor/32x32/apps/io.github.soddentrough.gpubench.png`
- `/usr/share/icons/hicolor/48x48/apps/io.github.soddentrough.gpubench.png`
- `/usr/share/icons/hicolor/64x64/apps/io.github.soddentrough.gpubench.png`
- `/usr/share/icons/hicolor/128x128/apps/io.github.soddentrough.gpubench.png`
- `/usr/share/icons/hicolor/256x256/apps/io.github.soddentrough.gpubench.png`
- `/usr/share/icons/hicolor/512x512/apps/io.github.soddentrough.gpubench.png`
- `/usr/share/icons/hicolor/scalable/apps/io.github.soddentrough.gpubench.svg`

### 1.4. Power & Sleep Inhibition (Critical for Compute Benchmarking)
When a long benchmark suite runs (e.g. multi-sample ray tracing or memory stress tests), display sleep or system suspension can corrupt GPU timings, drop DPM clocks, or trigger driver context resets.
- Mechanism (`cpp_src/utils/PlatformIntegration.cpp`): Uses `systemd-inhibit` / D-Bus `org.freedesktop.ScreenSaver.Inhibit` for the duration of the benchmark run and releases the inhibitor upon completion.

### 1.5. Desktop Notifications & Telemetry
Uses standard Freedesktop notifications (`notify-send` / `org.freedesktop.Notifications`):
- Notifies the user when a benchmark sweep completes while GPUBench is minimized or running on a secondary workspace.

### 1.6. CLI Ergonomics
- Bash, Zsh, and Fish shell completions for both `gpubench` and `gpubench-gui` are installed to `/usr/share/bash-completion/completions/`, `/usr/share/zsh/site-functions/`, and `/usr/share/fish/vendor_completions.d/`.
- Man pages `gpubench(1)` and `gpubench-gui(1)` are installed to `/usr/share/man/man1/`.

---

## 2. Windows (Win32, Shell, WinRT, and Windows 11/10 Experience)
On Windows, desktop integration provides clean Start Menu integration, proper file icons in Explorer, taskbar status, and resilient power management.

### 2.1. Executable Metadata & Manifest (`.rc` and `.manifest`)
Windows resource (`packaging/windows/gpubench.rc`) and application manifest (`packaging/windows/gpubench.manifest`) files are compiled and linked into both `gpubench.exe` and `gpubench-gui.exe` at build time:
- **Embedded Icon**: Displays the GPUBench logo (`packaging/windows/icon.ico`) in File Explorer and taskbar instead of the generic console icon.
- **Version Information**: Product name, company, version (`1.0.0.0`), description, and copyright visible in File Explorer's Properties dialog.
- **Application Manifest**:
  - **Per-Monitor V2 High-DPI Awareness**: Prevents blurry text or blurry GUI rendering on 4K/scaling setups (`dpiAwareness = PerMonitorV2`).
  - **Long Path Support**: (`longPathAware = true`).
  - **UTF-8 Code Page**: Sets active code page to UTF-8 to prevent character encoding corruption in terminal logs.

### 2.2. Taskbar Integration (`ITaskbarList3`)
Windows applications can use the `ITaskbarList3` COM interface:
- **Progress Bar**: `SetProgressValue(hwnd, completed, total)` displays a green progress bar directly on the taskbar icon.
- **State Changes**:
  - `TBPF_NORMAL` (green) during active benchmarking.
  - `TBPF_INDETERMINATE` (marquee) during shader compilation or pipeline cache warming.
  - `TBPF_ERROR` (red) if an out-of-memory or driver timeout occurs.
  - `TBPF_PAUSED` (yellow) during thermal cooldown intervals.

### 2.3. Power & Sleep State Management
Windows modern standby (Connected Standby) may attempt to turn off displays after several minutes of no keyboard/mouse input even when the GPU is under 100% compute load.
- **Win32 API** (`cpp_src/utils/PlatformIntegration.cpp`): Calls `SetThreadExecutionState` when starting a benchmark:
```cpp
// Prevent monitor sleep and system idle sleep during benchmark
SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
// Restore default power behavior when benchmarks finish
SetThreadExecutionState(ES_CONTINUOUS);
```

### 2.4. Windows Notifications
Delivers native Windows notifications upon batch completion with summary scores.

### 2.5. Shell File Association & NSIS Installer
Start Menu shortcuts created by NSIS (`CMakeLists.txt` CPack NSIS configuration):
- GPUBench GUI (`gpubench-gui.exe`)
- GPUBench CLI (`gpubench.exe`)
- Optional User `PATH` registration.

---

## 3. macOS (Aqua, Cocoa, Application Bundle, and Apple Silicon)
macOS users expect applications to conform to the Apple Human Interface Guidelines, self-contained `.app` bundles, and Dock behaviors.

### 3.1. Application Bundle Structure (`GPUBench.app`)
```text
GPUBench.app/
├── Contents/
│   ├── Info.plist
│   ├── PkgInfo
│   ├── MacOS/
│   │   ├── gpubench-gui       (Main GUI launcher)
│   │   └── gpubench           (CLI binary)
│   └── Resources/
│       ├── GPUBench.icns      (Multi-resolution icon asset: 16x16 to 1024x1024 @2x)
│       └── kernels/           (Shaders and kernel source files)
```

### 3.2. `Info.plist` Configuration
- **Bundle Identifier**: `io.github.soddentrough.gpubench`
- **HiDPI Support**: `NSHighResolutionCapable = true` (essential for crisp Retina display rendering).
- **Appearance**: `NSRequiresAquaSystemAppearance = false` (supports native macOS Dark Mode seamlessly).

### 3.3. Sleep Prevention (`IOPMLib` / `caffeinate`)
On macOS, system sleep is inhibited during benchmark execution via `caffeinate -di -w <pid>` or I/O Kit Power Management (`IOPMAssertionCreateWithName`):
```objc
IOPMAssertionID assertionID;
IOPMAssertionCreateWithName(
    kIOPMAssertionTypePreventUserIdleSystemSleep,
    kIOPMAssertionLevelOn,
    CFSTR("GPUBench is executing compute benchmark"),
    &assertionID
);
// Release when complete:
IOPMAssertionRelease(assertionID);
```

### 3.4. Notification Center
Dispatches notifications via `osascript -e 'display notification ...'` or `UNUserNotificationCenter`.

---

## 4. Architectural Comparison Across Platforms

| Feature / Domain | Linux (XDG / FreeDesktop) | Windows (Win32 / Modern Shell) | macOS (Aqua / Cocoa) |
| :--- | :--- | :--- | :--- |
| **App Bundle / Packaging** | RPM, DEB, Tarball, Flatpak | NSIS Installer (`.exe`), Portable (`.zip`) | `.app` Bundle, Drag-and-Drop `.dmg` |
| **Launcher Entry** | `/usr/share/applications/*.desktop` | Start Menu Shortcut (`.lnk`), App Paths | `/Applications/GPUBench.app`, LaunchServices |
| **Icon Format** | Freedesktop Hicolor PNGs & scalable SVG | Multi-res `.ico` (16x16 to 256x256) | Multi-res `.icns` (16x16 to 1024x1024 Retina) |
| **Window App Matching** | Wayland `app_id` / X11 `WM_CLASS` | `AppUserModelID` (`SetCurrentProcessExplicitAppUserModelID`) | `CFBundleIdentifier` in `Info.plist` |
| **Sleep / Display Guard** | `systemd-inhibit` / `org.freedesktop.ScreenSaver.Inhibit` | `SetThreadExecutionState(ES_SYSTEM_REQUIRED)` | `IOPMAssertionCreateWithName` / `caffeinate` |
| **Progress Indicator** | Unity/KDE Launcher D-Bus Progress | `ITaskbarList3::SetProgressValue` (Green/Red) | `NSApp.dockTile` badge label / mini-progress |
| **Completion Alerts** | `notify-send` / `org.freedesktop.Notifications` | Windows WinRT / Shell Toast | `UNUserNotificationCenter` / `osascript` |
| **File Association** | Shared MIME-Info (`application/x-gpubench+json`) | Registry (`HKCR\.gpubench`, `OpenWithProgids`) | `CFBundleDocumentTypes` in `Info.plist` |
| **Terminal Integration** | Shell completions (`bash`, `zsh`, `fish`), `man1` | Windows Terminal profile, `PATH` registration | `PATH` symlink in `/usr/local/bin/gpubench` |

---

## 5. Implementation Architecture Summary

1. **Linux Desktop Entry, AppStream & Icon Distribution**:
   - Standard resolution PNGs (16x16 through 512x512) and scalable SVG in `packaging/linux/icons/`.
   - Desktop entry `packaging/linux/io.github.soddentrough.gpubench.desktop` and AppStream metadata `packaging/linux/io.github.soddentrough.gpubench.metainfo.xml`.
   - CMake Linux install rules place `.desktop`, `.metainfo.xml`, icons, man pages (`gpubench.1`, `gpubench-gui.1`), and shell completions (`bash`, `zsh`, `fish`) into standard XDG paths.
   - Wayland `app_id` (`io.github.soddentrough.gpubench`) configured in `cpp_src/gui/VulkanContext.cpp`.
2. **Windows Version & Icon Resources**:
   - `packaging/windows/gpubench.rc` and `packaging/windows/gpubench.manifest` embed version `1.0.0.0`, application icon (`packaging/windows/icon.ico`), PerMonitorV2 High-DPI awareness, and UTF-8 code page support into both `gpubench.exe` and `gpubench-gui.exe`.
3. **Cross-Platform Power & Sleep Safeguards (`cpp_src/utils/PlatformIntegration.cpp`)**:
   - `ScopedSleepInhibitor` automatically suppresses idle and display sleep while benchmarks execute and restores system power policies on completion across Linux, Windows, and macOS.
