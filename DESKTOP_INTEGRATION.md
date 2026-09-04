1. Linux (FreeDesktop, XDG, Wayland, and Package Ecosystems)
Linux has the most modular desktop architecture (GNOME, KDE Plasma, COSMIC, wlroots/Sway/Hyprland). Proper desktop integration on Linux relies on FreeDesktop / XDG specifications.

1.1. Application Discovery & Launching
XDG Desktop Entry (gpubench.desktop or io.github.soddentrough.gpubench.desktop):
Install to /usr/share/applications/ (RPM and DEB packages).
Specifies categories: Categories=System;HardwareSettings;Benchmark;.
Executable target: Exec=gpubench-gui %F.
Terminal fallback action: Desktop entry actions can provide quick secondary launches (e.g., right-click dock icon → "Run Fast Benchmark CLI", "Open Documentation").
ini


[Desktop Entry]
Type=Application
Version=1.5
Name=GPUBench
GenericName=GPU Compute Benchmark
Comment=Modern cross-backend Vulkan, ROCm, and OpenCL benchmarking utility
Exec=gpubench-gui %F
Icon=io.github.soddentrough.gpubench
Terminal=false
Categories=System;HardwareSettings;Benchmark;
Keywords=GPU;Benchmark;Vulkan;ROCm;OpenCL;RayTracing;Compute;Hardware;
StartupNotify=true
StartupWMClass=gpubench-gui
MimeType=application/x-gpubench+json;
Actions=RunCLI;
[Desktop Action RunCLI]
Name=Run Benchmark (Terminal)
Exec=x-terminal-emulator -e gpubench
Wayland app_id and X11 WM_CLASS Alignment:
Wayland compositors (Mutter/GNOME, KWin/KDE) match running windows to .desktop files using the window's app_id.
In gpubench-gui (Iced), the window settings should explicitly set the application ID so the taskbar, dash, and Alt-Tab switcher show the correct app icon rather than a generic fallback gear icon:
rust


iced::window::Settings {
    platform_specific: iced::window::settings::PlatformSpecific {
        application_id: String::from("io.github.soddentrough.gpubench"),
    },
    ..Default::default()
}
1.2. AppStream Metainfo (metainfo.xml)
For inclusion in GNOME Software, KDE Discover, Fedora Software, and potential Flathub packaging:
Install to /usr/share/metainfo/io.github.soddentrough.gpubench.metainfo.xml.
Provides screenshots (from renders/), release notes, hardware developer tags, and URLs to the issue tracker.
1.3. XDG Icon Theme Layout
Extract multi-resolution PNGs and vector SVG from packaging/windows/icon.ico:
/usr/share/icons/hicolor/16x16/apps/io.github.soddentrough.gpubench.png
/usr/share/icons/hicolor/32x32/apps/io.github.soddentrough.gpubench.png
/usr/share/icons/hicolor/48x48/apps/io.github.soddentrough.gpubench.png
/usr/share/icons/hicolor/64x64/apps/io.github.soddentrough.gpubench.png
/usr/share/icons/hicolor/128x128/apps/io.github.soddentrough.gpubench.png
/usr/share/icons/hicolor/256x256/apps/io.github.soddentrough.gpubench.png
/usr/share/icons/hicolor/scalable/apps/io.github.soddentrough.gpubench.svg
1.4. Power & Sleep Inhibition (Critical for Compute Benchmarking)
When a long benchmark suite runs (e.g. multi-sample ray tracing or memory stress tests), display sleep or system suspension will corrupt GPU timings, drop DPM clocks, or trigger driver context resets.
Mechanism: Use D-Bus to call org.freedesktop.ScreenSaver.Inhibit or org.freedesktop.PowerManagement.Inhibit (or systemd-inhibit wrapper) for the duration of the benchmark run, and release the inhibitor cookie upon completion.
1.5. Desktop Notifications & Telemetry
Use standard D-Bus desktop notifications (org.freedesktop.Notifications / notify-rust):
Notify the user when a 10-minute sweep completes while GPUBench is minimized or running on a secondary workspace.
Summary: GPUBench Complete: 42.5 TFLOPS (FP32) | 637.7 GB/s (GDDR6).
Taskbar progress on KDE/GNOME:
Expose progress via com.canonical.Unity.LauncherEntry D-Bus interface to show a live circular or bar progress over the launcher icon.
1.6. CLI Ergonomics
Bash, Zsh, and Fish shell completions installed to /usr/share/bash-completion/completions/gpubench, /usr/share/zsh/site-functions/_gpubench, and /usr/share/fish/vendor_completions.d/gpubench.fish.
Man page installed to /usr/share/man/man1/gpubench.1.
2. Windows (Win32, Shell, WinRT, and Windows 11/10 Experience)
On Windows, user expectation centers on clean Start Menu integration, proper file icons in Explorer, taskbar status, and resilient power management.

2.1. Executable Metadata & Manifest (.rc and .manifest)
Currently gpubench.exe has no version resource. Adding a Windows resource file (packaging/windows/gpubench.rc) and embedding it at compile time provides:
Embedded Icon: Shows the GPUBench logo in File Explorer instead of the generic console icon.
Version Information: Product name, company, version (1.4.5.0), description, and copyright visible in File Explorer properties dialog.
Application Manifest:
Per-Monitor V2 High-DPI Awareness: Prevents blurry text or blurry GUI rendering on 4K/scaling setups (dpiAwareness = PerMonitorV2).
Long Path Support: (longPathAware = true).
UTF-8 Code Page: Sets active code page to UTF-8 to prevent character encoding corruptions in terminal logs.
2.2. Taskbar Integration (ITaskbarList3)
Modern Windows applications use the ITaskbarList3 COM interface:
Progress Bar: SetProgressValue(hwnd, completed, total) displays a green progress bar directly on the taskbar icon.
State Changes:
TBPF_NORMAL (green) during active benchmarking.
TBPF_INDETERMINATE (marquee) during shader compilation or pipeline cache warming.
TBPF_ERROR (red) if an out-of-memory or driver timeout occurs.
TBPF_PAUSED (yellow) during thermal cooldown intervals.
Thumbnail Toolbar: Fast quick-action buttons on the taskbar preview (e.g. "Stop", "Next Test").
2.3. Power & Sleep State Management
Windows aggressive modern standby (Connected Standby) will frequently attempt to turn off displays after 2–5 minutes of no keyboard/mouse input even when the GPU is at 100% compute load.
Win32 API: Call SetThreadExecutionState when starting a benchmark:
cpp


// Prevent monitor sleep and system idle sleep during benchmark
SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
// Restore default power behavior when benchmarks finish
SetThreadExecutionState(ES_CONTINUOUS);
2.4. Windows Notifications (Toast)
Deliver native Windows 10/11 Toast notifications via Windows Shell / WinRT:
Fires upon batch completion with summary scores.
Clicking the notification brings gpubench-gui to the foreground and focuses the results view.
2.5. Shell File Association & Registry
Associate .gpubench and .gpubench.json report files with GPUBench:
Double-clicking a saved benchmark run in Explorer opens it in gpubench-gui.
Icon handler associates .gpubench files with a custom "document" variant of the GPUBench icon.
Start Menu shortcuts created by NSIS:
GPUBench (GUI app)
GPUBench CLI (starts terminal session with gpubench --help)
Run All Benchmarks (executes RunAllBenchmarks.bat)
Add installation directory to User PATH (optional checkbox in installer).
3. macOS (Aqua, Cocoa, Application Bundle, and Apple Silicon)
macOS users expect applications to conform strictly to the Apple Human Interface Guidelines, self-contained .app bundles, and Dock behaviors.

3.1. Application Bundle Structure (GPUBench.app)
Rather than bare Unix binaries in /usr/bin, macOS requires an application bundle:



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
3.2. Info.plist Configuration
Bundle Identifier: io.github.soddentrough.gpubench
HiDPI Support: NSHighResolutionCapable = true (essential for crisp Retina display rendering).
Appearance: NSRequiresAquaSystemAppearance = false (supports native macOS Dark Mode seamlessly).
Document Types: Register custom file type com.soddentrough.gpubench.result (.gpubench) with CFBundleTypeRole = Viewer.
3.3. Dock Tile Integration
While running benchmarks:
Use NSApp.dockTile to draw dynamic progress badges or mini charts directly onto the Dock icon while minimized.
Bounce dock icon (NSApp requestUserAttention:NSInformationalRequest) when a long benchmark completes.
3.4. Sleep Prevention (IOPMLib)
On macOS, system sleep is inhibited using I/O Kit Power Management:
objc


IOPMAssertionID assertionID;
IOPMAssertionCreateWithName(
    kIOPMAssertionTypePreventUserIdleSystemSleep,
    kIOPMAssertionLevelOn,
    CFSTR("GPUBench is executing compute benchmark"),
    &assertionID
);
// Release when complete:
IOPMAssertionRelease(assertionID);
(Or execute caffeinate -w <pid> as a background guard process).
3.5. Notification Center
Dispatch notifications via UNUserNotificationCenter or AppleScript bridge (osascript -e 'display notification ...').
3.6. Packaging: Drag-and-Drop DMG
Standard macOS disk image (GPUBench-1.4.5-Darwin.dmg) configured with:
Custom branded DMG background image.
Left icon: GPUBench.app.
Right icon: Symlink to /Applications.
Clean window bounds and hidden sidebar.
4. Architectural Comparison Across Platforms
Feature / Domain	Linux (XDG / FreeDesktop)	Windows (Win32 / Modern Shell)	macOS (Aqua / Cocoa)
App Bundle / Packaging	RPM, DEB, Tarball (Flatpak potential)	NSIS Installer (.exe), Portable (.zip)	.app Bundle, Drag-and-Drop .dmg
Launcher Entry	/usr/share/applications/*.desktop	Start Menu Shortcut (.lnk), App Paths	/Applications/GPUBench.app, LaunchServices
Icon Format	Freedesktop Hicolor PNGs & scalable SVG	Multi-res .ico (16x16 to 256x256)	Multi-res .icns (16x16 to 1024x1024 Retina)
Window App Matching	Wayland app_id / X11 WM_CLASS	AppUserModelID (SetCurrentProcessExplicitAppUserModelID)	CFBundleIdentifier in Info.plist
Sleep / Display Guard	org.freedesktop.ScreenSaver.Inhibit	SetThreadExecutionState(ES_SYSTEM_REQUIRED)	IOPMAssertionCreateWithName / caffeinate
Progress Indicator	Unity/KDE Launcher D-Bus Progress	ITaskbarList3::SetProgressValue (Green/Red)	NSApp.dockTile badge label / mini-progress
Completion Alerts	org.freedesktop.Notifications	Windows WinRT / Shell Toast	UNUserNotificationCenter / Banner
File Association	Shared MIME-Info (application/x-gpubench+json)	Registry (HKCR\.gpubench, OpenWithProgids)	CFBundleDocumentTypes in Info.plist
Terminal Integration	Shell completions (bash, zsh, fish), man1	Windows Terminal profile, PATH registration	PATH symlink in /usr/local/bin/gpubench
5. Recommended Phased Implementation Plan
Phase 1: High-Impact Essentials (Linux & Windows Packaging Polish)
Linux Desktop Entry & Icon Distribution:
Extract standard resolution PNGs from packaging/windows/icon.ico into packaging/linux/icons/.
Create packaging/linux/io.github.soddentrough.gpubench.desktop.
Update CMakeLists.txt Linux install rules to place .desktop and icons into /usr/share/applications/ and /usr/share/icons/hicolor/.
Add Wayland app_id to gpubench-gui/src/main.rs.
Windows Version & Icon Resources:
Create packaging/windows/gpubench.rc and packaging/windows/gpubench.manifest with version 1.4.5.0, application icon, and High-DPI PerMonitorV2 support.
Link gpubench.rc into gpubench.exe in CMakeLists.txt when targeting Windows.
Phase 2: Power & Sleep Safeguards (Benchmark Reliability)
Implement cross-platform idle-sleep inhibitor module:
Linux: D-Bus screensaver/power inhibition.
Windows: SetThreadExecutionState.
macOS: IOPMAssertionCreateWithName.
Hook into BenchmarkRunner so any benchmark run automatically suppresses sleep while executing and restores power policies on completion.
Phase 3: Desktop Telemetry & Notifications
Add native desktop notification triggers when benchmark suites complete in gpubench-gui and gpubench.
Add Windows ITaskbarList3 progress reporting during multi-test runs.
macOS .app bundle restructuring and CPack DMG layout customization.
Would you like to begin by implementing Phase 1 (creating the Linux .desktop file, extracting the icons into the XDG hicolor theme hierarchy, updating CMakeLists.txt for RPM/DEB packages, and adding the Windows .rc resource and manifest for gpubench.exe)?
