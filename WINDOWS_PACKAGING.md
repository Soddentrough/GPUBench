# Windows Packaging for GPUBench

This document provides a quick reference for creating Windows distribution packages.

## What Was Added

The following Windows packaging infrastructure has been added to GPUBench:

### 1. CMake Packaging Configuration (`CMakeLists.txt`)
- Added CPack configuration for creating Windows packages
- Support for ZIP archives (portable distribution)
- Support for NSIS installers (professional installation)
- Version management (project version set to 1.0.0)
- Automatic architecture detection (win32/win64)

### 2. Automated Build Script (`packaging/windows/build_package.ps1`)
- PowerShell script for automated packaging
- Validates prerequisites (CMake, NSIS)
- Configures, builds, and packages in one command
- Options for ZIP, NSIS, or both package types
- Clean build option
- Colored output for easy status tracking

### 3. Documentation (`packaging/windows/README.md`)
- Comprehensive guide for Windows packaging
- Prerequisites and setup instructions
- Quick start examples
- Troubleshooting guide
- Distribution checklist

## Quick Start

### Method 1: Automated Script (Recommended)

```powershell
# From the project root directory
.\packaging\windows\build_package.ps1
```

This will:
1. Configure the project with CMake
2. Build in Release mode
3. Create both ZIP and NSIS packages (if NSIS is installed)

### Method 2: Manual Build

```powershell
# Create and enter build directory
mkdir build-release
cd build-release

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="GPUBench"

# Build
cmake --build . --config Release

# Create ZIP package
cpack -G ZIP -C Release

# Create NSIS installer (optional, requires NSIS)
cpack -G NSIS -C Release
```

## Output Packages

After running the packaging process, you'll get:

1. **ZIP Archive**: `GPUBench-1.0.0-win64.zip`
   - Portable, no installation required
   - Extract and run `bin\gpubench.exe`
   - Good for testing or users who prefer portable apps

2. **NSIS Installer**: `GPUBench-1.0.0-win64.exe` (if NSIS installed)
   - Professional installer with wizard
   - Automatic PATH configuration
   - Start Menu shortcuts
   - Uninstaller included
   - Default location: `C:\Program Files\GPUBench`

## Prerequisites

### Required
- Windows 10 or 11
- Visual Studio 2019+ with C++ desktop development
- CMake 3.16 or later
- Vulkan SDK or OpenCL SDK

### Optional (for NSIS installer)
- NSIS 3.x from https://nsis.sourceforge.io/

## Package Contents

Both package types include:

```
GPUBench/
├── RunAllBenchmarks.bat      # Easy execution script
├── bin/
│   └── gpubench.exe          # Main executable
└── share/gpubench/
    └── kernels/
        ├── vulkan/           # SPIR-V shaders (*.spv)
        └── opencl/           # OpenCL kernels (*.cl)
```

## Build Script Options

The PowerShell script accepts several parameters:

```powershell
# Create only ZIP package
.\packaging\windows\build_package.ps1 -PackageType zip

# Create only NSIS installer
.\packaging\windows\build_package.ps1 -PackageType nsis

# Create both (default)
.\packaging\windows\build_package.ps1 -PackageType both

# Clean build (removes existing build directory)
.\packaging\windows\build_package.ps1 -CleanBuild

# Use custom build directory
.\packaging\windows\build_package.ps1 -BuildDir "my-build"

# Debug build (default is Release)
.\packaging\windows\build_package.ps1 -BuildType Debug
```

## Testing the Package

### ZIP Package
1. Extract `GPUBench-1.0.0-win64.zip`
2. Double-click `RunAllBenchmarks.bat` in the root directory
3. The benchmarks will run and the window will stay open when finished

Alternatively, via command line:
```powershell
# Extract
Expand-Archive GPUBench-1.0.0-win64.zip -DestinationPath test

# Run the wrapper script
cd test\GPUBench
.\RunAllBenchmarks.bat
```

### NSIS Installer
1. Run the installer `GPUBench-1.0.0-win64.exe`
2. You can find "Run All Benchmarks" in the Start Menu
3. Or run `gpubench --list-benchmarks` from any terminal

## Customization

### Change Version Number
Edit `CMakeLists.txt`:
```cmake
project(GPUBench VERSION 1.0.0)  # Change version here
```

### Customize NSIS Installer
Edit the CPack NSIS settings in `CMakeLists.txt`:
- `CPACK_NSIS_DISPLAY_NAME` - Name shown in Add/Remove Programs
- `CPACK_NSIS_HELP_LINK` - Support URL
- `CPACK_NSIS_CONTACT` - Contact email
- `CPACK_NSIS_MENU_LINKS` - Start Menu shortcuts

### Add Custom Icon
1. Create or obtain a 256x256 icon
2. Convert to .ico format
3. Save as `packaging/windows/icon.ico`
4. The build system will automatically use it

See `packaging/windows/ICON_PLACEHOLDER.txt` for detailed instructions.

## Troubleshooting

### "NSIS not found"
- Install from https://nsis.sourceforge.io/
- Add to PATH or let the script auto-detect it
- Or create ZIP package only

### "Vulkan not found"
- Install Vulkan SDK from https://vulkan.lunarg.com/
- Ensure `VULKAN_SDK` environment variable is set
- Restart terminal after installation

### "No packages were created"
- Check build output for errors
- Ensure Release build succeeded
- Verify all kernel files are present

### Package won't run on target machine
- Ensure Visual C++ Redistributable is installed
- Check that GPU drivers are up to date
- Test on clean Windows VM before distributing

## Distribution Checklist

Before releasing:
- [ ] Test on clean Windows installation
- [ ] Verify all compute backends work
- [ ] Test with different GPU vendors (NVIDIA, AMD, Intel)
- [ ] Ensure all kernel files are included
- [ ] Test installer and uninstaller (NSIS)
- [ ] Verify PATH modification works
- [ ] Update version number
- [ ] Update contact information in CMakeLists.txt
- [ ] Include LICENSE and README files
- [ ] Document system requirements
- [ ] Create release notes

## Continuous Integration

For automated packaging in CI/CD (e.g., GitHub Actions):

```yaml
name: Build Windows Package

on: [push, release]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install Vulkan SDK
        run: |
          # Install Vulkan SDK
          
      - name: Build Package
        run: |
          mkdir build
          cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release
          cmake --build . --config Release
          cpack -G ZIP -C Release
          
      - name: Upload Artifacts
        uses: actions/upload-artifact@v2
        with:
          name: windows-package
          path: build/GPUBench-*.zip
```

## Support

For detailed information:
- Windows packaging guide: `packaging/windows/README.md`
- General installation: `INSTALL.md`
- CPack documentation: https://cmake.org/cmake/help/latest/module/CPack.html

For issues:
- Check build output for specific errors
- Verify all prerequisites are installed
- See troubleshooting sections in documentation
