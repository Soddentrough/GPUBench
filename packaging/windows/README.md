# Windows Packaging Guide for GPUBench

This guide explains how to create Windows binary packages and installers for GPUBench.

## Overview

GPUBench can be packaged for Windows in two formats:
1. **ZIP Archive** - Simple portable binary distribution
2. **NSIS Installer** - Professional installer with Start Menu shortcuts and uninstaller

## Prerequisites

### For Building
- Windows 10/11
- Visual Studio 2019 or later (with C++ desktop development)
- CMake 3.16 or later
- At least one GPU compute backend:
  - **Vulkan SDK** (recommended): https://vulkan.lunarg.com/
  - **OpenCL SDK** (alternative): Intel, AMD, or NVIDIA OpenCL SDK

### For Creating Installers (Optional)
- **NSIS** (Nullsoft Scriptable Install System): https://nsis.sourceforge.io/
  - Download and install NSIS 3.x
  - Add NSIS to your PATH or CMake will auto-detect it

## Quick Start: Create a ZIP Package

```powershell
# 1. Clone and navigate to the project
git clone <repository-url>
cd GPUBench

# 2. Create build directory
mkdir build-release
cd build-release

# 3. Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="GPUBench"

# 4. Build the project
cmake --build . --config Release

# 5. Create the package
cpack -G ZIP -C Release
```

This creates `GPUBench-X.X.X-win64.zip` in the build directory.

## Creating an NSIS Installer

```powershell
# Follow steps 1-4 above, then:

# 5. Create NSIS installer
cpack -G NSIS -C Release
```

This creates `GPUBench-X.X.X-win64.exe` installer.

## Automated Build Script

Use the provided PowerShell script for fully automated packaging:

```powershell
.\packaging\windows\build_package.ps1 -PackageType both
```

Options:
- `-PackageType zip` - Create only ZIP archive
- `-PackageType nsis` - Create only NSIS installer
- `-PackageType both` - Create both (default)

## Package Contents

Both package types include:

```
GPUBench/
├── bin/
│   ├── gpubench.exe           # Main executable
│   └── *.dll                  # Runtime dependencies (if any)
└── share/gpubench/
    └── kernels/
        ├── vulkan/            # Compiled SPIR-V shaders (*.spv)
        └── opencl/            # OpenCL source files (*.cl)
```

## Distribution

### ZIP Archive
- Portable, no installation required
- Extract anywhere and run `bin\gpubench.exe`
- Users need to add `bin` directory to PATH or use full path

### NSIS Installer
- Professional installation experience
- Creates Start Menu shortcuts
- Adds to system PATH automatically
- Includes uninstaller
- Default install location: `C:\Program Files\GPUBench`

## Testing the Package

### ZIP Package
```powershell
# Extract and test
Expand-Archive GPUBench-X.X.X-win64.zip -DestinationPath test
cd test\GPUBench\bin
.\gpubench.exe --list-benchmarks
```

### NSIS Installer
```powershell
# Run the installer
.\GPUBench-X.X.X-win64.exe

# After installation, test from anywhere
gpubench --list-benchmarks
```

## Customization

### Version Number
Edit `CMakeLists.txt` to set version:
```cmake
project(GPUBench VERSION 1.0.0)
```

### Package Metadata
Edit the CPack configuration in `CMakeLists.txt`:
- `CPACK_PACKAGE_VENDOR`
- `CPACK_PACKAGE_DESCRIPTION_SUMMARY`
- `CPACK_NSIS_DISPLAY_NAME`
- etc.

## Troubleshooting

### "NSIS not found"
- Install NSIS from https://nsis.sourceforge.io/
- Add NSIS to PATH: `C:\Program Files (x86)\NSIS`
- Or specify manually: `cmake -DCPACK_NSIS_EXECUTABLE="C:\Path\to\makensis.exe"`

### "Vulkan not found"
- Install Vulkan SDK from https://vulkan.lunarg.com/
- Ensure `VULKAN_SDK` environment variable is set
- Restart terminal/IDE after installation

### "DLL not found" when running packaged executable
- Ensure all runtime dependencies are included
- Check that the correct Visual C++ Redistributable is installed
- Consider static linking for easier distribution

### Package is too large
- Ensure Release build configuration
- Remove debug symbols: `-DCMAKE_BUILD_TYPE=Release`
- Consider compressing with 7-Zip or similar

## Distribution Checklist

Before distributing:
- [ ] Test on clean Windows VM/machine
- [ ] Verify all backends work (Vulkan/OpenCL)
- [ ] Check that all kernel files are included
- [ ] Test installation and uninstallation (for NSIS)
- [ ] Verify PATH modification works (for NSIS)
- [ ] Test with and without GPU drivers
- [ ] Include README and LICENSE files
- [ ] Document system requirements

## Advanced: Custom NSIS Scripts

For advanced installer customization, you can create a custom NSIS script:

1. Generate default script: `cpack -G NSIS --config CPackConfig.cmake`
2. Customize the generated script
3. Build manually: `makensis custom_installer.nsi`

## Continuous Integration

Example GitHub Actions workflow:

```yaml
- name: Build Windows Package
  run: |
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . --config Release
    cpack -G ZIP -C Release
```

## Support

For issues with:
- **Building**: Check Visual Studio and CMake configuration
- **Backends**: Verify GPU driver installation
- **Packaging**: Check CPack and NSIS setup

For more information, see:
- Main installation guide: `INSTALL.md`
- CMake packaging documentation: https://cmake.org/cmake/help/latest/module/CPack.html
