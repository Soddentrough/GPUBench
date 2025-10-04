# GPUBench Installation Guide

This document describes how to build and install GPUBench on your system.

## Prerequisites

At least one of the following compute backends must be available:
- **Vulkan**: Vulkan SDK with `glslc` compiler
- **OpenCL**: OpenCL development libraries
- **ROCm**: AMD ROCm with HIP support

## Building from Source

### 1. Clone and Configure

```bash
git clone <repository-url>
cd GPUBench
mkdir build && cd build
```

### 2. Configure with CMake

**Linux (default prefix: /usr/local):**
```bash
cmake ..
```

**Linux (custom prefix):**
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
```

**Windows:**
```bash
cmake .. -DCMAKE_INSTALL_PREFIX="C:\Program Files\GPUBench"
```

### 3. Build

```bash
make -j$(nproc)
```

On Windows, use:
```bash
cmake --build . --config Release
```

### 4. Install

**Linux (from the build directory):**
```bash
sudo make install
```

**Windows (run as Administrator from the build directory):**
```bash
cmake --build . --target install --config Release
```

**Note:** All commands in steps 2-4 should be run from the `build` directory.

## Installation Structure

The installation creates the following structure:

```
<PREFIX>/
├── bin/
│   └── gpubench                    # The executable
└── share/gpubench/
    └── kernels/
        ├── vulkan/                 # Compiled SPIR-V shaders (*.spv)
        ├── opencl/                 # OpenCL source files (*.cl)
        └── rocm/                   # Compiled HIP objects (*.o)
```

### Default Prefixes

- **Linux**: `/usr/local`
- **Windows**: `C:\Program Files\GPUBench`

## Running GPUBench

After installation, you can run GPUBench from anywhere:

```bash
# List available benchmarks
gpubench --list-benchmarks

# Run all benchmarks
gpubench

# Run specific benchmarks
gpubench -b FP32,FP64

# List available devices
gpubench --list-devices

# Run on a specific device
gpubench -d 0 -b FP32
```

## Kernel Path Resolution

GPUBench searches for kernel files in the following order:

1. **Environment variable** `GPUBENCH_KERNEL_PATH` (if set)
2. **Installed location**: `<PREFIX>/share/gpubench/kernels`
3. **Development fallback**: `./kernels` (relative to working directory)

### Using Custom Kernel Path

You can override the kernel path using an environment variable:

```bash
export GPUBENCH_KERNEL_PATH=/custom/path/to/kernels
gpubench -b FP32
```

This is useful for:
- Testing modified kernels
- Running from a non-standard installation
- Development and debugging

## Uninstalling

To uninstall GPUBench:

**Linux:**
```bash
cd build
sudo make uninstall
```

Or manually remove:
```bash
sudo rm /usr/local/bin/gpubench
sudo rm -rf /usr/local/share/gpubench
```

**Windows:**
Delete the installation directory:
```
C:\Program Files\GPUBench
```

## Development Build

For development, you can build without installing:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./gpubench --list-benchmarks
```

The binary will automatically use the `./kernels` directory in the source tree.

## Troubleshooting

### "Could not find kernel directory" error

If you see this warning, GPUBench couldn't locate the kernel files. Check:
1. The binary is installed correctly
2. Kernel files exist in `<PREFIX>/share/gpubench/kernels`
3. Try setting `GPUBENCH_KERNEL_PATH` environment variable

### Build Fails with "Cannot find Vulkan/OpenCL/ROCm"

Ensure at least one compute backend is installed:
- **Vulkan**: Install Vulkan SDK from https://vulkan.lunarg.com/
- **OpenCL**: Install OpenCL development packages (e.g., `ocl-icd-opencl-dev` on Debian/Ubuntu)
- **ROCm**: Install AMD ROCm from https://rocm.docs.amd.com/

### Permission Denied During Installation

On Linux, use `sudo` for system-wide installation:
```bash
sudo make install
```

Or use a prefix in your home directory:
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make install
```

Then add `$HOME/.local/bin` to your PATH.

## Platform-Specific Notes

### Linux
- Recommended prefix: `/usr/local` or `$HOME/.local`
- May need to add installation directory to PATH
- No special requirements

### Windows
- Recommended prefix: `C:\Program Files\GPUBench`
- Requires Administrator privileges for installation
- Add installation directory to system PATH for easy access

## Building with Specific Backends

You can disable backends by not installing their dependencies:

```bash
# Build with only Vulkan
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local

# CMake will automatically detect available backends
```

The build system will automatically detect and enable available backends.

## Creating Distribution Packages

### Windows Packages

GPUBench supports creating pre-packaged Windows binaries and installers. See `packaging/windows/README.md` for detailed instructions.

**Quick start:**

```powershell
# Automated build script (recommended)
.\packaging\windows\build_package.ps1

# Or manually:
mkdir build-release
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="GPUBench"
cmake --build . --config Release

# Create ZIP package
cpack -G ZIP -C Release

# Create NSIS installer (requires NSIS)
cpack -G NSIS -C Release
```

**Output:**
- `GPUBench-1.0.0-win64.zip` - Portable ZIP archive
- `GPUBench-1.0.0-win64.exe` - NSIS installer (if NSIS is installed)

### Linux Packages

GPUBench supports creating DEB and RPM packages:

```bash
mkdir build-release
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Create DEB package (Debian/Ubuntu)
cpack -G DEB

# Create RPM package (Fedora/RHEL)
cpack -G RPM

# Create tarball
cpack -G TGZ
```

**Output:**
- `GPUBench-1.0.0-Linux.deb`
- `GPUBench-1.0.0-Linux.rpm`
- `GPUBench-1.0.0-Linux.tar.gz`

### macOS Packages

```bash
mkdir build-release
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# Create DMG installer
cpack -G DragNDrop

# Create tarball
cpack -G TGZ
```

For more information on packaging, see:
- Windows packaging: `packaging/windows/README.md`
- CPack documentation: https://cmake.org/cmake/help/latest/module/CPack.html
