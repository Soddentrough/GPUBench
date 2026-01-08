# GPUBench

GPUBench is a high-performance cross-platform GPU benchmarking tool designed to measure the raw compute capabilities and memory bandwidth of modern graphics hardware. It supports multiple backends and wide range of data types, from double-precision floating point (FP64) down to 4-bit integers (INT4).

![GitHub Version](https://img.shields.io/github/v/release/Soddentrough/GPUBench)
![License](https://img.shields.io/github/license/Soddentrough/GPUBench)

## Features

- **Multi-Backend Support**: Benchmarks using Vulkan, OpenCL, and ROCm/HIP.
- **Comprehensive Data Types**: 
  - Floating Point: FP64, FP32, FP16, FP8, FP6, FP4
  - Integer: INT8, INT4
- **Memory Benchmarks**: Measure Device Memory Bandwidth, System Memory Bandwidth, and Cache performance.
- **Dynamic Loading**: Backends are loaded at runtime, making them optional and reducing installation dependencies.
- **Cross-Platform**: Built for Linux and Windows.

## Supported Backends

| Backend | Platform | Primary Use Case | Minimum Version |
| :--- | :--- | :--- | :--- |
| **Vulkan** | Linux, Windows | Standard cross-vendor compute | 1.4+ |
| **OpenCL** | Linux, Windows | Fallback cross-vendor compute | 1.2+ |
| **ROCm/HIP** | Linux | Native AMD performance | 6.4+ |

## Quick Start

### Prerequisites

Ensure you have the appropriate drivers and SDKs installed for the backends you wish to use. See [VERSION_REQUIREMENTS.md](VERSION_REQUIREMENTS.md) for details.

### Installation

Download the latest release from the [GitHub Releases](https://github.com/Soddentrough/GPUBench/releases) page or build from source following the [INSTALL.md](INSTALL.md) guide.

### Basic Usage

```bash
# List all available benchmarks
gpubench --list-benchmarks

# Run all benchmarks on the default device
gpubench

# Run specific benchmarks on a specific device
gpubench -d 0 -b FP32,FP16
```

## Documentation

- [Installation Guide](INSTALL.md) - Detailed build and install instructions.
- [Version Requirements](VERSION_REQUIREMENTS.md) - Software and hardware requirements.
- [Performance Analysis](PERFORMANCE_ANALYSIS.md) - Deep dive into how performance is measured.
- [OpenCL Backend](OPENCL_BACKEND.md) - Details on the OpenCL implementation.
- [Windows Packaging](WINDOWS_PACKAGING.md) - Instructions for Windows users.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
