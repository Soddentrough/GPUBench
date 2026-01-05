# GPUBench Environment Configuration Script
# This script detects the necessary build tools and configures the CMake environment.

Write-Host "=== GPUBench Windows Environment Configurator ===" -ForegroundColor Cyan

# 1. Detect MinGW / MSYS2
$mingwPath = ""
$potentialMingwPaths = @(
    "C:\msys64\mingw64\bin",
    "C:\msys64\ucrt64\bin",
    "C:\ProgramData\chocolatey\bin",
    "C:\Program Files\MinGW\bin"
)

foreach ($path in $potentialMingwPaths) {
    if (Test-Path "$path\gcc.exe") {
        $mingwPath = $path
        Write-Host "Found MinGW toolchain at: $mingwPath" -ForegroundColor Green
        break
    }
}

if ($mingwPath -eq "") {
    Write-Host "ERROR: MinGW (gcc.exe) not found in common locations." -ForegroundColor Red
    Write-Host "Please install MSYS2 (https://www.msys2.org/) or ensure MingW is in your PATH."
    exit 1
}

# Add to current session PATH
$env:PATH = "$mingwPath;C:\msys64\usr\bin;$env:PATH"

# 2. Detect Vulkan SDK
if (-not $env:VULKAN_SDK) {
    $vulkanPaths = Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    if ($vulkanPaths.Count -gt 0) {
        $env:VULKAN_SDK = $vulkanPaths[0].FullName
        Write-Host "Detected Vulkan SDK: $($env:VULKAN_SDK)" -ForegroundColor Green
        $env:PATH = "$($env:VULKAN_SDK)\Bin;$env:PATH"
    } else {
        Write-Host "WARNING: Vulkan SDK not found. Vulkan benchmarks will be disabled." -ForegroundColor Yellow
    }
} else {
    Write-Host "Using VULKAN_SDK from environment: $($env:VULKAN_SDK)" -ForegroundColor Green
    $env:PATH = "$($env:VULKAN_SDK)\Bin;$env:PATH"
}

# 3. Detect Generator (Ninja preferred)
$generator = "MinGW Makefiles"
if (Get-Command ninja -ErrorAction SilentlyContinue) {
    $generator = "Ninja"
    Write-Host "Using Ninja generator" -ForegroundColor Green
} else {
    Write-Host "Using MinGW Makefiles generator" -ForegroundColor Green
}

# 4. Run CMake Configuration
$buildDir = "build-release"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Write-Host "`nConfiguring CMake..." -ForegroundColor Cyan
cmake -B $buildDir -S . -G $generator -DCMAKE_BUILD_TYPE=Release

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSUCCESS: Environment configured and CMake project generated." -ForegroundColor Green
    Write-Host "You can now build using: cmake --build $buildDir" -ForegroundColor Cyan
} else {
    Write-Host "`nERROR: CMake configuration failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
