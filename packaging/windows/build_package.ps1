# GPUBench Windows Package Builder
# This script automates building Windows packages (ZIP and/or NSIS installer)

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("zip", "nsis", "both")]
    [string]$PackageType = "both",
    
    [Parameter(Mandatory=$false)]
    [string]$BuildDir = "build-release",
    
    [Parameter(Mandatory=$false)]
    [switch]$CleanBuild = $false,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("Release", "Debug")]
    [string]$BuildType = "Release"
)

# Color output functions
function Write-Success {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Red
}

# Check if CMake is available
function Test-CMake {
    try {
        $null = cmake --version
        return $true
    } catch {
        return $false
    }
}

# Check if NSIS is available (if needed)
function Test-NSIS {
    try {
        $nsisPath = "C:\Program Files (x86)\NSIS\makensis.exe"
        if (Test-Path $nsisPath) {
            return $true
        }
        # Try to find in PATH
        $null = Get-Command makensis -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Main script
Write-Info "=== GPUBench Windows Package Builder ==="
Write-Info ""

# Verify prerequisites
if (-not (Test-CMake)) {
    Write-Error-Custom "ERROR: CMake not found. Please install CMake 3.16 or later."
    exit 1
}

if (($PackageType -eq "nsis") -or ($PackageType -eq "both")) {
    if (-not (Test-NSIS)) {
        Write-Error-Custom "WARNING: NSIS not found. NSIS installer cannot be created."
        Write-Error-Custom "         Install NSIS from https://nsis.sourceforge.io/"
        if ($PackageType -eq "nsis") {
            Write-Error-Custom "         Falling back to ZIP package only."
            $PackageType = "zip"
        } else {
            Write-Error-Custom "         Will create ZIP package only."
            $PackageType = "zip"
        }
    }
}

# Check if we're in the project root
if (-not (Test-Path "CMakeLists.txt")) {
    Write-Error-Custom "ERROR: CMakeLists.txt not found. Please run this script from the project root."
    exit 1
}

# Clean build directory if requested
if ($CleanBuild -and (Test-Path $BuildDir)) {
    Write-Info "Cleaning build directory: $BuildDir"
    Remove-Item -Path $BuildDir -Recurse -Force
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    Write-Info "Creating build directory: $BuildDir"
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Change to build directory
Push-Location $BuildDir

try {
    # Configure with CMake
    Write-Info ""
    Write-Info "Step 1: Configuring project with CMake..."
    $cmakeArgs = @(
        "..",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_INSTALL_PREFIX=GPUBench"
    )
    
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    Write-Success "✓ Configuration complete"

    # Build the project
    Write-Info ""
    Write-Info "Step 2: Building project..."
    & cmake --build . --config $BuildType
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    Write-Success "✓ Build complete"

    # Create packages
    Write-Info ""
    Write-Info "Step 3: Creating packages..."
    
    $packagesCreated = @()
    
    # Create ZIP package
    if (($PackageType -eq "zip") -or ($PackageType -eq "both")) {
        Write-Info "  Creating ZIP package..."
        & cpack -G ZIP -C $BuildType
        if ($LASTEXITCODE -eq 0) {
            $zipFile = Get-ChildItem -Filter "GPUBench-*.zip" | Select-Object -First 1
            if ($zipFile) {
                $packagesCreated += $zipFile.Name
                Write-Success "  ✓ ZIP package created: $($zipFile.Name)"
            }
        } else {
            Write-Error-Custom "  ✗ ZIP package creation failed"
        }
    }
    
    # Create NSIS installer
    if (($PackageType -eq "nsis") -or ($PackageType -eq "both")) {
        Write-Info "  Creating NSIS installer..."
        & cpack -G NSIS -C $BuildType
        if ($LASTEXITCODE -eq 0) {
            $exeFile = Get-ChildItem -Filter "GPUBench-*.exe" | Select-Object -First 1
            if ($exeFile) {
                $packagesCreated += $exeFile.Name
                Write-Success "  ✓ NSIS installer created: $($exeFile.Name)"
            }
        } else {
            Write-Error-Custom "  ✗ NSIS installer creation failed"
        }
    }

    # Summary
    Write-Info ""
    Write-Success "=== Build Complete ==="
    Write-Info ""
    Write-Info "Build directory: $((Get-Location).Path)"
    Write-Info ""
    
    if ($packagesCreated.Count -gt 0) {
        Write-Success "Packages created:"
        foreach ($package in $packagesCreated) {
            Write-Success "  - $package"
        }
        
        Write-Info ""
        Write-Info "To test the ZIP package:"
        Write-Info "  Expand-Archive $($packagesCreated[0]) -DestinationPath test"
        Write-Info "  cd test\GPUBench\bin"
        Write-Info "  .\gpubench.exe --list-benchmarks"
        
        if ($packagesCreated.Count -gt 1) {
            Write-Info ""
            Write-Info "To test the installer:"
            Write-Info "  .\$($packagesCreated[1])"
        }
    } else {
        Write-Error-Custom "No packages were created successfully."
    }
    
} catch {
    Write-Error-Custom ""
    Write-Error-Custom "ERROR: $_"
    Pop-Location
    exit 1
}

# Return to original directory
Pop-Location

Write-Info ""
Write-Success "Done!"
