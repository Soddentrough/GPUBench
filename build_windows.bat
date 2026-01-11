@echo off
:: Robust Environment Detection for Windows
set "FOUND_COMPILER=0"

:: 1. Try common MingW locations
if exist "C:\msys64\mingw64\bin\gcc.exe" (
    set "PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%"
    set "FOUND_COMPILER=1"
) else if exist "C:\msys64\ucrt64\bin\gcc.exe" (
    set "PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%"
    set "FOUND_COMPILER=1"
)

:: 2. Try to find Vulkan SDK if not set
if "%VULKAN_SDK%"=="" (
    for /d %%i in (C:\VulkanSDK\*) do set "VULKAN_SDK=%%i"
)
if not "%VULKAN_SDK%"=="" (
    set "PATH=%VULKAN_SDK%\Bin;%PATH%"
)

:: Check for basic tools
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] CMake not found in PATH. Please install CMake.
    pause
    exit /b 1
)

:: Optional: Handle 'clean' argument
if "%1"=="clean" (
    echo [GPUBench] Cleaning build directory...
    if exist "build-release" rd /s /q build-release
)

:: Configure CMake (always run to pick up version changes)
echo [GPUBench] Configuring CMake...
    
:: Prefer Ninja if available
where ninja >nul 2>nul
if %errorlevel% == 0 (
    cmake -B build-release -S . -DCMAKE_BUILD_TYPE=Release -G "Ninja"
) else (
    cmake -B build-release -S . -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
)

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed.
    pause
    exit /b %errorlevel%
)

:: Build the project
echo [GPUBench] Building...
cmake --build build-release
if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    pause
    exit /b %errorlevel%
)

:: Create ZIP package
echo [GPUBench] Creating ZIP package...
cd build-release
cpack -G ZIP -C Release
if %errorlevel% neq 0 (
    echo [WARNING] Packaging failed.
) else (
    echo [GPUBench] ZIP package created successfully.
    for %%i in (GPUBench-*.zip) do echo [GPUBench] Package: build-release\%%i
)
cd ..

echo [GPUBench] Build successful! 
echo Binary is at: build-release\gpubench.exe