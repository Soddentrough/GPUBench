@echo off
echo [GPUBench] Building for Windows (Isolated Toolchain)...

:: Set isolated path to avoid conflicts with multiple SDKs/Environments
set PATH=C:\msys64\mingw64\bin;C:\msys64\usr\bin;C:\WINDOWS\system32;C:\WINDOWS

:: Check if build-release exists, if not create it
if not exist "build-release" (
    echo [GPUBench] Configuring CMake...
    cmake -B build-release -S . -DCMAKE_BUILD_TYPE=Release -G "Ninja"
    if %errorlevel% neq 0 (
        echo [ERROR] CMake configuration failed.
        pause
        exit /b %errorlevel%
    )
)

:: Build the project
echo [GPUBench] Building...
cmake --build build-release
if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    pause
    exit /b %errorlevel%
)

echo [GPUBench] Build successful! 
echo Binary is at: build-release\gpubench.exe