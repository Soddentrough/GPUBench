@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: GPUBench Windows Build & Execution Script
:: ============================================================================
:: Usage:
::   build_windows.bat                Build GPUBench and copy runtime dependencies
::   build_windows.bat clean          Clean build-release directory before building
::   build_windows.bat test           Build and list available Vulkan devices
::   build_windows.bat run [args...]  Build and execute gpubench.exe with args
::   build_windows.bat package        Build and create CPack ZIP release archive
:: Flags:
::   --no-pause                       Do not prompt on error (useful for scripts/CI)
:: ============================================================================

set "DO_PAUSE=1"
if defined CI set "DO_PAUSE=0"
set "ACTION=build"
set "RUN_ARGS="

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--no-pause" (
    set "DO_PAUSE=0"
    shift
    goto parse_args
)
if "%~1"=="/nopause" (
    set "DO_PAUSE=0"
    shift
    goto parse_args
)
if /i "%~1"=="clean" (
    echo [GPUBench] Cleaning build directory...
    if exist "build-release" rd /s /q build-release
    shift
    goto parse_args
)
if /i "%~1"=="test" (
    set "ACTION=test"
    shift
    goto parse_args
)
if /i "%~1"=="package" (
    set "ACTION=package"
    shift
    goto parse_args
)
if /i "%~1"=="run" (
    set "ACTION=run"
    shift
    goto collect_run_args
)
shift
goto parse_args

:collect_run_args
if "%~1"=="" goto args_done
set "RUN_ARGS=!RUN_ARGS! %1"
shift
goto collect_run_args

:args_done

:: 1. Detect MinGW toolchain
set "MINGW_BIN="
if exist "C:\msys64\mingw64\bin\gcc.exe" (
    set "MINGW_BIN=C:\msys64\mingw64\bin"
) else if exist "C:\msys64\ucrt64\bin\gcc.exe" (
    set "MINGW_BIN=C:\msys64\ucrt64\bin"
)

if not "%MINGW_BIN%"=="" (
    set "PATH=%MINGW_BIN%;C:\msys64\usr\bin;%PATH%"
)

:: 2. Detect CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    if exist "C:\Program Files\CMake\bin\cmake.exe" (
        set "PATH=C:\Program Files\CMake\bin;%PATH%"
    ) else (
        echo [ERROR] CMake not found in PATH or 'C:\Program Files\CMake\bin'.
        if "%DO_PAUSE%"=="1" pause
        exit /b 1
    )
)

:: 3. Detect Vulkan SDK
if "%VULKAN_SDK%"=="" (
    for /d %%i in (C:\VulkanSDK\*) do set "VULKAN_SDK=%%i"
)
if not "%VULKAN_SDK%"=="" (
    set "PATH=%VULKAN_SDK%\Bin;%PATH%"
)

:: 4. Detect Build Generator (Prefer Ninja)
where ninja >nul 2>nul
if %errorlevel% == 0 (
    set "CMAKE_GENERATOR=Ninja"
) else if exist "%MINGW_BIN%\ninja.exe" (
    set "CMAKE_GENERATOR=Ninja"
    set "PATH=%MINGW_BIN%;%PATH%"
) else (
    set "CMAKE_GENERATOR=MinGW Makefiles"
)

:: 5. Configure CMake
echo [GPUBench] Configuring with %CMAKE_GENERATOR%...
cmake -B build-release -S . -DCMAKE_BUILD_TYPE=Release -G "%CMAKE_GENERATOR%"
if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed.
    if "%DO_PAUSE%"=="1" pause
    exit /b %errorlevel%
)

:: 6. Build GPUBench
echo [GPUBench] Building Release target...
cmake --build build-release
if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    if "%DO_PAUSE%"=="1" pause
    exit /b %errorlevel%
)

:: 7. Ensure MinGW Runtime DLLs exist in build-release for standalone execution
if not "%MINGW_BIN%"=="" (
    for %%f in (libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll) do (
        if exist "%MINGW_BIN%\%%f" (
            copy /y "%MINGW_BIN%\%%f" "build-release\" >nul 2>&1
        )
    )
)

echo [GPUBench] Build successful: build-release\gpubench.exe

:: 8. Execute Requested Action
if "%ACTION%"=="test" (
    echo.
    echo [GPUBench] Listing available devices:
    build-release\gpubench.exe -l
    exit /b %errorlevel%
)

if "%ACTION%"=="run" (
    echo.
    echo [GPUBench] Running: build-release\gpubench.exe !RUN_ARGS!
    build-release\gpubench.exe !RUN_ARGS!
    exit /b !errorlevel!
)

if "%ACTION%"=="package" (
    echo.
    echo [GPUBench] Creating ZIP release package...
    pushd build-release
    cpack -G ZIP -C Release
    set "CPACK_ERR=!errorlevel!"
    popd
    if !CPACK_ERR! neq 0 (
        echo [WARNING] Packaging encountered errors.
    ) else (
        echo [GPUBench] Package created in build-release\
    )
)

exit /b 0