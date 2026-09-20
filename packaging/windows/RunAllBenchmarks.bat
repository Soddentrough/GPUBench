@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo   GPUBench - Automated Benchmark Execution
echo ===============================================================================
echo.

:: Get the directory where the batch script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check if gpubench.exe exists in the bin directory
if not exist "bin\gpubench.exe" (
    echo [ERROR] bin\gpubench.exe not found.
    echo Please ensure you are running this script from the GPUBench root directory.
    echo.
    pause
    exit /b 1
)

echo Running all benchmarks on the default device...
echo.

:: Run the benchmarks
"bin\gpubench.exe"

echo.
echo ===============================================================================
echo   Benchmarks completed.
echo ===============================================================================
echo.
echo You can now copy the results from this window.
echo Press any key to exit...
pause > nul
