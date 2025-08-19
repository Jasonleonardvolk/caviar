@echo off
REM test_tint.cmd - Quick test to verify Tint is installed correctly

echo === Testing Tint Installation ===
echo.

if exist tint.exe (
    echo [CHECK] tint.exe found
    echo.
    echo Version:
    tint.exe --version 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] tint.exe exists but won't run
        echo         May be wrong architecture or missing dependencies
    ) else (
        echo.
        echo [SUCCESS] Tint is working!
        echo.
        echo Testing WGSL validation:
        echo @vertex fn main() -^> @builtin^(position^) vec4^<f32^> { return vec4^<f32^>^(0.0^); } > test.wgsl
        tint.exe test.wgsl
        del test.wgsl 2>nul
    )
) else (
    echo [ERROR] tint.exe not found!
    echo.
    echo Please download it:
    echo   1. Run: powershell -ExecutionPolicy Bypass -File download_tint.ps1
    echo   2. Or manually download from:
    echo      https://dawn.googlesource.com/dawn/
    echo      https://github.com/google/dawn/releases
    echo   3. Place tint.exe in this directory
)

echo.
pause
