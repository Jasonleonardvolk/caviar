@echo off
REM Shader Validation Runner for Windows
REM Usage: validate_shaders.bat [iphone15|desktop|webgpu-baseline]

set TARGET=%1
if "%TARGET%"=="" set TARGET=iphone15

echo ========================================
echo Shader Quality Gate v2.0
echo Target Device: %TARGET%
echo ========================================

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Run validation
node tools\shaders\shader_quality_gate_v2.mjs ^
  --dir=frontend/ ^
  --strict ^
  --targets=msl,hlsl ^
  --limits=tools\shaders\device_limits.%TARGET%.json ^
  --report=build\shader_report.%TARGET%.json ^
  --junit=build\shader_report.%TARGET%.junit.xml

if %ERRORLEVEL% EQU 0 (
  echo.
  echo ✅ All shaders passed validation!
  echo 📄 Report saved to build\shader_report.%TARGET%.json
) else if %ERRORLEVEL% EQU 1 (
  echo.
  echo ❌ Shader validation failed!
  echo 📄 Check build\shader_report.%TARGET%.json for details
  exit /b 1
) else if %ERRORLEVEL% EQU 2 (
  echo.
  echo ⚠️ Warnings found in strict mode
  echo 📄 Check build\shader_report.%TARGET%.json for details
  exit /b 2
) else if %ERRORLEVEL% EQU 3 (
  echo.
  echo ❌ Device limit violations detected!
  echo 📄 Check build\shader_report.%TARGET%.json for details
  exit /b 3
)
