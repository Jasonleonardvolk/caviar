@echo off
echo MINIMAL Tint build - No unnecessary dependencies
echo =================================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tools

REM Stop any running git processes
taskkill /F /IM git.exe 2>nul

cd dawn

REM Only fetch the minimal required submodules for Tint
echo Fetching ONLY required dependencies for Tint...
git submodule update --init third_party/abseil-cpp
git submodule update --init third_party/glslang/src
git submodule update --init third_party/spirv-headers/src
git submodule update --init third_party/spirv-tools/src
git submodule update --init third_party/vulkan-headers/src

REM Create the missing template file if needed
if not exist third_party\glslang\src\build_info.h.tmpl (
    echo Creating missing glslang template...
    echo // Auto-generated build info > third_party\glslang\src\build_info.h.tmpl
    echo #define GLSLANG_VERSION_MAJOR 14 >> third_party\glslang\src\build_info.h.tmpl
    echo #define GLSLANG_VERSION_MINOR 0 >> third_party\glslang\src\build_info.h.tmpl
    echo #define GLSLANG_VERSION_PATCH 0 >> third_party\glslang\src\build_info.h.tmpl
    echo #define GLSLANG_VERSION_FLAVOR "" >> third_party\glslang\src\build_info.h.tmpl
)

echo.
echo Configuring minimal Tint build...
cmake -B build_minimal -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DDAWN_BUILD_SAMPLES=OFF ^
    -DDAWN_BUILD_NODE_BINDINGS=OFF ^
    -DDAWN_ENABLE_D3D11=OFF ^
    -DDAWN_ENABLE_D3D12=OFF ^
    -DDAWN_ENABLE_VULKAN=OFF ^
    -DDAWN_ENABLE_OPENGLES=OFF ^
    -DDAWN_ENABLE_OPENGL=OFF ^
    -DTINT_BUILD_TESTS=OFF ^
    -DTINT_BUILD_CMD_TOOLS=ON ^
    -DTINT_BUILD_GLSL_WRITER=OFF ^
    -DTINT_BUILD_MSL_WRITER=ON ^
    -DTINT_BUILD_HLSL_WRITER=ON

echo.
echo Building ONLY tint executable...
cmake --build build_minimal --config Release --target tint_cmd_tint_cmd

echo.
echo Looking for tint.exe...
for /r build_minimal %%f in (tint.exe) do (
    echo Found: %%f
    mkdir ..\tint 2>nul
    copy "%%f" ..\tint\tint.exe
    goto :found
)

echo ERROR: tint.exe not found
pause
exit /b 1

:found
echo.
echo Testing...
..\tint\tint.exe --version
if %errorlevel%==0 (
    echo SUCCESS! Tint is ready.
) else (
    echo Error running tint
)
pause
