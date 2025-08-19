@echo off
echo ULTRA MINIMAL - Get JUST Tint without the bloat
echo ================================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tools

echo Option 1: Try downloading pre-compiled Tint from Dawn CI
echo ---------------------------------------------------------
powershell -Command "& {try { Invoke-WebRequest -Uri 'https://ci.chromium.org/p/dawn/builders/ci/win-rel/latest' -OutFile 'tint_latest.html' } catch { Write-Host 'CI not accessible' }}"

echo.
echo Option 2: Clone ONLY what we need (no ANGLE!)
echo ----------------------------------------------
echo.

if not exist dawn_minimal (
    echo Cloning Dawn with minimal depth...
    git clone --depth 1 --single-branch https://dawn.googlesource.com/dawn dawn_minimal
    
    cd dawn_minimal
    
    echo Getting ONLY the required submodules (no ANGLE!)...
    git submodule update --init --depth 1 third_party/abseil-cpp
    git submodule update --init --depth 1 third_party/glslang/src  
    git submodule update --init --depth 1 third_party/spirv-headers/src
    git submodule update --init --depth 1 third_party/spirv-tools/src
    
    echo.
    echo Quick build with Ninja (if available) or Make...
    cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DTINT_BUILD_CMD_TOOLS=ON -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_NODE_BINDINGS=OFF -DTINT_BUILD_TESTS=OFF
    
    if %errorlevel% neq 0 (
        echo Ninja not found, trying with default generator...
        cmake -B build -DCMAKE_BUILD_TYPE=Release -DTINT_BUILD_CMD_TOOLS=ON -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_NODE_BINDINGS=OFF -DTINT_BUILD_TESTS=OFF
    )
    
    cmake --build build --config Release --target tint_cmd_tint_cmd
    
    cd ..
)

echo.
echo Option 3: Forget Tint - proceed with shader work
echo -------------------------------------------------
echo The shader validation will work without Tint.
echo You'll just see "Tint not found" warnings.
echo.
pause
