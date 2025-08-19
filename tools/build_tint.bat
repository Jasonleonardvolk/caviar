@echo off
echo Building Tint from source...
echo ============================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tools

REM Check prerequisites
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git not found. Install Git for Windows first.
    pause
    exit /b 1
)

where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake not found. Install CMake first.
    echo Download from: https://cmake.org/download/
    pause
    exit /b 1
)

REM Clone Dawn repository
if not exist dawn (
    echo Cloning Dawn repository...
    git clone https://dawn.googlesource.com/dawn
    if %errorlevel% neq 0 (
        echo Failed to clone. Trying GitHub mirror...
        git clone https://github.com/google/dawn.git
    )
)

cd dawn

REM Update dependencies
echo Fetching dependencies...
python tools\fetch_dawn_dependencies.py
if %errorlevel% neq 0 (
    echo Trying without Python script...
    git submodule update --init --recursive
)

REM Configure build
echo Configuring build...
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_NODE_BINDINGS=OFF -DTINT_BUILD_TESTS=OFF

REM Build just Tint
echo Building Tint (this will take a few minutes)...
cmake --build build --config Release --target tint

REM Copy tint.exe to tools directory
if exist build\Release\tint.exe (
    copy build\Release\tint.exe ..\tint\tint.exe
    echo SUCCESS: Tint built and installed to C:\Users\jason\Desktop\tori\kha\tools\tint\
) else if exist build\tint.exe (
    copy build\tint.exe ..\tint\tint.exe
    echo SUCCESS: Tint built and installed to C:\Users\jason\Desktop\tori\kha\tools\tint\
) else (
    echo ERROR: Could not find built tint.exe
    echo Check build\Release\ or build\ directories
)

cd ..
pause
