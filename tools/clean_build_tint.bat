@echo off
echo Fixing Dawn/Tint build with proper dependencies...
echo ==================================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tools

REM Delete the corrupted tint.exe
if exist tint\tint.exe (
    echo Removing incompatible tint.exe...
    del /f tint\tint.exe
)

cd dawn

REM Clean up bad build directory
echo Cleaning build directory...
rmdir /s /q build 2>nul

echo.
echo Updating git submodules (this is critical)...
git submodule sync
git submodule update --init --recursive

REM Alternative if git submodule fails
if %errorlevel% neq 0 (
    echo Git submodule update failed, trying fetch_dawn_dependencies.py...
    python tools\fetch_dawn_dependencies.py
)

echo.
echo Creating fresh build configuration...
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_NODE_BINDINGS=OFF -DTINT_BUILD_TESTS=OFF -DTINT_BUILD_CMD_TOOLS=ON

if %errorlevel% neq 0 (
    echo CMake configuration failed!
    echo Make sure all dependencies are installed:
    echo - Git
    echo - CMake
    echo - Visual Studio 2022 with C++ tools
    echo - Python
    pause
    exit /b 1
)

echo.
echo Building Tint executable...
cmake --build build --config Release --target tint_cmd_tint_cmd

echo.
echo Searching for built tint.exe...

REM Check multiple possible locations
if exist build\src\tint\cmd\Release\tint.exe (
    mkdir ..\tint 2>nul
    copy build\src\tint\cmd\Release\tint.exe ..\tint\tint.exe
    echo Found at: build\src\tint\cmd\Release\tint.exe
    goto :test
)

if exist build\Release\tint.exe (
    mkdir ..\tint 2>nul
    copy build\Release\tint.exe ..\tint\tint.exe
    echo Found at: build\Release\tint.exe
    goto :test
)

REM Search for tint.exe anywhere in build
for /r build %%f in (tint.exe) do (
    echo Found: %%f
    mkdir ..\tint 2>nul
    copy "%%f" ..\tint\tint.exe
    goto :test
)

echo ERROR: Could not find tint.exe after build
pause
exit /b 1

:test
echo.
echo Testing Tint...
..\tint\tint.exe --version
if %errorlevel%==0 (
    echo.
    echo SUCCESS: Tint is working!
    echo.
    echo Remember to add to PATH: C:\Users\jason\Desktop\tori\kha\tools\tint
) else (
    echo ERROR: tint.exe not working correctly
)

pause
