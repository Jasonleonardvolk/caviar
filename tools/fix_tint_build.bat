@echo off
echo Fixing Tint build...
echo ====================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tools\dawn

if not exist build (
    echo ERROR: Build directory not found. Run initial setup first.
    pause
    exit /b 1
)

echo Building Tint (correct target)...
echo.

REM Try different target names
echo Attempting build with tint_cmd target...
cmake --build build --config Release --target tint_cmd
if exist build\Release\tint.exe goto :found

echo Attempting build with tint_exe target...
cmake --build build --config Release --target tint_exe
if exist build\Release\tint.exe goto :found

echo Attempting to build ALL targets (will take longer)...
cmake --build build --config Release
if exist build\Release\tint.exe goto :found

REM Check various possible locations
echo.
echo Searching for tint.exe in build directory...
dir /s /b build\*.exe | findstr /i tint

if exist build\Release\tint_cmd.exe (
    copy build\Release\tint_cmd.exe ..\tint\tint.exe
    echo Copied tint_cmd.exe as tint.exe
    goto :success
)

if exist build\Release\tint_exe.exe (
    copy build\Release\tint_exe.exe ..\tint\tint.exe
    echo Copied tint_exe.exe as tint.exe
    goto :success
)

if exist build\tint_cmd.exe (
    copy build\tint_cmd.exe ..\tint\tint.exe
    echo Copied tint_cmd.exe as tint.exe
    goto :success
)

echo ERROR: Could not find tint executable
echo Try running: cmake --build build --config Release
echo Then look for any .exe with tint in the name
pause
exit /b 1

:found
copy build\Release\tint.exe ..\tint\tint.exe

:success
echo.
echo SUCCESS: Tint installed to C:\Users\jason\Desktop\tori\kha\tools\tint\
..\tint\tint.exe --version
pause
