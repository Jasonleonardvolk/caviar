@echo off
echo Building Tint with correct target...
echo =====================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tools\dawn

REM Build the tint executable
echo Building tint_cmd_tint_cmd target...
cmake --build build --config Release --target tint_cmd_tint_cmd

REM The executable should be in build\src\tint\Release or build\Release
echo.
echo Searching for tint.exe...

if exist build\src\tint\Release\tint.exe (
    mkdir ..\tint 2>nul
    copy build\src\tint\Release\tint.exe ..\tint\tint.exe
    echo SUCCESS: Found and copied tint.exe
    goto :test
)

if exist build\Release\tint.exe (
    mkdir ..\tint 2>nul
    copy build\Release\tint.exe ..\tint\tint.exe
    echo SUCCESS: Found and copied tint.exe
    goto :test
)

REM Search for any tint executable
for /r build %%f in (tint.exe) do (
    echo Found: %%f
    mkdir ..\tint 2>nul
    copy "%%f" ..\tint\tint.exe
    echo Copied to tools\tint\tint.exe
    goto :test
)

echo ERROR: tint.exe not found after build
echo Searching for any exe files in build directory...
dir /s /b build\*.exe | findstr /i tint
pause
exit /b 1

:test
echo.
echo Testing Tint...
..\tint\tint.exe --version
if %errorlevel%==0 (
    echo.
    echo SUCCESS: Tint is working!
    echo Location: C:\Users\jason\Desktop\tori\kha\tools\tint\tint.exe
) else (
    echo WARNING: tint.exe exists but may not be working correctly
)

pause
