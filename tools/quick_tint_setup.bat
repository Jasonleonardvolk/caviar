@echo off
REM Super Quick Tint Install - The 2-minute way
REM
REM OPTION 1 (Fastest): Download pre-built binary
REM -----------------------------------------------
REM 1. Go to: https://github.com/google/dawn/releases
REM 2. Download latest Windows release (dawn-XXXX-windows-amd64.zip)
REM 3. Extract just tint.exe to C:\Users\jason\Desktop\tori\kha\tools\tint\
REM 4. Run this batch file to add to PATH

set TINT_DIR=C:\Users\jason\Desktop\tori\kha\tools\tint

REM Check if tint.exe exists
if exist "%TINT_DIR%\tint.exe" (
    echo Found tint.exe at %TINT_DIR%
    echo Testing tint...
    "%TINT_DIR%\tint.exe" --version
    
    echo.
    echo Adding to PATH for this session...
    set PATH=%PATH%;%TINT_DIR%
    
    echo.
    echo To add permanently, run:
    echo   setx PATH "%PATH%;%TINT_DIR%"
    echo.
    echo SUCCESS: Tint ready to use!
) else (
    echo ERROR: tint.exe not found at %TINT_DIR%\tint.exe
    echo.
    echo Quick fix:
    echo 1. Download: https://github.com/google/dawn/releases
    echo 2. Get the latest dawn-XXXX-windows-amd64.zip
    echo 3. Extract tint.exe to: %TINT_DIR%\
    echo 4. Run this script again
)

pause
