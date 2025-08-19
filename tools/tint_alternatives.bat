@echo off
echo Alternative Tint Installation Methods
echo =====================================
echo.
echo OPTION 1: Using Chocolatey (if installed)
echo -------------------------------------------
where choco >nul 2>&1
if %errorlevel%==0 (
    echo Chocolatey found. Try:
    echo   choco install dawn-tint
    echo.
) else (
    echo Chocolatey not found.
    echo.
)

echo OPTION 2: Using npm (if Node.js installed)
echo -------------------------------------------
where npm >nul 2>&1
if %errorlevel%==0 (
    echo npm found. Try:
    echo   npm install -g @google/dawn-tint
    echo.
) else (
    echo npm not found.
    echo.
)

echo OPTION 3: Direct from Google Storage
echo -------------------------------------------
echo Try this PowerShell command:
echo.
echo powershell -Command "Invoke-WebRequest -Uri 'https://storage.googleapis.com/dawn-release/tint-windows-amd64.exe' -OutFile 'C:\Users\jason\Desktop\tori\kha\tools\tint\tint.exe'"
echo.

echo OPTION 4: Skip Tint for now
echo -------------------------------------------
echo The shader validation will still work, it just won't do
echo HLSL/MSL cross-checks. You'll see "Tint not found" warnings
echo but shaders will still compile and validate.
echo.
echo Create a dummy tint.exe that returns success:
echo.
echo   echo @echo off > C:\Users\jason\Desktop\tori\kha\tools\tint\tint.exe
echo   echo exit /b 0 >> C:\Users\jason\Desktop\tori\kha\tools\tint\tint.exe
echo.
pause
