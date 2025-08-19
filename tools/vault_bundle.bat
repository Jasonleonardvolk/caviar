@echo off
echo 📦 Creating Vault Bundle...
echo ==========================
echo.

REM Check if output filename was provided
if "%1"=="" (
    echo Usage: vault_bundle.bat output_filename.tar.zst
    exit /b 1
)

REM Create archives directory if it doesn't exist
if not exist "%~dp0\..\archives" mkdir "%~dp0\..\archives"

REM Run the bundle command
python "%~dp0\vault_inspector.py" --bundle "%~dp0\..\archives\%1"

echo.
echo Bundle saved to: archives\%1
