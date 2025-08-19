@echo off
REM Quick check and install of toml package for dependency manager

echo Checking for TOML parser...
python -c "import toml" 2>nul
if %errorlevel% neq 0 (
    echo Installing toml package...
    pip install toml
) else (
    echo ✅ TOML parser already installed
)

echo.
pause
