@echo off
echo ========================================
echo    Installing Penrose Dependencies
echo ========================================
echo.

echo [1/3] Installing scipy (sparse matrix operations)...
python -m pip install scipy
if %errorlevel% neq 0 (
    echo ERROR: Failed to install scipy
    pause
    exit /b 1
)
echo SUCCESS: scipy installed
echo.

echo [2/3] Installing zstandard (compression)...
python -m pip install zstandard
if %errorlevel% neq 0 (
    echo ERROR: Failed to install zstandard
    pause
    exit /b 1
)
echo SUCCESS: zstandard installed
echo.

echo [3/3] Installing numba (JIT compilation - optional but recommended)...
python -m pip install numba
if %errorlevel% neq 0 (
    echo WARNING: Failed to install numba (optional - Penrose will work without it)
    echo          You'll still get great performance, just not the maximum possible
) else (
    echo SUCCESS: numba installed (2x additional speedup enabled!)
)
echo.

echo ========================================
echo    Verifying Penrose Installation
echo ========================================
echo.

python verify_penrose.py

echo.
echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo Next step: Restart TORI with
echo   python enhanced_launcher.py
echo.
pause
