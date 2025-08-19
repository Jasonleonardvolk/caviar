@echo off
REM TORI Self-Transformation System Startup
REM Initializes phase-coherent cognition components

echo ========================================
echo TORI Self-Transformation System
echo ========================================
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ and add to PATH
    pause
    exit /b 1
)

REM Run startup script
echo Starting self-transformation components...
echo.

python startup_self_transformation.py

if errorlevel 1 (
    echo.
    echo ERROR: Self-transformation startup failed
    echo Check the error messages above
) else (
    echo.
    echo SUCCESS: Self-transformation system ready
    echo.
    echo You can now:
    echo   - Run demo: python demo_self_transformation.py
    echo   - Run tests: python test_self_transformation.py
    echo   - Check state: type self_transformation_state.json
)

echo.
pause
