@echo off
echo ========================================
echo CRITICAL DEPENDENCY FIX
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing critical missing dependencies...
echo.

echo [1/5] Installing sse-starlette for API streaming...
pip install "sse-starlette>=1.8,<2"

echo.
echo [2/5] Installing PyPDF2 for PDF processing...
pip install "pypdf2>=3"

echo.
echo [3/5] Installing deepdiff for Phase 6...
pip install "deepdiff>=6.7"

echo.
echo [4/5] Installing test dependencies...
pip install reportlab aiofiles

echo.
echo [5/5] Attempting GUDHI (may fail on Windows without C++ tools)...
pip install gudhi --config-settings="--define=GUDHI_WITH_OPENMP=OFF" || echo "GUDHI installation failed - not critical"

echo.
echo Updating requirements files...
pip freeze > requirements-installed.txt

echo.
echo ✅ Critical dependencies installed!
echo.
echo Next: Run fix_phase7_soliton.bat to fix FractalSolitonMemory
pause
