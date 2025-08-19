@echo off
echo ========================================
echo TORI CRITICAL FIX - ALL ISSUES
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo STEP 1: Installing ALL critical dependencies...
echo ========================================
pip install "sse-starlette>=1.8,<2" "pypdf2>=3" "deepdiff>=6.7" reportlab aiofiles

echo.
echo Attempting GUDHI (may fail on Windows)...
pip install gudhi --config-settings="--define=GUDHI_WITH_OPENMP=OFF" || echo "GUDHI failed - not critical"

echo.
echo STEP 2: Critical fixes applied...
echo ========================================
echo ✅ FractalSolitonMemory.add_concept method added
echo ✅ Vite config updated to use port 5174
echo ✅ Requirements-dev.txt updated

echo.
echo STEP 3: Testing fixes...
echo ========================================
echo Testing Python imports...
python -c "import sse_starlette; print('✅ sse_starlette OK')"
python -c "import PyPDF2; print('✅ PyPDF2 OK')"
python -c "import deepdiff; print('✅ deepdiff OK')"
python -c "from python.core.fractal_soliton_memory import FractalSolitonMemory; print('✅ FractalSolitonMemory OK')"

echo.
echo STEP 4: Next steps...
echo ========================================
echo.
echo 1. Frontend port issue:
echo    - Vite config already updated to use port 5174
echo    - The launcher needs a small fix to recognize this
echo.
echo 2. Start TORI:
echo    python enhanced_launcher.py --no-browser
echo.
echo 3. Test the API:
echo    curl http://localhost:8003/api/health
echo.
echo 4. Test the diff endpoint:
echo    .\test_diff_endpoint.bat
echo.
echo 5. Check the logs for:
echo    - "✅ Penrose engine initialized (rust)"
echo    - "[Prajna] Concept mesh diff routes loaded successfully!"
echo    - "[lattice] oscillators=" messages
echo.
pause
