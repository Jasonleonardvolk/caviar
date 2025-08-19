@echo off
REM Batch script to fix NumPy ABI issues - Windows Command Prompt version
REM This completely bypasses Poetry and uses pip directly
REM Author: Enhanced Assistant
REM Date: 2025-08-06

cls
echo.
echo ===========================================================
echo      DIRECT FIX FOR NUMPY ABI ISSUES (BATCH VERSION)
echo ===========================================================
echo.
echo CRITICAL: You were using NumPy 2.2.6 - THIS IS WRONG!
echo We need to use NumPy 1.26.4 for spaCy compatibility!
echo.
echo This script will:
echo   1. Uninstall all problematic packages
echo   2. Install NumPy 1.26.4 (the CORRECT version)
echo   3. Reinstall spaCy and dependencies
echo.
pause

REM Check if in virtual environment
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found!
    echo Please create a virtual environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    pause
    exit /b 1
)

echo.
echo Step 1: Uninstalling problematic packages...
echo ----------------------------------------
.venv\Scripts\pip uninstall -y spacy spacy-legacy spacy-loggers thinc cymem murmurhash preshed numpy scipy pandas matplotlib

echo.
echo Step 2: Clearing pip cache...
echo ----------------------------------------
.venv\Scripts\pip cache purge

echo.
echo Step 3: Installing NumPy 1.26.4 (CORRECT VERSION)...
echo ----------------------------------------
.venv\Scripts\pip install numpy==1.26.4 --no-cache-dir

REM Verify NumPy version
echo.
echo Verifying NumPy installation...
.venv\Scripts\python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo.
echo Step 4: Installing scipy, pandas, matplotlib...
echo ----------------------------------------
.venv\Scripts\pip install scipy==1.14.1 pandas==2.2.3 matplotlib==3.9.2 --no-cache-dir

echo.
echo Step 5: Installing spaCy dependencies...
echo ----------------------------------------
.venv\Scripts\pip install Cython cymem murmurhash preshed --no-cache-dir

echo.
echo Step 6: Installing thinc...
echo ----------------------------------------
.venv\Scripts\pip install thinc==8.2.4 --no-cache-dir

echo.
echo Step 7: Installing spaCy...
echo ----------------------------------------
.venv\Scripts\pip install spacy==3.7.5 --no-cache-dir

echo.
echo Step 8: Final verification...
echo ----------------------------------------
.venv\Scripts\python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
.venv\Scripts\python -c "import spacy; print(f'spaCy: {spacy.__version__}')"
.venv\Scripts\python -c "import thinc; print(f'thinc: {thinc.__version__}')"

echo.
echo Testing spaCy functionality...
.venv\Scripts\python -c "import spacy; nlp = spacy.blank('en'); doc = nlp('Test'); print('spaCy test: SUCCESS')"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===========================================================
    echo                    FIX COMPLETED SUCCESSFULLY!
    echo ===========================================================
    echo.
    echo Your environment now has:
    echo   - NumPy 1.26.4 (compatible version)
    echo   - spaCy 3.7.5 (working)
    echo   - All dependencies properly installed
    echo.
    echo Next steps:
    echo   1. Activate your environment: .venv\Scripts\activate
    echo   2. Download spaCy models: python -m spacy download en_core_web_sm
    echo   3. Run your application
) else (
    echo.
    echo ===========================================================
    echo                    FIX FAILED - SEE ERRORS ABOVE
    echo ===========================================================
    echo.
    echo Try running the nuclear fix instead:
    echo   powershell -ExecutionPolicy Bypass -File nuclear_fix_numpy_abi.ps1
)

echo.
pause