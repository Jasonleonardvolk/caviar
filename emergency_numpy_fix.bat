@echo off
REM ============================================
REM EMERGENCY NUMPY ABI FIX - QUICK & DIRTY
REM ============================================
REM Use this when the main fix scripts fail or you need the fastest possible fix
REM This performs a minimal, aggressive cleanup and reinstall

echo.
echo ============================================
echo     EMERGENCY NUMPY ABI FIX
echo ============================================
echo.
echo WARNING: This is an aggressive fix that will:
echo   - Force uninstall ALL numpy-dependent packages
echo   - Delete the entire virtual environment
echo   - Clear ALL caches
echo   - Reinstall everything from scratch
echo.
echo This should take 3-5 minutes.
echo.

set /p confirm="Are you SURE you want to continue? (type YES): "
if /i not "%confirm%"=="YES" (
    echo.
    echo Fix cancelled.
    exit /b 0
)

echo.
echo [1/6] Killing any Python processes...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2/6] Force removing packages...
echo   Removing numpy and all dependent packages...

REM Try to uninstall via pip first (faster)
python -m pip uninstall -y numpy scipy pandas scikit-learn matplotlib thinc spacy torch tensorflow numba 2>nul
poetry run pip uninstall -y numpy scipy pandas scikit-learn matplotlib thinc spacy torch tensorflow numba 2>nul

echo.
echo [3/6] Deleting virtual environment...
if exist ".venv" (
    rmdir /s /q ".venv"
    echo   Virtual environment deleted.
) else (
    echo   No .venv found.
)

echo.
echo [4/6] Clearing all caches...

REM Clear pip cache
python -m pip cache purge 2>nul
poetry run pip cache purge 2>nul

REM Clear poetry cache
poetry cache clear pypi --all -n 2>nul

REM Clear Python cache files
echo   Removing __pycache__ directories...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo   Removing .pyc files...
del /s /q *.pyc 2>nul

REM Clear temp files
if exist "%TEMP%\pip-*" del /s /q "%TEMP%\pip-*" 2>nul

echo.
echo [5/6] Reinstalling packages with Poetry...
echo   This will take a few minutes...

REM First ensure Poetry itself is working
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Poetry is not available!
    echo Please install Poetry first: https://python-poetry.org/docs/#installation
    pause
    exit /b 1
)

REM Check for pyproject.toml
if not exist "pyproject.toml" (
    echo.
    echo ERROR: pyproject.toml not found!
    echo Current directory: %CD%
    echo Expected location: C:\Users\jason\Desktop\tori\kha
    pause
    exit /b 1
)

REM Force a fresh install
poetry install --no-cache

echo.
echo [6/6] Quick verification...

REM Test critical imports
poetry run python -c "import numpy; print(f'Numpy {numpy.__version__} OK')" 2>nul
if %errorlevel% equ 0 (
    echo   Numpy: OK
) else (
    echo   Numpy: FAILED
)

poetry run python -c "import spacy; print(f'Spacy {spacy.__version__} OK')" 2>nul
if %errorlevel% equ 0 (
    echo   Spacy: OK
) else (
    echo   Spacy: FAILED - This may be OK if spacy is optional
)

poetry run python -c "import thinc; print(f'Thinc {thinc.__version__} OK')" 2>nul
if %errorlevel% equ 0 (
    echo   Thinc: OK
) else (
    echo   Thinc: FAILED - This may be OK if thinc is optional
)

echo.
echo ============================================
echo          EMERGENCY FIX COMPLETE
echo ============================================
echo.
echo Next steps:
echo 1. Run full verification:
echo    poetry run python verify_numpy_abi.py
echo.
echo 2. If verification passes, launch your app:
echo    poetry run python enhanced_launcher.py --api full --require-penrose --enable-hologram --hologram-audio
echo.
echo 3. If issues persist, try the full fix:
echo    numpy_abi_fix.bat
echo.

pause