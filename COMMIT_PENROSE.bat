@echo off
echo ========================================
echo TORI Penrose - Commit and Push
echo ========================================
echo.

REM Always start from project root
cd /d C:\Users\jason\Desktop\tori\kha

REM Activate venv first
echo Activating virtual environment...
call .venv\Scripts\activate

REM Verify we're in venv
echo.
echo Verifying environment:
where python
python -c "import sys; print('Using:', sys.executable)"

REM Quick test that Penrose works
echo.
echo Testing Penrose import...
python -c "import penrose_engine_rs, sys; print('SUCCESS: Rust backend from', sys.executable)"

if errorlevel 1 (
    echo ERROR: Penrose import failed!
    pause
    exit /b 1
)

REM Git operations
echo.
echo ========================================
echo Git Operations
echo ========================================
echo.

echo Adding files...
git add concept_mesh/penrose_rs/Cargo.toml
git add concept_mesh/penrose_rs/src/lib.rs
git add .github/workflows/build-penrose.yml

echo.
echo Current git status:
git status

echo.
echo Ready to commit with message:
echo "feat: Penrose Rust engine - remove OpenBLAS, add CI workflow"
echo.
echo Press ENTER to commit and push, or Ctrl+C to cancel...
pause >nul

git commit -m "feat: Penrose Rust engine - remove OpenBLAS, add CI workflow"
git push

echo.
echo ========================================
echo DONE! Check GitHub Actions
echo ========================================
echo.
echo Next steps:
echo 1. Go to: https://github.com/Jasonleonardvolk/Tori/actions
echo 2. Watch "Build Penrose Wheels" workflow
echo 3. Once green, we'll do Phase 2 (README updates)
echo.
pause
