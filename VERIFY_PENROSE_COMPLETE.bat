@echo off
echo ========================================
echo Penrose Integration - Final Verification
echo ========================================
echo.

REM Always start from project root
cd /d C:\Users\jason\Desktop\tori\kha

REM Activate venv
call .venv\Scripts\activate

echo [1/3] Testing Wheel Import in venv...
echo =====================================
python -c "import penrose_engine_rs, sys; print('SUCCESS: Rust backend from', sys.executable)"
if errorlevel 1 (
    echo FAILED: Could not import penrose_engine_rs
    pause
    exit /b 1
)
echo.

echo [2/3] Running Penrose Performance Test...
echo ========================================
if exist "tests\test_penrose.py" (
    pytest tests\test_penrose.py -q
) else (
    echo Creating basic performance test...
    mkdir tests 2>nul
    echo import penrose_engine_rs > tests\test_penrose.py
    echo import time >> tests\test_penrose.py
    echo def test_penrose_performance(): >> tests\test_penrose.py
    echo     start = time.perf_counter() >> tests\test_penrose.py
    echo     result = penrose_engine_rs.compute_similarity([1.0]*512, [1.0]*512) >> tests\test_penrose.py
    echo     elapsed = time.perf_counter() - start >> tests\test_penrose.py
    echo     print(f'\nCompute similarity took {elapsed*1000:.3f}ms') >> tests\test_penrose.py
    echo     assert elapsed ^< 0.003  # Should be under 3ms >> tests\test_penrose.py
    echo     assert abs(result - 1.0) ^< 0.001  # Should be ~1.0 >> tests\test_penrose.py
    
    pip install pytest >nul 2>&1
    pytest tests\test_penrose.py -q
)
echo.

echo [3/3] Testing Full App with Rust Requirement...
echo ==============================================
echo Starting enhanced launcher (5 second test)...
echo.
start /B python enhanced_launcher.py --require-penrose --no-browser > launcher_test.log 2>&1
timeout /t 5 /nobreak >nul
taskkill /F /IM python.exe >nul 2>&1

echo Checking launcher log for Rust initialization...
findstr /C:"Penrose engine initialized (rust)" launcher_test.log >nul
if errorlevel 1 (
    echo FAILED: Launcher did not initialize with Rust backend
    echo Check launcher_test.log for details
    type launcher_test.log
) else (
    echo SUCCESS: App boots with Rust backend!
    findstr /C:"Penrose" launcher_test.log
)
del launcher_test.log 2>nul
echo.

echo ========================================
echo LOCAL VERIFICATION COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Push commits if not already done:
echo    git push
echo.
echo 2. Check GitHub Actions:
echo    https://github.com/Jasonleonardvolk/Tori/actions
echo.
echo 3. Look for green "Build Penrose Wheels" workflow
echo.
echo 4. Download wheel artifacts to verify builds
echo.
echo If all checks passed, Phase 1 is COMPLETE!
echo.
pause
