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
REM Create simple performance test
if not exist "tests\test_penrose_simple.py" (
    mkdir tests 2>nul
    (
        echo import penrose_engine_rs
        echo import time
        echo print("Testing performance...")
        echo v1 = [1.0] * 512
        echo v2 = [1.0] * 512
        echo start = time.perf_counter()
        echo result = penrose_engine_rs.compute_similarity(v1, v2)
        echo elapsed = time.perf_counter() - start
        echo print(f"Single similarity: {elapsed*1000:.3f}ms")
        echo print(f"Result: {result}")
        echo if elapsed ^< 0.003:
        echo     print("PASS: Performance is good!")
        echo else:
        echo     print("FAIL: Too slow")
    ) > tests\test_penrose_simple.py
)
python tests\test_penrose_simple.py
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
    type launcher_test.log | findstr "Penrose"
) else (
    echo SUCCESS: App boots with Rust backend!
    type launcher_test.log | findstr "Penrose"
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
echo If all checks passed, Phase 1 is COMPLETE!
echo.
pause
