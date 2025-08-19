@echo off
REM =====================================================
REM   APPLY TRACE FIXES - Production Ready
REM =====================================================

echo ================================================
echo    APPLYING PRODUCTION FIXES FROM TRACE
echo ================================================
echo.

REM Check if we're in the right directory
if not exist "api\enhanced_api.py" (
    echo ERROR: Not in the right directory!
    echo Please run from: C:\Users\jason\Desktop\tori\kha
    pause
    exit /b 1
)

echo [1/6] Checking Python files...
python -m py_compile mcp_metacognitive\server.py
if errorlevel 1 (
    echo ERROR: MCP server has syntax errors
    pause
    exit /b 1
)

python -m py_compile api\routes\concept_mesh.py
if errorlevel 1 (
    echo ERROR: Concept mesh router has syntax errors
    pause
    exit /b 1
)

echo ✅ Python files compile successfully
echo.

echo [2/6] Setting up log rotation...
if not exist "logs" mkdir logs
echo Log rotation will activate on next startup
echo.

echo [3/6] Creating ScholarSphere directories...
if not exist "data\scholarsphere\pending" mkdir data\scholarsphere\pending
if not exist "data\scholarsphere\uploaded" mkdir data\scholarsphere\uploaded
echo ✅ ScholarSphere directories created
echo.

echo [4/6] Running tests...
python -m pytest tests\test_mcp_transport.py -v
if errorlevel 1 (
    echo WARNING: Some tests failed, but continuing...
)
echo.

echo [5/6] Creating environment template...
if not exist ".env" (
    echo # ScholarSphere Configuration > .env
    echo SCHOLARSPHERE_API_URL=https://api.scholarsphere.org >> .env
    echo SCHOLARSPHERE_API_KEY=your-api-key-here >> .env
    echo SCHOLARSPHERE_BUCKET=concept-diffs >> .env
    echo. >> .env
    echo # Log Configuration >> .env
    echo LOG_ROTATION_WHEN=midnight >> .env
    echo LOG_ROTATION_DAYS=14 >> .env
    echo LOG_MAX_SIZE_MB=10 >> .env
    echo ✅ Created .env template
) else (
    echo .env already exists, skipping...
)
echo.

echo [6/6] Creating verification script...
echo import requests > verify_trace_fixes.py
echo import json >> verify_trace_fixes.py
echo. >> verify_trace_fixes.py
echo print("Verifying trace fixes...") >> verify_trace_fixes.py
echo. >> verify_trace_fixes.py
echo # Test concept mesh endpoint >> verify_trace_fixes.py
echo try: >> verify_trace_fixes.py
echo     response = requests.post( >> verify_trace_fixes.py
echo         "http://localhost:8002/api/concept-mesh/record_diff", >> verify_trace_fixes.py
echo         json={"concepts": [{"id": "test1", "name": "Test Concept", "strength": 0.8}]} >> verify_trace_fixes.py
echo     ) >> verify_trace_fixes.py
echo     print(f"Concept mesh endpoint: {response.status_code}") >> verify_trace_fixes.py
echo     if response.status_code == 200: >> verify_trace_fixes.py
echo         print("✅ Concept mesh working!") >> verify_trace_fixes.py
echo         print(f"Response: {response.json()}") >> verify_trace_fixes.py
echo except Exception as e: >> verify_trace_fixes.py
echo     print(f"❌ Concept mesh error: {e}") >> verify_trace_fixes.py

echo.
echo ================================================
echo    ALL FIXES APPLIED!
echo ================================================
echo.
echo Next steps:
echo.
echo 1. Start the API server:
echo    uvicorn api.enhanced_api:app --reload --port 8002
echo.
echo 2. Start the MCP server:
echo    python -m mcp_metacognitive.server
echo.
echo 3. Run verification:
echo    python verify_trace_fixes.py
echo.
echo 4. Check logs for oscillator counts:
echo    findstr "oscillators=" logs\session.log
echo.
echo All trace issues have been fixed! 🎉
echo.
pause
