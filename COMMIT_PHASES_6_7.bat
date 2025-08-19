@echo off
echo ========================================
echo Phase 6-8 Implementation Complete
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing required dependencies...
pip install aiofiles torch httpx pytest pytest-asyncio

echo.
echo Files created:
echo - tests\test_penrose.py (Performance regression tests)
echo - api\concept_mesh_diff.py (Phase 6 diff endpoint)
echo - python\core\fractal_soliton_events.py (Phase 7 events)
echo - python\core\lattice_evolution_subscriber.py (Oscillator updates)
echo - tests\test_phase6_7_integration.py (Integration tests)
echo.

echo Running tests...
echo.
echo [1/2] Penrose speed test...
pytest tests\test_penrose.py -v

echo.
echo [2/2] Phase 6-7 integration test (requires API running)...
echo Note: Start TORI first with: python enhanced_launcher.py --no-browser
echo Then run: python tests\test_phase6_7_integration.py
echo.

echo ========================================
echo Commit commands:
echo ========================================
echo git add -A
echo git commit -m "feat: implement Phases 6-7 - ScholarSphere diff and oscillator feed"
echo git push
echo.

echo Integration steps:
echo 1. The concept mesh diff endpoint has been added to prajna_api.py
echo 2. The lattice runner now subscribes to concept events
echo 3. See api\phase6_7_integration_helper.py for integration code
echo.

echo Next: After starting TORI, ingest a PDF and watch for:
echo - "Diff queued" messages in logs
echo - "[lattice] concept_oscillators=" count increasing
echo.
pause
