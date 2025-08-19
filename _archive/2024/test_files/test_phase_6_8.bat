@echo off
echo ========================================
echo Testing Phase 6-8 Fixes
echo ========================================
echo.

echo [1/3] Testing diff route...
echo -----------------------------
powershell -Command "Invoke-RestMethod -Uri 'http://localhost:8003/api/concept-mesh/record_diff' -Method POST -Headers @{'Content-Type'='application/json'} -Body '{\"record_id\":\"smoke\"}' | ConvertTo-Json"

echo.
echo [2/3] Testing hologram bridge SSE...
echo -----------------------------
echo Starting SSE stream (press Ctrl+C to stop after seeing a few pings)...
timeout /t 2 >nul
powershell -Command "Invoke-WebRequest -Uri 'http://localhost:8003/holo_renderer/events' -Method GET -TimeoutSec 10"

echo.
echo [3/3] Checking available endpoints...
echo -----------------------------
powershell -Command "Invoke-RestMethod -Uri 'http://localhost:8003/api/health' -Method GET | ConvertTo-Json"

echo.
echo ========================================
echo Tests Complete!
echo ========================================
pause
