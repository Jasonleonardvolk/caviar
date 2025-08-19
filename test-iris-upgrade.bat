@echo off
echo === IRIS UPGRADE SMOKE TEST ===
echo.

echo [1/5] Checking device matrix configuration...
if exist "tori_ui_svelte\config\device-matrix.json" (
    echo ✓ Device matrix found
) else (
    echo ✗ Device matrix missing
)

echo.
echo [2/5] Testing UA endpoint...
curl -s http://localhost:3000/ua > nul 2>&1
if %errorlevel%==0 (
    echo ✓ UA endpoint responding
) else (
    echo ✗ UA endpoint not available (start dev server first)
)

echo.
echo [3/5] Checking Penrose service...
if exist "services\penrose\main.py" (
    echo ✓ Penrose service files present
) else (
    echo ✗ Penrose service missing
)

echo.
echo [4/5] Verifying WebGPU components...
if exist "tori_ui_svelte\src\lib\webgpu\init.ts" (
    echo ✓ WebGPU init module found
) else (
    echo ✗ WebGPU init missing
)

echo.
echo [5/5] Checking thermal governor...
if exist "tori_ui_svelte\src\lib\runtime\thermalGovernor.ts" (
    echo ✓ Thermal governor implemented
) else (
    echo ✗ Thermal governor missing
)

echo.
echo === TEST COMPLETE ===
echo.
echo To run full verification:
echo   1. Start dev server: cd tori_ui_svelte && pnpm dev
echo   2. Start Penrose: cd services\penrose && python -m uvicorn main:app --port 7401
echo   3. Open http://localhost:3000/hologram
echo.
pause