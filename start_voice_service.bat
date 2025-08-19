@echo off
echo.
echo ================================================================
echo  🗣️ STARTING VOICE SERVICE (Port 8101) 
echo ================================================================
echo.
echo Voice service uses: concept_mesh_8101.json
echo This prevents collision with Memory service (5173)
echo.

REM Set parameterized mesh path for Voice service
set CONCEPT_MESH_PATH=concept_mesh_8101.json

echo ✅ Environment configured:
echo    CONCEPT_MESH_PATH=%CONCEPT_MESH_PATH%
echo.

echo 🚀 Starting Voice service on port 8101...
echo.

REM Start the Voice service with the proper mesh path
python -m uvicorn prajna.api.prajna_api:app --host 0.0.0.0 --port 8101 --reload

echo.
echo 🛑 Voice service stopped.
pause
