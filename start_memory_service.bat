@echo off
echo.
echo ================================================================
echo  🧠 STARTING MEMORY SERVICE (Port 5173)
echo ================================================================
echo.
echo Memory service uses: concept_mesh_5173.json
echo This prevents collision with Voice service (8101)
echo.

REM Set parameterized mesh path for Memory service
set CONCEPT_MESH_PATH=concept_mesh_5173.json

echo ✅ Environment configured:
echo    CONCEPT_MESH_PATH=%CONCEPT_MESH_PATH%
echo.

echo 🚀 Starting Memory service on port 5173...
echo.

REM Start the Memory service with the proper mesh path
python -m uvicorn ingest_pdf.cognitive_interface:app --host 0.0.0.0 --port 5173 --reload

echo.
echo 🛑 Memory service stopped.
pause
