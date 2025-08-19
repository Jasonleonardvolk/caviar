@echo off
echo.
echo ========================================
echo Testing Soliton API Endpoints
echo ========================================
echo.

echo 1. Testing /api/soliton/health...
curl -s -o nul -w "   HTTP Status: %%{http_code}\n" http://localhost:5173/api/soliton/health
echo.

echo 2. Testing /api/soliton/diagnostic...
curl -s -o nul -w "   HTTP Status: %%{http_code}\n" http://localhost:5173/api/soliton/diagnostic
echo.

echo 3. Testing /api/soliton/init...
curl -s -o nul -w "   HTTP Status: %%{http_code}\n" -X POST -H "Content-Type: application/json" -d "{\"userId\":\"test_user\"}" http://localhost:5173/api/soliton/init
echo.

echo ========================================
echo.
echo If you see 500 errors above, run:
echo    RUN_GREMLIN_FIX.ps1
echo.
echo If you see connection errors, start the API:
echo    cd C:\Users\jason\Desktop\tori\kha
echo    uvicorn api.main:app --port 5173 --reload
echo.
pause
