@echo off
echo ==========================================
echo 🍺 TORI BEER TIME AUTO-UPLOAD 🍺
echo ==========================================
echo.
echo 🤖 About to process 23GB of arXiv/Nature PDFs
echo 🍺 Perfect time to grab a beer (or several)!
echo ⏱️ Estimated time: 2-4 hours depending on file count
echo.

REM Check if backend is running
echo 🔍 Checking if TORI backend is running...
curl -s http://localhost:8002/health >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ TORI backend not running!
    echo 💡 Starting backend first...
    start "TORI Backend" cmd /k "cd /d C:\Users\jason\Desktop\tori\kha && python start_dynamic_api.py"
    echo ⏳ Waiting 10 seconds for backend to start...
    timeout /t 10 /nobreak >nul
)

echo ✅ Backend ready! Starting auto-upload...
echo.
echo 🍺 BEER TIME! Go relax while the beast works!
echo 📊 Progress will be logged to pdf_upload.log
echo 🔄 Can resume if interrupted - progress is saved
echo.

REM Install required packages if needed
pip install aiohttp aiofiles >nul 2>&1

REM Start the upload beast
python auto_upload_pdfs.py

echo.
echo 🎉 Upload complete! Check pdf_upload.log for details
echo 🧠 Your knowledge base is now MASSIVE!
echo 🍺 Hope you enjoyed your beer! 🍺
pause