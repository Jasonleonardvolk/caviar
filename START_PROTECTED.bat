@echo off
echo 🔧 TORI EXTRACTION CRASH PREVENTION
echo ===================================
echo.
echo This version adds timeout and memory protection to prevent crashes
echo during the concept extraction process
echo.
echo ⏰ Extraction timeout: 120 seconds  
echo 🧠 Memory protection: enabled
echo 📏 Text size limit: 1MB to prevent memory issues
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo 🚀 Starting protected API server...
python protected_server.py

echo.
echo 🔧 Protected server stopped
pause
