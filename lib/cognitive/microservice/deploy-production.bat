@echo off
echo 🚀 TORI Cognitive System - Production Deployment
echo ================================================
echo.
echo This will set up and deploy your complete cognitive system:
echo  • Install all dependencies
echo  • Start both services (Node.js + FastAPI)
echo  • Run comprehensive tests
echo  • Generate deployment report
echo.
echo Press any key to begin deployment...
pause >nul

cd /d "C:\Users\jason\Desktop\tori\kha\lib\cognitive\microservice"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python first.
    pause
    exit /b 1
)

echo 🐍 Starting production deployment with Python...
python deploy_production.py

echo.
echo 📋 Deployment completed!
echo Check the deployment report for details.
pause
