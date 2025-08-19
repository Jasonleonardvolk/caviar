@echo off
REM TORI Full-Spectrum Video Ingestion System Startup Script
REM This script starts the complete video processing system

echo.
echo ========================================================
echo   TORI Full-Spectrum Video Ingestion System
echo ========================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if video requirements are installed
echo 📦 Checking dependencies...
python -c "import whisper, cv2, fastapi" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Video processing dependencies not found
    echo Would you like to install them now? [Y/N]
    set /p choice=
    if /i "%choice%"=="Y" (
        echo 🔄 Installing dependencies...
        python setup_video_system.py --install-models
    ) else (
        echo ❌ Cannot start without dependencies
        pause
        exit /b 1
    )
)

REM Start the video ingestion system
echo.
echo 🚀 Starting TORI Video Ingestion System...
echo.
echo 🎬 Features available:
echo   - Video/Audio file processing
echo   - Real-time streaming
echo   - AI transcription with Whisper
echo   - Visual context analysis
echo   - Ghost Collective reflections
echo   - Memory system integration
echo.
echo 📖 Once started, visit: http://localhost:8080
echo 📚 API Documentation: http://localhost:8080/docs
echo.
echo Press Ctrl+C to stop the system
echo.

REM Start the main application
python main_video.py

echo.
echo 🛑 Video Ingestion System stopped
pause
