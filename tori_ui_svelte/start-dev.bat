@echo off
echo 🚀 Starting TORI Development Server...
echo.

REM Check if node_modules exists
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    call npm install
    echo.
)

echo 🌌 Starting TORI Genesis UI...
echo ✨ Light theme active
echo 🧠 Memory system ready
echo 👻 Ghost AI initialized
echo.

echo Opening TORI at http://localhost:5173
echo Press Ctrl+C to stop the server
echo.

call npm run dev