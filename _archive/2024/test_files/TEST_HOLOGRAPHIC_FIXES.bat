@echo off
echo 🔧 Starting TORI Development Server with All Fixes Applied...
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tori_ui_svelte

echo 📦 Installing dependencies...
call npm install

echo 🎯 Running development server...
call npm run dev

pause
