@echo off
echo 🚀 TORI PRODUCTION DEPLOYMENT
echo ============================

cd /d "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte"

echo 📦 Installing dependencies...
call npm install

echo 🔧 Building frontend...
call npm run build

echo 🚀 Starting backend server...
start "TORI Backend" node server.js

echo 🌐 Starting frontend preview...
call npm run preview

echo ✅ TORI is now running in production mode!
echo Frontend: http://localhost:4173
echo Backend: http://localhost:3001
pause