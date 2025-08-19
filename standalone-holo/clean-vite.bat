@echo off
echo Cleaning Vite cache...

:: Kill any running node processes
taskkill /F /IM node.exe 2>nul

:: Remove .vite cache
if exist "node_modules\.vite" (
    rmdir /s /q "node_modules\.vite" 2>nul
    echo Cleared .vite cache
)

:: Remove Vite temp files
if exist "node_modules\.vite" (
    del /f /s /q "node_modules\.vite\*" 2>nul
    rmdir /s /q "node_modules\.vite" 2>nul
)

echo Vite cache cleared!
echo You can now run: npm run dev