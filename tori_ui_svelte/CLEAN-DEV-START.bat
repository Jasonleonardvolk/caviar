@echo off
echo === CLEAN DEV SERVER SETUP (RUN AS ADMIN) ===
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo [1/6] Killing processes...
taskkill /F /IM node.exe 2>nul
taskkill /F /IM vite.exe 2>nul
echo   [OK] Processes terminated

echo.
echo [2/6] Removing old files...
rmdir /S /Q node_modules 2>nul
rmdir /S /Q "node_modules\.ignored" 2>nul
del /F package-lock.json 2>nul
del /F yarn.lock 2>nul
echo   [OK] Old files removed

echo.
echo [3/6] Setting up pnpm...
call corepack enable
call corepack prepare pnpm@latest --activate
echo   [OK] pnpm activated

echo.
echo [4/6] Installing dependencies...
call pnpm install
echo   [OK] Dependencies installed

echo.
echo [5/6] Installing adapters...
call pnpm add -D @sveltejs/adapter-node @sveltejs/adapter-auto
echo   [OK] Adapters installed

echo.
echo [6/6] Syncing SvelteKit...
call pnpm exec svelte-kit sync
echo   [OK] SvelteKit synced

echo.
echo ========================================
echo  SETUP COMPLETE - Starting dev server
echo ========================================
echo.

echo Starting on http://localhost:5173
call pnpm dev -- --host 0.0.0.0 --port 5173