@echo off
echo Installing D3 dependencies for ConceptGraph component...
echo.

cd /d D:\Dev\kha\tori_ui_svelte

echo Installing d3...
call pnpm add d3

echo.
echo Installing @types/d3 as dev dependency...
call pnpm add -D @types/d3

echo.
echo D3 dependencies installed successfully!
echo.
pause
