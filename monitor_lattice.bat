@echo off
echo === OSCILLATOR LATTICE MONITOR ===
echo.
echo This script continuously monitors the oscillator lattice status
echo to help diagnose and confirm the fix is working.
echo.
cd C:\Users\jason\Desktop\tori\kha
poetry run python monitor_lattice.py
echo.
pause
