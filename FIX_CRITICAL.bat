@echo off
echo Fixing Critical TORI Components...

REM Fix 1: Create data directory and concept_db.json
if not exist "data" mkdir data
if not exist "data\concept_db.json" echo {} > data\concept_db.json
echo [OK] Data directory and concept_db.json created

REM Fix 2: Set PYTHONPATH to include project root for Penrose
set PYTHONPATH=%CD%;%PYTHONPATH%
echo [OK] PYTHONPATH updated for Penrose imports

REM Fix 3: Run the comprehensive fix script
python fix_critical_components.py

echo.
echo All fixes applied! Now restart TORI:
echo   python enhanced_launcher.py
pause
