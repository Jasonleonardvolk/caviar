@echo off
echo Testing all fixes for MCP stack (including error_handling.py fix)...
cd C:\Users\jason\Desktop\tori\kha
echo.
echo RUNNING ENHANCED LAUNCHER WITH ALL FIXES APPLIED
echo.
poetry run python enhanced_launcher.py --no-browser
pause
