@echo off
REM Quick type-check runner with output saving
REM Saves results with timestamp to type-check-results_TIMESTAMP.txt

echo Running type-check and saving results...
powershell -ExecutionPolicy Bypass -File run-type-check.ps1
pause
