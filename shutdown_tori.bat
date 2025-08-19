@echo off
REM Quick shutdown script for TORI

echo Shutting down TORI...

REM Kill Python processes
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM pythonw.exe /T 2>nul

REM Kill Node processes
taskkill /F /IM node.exe /T 2>nul

REM Kill by window title
taskkill /FI "WINDOWTITLE eq TORI*" /F 2>nul
taskkill /FI "WINDOWTITLE eq *enhanced_launcher*" /F 2>nul
taskkill /FI "WINDOWTITLE eq Administrator*" /F 2>nul

echo Done!
