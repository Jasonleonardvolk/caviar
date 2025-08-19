@echo off
echo Cleaning dev_install.ps1 file...

REM Remove the old file
del scripts\dev_install.ps1 2>nul

REM Create clean file without BOM
powershell -Command "[System.IO.File]::WriteAllText('scripts\dev_install.ps1', (Get-Content 'clean_install.ps1' -Raw), [System.Text.UTF8Encoding]::new($false))"

echo Done! File cleaned.
echo.
echo Run: .\scripts\dev_install.ps1
pause
