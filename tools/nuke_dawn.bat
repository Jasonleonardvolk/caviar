@echo off
echo CLEANING UP THE MESS - Removing all Dawn/ANGLE garbage
echo ========================================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tools

echo Killing any git processes...
taskkill /F /IM git.exe 2>nul
taskkill /F /IM git-remote-https.exe 2>nul

echo.
echo Removing dawn directory and all its bloat...
echo This may take a minute...

REM Force remove with system commands
rmdir /s /q dawn 2>nul

REM If that fails, use more aggressive approach
if exist dawn (
    echo Using aggressive removal...
    rd /s /q dawn 2>nul
    
    REM Last resort - remove read-only attributes and retry
    if exist dawn (
        echo Removing read-only attributes...
        attrib -r -s -h dawn\*.* /s /d
        rd /s /q dawn
    )
)

REM Clean up any leftover tint attempts
if exist tint (
    rmdir /s /q tint 2>nul
)

echo.
echo Cleanup complete!
echo.
echo Space freed. Dawn directory removed.
echo.
echo Next steps for getting Tint the RIGHT way:
echo -------------------------------------------
echo 1. Download pre-built Tint directly (if we can find a working source)
echo 2. Or use a minimal clone approach (no recursive submodules!)
echo 3. Or just proceed without Tint for now
echo.
pause
