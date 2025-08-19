@echo off
echo Running TORI Patch Verification...
echo.
poetry run python verify_patches.py
echo.
echo ==========================================
echo.
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS! All patches applied successfully!
    echo.
    echo You can now launch TORI with:
    echo   poetry run python enhanced_launcher.py --api full --enable-hologram
) else (
    echo Some patches still need to be applied.
    echo Review the output above for details.
)
echo.
pause
