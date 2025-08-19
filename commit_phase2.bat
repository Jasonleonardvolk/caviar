@echo off
echo Committing entropy-prune fix and tagging v0.9.6-dev-install...
echo.

REM Add all changes
git add -A

REM Commit with descriptive message
git commit -m "fix: disable entropy-prune by default with opt-in flag

- Commented out entropy-prune from pyproject.toml and requirements-dev.txt
- Added TORI_DISABLE_ENTROPY_PRUNE=1 to enhanced_launcher.py
- Added env check to pruning.py to raise ImportError when disabled
- Allows system to run without this optional dependency"

REM Tag the release
git tag v0.9.6-dev-install

echo.
echo ✅ Changes committed and tagged!
echo.
echo To push:
echo   git push origin main --tags
echo.
pause
