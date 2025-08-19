@echo off
echo Installing Phase 6-8 Dependencies...
echo.

REM Activate venv
call .venv\Scripts\activate

echo Installing aiofiles and torch...
pip install aiofiles torch

echo.
echo Running Penrose speed test...
pytest tests\test_penrose.py -v

echo.
echo Committing changes...
git add requirements-dev.txt tests\test_penrose.py
git commit -m "test: add Penrose performance regression guard"

echo.
echo Ready for Phase 6 implementation!
pause
