@echo off
echo ========================================
echo FIX: Entropy Pruning Module
echo ========================================
echo.

REM Activate venv
call .venv\Scripts\activate

echo Option 1: Installing entropy-prune module...
pip install entropy-prune==0.2.1

echo.
echo Testing import...
python -c "import entropy_prune; print('✅ entropy-prune OK')" || (
    echo.
    echo ❌ entropy-prune installation failed
    echo.
    echo Option 2: Disabling entropy pruning...
    echo Set TORI_DISABLE_ENTROPY_PRUNE=1 to disable
)

echo.
echo ✅ Fix applied!
echo.
echo To disable entropy pruning permanently, add to enhanced_launcher.py:
echo os.environ['TORI_DISABLE_ENTROPY_PRUNE'] = '1'
echo.
pause
