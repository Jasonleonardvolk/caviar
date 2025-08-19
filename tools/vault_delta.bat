@echo off
echo 📊 Comparing Vault Snapshots...
echo ==============================
echo.

REM Check if both files were provided
if "%2"=="" (
    echo Usage: vault_delta.bat old_snapshot new_snapshot
    echo.
    echo Example: vault_delta.bat vault_20250701.json vault_20250702.json
    echo          vault_delta.bat session_20250701.jsonl session_20250702.jsonl
    exit /b 1
)

python "%~dp0\vault_inspector.py" --delta %1 %2
