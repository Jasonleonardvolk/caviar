@echo off
REM Batch file to run ψ-Sync tests on Windows
echo Running ψ-Sync Stability Monitoring System tests...

if "%1"=="" (
    echo Running all tests...
    python alan_backend/banksy/run_psi_sync_tests.py all
) else (
    echo Running %1 test...
    python alan_backend/banksy/run_psi_sync_tests.py %1
)

echo Tests completed.
pause
