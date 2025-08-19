@echo off
REM ===================================================
REM   15-MINUTE ACTION PLAN for v0.12.0
REM ===================================================

cls
echo.
echo    ╔══════════════════════════════════════════╗
echo    ║   15-MINUTE FINAL SPRINT for v0.12.0    ║
echo    ╚══════════════════════════════════════════╝
echo.
echo    Everything is GREEN except 4 quick items:
echo.
echo    1. Archive scripts (1 min)
echo    2. Update README badge (2 min)  
echo    3. Rename concept_mesh_rs (2 min)
echo    4. Tag v0.12.0-pre-albert (1 min)
echo.
echo    Ready? Let's do this!
echo.

pause

REM Quick status check
echo.
echo Running quick status check...
python check_merge_gate_status.py
echo.

echo Press any key to run the FINAL MERGE GATE...
pause >nul

REM Run the main script
call FINAL_MERGE_GATE.bat

echo.
echo ════════════════════════════════════════════════
echo.
echo 🎯 ALMOST DONE! Final manual steps:
echo.
echo 1. Update concept-mesh/Cargo.toml:
echo    name = "concept_mesh_rs"
echo.
echo 2. Run: python update_readme_badge.py
echo    (or manually edit README.md)
echo.
echo 3. Push to GitHub:
echo    git push origin main
echo    git push origin v0.12.0-pre-albert
echo.
echo Then create the Albert sprint issue!
echo.
echo ════════════════════════════════════════════════
pause
