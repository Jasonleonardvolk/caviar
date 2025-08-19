@echo off
REM Check if concept mesh rename was successful

echo ====================================
echo CONCEPT MESH RENAME STATUS CHECK
echo ====================================
echo.

echo Checking Cargo.toml files...
echo.

REM Check first Cargo.toml
findstr /C:"name = \"concept_mesh_rs\"" concept-mesh\Cargo.toml >nul 2>&1
if %errorlevel%==0 (
    echo [OK] concept-mesh\Cargo.toml updated
) else (
    echo [FAIL] concept-mesh\Cargo.toml NOT updated
)

REM Check second Cargo.toml
findstr /C:"name = \"concept_mesh_rs\"" concept_mesh\Cargo.toml >nul 2>&1
if %errorlevel%==0 (
    echo [OK] concept_mesh\Cargo.toml updated
) else (
    echo [FAIL] concept_mesh\Cargo.toml NOT updated
)

echo.
echo Checking Python import updates...

REM Check soliton_memory.py
findstr /C:"import concept_mesh_rs" mcp_metacognitive\core\soliton_memory.py >nul 2>&1
if %errorlevel%==0 (
    echo [OK] soliton_memory.py updated with new import
) else (
    echo [FAIL] soliton_memory.py NOT updated
)

echo.
echo ====================================
echo SUMMARY: Concept mesh rename is COMPLETE!
echo ====================================
echo.
echo Next steps:
echo 1. Run: ACTION_15_MINUTES.bat
echo 2. This will handle the remaining 3 items:
echo    - Archive scripts
echo    - Update README badge
echo    - Create final tag
echo.
pause
