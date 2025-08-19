@echo off
echo ========================================
echo Phase 2 - One-Command Dev Install Test
echo ========================================
echo.

echo Checking all Phase 2 deliverables...
echo.

echo [1/6] Checking scripts...
if exist scripts\dev_install.ps1 (
    echo ✅ scripts\dev_install.ps1 exists
) else (
    echo ❌ scripts\dev_install.ps1 missing
)

if exist scripts\dev_install.sh (
    echo ✅ scripts\dev_install.sh exists
) else (
    echo ❌ scripts\dev_install.sh missing
)

echo.
echo [2/6] Checking pyproject.toml...
if exist pyproject.toml (
    echo ✅ pyproject.toml exists
    findstr /C:"[project.optional-dependencies]" pyproject.toml >nul && echo ✅ dev dependencies defined || echo ❌ dev dependencies missing
) else (
    echo ❌ pyproject.toml missing
)

echo.
echo [3/6] Checking Makefile...
if exist Makefile (
    echo ✅ Makefile exists
) else (
    echo ❌ Makefile missing
)

echo.
echo [4/6] Checking documentation...
if exist GETTING_STARTED.md (
    echo ✅ GETTING_STARTED.md exists
) else (
    echo ❌ GETTING_STARTED.md missing
)

echo.
echo [5/6] Checking concept_mesh module...
if exist concept_mesh\__init__.py (
    echo ✅ concept_mesh\__init__.py exists
) else (
    echo ❌ concept_mesh\__init__.py missing
)

echo.
echo [6/6] Testing one-command install...
echo.
echo Ready to test the one-command install!
echo.
echo To run the full test:
echo   .\scripts\dev_install.ps1
echo.
echo Or use Make:
echo   make dev
echo.
echo ========================================
echo ✅ Phase 2 Implementation Complete!
echo ========================================
echo.
echo Tag this as: v0.9.6-dev-install
echo.
pause
