@echo off
echo 🕸️ Comparing Vault with ConceptMesh...
echo =====================================
echo.

REM Check if mesh file was provided
if "%1"=="" (
    echo Usage: vault_mesh_compare.bat mesh_file.json
    echo.
    echo Example: vault_mesh_compare.bat concept_mesh_data.json
    exit /b 1
)

python "%~dp0\vault_inspector.py" --compare-mesh %1

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Inconsistencies found between Vault and Mesh!
) else (
    echo.
    echo ✅ Vault and Mesh are consistent!
)
