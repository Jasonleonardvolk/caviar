@echo off
echo Fixing test_concept_mesh_3 to use correct API port...
echo.

REM Rename original file as backup
if exist test_concept_mesh_3 (
    copy test_concept_mesh_3 test_concept_mesh_3.backup
    echo Created backup: test_concept_mesh_3.backup
)

REM Copy fixed version to original name
copy test_concept_mesh_3_fixed.py test_concept_mesh_3
echo Updated test_concept_mesh_3 with port discovery fix

REM Make it executable (Windows doesn't need chmod, but we'll use Python shebang)
echo.
echo Test script fixed! It will now automatically detect port 8002 from api_port.json
echo.
echo You can now run the test with:
echo   python test_concept_mesh_3
echo.
echo Or with manual port override:
echo   python test_concept_mesh_3 --api-port 8002
echo.
pause
