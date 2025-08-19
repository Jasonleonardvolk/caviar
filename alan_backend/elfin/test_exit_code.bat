@echo off
python alan_backend\elfin\check_circular_refs.py alan_backend\elfin\analyzer\test_files\direct_circular.elfin
echo Exit code: %ERRORLEVEL%
