@echo off
echo ================================================================================
echo TESTING WORKING TOPOLOGIES (Avoiding Ramanujan hang)
echo ================================================================================
echo.

cd /d "C:\Users\jason\Desktop\tori\kha"

echo To stop the hanging test, press Ctrl+C
echo.
echo Running test with working topologies...

python test_working_topologies.py

pause
