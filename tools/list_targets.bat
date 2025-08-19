@echo off
echo Checking available CMake targets...
echo ===================================
echo.

cd /d C:\Users\jason\Desktop\tori\kha\tools\dawn\build

echo Listing all available targets:
echo.
cmake --build . --target help

echo.
echo ===================================
echo.
echo Look for any target with 'tint' in the name above.
echo Then run: cmake --build . --config Release --target [TARGET_NAME]
echo.
pause
