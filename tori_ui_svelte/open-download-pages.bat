@echo off
echo =====================================
echo   Opening AR Model Download Pages
echo =====================================
echo.
echo This will open 3 browser tabs with the model download pages.
echo.
echo Remember to:
echo   1. Download each model
echo   2. Extract the ZIP files
echo   3. Place files in the /raw directories
echo.
pause

echo Opening CGTrader Sneaker page...
start https://www.cgtrader.com/free-3d-models/sports/equipment/sneaker

timeout /t 2 /nobreak >nul

echo Opening Sketchfab Watch page...
start https://sketchfab.com/3d-models/low-poly-watch-85e3fe10d5e9411fbcd409f7eab6524f

timeout /t 2 /nobreak >nul

echo Opening Sketchfab Perfume page...
start https://sketchfab.com/3d-models/3d-perfume-bottle-for-product-visualizations-6e58cd19fb4f49829411c7f97df14f6e

echo.
echo =====================================
echo   Download pages opened!
echo =====================================
echo.
echo After downloading, place files in:
echo.
echo Sneaker: assets\3d\luxury\sneakers\free\cgtrader_generic_sneaker\raw\
echo Watch:   assets\3d\luxury\watches\free\sketchfab_lowpoly_watch\raw\
echo Perfume: assets\3d\luxury\perfume\free\sketchfab_perfume_bottle\raw\
echo.
echo Then run: process-ar-models.bat
echo.
pause